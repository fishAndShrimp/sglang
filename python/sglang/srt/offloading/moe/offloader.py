"""BaseOffloader facade. Backend injection is deliberate until M0 is validated."""

import logging

from ..base import BaseOffloader
from .config import ExpertCatalog
from .models import DeepSeekV4Adapter
from .owner import MoEOffloadOwner
from .periodic_lfu import PeriodicLFUPolicy
from .policy import TensorLFUPolicy

logger = logging.getLogger(__name__)


class MoEOffloaderV2(BaseOffloader):
    """Shares V2's framework contract, not its whole-module rotation policy.

    No default accelerator backend is installed. prepare_layer must arrange
    bounded host loading; finalize must supply a full-graph-capable binding.
    """

    def __init__(
        self, config, backend, *, adapter=None, policy=None, periodic_lfu_config=None
    ):
        if policy is not None and periodic_lfu_config is not None:
            raise ValueError("Supply either policy or periodic_lfu_config, not both")
        self.config = config
        self.backend = backend
        bound_adapter = getattr(backend, "model_adapter", None)
        if (
            adapter is not None
            and bound_adapter is not None
            and adapter is not bound_adapter
        ):
            raise ValueError(
                "Use the backend's model adapter for registration and loading"
            )
        self.adapter = adapter or bound_adapter or DeepSeekV4Adapter()
        self.policy = policy or TensorLFUPolicy()
        self.periodic_lfu_config = periodic_lfu_config
        self.owner = None
        self._runtime = None
        self._layers = None
        self._catalog = None
        self._failed = False
        self._installed = False
        self._closed = False

    def wrap_modules(
        self,
        all_modules_generator,
        submodule_accessor=None,
        whitelist_param_names_creator=None,
    ):
        if self._layers is not None or self._failed or self._closed:
            raise RuntimeError("MoE modules may only be registered once")
        if submodule_accessor is not None or whitelist_param_names_creator is not None:
            raise ValueError("MoE registration uses expert formats, not V2 whitelists")
        modules, specs, offloaded = [], [], []
        model_plan = getattr(self.backend, "model_plan", None)
        try:
            for module in all_modules_generator:
                spec = self.adapter.describe(module)
                if (spec.ep_size, spec.ep_rank) != (
                    self.config.ep_size,
                    self.config.ep_rank,
                ):
                    raise ValueError("Model and offloader EP topology disagree")
                if model_plan is not None and model_plan.layer(spec.layer_id) != spec:
                    raise ValueError(
                        "Constructed expert layer differs from the model plan"
                    )
                if any(existing.layer_id == spec.layer_id for existing in specs):
                    raise ValueError("Duplicate expert layer")
                specs.append(spec)
                logger.debug(
                    "MoE offload layer.register: rank=%s layer=%s resident=%s "
                    "local_experts=%s global_start=%s",
                    spec.ep_rank,
                    spec.layer_id,
                    spec.resident,
                    spec.local_experts,
                    spec.global_start,
                )
                if not spec.resident:
                    self.backend.prepare_layer(module, spec)
                    offloaded.append((module, spec))
                modules.append(module)
            self._catalog = ExpertCatalog(tuple(specs))
            if model_plan is not None and self._catalog != model_plan:
                raise ValueError(
                    "Constructed expert catalog differs from the model plan"
                )
            if not self._catalog.num_experts:
                raise ValueError("No eligible routed experts to offload")
            self._layers = tuple(offloaded)
        except BaseException:
            self._failed = True
            self.backend.close_unbound()
            raise
        return modules

    def post_init(self):
        if (
            self.owner is not None
            or self._layers is None
            or self._failed
            or self._closed
        ):
            raise RuntimeError("Register modules before finalizing exactly once")
        runtime = None
        try:
            if self.periodic_lfu_config is not None:
                # Layer registration determines rank-local keys. Bind the policy
                # only now, before the backend allocates state or captures ops.
                self.policy = PeriodicLFUPolicy(self.periodic_lfu_config, self._catalog)
            runtime = self.backend.finalize(
                self._layers, self._catalog, self.config, self.policy
            )
            self._runtime = runtime
            self.owner = MoEOffloadOwner(self.config, self._catalog, runtime)
            logger.info(
                "MoE offload ready: rank=%s offloaded_layers=%s experts=%s policy=%s "
                "memory_bytes=%s",
                self.config.ep_rank,
                len(self._layers),
                self._catalog.num_experts,
                type(self.policy).__name__,
                {
                    kind.value: used
                    for kind, used in self.owner.allocation_ledger.usage.items()
                },
            )
        except BaseException:
            self._failed = True
            if runtime is not None:
                runtime.close()
                self._runtime = None
            else:
                self.backend.close_unbound()
            raise

    def bind_model_runner(self, model_runner):
        if self.owner is None or self._installed or self._failed or self._closed:
            raise RuntimeError("Finalize before binding model hooks exactly once")
        try:
            # Existing weight-update entry points already honor this model-owned
            # guard. The backend must additionally reject transaction preparation
            # paths before they can rebind/repack captured weights.
            model = model_runner.model
            if getattr(model, "_derived_weight_cache_error", None) is None:
                model._derived_weight_cache_error = (
                    "Online weight updates are not supported with MoE offload; "
                    "pinned host weights and captured expert bindings are immutable."
                )
            self.owner.runtime.install_model_hooks(self, model_runner)
            self._installed = True
            logger.info("MoE offload model hooks bound: rank=%s", self.config.ep_rank)
        except BaseException:
            self._failed = True
            # Binding is part of startup. A failed hook transaction must also
            # retire the already-finalized native resources. Failed drain keeps
            # the owner/runtime reachable for an explicit close retry.
            self.close()
            raise

    def bind_graph_backend(self, backend, graph_runner):
        if not self._installed or self._failed or self._closed:
            raise RuntimeError("Bind MoE model hooks before graph capture")
        from .graph import GraphLeaseBackend

        return GraphLeaseBackend(backend, self.owner, graph_runner)

    def forward_scope(self, forward_batch):
        if not self._installed or self._failed or self._closed:
            raise RuntimeError("MoE runtime/model hooks have not been bound")
        return self.owner.runtime.forward_scope(self.owner, forward_batch)

    @property
    def forbid_copy_engine_usage(self):
        return self.backend.forbid_copy_engine_usage

    def close(self):
        if self._closed:
            return
        if self.owner is not None:
            self.owner.close()
        elif self._runtime is not None:
            self._runtime.close()
        else:
            self.backend.close_unbound()
        self._runtime = None
        self._closed = True
