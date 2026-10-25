"""Model topology and entry binding, independent of device and weight format.

The catalog is the pre-allocation model plan. Actual constructed layers are
checked against it before weights can be materialized. Entry bindings own only
model hooks; invocation, graph and transfer lifetimes belong to the runtime.
"""

from abc import ABC, abstractmethod
from functools import wraps
from weakref import ref

from .config import ExpertCatalog, ExpertLayerSpec


class ModelAdapter(ABC):
    # Some model implementations call a layer method without Module.__call__.
    direct_layer_entries = ()

    @abstractmethod
    def plan_model(self, hf_config, config):
        """Return the immutable ExpertCatalog before full expert allocation."""

    @abstractmethod
    def describe(self, decoder):
        """Validate a constructed layer and return its ExpertLayerSpec."""

    @abstractmethod
    def expert_module(self, decoder):
        """Locate the routed expert module without exposing model structure."""

    def bind_entries(self, runtime, layers):
        return ModelEntryBinding(self, runtime, layers)


class DeepSeekV4Adapter(ModelAdapter):
    direct_layer_entries = ("forward_hc_pre_from_prev",)

    def plan_model(self, hf_config, config):
        if getattr(hf_config, "model_type", "") != "deepseek_v4":
            raise ValueError("The DeepSeek V4 adapter requires a deepseek_v4 model")
        count = hf_config.num_hidden_layers
        if type(count) is not int or count <= 0:
            raise ValueError("The model must have a positive layer count")
        resident = max(3, getattr(hf_config, "num_hash_layers", 0))
        catalog = ExpertCatalog(
            tuple(
                ExpertLayerSpec(
                    layer_id,
                    hf_config.n_routed_experts,
                    config.ep_size,
                    config.ep_rank,
                    resident=layer_id < resident,
                )
                for layer_id in range(count)
            )
        )
        if not catalog.num_experts:
            raise ValueError("The model has no offloadable routed experts")
        return catalog

    def expert_module(self, decoder):
        mlp = getattr(decoder, "mlp", None)
        if mlp is None or not getattr(mlp, "is_deepseek_v4", False):
            raise ValueError("The DeepSeek V4 adapter requires DeepSeek V4 layers")
        return mlp.experts

    def describe(self, decoder):
        experts = self.expert_module(decoder)
        mlp = decoder.mlp
        if experts.moe_tp_size != 1:
            raise ValueError("MoE TP is not supported")
        if getattr(mlp, "is_nextn", False):
            raise ValueError("Speculative/MTP experts are not supported")
        if getattr(experts, "num_fused_shared_experts", 0):
            raise ValueError("Keep shared experts separate and resident")
        spec = ExpertLayerSpec(
            layer_id=mlp.layer_id,
            global_experts=experts._num_global_routed,
            ep_size=experts.moe_ep_size,
            ep_rank=experts.moe_ep_rank,
            resident=mlp.layer_id < 3 or getattr(mlp, "is_hash", False),
        )
        if experts._num_local_routed != spec.local_experts:
            raise ValueError("Elastic/remapped EP ownership is not supported")
        return spec


class ModelEntryBinding:
    """Install and transactionally retire model entries for the shared runtime.

    Hook closures hold a weak runtime reference. The runtime owns this binding;
    removing it restores original instance/class lookup without cyclic GC.
    """

    def __init__(self, adapter, runtime, layers):
        self._runtime = ref(runtime)
        self._hooks, self._methods, self._experts = [], [], []
        self._closed = False
        found = set()
        try:
            for decoder, spec in layers:
                if spec.layer_id in found:
                    raise ValueError("Duplicate model layer entry")
                if spec.layer_id not in runtime.layer_programs:
                    raise ValueError("Model layer has no bound offload program")
                experts = adapter.expert_module(decoder)
                if getattr(experts, "_moe_offload_runtime", None) is not None:
                    raise ValueError("Experts already belong to another offloader")
                found.add(spec.layer_id)
                for name in adapter.direct_layer_entries:
                    original = getattr(decoder, name, None)
                    if original is not None:
                        previous = decoder.__dict__.get(name)
                        self._methods.append(
                            (decoder, name, name in decoder.__dict__, previous)
                        )
                        setattr(decoder, name, self._entry(original, spec.layer_id))
                self._hooks.append(
                    decoder.register_forward_pre_hook(self._pre_hook(spec.layer_id))
                )
                experts._moe_offload_runtime = runtime
                self._experts.append(experts)
            if found != set(runtime.layer_programs):
                raise ValueError("Model does not contain every bound offload layer")
        except BaseException:
            self.close()
            raise

    def _pre_hook(self, layer_id):
        runtime_ref = self._runtime

        def hook(module, args):
            runtime = runtime_ref()
            if runtime is None:
                raise RuntimeError("The offload runtime has been retired")
            runtime._enter_layer(layer_id)

        return hook

    def _entry(self, original, layer_id):
        runtime_ref = self._runtime

        @wraps(original)
        def call(*args, **kwargs):
            runtime = runtime_ref()
            if runtime is None:
                raise RuntimeError("The offload runtime has been retired")
            runtime._enter_layer(layer_id)
            return original(*args, **kwargs)

        return call

    def close(self):
        if self._closed:
            return
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        for module, name, had_instance, previous in reversed(self._methods):
            if had_instance:
                setattr(module, name, previous)
            else:
                delattr(module, name)
        self._methods.clear()
        runtime = self._runtime()
        for experts in self._experts:
            if getattr(experts, "_moe_offload_runtime", None) is runtime:
                del experts._moe_offload_runtime
        self._experts.clear()
        self._closed = True
