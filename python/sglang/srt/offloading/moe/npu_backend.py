"""Assemble native V4 FP4 storage, ACL/MemFabric transport and Ascend execution."""

from __future__ import annotations

from dataclasses import dataclass, fields
from types import SimpleNamespace

from ..transfer import (
    ComposedTransferBackend,
    CopySourceKind,
    DeviceCopyPlan,
    HostCopyPlan,
    HostCopyRegion,
)
from .config import ExpertLayerSpec, Phase, fixed_waves
from .execution import (
    CacheUpdateCoordinator,
    ExecutionRequest,
    ExecutionReservation,
    ExpertWeightAccess,
    SourcePolicy,
)
from .interfaces import MoEBackend
from .npu_config import validate_native_environment
from .npu_runtime import NpuBoundMoERuntime, NpuDeviceRuntime
from .npu_transfer import (
    MemFabricCopyComponent,
    MemFabricDeviceTransferOps,
    MemFabricTransferBindings,
)
from .prefill import BoundedPrefillWeightPipeline, PrefillPipelineConfig
from .residency import BoundExecutionWave, TensorResidencyOps
from .resources import Allocation, ResourceKind
from .runtime import DecodeStateSnapshot
from .state import ResidencyState


def _tensor_allocations(prefix, tensors, kind=ResourceKind.METADATA):
    result, seen = [], set()
    for name, tensor in tensors:
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        if storage.nbytes() and key not in seen:
            seen.add(key)
            result.append(
                Allocation(prefix + name, kind, storage.nbytes(), key, tensor)
            )
    return tuple(result)


@dataclass(frozen=True)
class _PrefillTarget:
    binding: object
    offset: int


class NpuMoEBackend(MoEBackend):
    """One model/rank, fixed expert format, single immutable registered master."""

    def __init__(self, settings, config, hf_config, device):
        import torch

        validate_native_environment()
        self.settings, self.config, self.hf_config = settings, config, hf_config
        self.device = torch.device(device)
        if self.device.type != "npu" or self.device.index is None:
            raise ValueError("Native V4 offload requires an explicit NPU device")
        if getattr(hf_config, "model_type", "") != "deepseek_v4":
            raise ValueError("Native MoE offload supports DeepSeek V4 only")
        self.resident_layers = max(3, getattr(hf_config, "num_hash_layers", 0))
        self.num_layers = hf_config.num_hidden_layers
        self.global_experts = hf_config.n_routed_experts
        if self.num_layers <= self.resident_layers:
            raise ValueError("The model has no offloadable routed experts")
        self.layer_ids = tuple(range(self.resident_layers, self.num_layers))
        self.format = self.host_pool = self.device_runtime = None
        self.layers = {}
        self.runtime = None
        self._unbound_resources = []
        self._closed = False

    def validate_quant_method(self, layer, method):
        """Reject unhandled expert formats before their first weight allocation."""
        if layer.layer_id < self.resident_layers:
            return
        if layer.layer_id not in self.layer_ids:
            raise ValueError("Offloaded expert layer is outside the fixed catalog")
        from sglang.srt.hardware_backend.npu.quantization.fp4_moe_methods import (
            NPUW4A8MXFP4FusedMoEMethod,
        )

        if not isinstance(method, NPUW4A8MXFP4FusedMoEMethod):
            raise ValueError(
                "Native V4 MoE offload requires NPUW4A8MXFP4FusedMoEMethod "
                f"before allocating layer {layer.layer_id}, but resolved "
                f"{type(method).__name__}. Disable FP4 dequantization and "
                "expert quantization overrides; Ascend arch35 is required."
            )

    def prepare_npu_fp4_weights(
        self,
        layer,
        *,
        method,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        extra_weight_attrs,
    ):
        from .npu_storage import MemFabricHostPool
        from .v4_format import NpuV4FP4Format, V4LayerStorage

        layer_id = layer.layer_id
        if layer_id < self.resident_layers:
            return False
        if self.runtime is not None or self._closed or layer_id in self.layers:
            raise RuntimeError("Expert storage must be registered exactly once")
        spec = ExpertLayerSpec(
            layer_id, self.global_experts, self.config.ep_size, self.config.ep_rank
        )
        if (
            layer_id not in self.layer_ids
            or num_experts != spec.local_experts
            or layer.moe_tp_size != 1
            or layer.num_fused_shared_experts
        ):
            raise ValueError(
                "Native V4 storage requires unchanged local routed expert ownership"
            )
        dimensions = (hidden_size, intermediate_size_per_partition)
        if self.format is None:
            from .native_acl import get_native_acl

            self.native_acl = get_native_acl(self.settings.native_extension_directory)
            self.device_runtime = NpuDeviceRuntime(self.device)
            # One-expert NZ format probe and conversion workspace, never E experts.
            self.format = NpuV4FP4Format(
                *dimensions,
                self.device,
                native_acl=self.native_acl,
                workspace_limit_bytes=self.settings.workspace_limit_bytes,
            )
            self._dimensions = dimensions
            required = len(self.layer_ids) * self.format.host_nbytes(spec.local_experts)
            self.host_pool = MemFabricHostPool(
                required,
                self.device.index,
                physical_bytes=self.settings.host_pool_gib * 1024**3,
                native_acl=self.native_acl,
            )
        elif dimensions != self._dimensions:
            raise ValueError(
                "Offloaded V4 layers must share one physical expert format"
            )
        storage = V4LayerStorage(
            layer, spec, self.format, self.host_pool, extra_weight_attrs
        )
        self.layers[layer_id] = storage
        return True

    def prepare_layer(self, module, spec):
        storage = self.layers.get(spec.layer_id)
        if (
            storage is None
            or getattr(module.mlp.experts, "_moe_offload_storage", None) is not storage
        ):
            raise RuntimeError("Routed weights were not intercepted before allocation")
        if storage.spec != spec:
            raise ValueError("Constructed expert ownership differs from the catalog")

    def finalize(self, layers, catalog, config, policy):
        import torch
        from .native_acl import make_acl_host_ops
        from .npu_experts import NPUExpertBank, build_npu_layer_programs

        if self.runtime is not None or self._closed:
            raise RuntimeError("Backend already finalized or closed")
        if set(self.layers) != {s.layer_id for s in catalog.offloaded_layers}:
            raise ValueError("Incomplete native expert storage registration")
        for storage in self.layers.values():
            storage.host_store.require_device_access()
        k = config.hot_slots(catalog)
        state = ResidencyState.allocate(
            catalog.num_experts, k, config.execution_slots, self.device
        )
        error = torch.zeros((), dtype=torch.int32, device=self.device)
        hot = self.format.allocate_bank(k, ResourceKind.HOT_WEIGHTS, "hot")
        execution = self.format.allocate_bank(
            config.execution_slots, ResourceKind.EXECUTION_WEIGHTS, "execution"
        )
        self._unbound_resources.extend((hot, execution))
        component_names = tuple(self.format.component_nbytes)
        hosts = {
            name: tuple(
                cap
                for spec in catalog.offloaded_layers
                for cap in self.layers[spec.layer_id].component_capabilities[name]
            )
            for name in component_names
        }
        bindings = []

        def bind(bank, capacity, name, valid=None):
            components = tuple(
                MemFabricCopyComponent(
                    hosts[key],
                    hot.component_regions[key],
                    bank.component_regions[key],
                    self.format.component_nbytes[key],
                    key,
                )
                for key in component_names
            )
            result = MemFabricTransferBindings(
                components,
                capacity=capacity,
                hot_generations=state.slot_generation,
                error_status=error,
                device=self.device,
                name=name,
            )
            if valid is not None:
                result.expert_bank = NPUExpertBank(bank.quant_info(), valid, error)
            bindings.append(result)
            return result

        update = bind(hot, k, "hot.update")
        bound_waves, waves_by_layer = [], {}
        for spec in catalog.offloaded_layers:
            waves = fixed_waves(spec.local_experts, config.execution_slots)
            waves_by_layer[spec.layer_id] = waves
            for wave in waves:
                count = wave.local_end - wave.local_start
                bank = execution.view(0, count)
                transfer_binding = bind(
                    bank,
                    count,
                    f"decode.{spec.layer_id}.{wave.index}",
                    state.execution_valid[:count],
                )
                bound_waves.append(
                    BoundExecutionWave(spec.layer_id, wave, 0, transfer_binding)
                )

        # Prefill borrows the hot pool when available. For ratio=0, partition the
        # explicitly budgeted execution allocation into non-overlapping A/B.
        group = min(
            self.settings.prefill_group_size, catalog.offloaded_layers[0].local_experts
        )
        if k >= group:
            a = hot.view(0, min(k, catalog.offloaded_layers[0].local_experts))
            b = execution.view(0, group)
        elif config.execution_slots >= 2 * group:
            a = execution.view(0, config.execution_slots - group)
            b = execution.view(config.execution_slots - group, group)
        else:
            raise ValueError(
                "Prefill requires a hot A bank or execution_slots >= 2 * prefill_group_size"
            )
        prefill_banks = {0: a, 1: b}
        prefill_valid = {
            index: torch.ones(bank.capacity, dtype=torch.bool, device=self.device)
            for index, bank in prefill_banks.items()
        }

        def prefill_binding(bank_id, offset, count):
            bank = prefill_banks[bank_id].view(offset, count)
            return SimpleNamespace(
                bank=bank,
                bank_id=bank_id,
                offset=offset,
                count=count,
                error_status=error,
                hot_generations=state.slot_generation,
                expert_bank=NPUExpertBank(
                    bank.quant_info(),
                    prefill_valid[bank_id][offset : offset + count],
                    error,
                ),
            )

        def make_request(layer_id, wave, bank_offset, context):
            count = wave.local_end - wave.local_start
            binding = prefill_binding(wave.bank, bank_offset, count)
            return ExecutionRequest(
                layer_id,
                _PrefillTarget(binding, bank_offset),
                wave,
                Phase.PREFILL,
                SourcePolicy.HOST_ONLY,
            )

        def prepare_prefill(request, context):
            target = request.local_demand
            if not isinstance(target, _PrefillTarget):
                raise TypeError("Prefill requires a bounded physical bank target")
            bank = target.binding.bank
            storage = self.layers[request.layer_id]
            regions = []
            for name in component_names:
                for lane, expert in enumerate(
                    range(request.wave.local_start, request.wave.local_end)
                ):
                    source = storage.component_capabilities[name][expert]
                    destination = bank.component_regions[name][lane]
                    nbytes = self.format.component_nbytes[name]
                    regions.append(
                        HostCopyRegion(
                            source,
                            destination,
                            0,
                            0,
                            nbytes,
                            source.nbytes,
                            destination.nbytes,
                        )
                    )
            return ExecutionReservation(
                HostCopyPlan(tuple(regions), (storage, bank)), target.binding
            )

        def combine_bindings(layer_id, wave, leases, context):
            offset = 0
            for lease in leases:
                binding = lease.bindings
                if binding.bank_id != wave.bank or binding.offset != offset:
                    raise ValueError(
                        "Initial expert microbatches must occupy adjacent A slots"
                    )
                offset += binding.count
            if offset != wave.local_end - wave.local_start:
                raise ValueError("Incomplete first prefill group")
            return prefill_binding(wave.bank, 0, offset)

        ops = TensorResidencyOps(
            state,
            catalog,
            execution_waves=bound_waves,
            update_bindings=update,
            error_status=error,
            prepare_prefill=prepare_prefill,
        )

        # Native operations retain into the current context, including worker
        # submissions. Capture heads install a new context before any program.
        def retain(resource):
            self.runtime.context.retain(resource)

        host_ops = make_acl_host_ops(
            retain=retain, module=self.native_acl, source_location="unregistered"
        )
        device_ops = MemFabricDeviceTransferOps(retain=retain)
        transfer = ComposedTransferBackend(host_ops=host_ops, device_ops=device_ops)
        access = ExpertWeightAccess(ops, transfer)
        policy_state = policy.allocate_state(catalog.num_experts, self.device)
        targets = torch.full((k,), -1, dtype=torch.int64, device=self.device)
        updates = CacheUpdateCoordinator(policy, policy_state, state, targets, access)
        snapshot = DecodeStateSnapshot(state, policy, policy_state, error_status=error)
        restore_plan = DeviceCopyPlan(
            torch.full(
                (k,), int(CopySourceKind.HOST), dtype=torch.int64, device=self.device
            ),
            torch.full((k,), -1, dtype=torch.int64, device=self.device),
            torch.arange(k, dtype=torch.int64, device=self.device),
            torch.zeros(k, dtype=torch.bool, device=self.device),
            torch.full((k,), -1, dtype=torch.int64, device=self.device),
            torch.zeros((), dtype=torch.int32, device=self.device),
            k,
        )

        def reload_weights(saved, context):
            restore_plan.source_index.copy_(saved.slot_to_key)
            restore_plan.valid.copy_(saved.slot_valid)
            restore_plan.count.copy_(restore_plan.valid.sum(dtype=torch.int32))
            ready = context.record_compute_completion()
            ready.completion.wait_on(context.copy_stream)
            return transfer.materialize_into(restore_plan, update, context.copy_stream)

        programs, pipelines, program_resources = {}, [], []
        shared_layout = None
        for module, spec in layers:
            pipeline = BoundedPrefillWeightPipeline(
                access,
                PrefillPipelineConfig(
                    local_experts=spec.local_experts,
                    initial_capacity=a.capacity,
                    group_size=group,
                    secondary_capacity=b.capacity,
                    microbatch_size=self.settings.prefill_microbatch_size,
                    max_inflight=self.settings.prefill_max_inflight,
                ),
                make_request=make_request,
                combine_bindings=combine_bindings,
                worker_scope=lambda ctx: self.device_runtime.stream_scope(
                    ctx.copy_stream
                ),
                record_copy_completion=lambda ctx: self.device_runtime.record(
                    ctx.copy_stream
                ),
            )
            programs[spec.layer_id] = build_npu_layer_programs(
                module.mlp.experts,
                access=access,
                residency=ops,
                decode_waves=waves_by_layer[spec.layer_id],
                pipeline=pipeline,
                max_dispatch_rows=self.settings.max_dispatch_rows,
                device=self.device,
                shared_layout=shared_layout,
            )
            pipelines.append(pipeline)
            if shared_layout is None:
                shared_layout = programs[spec.layer_id].layout
                program_resources.extend(programs[spec.layer_id].allocations())
        resources = (
            list(self.host_pool.allocations())
            + list(hot.allocations())
            + list(execution.allocations())
        )
        resources += (
            list(state.allocations())
            + list(ops.allocations())
            + list(snapshot.allocations())
        )
        for binding in bindings:
            resources.extend(binding.allocations())
        tensors = [("error", error), ("target_keys", targets)]
        tensors.extend(
            ("policy." + f.name, getattr(policy_state, f.name))
            for f in fields(policy_state)
        )
        tensors.extend(
            ("restore." + f.name, getattr(restore_plan, f.name))
            for f in fields(restore_plan)
            if f.name != "capacity"
        )
        tensors.extend(
            (f"prefill.valid.{key}", tensor) for key, tensor in prefill_valid.items()
        )
        resources.extend(_tensor_allocations("npu.", tensors))
        resources.extend(program_resources)
        self.runtime = NpuBoundMoERuntime(
            device_runtime=self.device_runtime,
            residency=state,
            residency_ops=ops,
            cache_updates=updates,
            snapshot=snapshot,
            reload_weights=reload_weights,
            layer_programs=programs,
            resources=resources,
            expert_nbytes=self.format.expert_nbytes,
            storage=tuple(self.layers.values()),
            pipelines=pipelines,
            release=self._release,
        )
        return self.runtime

    def _release(self):
        for storage in self.layers.values():
            storage.close()
        if self.host_pool is not None:
            self.host_pool.close()
        self._unbound_resources.clear()
        self._closed = True

    def close_unbound(self):
        if self._closed:
            return
        if self.device_runtime is not None:
            self.device_runtime.synchronize()
        self._release()
