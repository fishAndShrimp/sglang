"""Shared NPU bundle allocation, bounded conversion and checkpoint lifecycle."""

import math
import threading
from dataclasses import dataclass
from types import MappingProxyType

from ..formats import BundleComponent, BundleSchema, ExpertFormat
from ..resources import Allocation, ResourceKind
from ..storage import HostExpertStore, HostMemoryCapability
from .storage import _align
from .transfer import DeviceMemoryRegion


@dataclass(frozen=True)
class _Layout:
    raw_shape: tuple
    runtime_shape: tuple
    runtime_stride: tuple
    dtype: object
    nbytes: int


class TorchNpuFormatAdapter:
    """Use only the deployed torch_npu format and actual ACL byte-copy APIs."""

    def __init__(self, device, native_acl):
        import torch
        import torch_npu

        self.torch, self.torch_npu = torch, torch_npu
        self.device = torch.device(device)
        if self.device.type != "npu" or self.device.index is None:
            raise ValueError("Native expert format requires an explicit NPU device")
        if native_acl is None:
            from .native_acl import get_native_acl

            native_acl = get_native_acl()
        self.native_acl = native_acl

    def to_device(self, tensor):
        return tensor.to(self.device)

    def synchronize(self):
        # aclrtMemcpy does not order itself after torch_npu's queued format cast.
        event = self.torch.npu.Event()
        event.record(self.torch.npu.current_stream(self.device))
        event.synchronize()

    def begin_workspace(self):
        baseline = self.torch.npu.memory_allocated(self.device)
        self.torch.npu.reset_peak_memory_stats(self.device)
        return baseline

    def end_workspace(self, baseline):
        return max(0, self.torch.npu.max_memory_allocated(self.device) - baseline)

    def physical_nbytes(self, tensor):
        if tensor.element_size() != 1:
            raise RuntimeError("Opaque native carrier must remain one byte per element")
        # get_storage_size uses the backend descriptor. With byte carriers its
        # element count and byte count coincide; also verify allocator extent.
        descriptor_size = int(self.torch.ops.npu.get_storage_size(tensor))
        allocator_size = tensor.untyped_storage().nbytes()
        if descriptor_size != allocator_size or tensor.storage_offset() != 0:
            raise RuntimeError("NPU format descriptor and physical allocation disagree")
        return allocator_size

    def copy_to_host(self, destination, source, nbytes):
        self.native_acl.memcpy_sync(
            destination,
            nbytes,
            source.untyped_storage().data_ptr(),
            nbytes,
            "device_to_host",
            self.device.index,
        )


class NpuExpertFormat(ExpertFormat):
    """Fixed bank allocation and byte budgets shared by FP4 and INT8 formats."""

    host_component_nbytes = {}

    def _bind_schema(
        self,
        format_id,
        *,
        checkpoint_dtypes,
        shards,
        conversion_groups,
        new_outputs,
        runtime_names,
        dispatch_config,
    ):
        components = tuple(
            BundleComponent(
                name,
                layout.raw_shape,
                layout.runtime_shape,
                layout.runtime_stride,
                layout.dtype,
                layout.nbytes,
                checkpoint_dtypes[name],
                self.host_component_nbytes.get(name, layout.nbytes),
                tuple(shards[name]),
                runtime_names.get(name, name),
                tuple(self.checkpoint_attributes(name).items()),
            )
            for name, layout in self.layouts.items()
        )
        self.schema = BundleSchema(
            format_id,
            components,
            tuple(conversion_groups),
            tuple(new_outputs),
            tuple(dispatch_config.items()),
        )
        self.layouts = MappingProxyType({c.name: c for c in components})
        self.component_nbytes = MappingProxyType({c.name: c.nbytes for c in components})
        self.host_component_nbytes = MappingProxyType(
            {c.name: c.host_nbytes for c in components}
        )
        self.expert_nbytes = self.schema.expert_nbytes

    def checkpoint_view(self, allocation, name, experts):
        import torch

        component = self.layouts[name]
        typed = allocation.view(component.checkpoint_dtype)
        shape = component.raw_shape[1:]
        stride = tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))
        extent = component.host_nbytes // typed.element_size()
        return torch.as_strided(typed, (experts,) + shape, (extent,) + stride)

    def quant_info(self, tensors):
        from sglang.srt.layers.moe.moe_runner.ascend import AscendQuantInfo

        return AscendQuantInfo(
            **{
                component.runtime_name: tensors[component.name]
                for component in self.schema.components
            }
        )

    def validate_master(self, raw_components):
        """Optional format-wide validation before the first in-place conversion."""

    def repack_into(self, raw_components, expert, destinations):
        for names in self.schema.conversion_groups:
            baseline = self.adapter.begin_workspace()
            inputs = tuple(
                self.adapter.to_device(raw_components[name][expert : expert + 1])
                for name in names
            )
            converted = self.adapter.prepare(*inputs)
            self.adapter.synchronize()
            known_bytes = sum(value.numel() * value.element_size() for value in inputs)
            known_bytes += sum(
                self.layouts[name].nbytes
                for name in names
                if name in self.schema.new_outputs
            )
            self._check_workspace(baseline, known_bytes)
            if len(converted) != len(names):
                raise RuntimeError(
                    "Native conversion returned an incomplete projection"
                )
            # Validate the complete projection before overwriting the sole master.
            for name, value in zip(names, converted):
                layout = self.layouts[name]
                if (
                    tuple(value.shape) != layout.runtime_shape
                    or tuple(value.stride()) != layout.runtime_stride
                    or value.dtype != layout.dtype
                    or self.adapter.physical_nbytes(value) != layout.nbytes
                ):
                    raise RuntimeError(
                        "Native conversion changed after the format probe"
                    )
            for name, value in zip(names, converted):
                target = destinations[name][expert]
                self.adapter.copy_to_host(target.host_address, value, target.nbytes)
            del inputs, converted, value

    def _check_workspace(self, baseline, known_bytes):
        measured = self.adapter.end_workspace(baseline)
        self.workspace_nbytes = max(self.workspace_nbytes, known_bytes, measured)
        if (
            self.workspace_limit_bytes is not None
            and self.workspace_nbytes > self.workspace_limit_bytes
        ):
            raise ValueError(
                "Native expert conversion exceeds the loader workspace budget"
            )

    def host_nbytes(self, experts):
        if type(experts) is not int or experts <= 0:
            raise ValueError("Local expert count must be positive")
        return sum(
            _align(layout.host_nbytes * experts, layout.alignment)
            for name, layout in self.layouts.items()
        )

    def allocations(self):
        # Transient loader peak, accounted separately from persistent graph HBM.
        return ()

    def allocate_bank(self, capacity, kind, name):
        if type(capacity) is not int or capacity < 0:
            raise ValueError("Bank capacity must be a nonnegative integer")
        if kind not in (ResourceKind.HOT_WEIGHTS, ResourceKind.EXECUTION_WEIGHTS):
            raise ValueError("Expert banks must have a weight resource kind")
        tensors = {}
        for component, layout in self.layouts.items():
            tensor = self.adapter.empty_component(component, layout, capacity)
            if (
                capacity
                and self.adapter.physical_nbytes(tensor) != capacity * layout.nbytes
            ):
                raise RuntimeError(
                    "Native bank does not have independent fixed expert extents"
                )
            if tuple(tensor.shape) != (capacity,) + layout.runtime_shape[1:]:
                raise RuntimeError(
                    "Native bank shape differs from the selected expert format"
                )
            if capacity and tuple(tensor.stride()[1:]) != layout.runtime_stride[1:]:
                raise RuntimeError("Native bank strides differ from the probed format")
            tensors[component] = tensor
        return ExpertBank(self, tensors, capacity, kind, name)


class ExpertBank:
    def __init__(self, format, tensors, capacity, kind, name, *, owner=None, start=0):
        self.format, self.tensors, self.capacity = format, tensors, capacity
        self.kind, self.name = kind, name
        self._owner, self._start = owner, start
        self.expert_nbytes = format.expert_nbytes
        self.component_regions = {}
        for component, tensor in tensors.items():
            setattr(self, component, tensor)
            extent = format.component_nbytes[component]
            self.component_regions[component] = tuple(
                DeviceMemoryRegion(
                    tensor.data_ptr() + index * extent, extent, tensor.device, tensor
                )
                for index in range(capacity)
            )

    def view(self, start, count):
        if (
            any(type(x) is not int for x in (start, count))
            or start < 0
            or count < 0
            or start + count > self.capacity
        ):
            raise ValueError("Expert view lies outside its fixed bank")
        tensors = {}
        for name, tensor in self.tensors.items():
            value = tensor.narrow(0, start, count)
            if count and (
                value.untyped_storage().data_ptr()
                != tensor.untyped_storage().data_ptr()
                or value.data_ptr()
                != tensor.data_ptr() + start * self.format.component_nbytes[name]
            ):
                raise RuntimeError(
                    "Native expert slicing did not preserve physical byte extents"
                )
            tensors[name] = value
        return ExpertBank(
            self.format, tensors, count, self.kind, self.name, owner=self, start=start
        )

    def slice(self, start, end):
        return self.view(start, end - start)

    def quant_info(self):
        return self.format.quant_info(self.tensors)

    def allocations(self):
        if self._owner is not None:
            return ()
        return tuple(
            Allocation(
                f"{self.name}.{name}",
                self.kind,
                tensor.untyped_storage().nbytes(),
                (str(tensor.device), tensor.untyped_storage().data_ptr()),
                self,
            )
            for name, tensor in self.tensors.items()
            if tensor.numel()
        )


def _sealed_loader(*args, **kwargs):
    # No reference to the storage, layer or checkpoint bookkeeping survives seal.
    raise RuntimeError("Host expert master is not writable")


class ExpertLayerStorage:
    """One checkpoint state machine for every native format recipe."""

    def __init__(self, layer, spec, format, host_pool, extra_weight_attrs):
        import torch

        if spec.resident or getattr(layer, "_moe_offload_storage", None) is not None:
            raise ValueError("Only new offloaded routed experts may use this storage")
        self.layer, self.spec, self.format, self.host_pool = (
            layer,
            spec,
            format,
            host_pool,
        )
        self.expert_nbytes = format.expert_nbytes
        self.host_store = HostExpertStore()
        self.component_capabilities, self.raw_components = {}, {}
        self._loaded, self._checkpoint_state = set(), {}
        self._parameter_names = []
        self._state = "loading"
        self._lock = threading.RLock()
        if any(hasattr(layer, name) for name in format.layouts):
            raise ValueError("Expert parameters already exist before host interception")
        try:
            for name, layout in format.layouts.items():
                host_extent = format.host_component_nbytes.get(name, layout.nbytes)
                allocation = host_pool.allocate(
                    host_extent * spec.local_experts, alignment=layout.alignment
                )
                allocation.tensor.zero_()
                raw = format.checkpoint_view(
                    allocation.tensor, name, spec.local_experts
                )
                parameter = torch.nn.Parameter(raw, requires_grad=False)
                for key, value in extra_weight_attrs.items():
                    if key != "weight_loader":
                        setattr(parameter, key, value)
                parameter.weight_loader = self._loader(name)
                for key, value in layout.attributes:
                    setattr(parameter, key, value)
                layer.register_parameter(name, parameter)
                self._parameter_names.append(name)
                self.raw_components[name] = raw
                self.host_store.add(
                    spec.layer_id,
                    name,
                    allocation.tensor,
                    capability=allocation.capability,
                )
                self.component_capabilities[name] = tuple(
                    HostMemoryCapability(
                        raw.data_ptr() + index * host_extent,
                        layout.nbytes,
                        True,
                        allocation.capability.device_pointer(
                            raw.data_ptr() + index * host_extent, layout.nbytes
                        ),
                        host_pool,
                    )
                    for index in range(spec.local_experts)
                )
        except BaseException:
            self.close()
            raise
        layer._moe_offload_storage = self

    def _loader(self, name):
        def load(
            param,
            loaded_weight,
            weight_name=None,
            shard_id=None,
            expert_id=None,
            **kwargs,
        ):
            with self._lock:
                if self._state != "loading":
                    raise RuntimeError("Host expert master is not writable")
                if (
                    type(expert_id) is not int
                    or not 0 <= expert_id < self.spec.global_experts
                ):
                    raise ValueError(
                        "Checkpoint expert ID is outside the global catalog"
                    )
                if (
                    not self.spec.global_start
                    <= expert_id
                    < self.spec.global_start + self.spec.local_experts
                ):
                    return
                local = self.spec.local_id(expert_id)
                if shard_id not in self.format.layouts[name].shards:
                    raise ValueError(
                        "Checkpoint shard does not match the expert component"
                    )
                key = name, local, shard_id
                if key in self._loaded:
                    raise ValueError("Duplicate expert checkpoint component")
                if loaded_weight.device.type != "cpu":
                    raise ValueError("Bounded loading requires CPU checkpoint tensors")
                metadata = self._checkpoint_state.setdefault(local, {})
                self.format.write_checkpoint(
                    name,
                    self.raw_components[name][local],
                    loaded_weight,
                    shard_id,
                    metadata,
                )
                self._loaded.add(key)

        return load

    def finalize(self):
        with self._lock:
            if self._state == "sealed":
                return
            if self._state != "loading":
                raise RuntimeError(
                    "The host master cannot be finalized after a failed repack or close"
                )
            if (
                len(self._loaded)
                != self.format.schema.required_shards * self.spec.local_experts
            ):
                raise RuntimeError(
                    "Incomplete checkpoint: every expert needs all "
                    f"{self.format.schema.required_shards} component shards"
                )
            self.format.validate_master(self.raw_components)
            try:
                for expert in range(self.spec.local_experts):
                    self.format.repack_into(
                        self.raw_components, expert, self.component_capabilities
                    )
                self.host_store.seal()
                self.host_store.require_device_access()
                if hasattr(self.layer, "dispatcher"):
                    self.layer.dispatcher.set_quant_config(
                        dict(self.format.schema.dispatch_config)
                    )
            except BaseException:
                # A prefix may already contain native bytes; retry is unsafe.
                self._state = "failed"
                raise
            self._state = "sealed"
            for name in self._parameter_names:
                getattr(self.layer, name).weight_loader = _sealed_loader
            self._loaded = self._checkpoint_state = None
            self.raw_components.clear()

    def allocations(self):
        # The model-wide pool reports its physical reservation once.
        return ()

    def close(self):
        """Retire module-visible host views before the caller frees the drained pool."""
        import torch

        with self._lock:
            if self._state == "closed":
                return
            for name in self._parameter_names:
                parameter = getattr(self.layer, name)
                parameter.data = torch.empty(0, dtype=parameter.dtype, device="cpu")
                if hasattr(parameter, "weight_loader"):
                    del parameter.weight_loader
            if getattr(self.layer, "_moe_offload_storage", None) is self:
                del self.layer._moe_offload_storage
            self._parameter_names.clear()
            self.raw_components.clear()
            self.component_capabilities.clear()
            self._loaded = self._checkpoint_state = None
            self.host_store = self.host_pool = self.layer = None
            self._state = "closed"
