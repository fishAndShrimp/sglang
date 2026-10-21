"""Bounded loading of the repository's native V4 MXFP4 expert format.

Checkpoint uint8 FP4 and E8M0 bytes are written directly to the registered host
master. Each expert is converted with the same prepare_w4a8_mxfp_weight used by
the resident NPU path, then its physical NZ bytes are copied back in place.
No Python dequantization or full-layer NPU staging is performed. The carrier
dtype is not a claim about numeric precision: the native GMM interprets FP4.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from types import MappingProxyType

from .npu_storage import _align
from .npu_transfer import DeviceMemoryRegion
from .resources import Allocation, ResourceKind
from .storage import HostExpertStore, HostMemoryCapability

_WEIGHTS = ("w13_weight", "w2_weight")
_SCALES = ("w13_weight_scale_inv", "w2_weight_scale_inv")


@dataclass(frozen=True)
class _Layout:
    raw_shape: tuple
    runtime_shape: tuple
    runtime_stride: tuple
    dtype: object
    nbytes: int


class _TorchNpuFormatAdapter:
    """Use only the deployed torch_npu format and actual ACL byte-copy APIs."""

    def __init__(self, device, native_acl):
        import torch
        import torch_npu

        self.torch, self.torch_npu = torch, torch_npu
        self.device = torch.device(device)
        if self.device.type != "npu" or self.device.index is None:
            raise ValueError("Native V4 format requires an explicit NPU device")
        if native_acl is None:
            from .native_acl import get_native_acl

            native_acl = get_native_acl()
        self.native_acl = native_acl

    def to_device(self, tensor):
        return tensor.to(self.device)

    def prepare(self, weight, scale):
        from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
            prepare_w4a8_mxfp_weight,
        )
        from sglang.srt.hardware_backend.npu.utils import NPUACLFormat

        converted, scales = prepare_w4a8_mxfp_weight(
            weight, scale, npu_format=NPUACLFormat.ACL_FORMAT_FRACTAL_NZ
        )
        if int(self.torch_npu.get_npu_format(converted)) != 29:
            raise RuntimeError("Native FP4 conversion did not produce FRACTAL_NZ")
        return converted, scales

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
            raise RuntimeError("V4 native carrier must remain one byte per element")
        # get_storage_size uses the backend descriptor. With byte carriers its
        # element count and byte count coincide; also verify allocator extent.
        descriptor_size = int(self.torch.ops.npu.get_storage_size(tensor))
        allocator_size = tensor.untyped_storage().nbytes()
        if descriptor_size != allocator_size or tensor.storage_offset() != 0:
            raise RuntimeError("NPU format descriptor and physical allocation disagree")
        return allocator_size

    def empty_component(self, name, layout, capacity):
        torch = self.torch
        if capacity == 0:
            return torch.empty(
                (0,) + layout.runtime_shape[1:], dtype=layout.dtype, device=self.device
            )
        if name in _WEIGHTS:
            # The resident path transposes AFTER npu_format_cast. Allocate the
            # exact probed pre-transpose shape and carrier dtype, then same view.
            shape = (capacity,) + layout.runtime_shape[1:]
            shape = shape[:-2] + (shape[-1], shape[-2])
            return torch.ops.npu.empty_with_format(
                list(shape), dtype=layout.dtype, device=self.device, acl_format=29
            ).transpose(-1, -2)
        raw = torch.empty(
            (capacity,) + layout.raw_shape[1:], dtype=layout.dtype, device=self.device
        )
        return _scale_view(raw)

    def copy_to_host(self, destination, source, nbytes):
        self.native_acl.memcpy_sync(
            destination,
            nbytes,
            source.untyped_storage().data_ptr(),
            nbytes,
            "device_to_host",
            self.device.index,
        )


def _scale_view(raw):
    e, n, k32 = raw.shape
    if k32 % 2:
        raise ValueError("Native V4 scales require paired block-32 exponents")
    return raw.view(e, n, k32 // 2, 2).transpose(1, 2)


class NpuV4FP4Format:
    """One-expert native probe, fixed bank descriptors, bounded repacking.

    ``adapter`` is an explicit dependency-injection seam for CPU contract tests;
    the production default always imports torch_npu and fails on unsupported
    layouts. Probed metadata, never probe weights, remain resident in this object.
    """

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        device,
        *,
        native_acl=None,
        adapter=None,
        workspace_limit_bytes=None,
    ):
        import torch

        if any(
            type(d) is not int or d <= 0 or d % 64
            for d in (hidden_size, intermediate_size)
        ):
            raise ValueError(
                "V4 padded hidden/intermediate dimensions must divide by 64"
            )
        self.hidden_size, self.intermediate_size = hidden_size, intermediate_size
        self.device = device
        self.adapter = adapter or _TorchNpuFormatAdapter(device, native_acl)
        self.workspace_limit_bytes = workspace_limit_bytes
        raw_shapes = {
            _WEIGHTS[0]: (1, 2 * intermediate_size, hidden_size // 2),
            _WEIGHTS[1]: (1, hidden_size, intermediate_size // 2),
            _SCALES[0]: (1, 2 * intermediate_size, hidden_size // 32),
            _SCALES[1]: (1, hidden_size, intermediate_size // 32),
        }
        # Reject impossible budgets before even the first H2D input allocation.
        input_peak = max(
            math.prod(raw_shapes[w]) + math.prod(raw_shapes[s])
            for w, s in zip(_WEIGHTS, _SCALES)
        )
        if workspace_limit_bytes is not None and input_peak > workspace_limit_bytes:
            raise ValueError("One-expert V4 input exceeds the loader workspace budget")
        self.layouts = {}
        self.workspace_nbytes = 0
        for weight_name, scale_name in zip(_WEIGHTS, _SCALES):
            baseline = self._begin_workspace()
            raw_weight = torch.zeros(
                raw_shapes[weight_name], dtype=torch.uint8, device="cpu"
            )
            raw_scale = torch.zeros(
                raw_shapes[scale_name], dtype=torch.uint8, device="cpu"
            )
            device_weight = self.adapter.to_device(raw_weight)
            device_scale = self.adapter.to_device(raw_scale)
            weight, scale = self.adapter.prepare(device_weight, device_scale)
            self.adapter.synchronize()
            self._record_layout(weight_name, raw_shapes[weight_name], weight)
            self._record_layout(scale_name, raw_shapes[scale_name], scale)
            if (
                scale.untyped_storage().data_ptr()
                != device_scale.untyped_storage().data_ptr()
            ):
                raise RuntimeError(
                    "Native V4 scale reshape unexpectedly copied storage"
                )
            peak = (
                math.prod(raw_shapes[weight_name])
                + math.prod(raw_shapes[scale_name])
                + self.layouts[weight_name].nbytes
            )
            self._check_workspace(baseline, peak)
            del weight, scale, device_weight, device_scale, raw_weight, raw_scale
        self.component_nbytes = MappingProxyType(
            {name: value.nbytes for name, value in self.layouts.items()}
        )
        self.expert_nbytes = sum(self.component_nbytes.values())

    def _begin_workspace(self):
        begin = getattr(self.adapter, "begin_workspace", None)
        return begin() if begin is not None else None

    def _check_workspace(self, baseline, known_bytes):
        measured = self.adapter.end_workspace(baseline) if baseline is not None else 0
        self.workspace_nbytes = max(self.workspace_nbytes, known_bytes, measured)
        if (
            self.workspace_limit_bytes is not None
            and self.workspace_nbytes > self.workspace_limit_bytes
        ):
            raise ValueError("Native V4 conversion exceeds the loader workspace budget")

    def _record_layout(self, name, raw_shape, tensor):
        nbytes = self.adapter.physical_nbytes(tensor)
        if nbytes < math.prod(raw_shape) or nbytes % 64:
            raise RuntimeError(
                "Native component cannot hold the aligned checkpoint bytes"
            )
        self.layouts[name] = _Layout(
            raw_shape, tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, nbytes
        )

    def host_nbytes(self, experts):
        if type(experts) is not int or experts <= 0:
            raise ValueError("Local expert count must be positive")
        return sum(
            _align(layout.nbytes * experts, 64) for layout in self.layouts.values()
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
                    "Native bank shape differs from the resident FP4 format"
                )
            if capacity and tuple(tensor.stride()[1:]) != layout.runtime_stride[1:]:
                raise RuntimeError("Native bank strides differ from the probed format")
            tensors[component] = tensor
        return V4ExpertBank(self, tensors, capacity, kind, name)

    def repack_into(self, raw_components, expert, destinations):
        for weight_name, scale_name in zip(_WEIGHTS, _SCALES):
            baseline = self._begin_workspace()
            weight_input = self.adapter.to_device(
                raw_components[weight_name][expert : expert + 1]
            )
            scale_input = self.adapter.to_device(
                raw_components[scale_name][expert : expert + 1]
            )
            weight, scale = self.adapter.prepare(weight_input, scale_input)
            self.adapter.synchronize()
            known_bytes = (
                math.prod(self.layouts[weight_name].raw_shape)
                + math.prod(self.layouts[scale_name].raw_shape)
                + self.layouts[weight_name].nbytes
            )
            self._check_workspace(baseline, known_bytes)
            # ABI drift must fail before overwriting the only host master.
            for name, tensor in ((weight_name, weight), (scale_name, scale)):
                layout = self.layouts[name]
                if (
                    tuple(tensor.shape) != layout.runtime_shape
                    or tensor.dtype != layout.dtype
                    or tuple(tensor.stride()) != layout.runtime_stride
                    or self.adapter.physical_nbytes(tensor) != layout.nbytes
                ):
                    raise RuntimeError(
                        "Native V4 conversion changed after the format probe"
                    )
            for name, tensor in ((weight_name, weight), (scale_name, scale)):
                destination = destinations[name][expert]
                self.adapter.copy_to_host(
                    destination.host_address, tensor, destination.nbytes
                )
            del weight, scale, weight_input, scale_input


class V4ExpertBank:
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
                    tensor.data_ptr() + index * extent, extent, tensor.device, self
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
        return V4ExpertBank(
            self.format, tensors, count, self.kind, self.name, owner=self, start=start
        )

    def slice(self, start, end):
        return self.view(start, end - start)

    def quant_info(self):
        from sglang.srt.layers.moe.moe_runner.ascend import AscendQuantInfo

        return AscendQuantInfo(
            w13_weight=self.w13_weight,
            w2_weight=self.w2_weight,
            w13_weight_scale=self.w13_weight_scale_inv,
            w2_weight_scale=self.w2_weight_scale_inv,
        )

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


class V4LayerStorage:
    """Host-backed parameters and strict per-expert checkpoint completeness."""

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
        self._loaded, self._intermediate_sizes = set(), {}
        self._sealed = self._failed = self._closed = False
        self._lock = threading.RLock()
        if any(hasattr(layer, name) for name in format.layouts):
            raise ValueError("Expert parameters already exist before host interception")
        try:
            for name, layout in format.layouts.items():
                allocation = host_pool.allocate(layout.nbytes * spec.local_experts)
                allocation.tensor.zero_()
                raw_shape = (spec.local_experts,) + layout.raw_shape[1:]
                inner_stride = (raw_shape[-1], 1)
                raw = torch.as_strided(
                    allocation.tensor, raw_shape, (layout.nbytes,) + inner_stride
                )
                parameter = torch.nn.Parameter(raw, requires_grad=False)
                for key, value in extra_weight_attrs.items():
                    if key != "weight_loader":
                        setattr(parameter, key, value)
                parameter.weight_loader = self._loader(name)
                if name in _SCALES:
                    # FusedMoeWeightScaleSupported.BLOCK.value; no heavy runner import.
                    parameter.quant_method = "block"
                    parameter.format_ue8m0 = False
                layer.register_parameter(name, parameter)
                self.raw_components[name] = raw
                self.host_store.add(
                    spec.layer_id,
                    name,
                    allocation.tensor,
                    capability=allocation.capability,
                )
                self.component_capabilities[name] = tuple(
                    HostMemoryCapability(
                        raw.data_ptr() + index * layout.nbytes,
                        layout.nbytes,
                        True,
                        allocation.capability.device_pointer(
                            raw.data_ptr() + index * layout.nbytes, layout.nbytes
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
        def load_unlocked(
            param,
            loaded_weight,
            weight_name=None,
            shard_id=None,
            expert_id=None,
            **kwargs,
        ):
            import torch

            if self._sealed or self._failed or self._closed:
                raise RuntimeError("Host expert master is not writable")
            if (
                type(expert_id) is not int
                or not 0 <= expert_id < self.spec.global_experts
            ):
                raise ValueError("Checkpoint expert ID is outside the global catalog")
            if (
                not self.spec.global_start
                <= expert_id
                < self.spec.global_start + self.spec.local_experts
            ):
                return
            local = self.spec.local_id(expert_id)
            allowed = ("w1", "w3") if name.startswith("w13_") else ("w2",)
            if shard_id not in allowed:
                raise ValueError("Checkpoint shard does not match the expert component")
            key = name, local, shard_id
            if key in self._loaded:
                raise ValueError("Duplicate expert checkpoint component")
            if loaded_weight.device.type != "cpu":
                raise ValueError("Bounded V4 loading requires CPU checkpoint tensors")
            if (
                loaded_weight.dtype == getattr(torch, "float8_e8m0fnu", None)
                and name in _SCALES
            ):
                loaded_weight = loaded_weight.view(torch.uint8)
            if loaded_weight.dtype != torch.uint8 or loaded_weight.ndim != 2:
                raise ValueError(
                    "Expected two-dimensional packed FP4/E8M0 checkpoint bytes"
                )
            target = self.raw_components[name][local]
            h, intermediate = self.format.hidden_size, self.format.intermediate_size
            if shard_id in ("w1", "w3"):
                expected_columns = h // (32 if name in _SCALES else 2)
                original_i = loaded_weight.shape[0]
                if (
                    loaded_weight.shape[1] != expected_columns
                    or not 0 < original_i <= intermediate
                ):
                    raise ValueError(
                        "Checkpoint gate/up shape differs from the V4 format"
                    )
                target = target.narrow(
                    0, 0 if shard_id == "w1" else intermediate, original_i
                )
            else:
                original_i = loaded_weight.shape[1] * (32 if name in _SCALES else 2)
                if loaded_weight.shape[0] != h or not 0 < original_i <= intermediate:
                    raise ValueError("Checkpoint down shape differs from the V4 format")
                target = target[:, : loaded_weight.shape[1]]
            if original_i % 32:
                raise ValueError("Checkpoint intermediate dimension must divide by 32")
            previous_i = self._intermediate_sizes.get(local, original_i)
            if previous_i != original_i:
                raise ValueError(
                    "Checkpoint weight and scale padding dimensions disagree"
                )
            target.copy_(loaded_weight)
            self._intermediate_sizes[local] = original_i
            self._loaded.add(key)

        def load(*args, **kwargs):
            # DeepSeek V4 may dispatch checkpoint loads on a ThreadPoolExecutor.
            # Validation and publication must be atomic across duplicate shards.
            with self._lock:
                return load_unlocked(*args, **kwargs)

        return load

    def finalize(self):
        with self._lock:
            return self._finalize()

    def _finalize(self):
        if self._sealed:
            return
        if self._failed or self._closed:
            raise RuntimeError(
                "The host master cannot be finalized after a failed repack"
            )
        if len(self._loaded) != 6 * self.spec.local_experts:
            raise RuntimeError(
                "Incomplete V4 checkpoint: every expert needs all weights and scales"
            )
        for name in _SCALES:
            if not bool(self.raw_components[name].any()):
                raise RuntimeError("FP4 expert weight scales are all zero")
        try:
            for expert in range(self.spec.local_experts):
                self.format.repack_into(
                    self.raw_components, expert, self.component_capabilities
                )
        except BaseException:
            # Earlier experts may already have native NZ bytes. A retry would
            # reinterpret them as checkpoint ND, corrupting the sole master.
            self._failed = True
            raise
        self.host_store.seal()
        self.host_store.require_device_access()
        self._sealed = True
        if hasattr(self.layer, "dispatcher"):
            self.layer.dispatcher.set_quant_config(
                {
                    "normal_dispatcher_output_dtype": "bf16",
                    "low_latency_dispatcher_output_dtype": "mxfp8",
                }
            )

    def allocations(self):
        # The model-wide MemFabricHostPool reports its physical reservation once.
        return ()

    def close(self):
        """Retire module-visible host views before the drained pool is freed."""
        import torch

        if self._closed:
            return
        for name in self.raw_components:
            parameter = getattr(self.layer, name)
            parameter.data = torch.empty(0, dtype=torch.uint8, device="cpu")
            if hasattr(parameter, "weight_loader"):
                del parameter.weight_loader
        self.raw_components.clear()
        self.component_capabilities.clear()
        self.host_store = None
        self._closed = True
