"""Bounded ModelSlim W8A8_DYNAMIC loading into the shared V4 host master.

Checkpoint scales are assembled in FP32, then converted exactly like the resident
W8A8 path. The host extent reserves max(checkpoint bytes, runtime bytes); its
charged padding avoids a second full host store. Only one expert/projection is
converted at a time. Offsets must be zero: the current INT8 GMM is symmetric.
"""

from __future__ import annotations

from types import MappingProxyType

import torch

from .storage import _align
from .v4_format import (
    NpuExpertFormat,
    V4LayerStorage,
    _Layout,
    _TorchNpuFormatAdapter,
)


class _TorchNpuW8A8FormatAdapter(_TorchNpuFormatAdapter):
    def prepare(self, weight, scale, offset):
        from sglang.srt.hardware_backend.npu.utils import npu_format_cast

        # W8A8 transposes BEFORE casting, unlike the native FP4 carrier path.
        weight = npu_format_cast(weight.transpose(1, 2).contiguous())
        if int(self.torch_npu.get_npu_format(weight)) != 29:
            raise RuntimeError("W8A8 offload requires the probed FRACTAL_NZ layout")
        return weight, scale.squeeze(-1).to(torch.bfloat16), offset.squeeze(-1)

    def physical_nbytes(self, tensor):
        if tensor.element_size() == 1:
            return super().physical_nbytes(tensor)
        # Only weights are opaque NZ. Typed scales/offsets must be ordinary ND
        # allocations, so their physical bytes can be verified without guessing
        # the units of get_storage_size for a multi-byte carrier.
        size = tensor.untyped_storage().nbytes()
        if (
            int(self.torch_npu.get_npu_format(tensor)) != 2
            or not tensor.is_contiguous()
            or tensor.storage_offset() != 0
            or size != tensor.numel() * tensor.element_size()
        ):
            raise RuntimeError("W8A8 metadata must own contiguous ND storage")
        return size

    def empty_component(self, name, layout, capacity):
        shape = (capacity,) + layout.runtime_shape[1:]
        if capacity and name.endswith("_weight"):
            return torch.ops.npu.empty_with_format(
                list(shape), dtype=layout.dtype, device=self.device, acl_format=29
            )
        return torch.empty(shape, dtype=layout.dtype, device=self.device)


class NpuV4W8A8Format(NpuExpertFormat):
    """One homogeneous INT8 format, with six complete per-expert components."""

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        device,
        *,
        native_acl=None,
        workspace_limit_bytes=None,
    ):
        if any(
            type(d) is not int or d <= 0 or d % 64
            for d in (hidden_size, intermediate_size)
        ):
            raise ValueError("W8A8 hidden/intermediate dimensions must divide by 64")
        self.hidden_size, self.intermediate_size = hidden_size, intermediate_size
        self.device = device
        self.workspace_limit_bytes = workspace_limit_bytes
        self.adapter = _TorchNpuW8A8FormatAdapter(device, native_acl)
        self.layouts, self.checkpoint_dtypes, self.host_component_nbytes = {}, {}, {}
        self.workspace_nbytes = 0
        for prefix, n, k in (
            ("w13", 2 * intermediate_size, hidden_size),
            ("w2", hidden_size, intermediate_size),
        ):
            # Input weights/scales/offset plus the transpose/NZ result and BF16
            # scales; the probe checks additional allocator padding as well.
            minimum_peak = 2 * n * k + 10 * n
            if (
                workspace_limit_bytes is not None
                and minimum_peak > workspace_limit_bytes
            ):
                raise ValueError("One-expert W8A8 conversion exceeds loader workspace")
            baseline = self.adapter.begin_workspace()
            inputs = (
                self.adapter.to_device(torch.zeros((1, n, k), dtype=torch.int8)),
                self.adapter.to_device(torch.ones((1, n, 1), dtype=torch.float32)),
                self.adapter.to_device(torch.zeros((1, n, 1), dtype=torch.float32)),
            )
            converted = self.adapter.prepare(*inputs)
            self.adapter.synchronize()
            for suffix, raw, value in zip(
                ("weight", "weight_scale", "weight_offset"), inputs, converted
            ):
                name = f"{prefix}_{suffix}"
                size = self.adapter.physical_nbytes(value)
                if size < value.numel() * value.element_size() or size % 64:
                    raise RuntimeError("W8A8 component has an invalid physical extent")
                self.layouts[name] = _Layout(
                    tuple(raw.shape),
                    tuple(value.shape),
                    tuple(value.stride()),
                    value.dtype,
                    size,
                )
                self.checkpoint_dtypes[name] = raw.dtype
                self.host_component_nbytes[name] = max(
                    size, _align(raw.numel() * raw.element_size(), 64)
                )
            # Count all distinct allocations, including the intermediate ND
            # transpose which may have existed during the native format cast.
            known_peak = sum(x.numel() * x.element_size() for x in inputs)
            known_peak += sum(
                self.layouts[f"{prefix}_{s}"].nbytes for s in ("weight", "weight_scale")
            )
            self._check_workspace(baseline, max(minimum_peak, known_peak))
            del inputs, converted, raw, value
        self.component_nbytes = MappingProxyType(
            {name: layout.nbytes for name, layout in self.layouts.items()}
        )
        self.expert_nbytes = sum(self.component_nbytes.values())

    def checkpoint_view(self, allocation, name, experts):
        dtype = self.checkpoint_dtypes[name]
        typed = allocation.view(dtype)
        shape = self.layouts[name].raw_shape
        stride = (shape[2], 1)
        extent = self.host_component_nbytes[name] // typed.element_size()
        return torch.as_strided(typed, (experts,) + shape[1:], (extent,) + stride)

    def checkpoint_attributes(self, name):
        return {"quant_method": "channel"} if not name.endswith("_weight") else {}

    def quant_info(self, tensors):
        from sglang.srt.layers.moe.moe_runner.ascend import AscendQuantInfo

        return AscendQuantInfo(**tensors)

    def repack_into(self, raw_components, expert, destinations):
        for prefix in ("w13", "w2"):
            names = tuple(
                f"{prefix}_{s}" for s in ("weight", "weight_scale", "weight_offset")
            )
            baseline = self.adapter.begin_workspace()
            inputs = tuple(
                self.adapter.to_device(raw_components[name][expert : expert + 1])
                for name in names
            )
            converted = self.adapter.prepare(*inputs)
            self.adapter.synchronize()
            known_peak = sum(x.numel() * x.element_size() for x in inputs)
            known_peak += sum(self.layouts[name].nbytes for name in names[:2])
            self._check_workspace(baseline, known_peak)
            # Validate the complete projection before replacing checkpoint bytes.
            for name, value in zip(names, converted):
                layout = self.layouts[name]
                if (
                    tuple(value.shape) != layout.runtime_shape
                    or tuple(value.stride()) != layout.runtime_stride
                    or value.dtype != layout.dtype
                    or self.adapter.physical_nbytes(value) != layout.nbytes
                ):
                    raise RuntimeError("W8A8 conversion changed after the format probe")
            for name, value in zip(names, converted):
                target = destinations[name][expert]
                self.adapter.copy_to_host(target.host_address, value, target.nbytes)
            del inputs, converted, value


class W8A8LayerStorage(V4LayerStorage):
    """Reuse host ownership/registration, with a typed nine-component loader."""

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
                if self._sealed or self._failed or self._closed:
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
                allowed = ("w1", "w3") if name.startswith("w13_") else ("w2",)
                if shard_id not in allowed:
                    raise ValueError(
                        "Checkpoint shard does not match the expert component"
                    )
                key = name, local, shard_id
                if key in self._loaded:
                    raise ValueError("Duplicate expert checkpoint component")
                if loaded_weight.device.type != "cpu":
                    raise ValueError(
                        "Bounded W8A8 loading requires CPU checkpoint tensors"
                    )
                target = self.raw_components[name][local]
                if name.startswith("w13_"):
                    i = self.format.intermediate_size
                    target = target.narrow(0, 0 if shard_id == "w1" else i, i)
                if name.endswith("_weight"):
                    if (
                        loaded_weight.dtype != torch.int8
                        or loaded_weight.shape != target.shape
                    ):
                        raise ValueError(
                            "Expected checkpoint INT8 [out, in] expert weight"
                        )
                else:
                    if loaded_weight.dtype not in (
                        torch.float32,
                        torch.bfloat16,
                        torch.float16,
                    ):
                        raise ValueError("Expected floating per-channel scale/offset")
                    if loaded_weight.shape == target.shape[:-1]:
                        loaded_weight = loaded_weight.unsqueeze(-1)
                    if loaded_weight.shape != target.shape:
                        raise ValueError(
                            "Per-channel metadata shape differs from the expert"
                        )
                    if not bool(torch.isfinite(loaded_weight).all()):
                        raise ValueError("Expert scale/offset must be finite")
                    if name.endswith("_offset"):
                        if bool(loaded_weight.ne(0).any()):
                            raise ValueError(
                                "W8A8 offload supports symmetric zero offsets only"
                            )
                    elif bool(loaded_weight.lt(0).any()):
                        raise ValueError("Expert scales must be nonnegative")
                target.copy_(loaded_weight)
                self._loaded.add(key)

        return load

    def _finalize(self):
        if self._sealed:
            return
        if self._failed or self._closed:
            raise RuntimeError(
                "The host master cannot be finalized after a failed repack"
            )
        if len(self._loaded) != 9 * self.spec.local_experts:
            raise RuntimeError(
                "Incomplete W8A8 checkpoint: each expert needs nine components"
            )
        try:
            for expert in range(self.spec.local_experts):
                self.format.repack_into(
                    self.raw_components, expert, self.component_capabilities
                )
        except BaseException:
            self._failed = True
            raise
        self.host_store.seal()
        self.host_store.require_device_access()
        self._sealed = True
        if hasattr(self.layer, "dispatcher"):
            # Keep EP payload BF16 in both modes; W13 quantizes per-token inside
            # the graph. Do not feed MXFP8 activations into an INT8 GMM.
            self.layer.dispatcher.set_quant_config(
                {
                    "normal_dispatcher_output_dtype": "bf16",
                    "low_latency_dispatcher_output_dtype": "bf16",
                }
            )
