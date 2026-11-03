"""ModelSlim per-channel INT4/INT8 experts, using the resident conversion.

Keep all checkpoint metadata, including scale_bias. MXFP4 and groupwise INT4
are different formats and must not enter this adapter.
"""

import torch

from ..formats import CheckpointComponent
from .format import NpuExpertFormat, TorchNpuFormatAdapter


class _W4A8Adapter(TorchNpuFormatAdapter):
    def prepare(self, weight, scale, offset, bias):
        from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
            NPUW4A8Int8MoEMethod,
        )

        layer = torch.nn.Module()
        for suffix, value in zip(
            ("weight", "weight_scale", "weight_offset", "scale_bias"),
            (weight, scale, offset, bias),
        ):
            layer.register_parameter("w2_" + suffix, torch.nn.Parameter(value, requires_grad=False))
        # Both projections use identical post-load operations. w2 avoids setting
        # a dispatcher on this temporary module; the real layer sets it at seal.
        NPUW4A8Int8MoEMethod(is_per_channel_weight=True).process_weights_after_loading(layer, "w2")
        return layer.w2_weight, layer.w2_weight_scale, layer.w2_weight_offset, layer.w2_scale_bias

    def physical_nbytes(self, tensor):
        size = tensor.untyped_storage().nbytes()
        if tensor.storage_offset() != 0 or size < tensor.numel() * tensor.element_size():
            raise RuntimeError("Invalid W4A8 component extent")
        fmt = int(self.torch_npu.get_npu_format(tensor))
        if tensor.dtype == torch.int32:
            if fmt != 29:
                raise RuntimeError("W4A8 packed weight requires FRACTAL_NZ")
        elif fmt != 2 or not tensor.is_contiguous() or size != tensor.numel() * tensor.element_size():
            raise RuntimeError("W4A8 metadata requires contiguous ND storage")
        return size

    def empty_component(self, name, layout, capacity):
        shape = (capacity,) + layout.runtime_shape[1:]
        if capacity and name.endswith("_weight"):
            # Match resident INT8 NZ -> INT32 view packing exactly.
            carrier_shape = list(shape)
            carrier_shape[-1] *= 4
            return torch.ops.npu.empty_with_format(
                carrier_shape, dtype=torch.int8, device=self.device, acl_format=29
            ).view(torch.int32)
        return torch.empty(shape, dtype=layout.dtype, device=self.device)


class NpuV4W4A8Format(NpuExpertFormat):
    def __init__(self, hidden_size, intermediate_size, device, *, native_acl=None,
                 workspace_limit_bytes=None):
        if any(type(d) is not int or d <= 0 or d % 64 for d in (hidden_size, intermediate_size)):
            raise ValueError("W4A8 dimensions must be positive multiples of 64")
        self.hidden_size, self.intermediate_size = hidden_size, intermediate_size
        self.device, self.workspace_limit_bytes = device, workspace_limit_bytes
        self.adapter = _W4A8Adapter(device, native_acl)
        groups, minimum = [], []
        for prefix, n, k, bias_parts in (
            ("w13", 2 * intermediate_size, hidden_size, 1),
            ("w2", hidden_size, intermediate_size, 16),
        ):
            shards = ("w1", "w3") if prefix == "w13" else ("w2",)
            groups.append(tuple(CheckpointComponent(
                f"{prefix}_{suffix}", shape, dtype, shards, f"{prefix}_{suffix}", attrs, initial,
            ) for suffix, shape, dtype, attrs, initial in (
                ("weight", (1, n // 2, k), torch.int8, (), 0),
                ("weight_scale", (1, n, 1), torch.float32, (("quant_method", "channel"),), 1),
                ("weight_offset", (1, n, 1), torch.float32, (("quant_method", "channel"),), 0),
                ("scale_bias", (1, n, bias_parts), torch.float32, (), 0),
            )))
            minimum.append(2 * n * k + 32 * n * bias_parts)
        self._probe(
            "ascend.v4.w4a8.int4.nz.v1", groups, minimum_workspace=minimum,
            new_outputs=tuple(c.name for group in groups for c in group if not c.name.endswith("_offset")),
            dispatch_config={"dispatcher_output_dtype": "int8"},
        )

    def write_checkpoint(self, name, target, loaded_weight, shard_id, metadata):
        if name.startswith("w13_"):
            half = target.shape[0] // 2
            target = target.narrow(0, 0 if shard_id == "w1" else half, half)
        if name.endswith("_weight"):
            if loaded_weight.dtype != torch.int8:
                raise ValueError("Expected packed INT4 in an INT8 checkpoint tensor")
        else:
            if loaded_weight.dtype != torch.float32 or not bool(torch.isfinite(loaded_weight).all()):
                raise ValueError("W4A8 metadata must be finite FP32")
            if name.endswith("_offset") and bool(loaded_weight.ne(0).any()):
                raise ValueError("Only symmetric per-channel W4A8 offsets are supported")
            if name.endswith("_scale") and bool(loaded_weight.lt(0).any()):
                raise ValueError("W4A8 scales must be nonnegative")
        if loaded_weight.shape != target.shape:
            raise ValueError(f"W4A8 checkpoint shape mismatch for {name}: {loaded_weight.shape} != {target.shape}")
        target.copy_(loaded_weight)
