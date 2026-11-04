"""Bounded ModelSlim W4A8_DYNAMIC expert loading using resident conversion.

Weights retain ModelSlim's packed INT8 checkpoint carrier, then use the same
NZ cast and INT32 view as resident W4A8. Per-group scales are folded into the
second-scale allocation, whose extent reserves both checkpoint and runtime
bytes. The original channel scale and unused offsets remain charged metadata;
they are not exposed to the grouped matmul. No second full host weight store is
created, and conversion is limited to one expert/projection at a time.
"""

from __future__ import annotations

import torch

from ..formats import CheckpointComponent
from .format import NpuExpertFormat, TorchNpuFormatAdapter


class _TorchNpuW4A8FormatAdapter(TorchNpuFormatAdapter):
    def __init__(self, device, native_acl, *, is_per_channel_weight):
        super().__init__(device, native_acl)
        from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
            NPUW4A8Int8MoEMethod,
        )

        self.is_per_channel_weight = is_per_channel_weight
        self.kernel = NPUW4A8Int8MoEMethod(
            is_per_channel_weight=is_per_channel_weight, activation_use_clip=False
        )

    def prepare(
        self, weight, scale, offset, bias, scale_second=None, offset_second=None
    ):
        # Use the resident implementation, including its scale bit packing,
        # FP16 rounding for group scales and scale-bias reduction. A temporary
        # module owns only this projection and does not retain any conversion.
        layer = torch.nn.Module()
        values = {
            "weight": weight,
            "weight_scale": scale,
            "weight_offset": offset,
            "scale_bias": bias,
        }
        if not self.is_per_channel_weight:
            if scale_second is None or offset_second is None:
                raise ValueError("Per-group W4A8 requires both second-scale components")
            values.update(
                weight_scale_second=scale_second,
                weight_offset_second=offset_second,
            )
        elif scale_second is not None or offset_second is not None:
            raise ValueError("Per-channel W4A8 cannot consume second-scale components")
        for suffix, value in values.items():
            layer.register_parameter(
                f"w2_{suffix}", torch.nn.Parameter(value, requires_grad=False)
            )
        self.kernel.process_weights_after_loading(layer, "w2")
        converted_weight = layer.w2_weight.data
        if int(self.torch_npu.get_npu_format(converted_weight)) != 29:
            raise RuntimeError("W4A8 offload requires the probed FRACTAL_NZ layout")
        converted_scale = layer.w2_weight_scale.data
        converted_bias = layer.w2_scale_bias.data
        if self.is_per_channel_weight:
            return converted_weight, converted_scale, offset, converted_bias
        return (
            converted_weight,
            scale,
            offset,
            converted_bias,
            converted_scale,
            offset_second,
        )

    def physical_nbytes(self, tensor):
        if tensor.dtype == torch.int32:
            if int(self.torch_npu.get_npu_format(tensor)) != 29:
                raise RuntimeError("W4A8 packed weights must retain the NZ descriptor")
            # Inspect the byte view so get_storage_size has unambiguous units.
            # This is the inverse of the resident INT8 -> INT32 dtype view.
            return super().physical_nbytes(tensor.view(torch.int8))
        size = tensor.untyped_storage().nbytes()
        if (
            int(self.torch_npu.get_npu_format(tensor)) != 2
            or not tensor.is_contiguous()
            or tensor.storage_offset() != 0
            or size != tensor.numel() * tensor.element_size()
        ):
            raise RuntimeError("W4A8 metadata must own contiguous ND storage")
        return size

    def empty_component(self, name, layout, capacity):
        shape = (capacity,) + layout.runtime_shape[1:]
        if capacity and name.endswith("_weight"):
            # Allocate the same INT8 NZ carrier that resident conversion packs.
            # Directly allocating an INT32 NZ tensor can select a different tile.
            packed_shape = list(shape)
            packed_shape[-1] *= 4
            carrier = torch.ops.npu.empty_with_format(
                packed_shape, dtype=torch.int8, device=self.device, acl_format=29
            )
            return carrier.view(torch.int32)
        return torch.empty(shape, dtype=layout.dtype, device=self.device)


class NpuW4A8Format(NpuExpertFormat):
    """Packed INT4 weights with INT8 activations and complete scale/bias state."""

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        device,
        *,
        group_size=0,
        tp_size=1,
        activation_use_clip=False,
        native_acl=None,
        workspace_limit_bytes=None,
    ):
        if activation_use_clip:
            raise ValueError(
                "W4A8 MoE offload does not support activation_use_clip=True"
            )
        if any(
            type(d) is not int or d <= 0 or d % 64
            for d in (hidden_size, intermediate_size)
        ):
            raise ValueError("W4A8 hidden/intermediate dimensions must divide by 64")
        if type(tp_size) is not int or tp_size <= 0 or 16 % tp_size:
            raise ValueError("W4A8 scale-bias tensor parallel size must divide 16")
        if type(group_size) is not int or group_size < 0:
            raise ValueError("W4A8 group size must be a nonnegative integer")
        if group_size and any(d % group_size for d in (hidden_size, intermediate_size)):
            raise ValueError("W4A8 group size must divide both projection input widths")
        self.hidden_size, self.intermediate_size = hidden_size, intermediate_size
        self.group_size, self.tp_size = group_size, tp_size
        self.device = device
        self.workspace_limit_bytes = workspace_limit_bytes
        self.adapter = _TorchNpuW4A8FormatAdapter(
            device, native_acl, is_per_channel_weight=group_size == 0
        )
        groups, minimum, new_outputs = [], [], []
        for prefix, n, k, bias_width in (
            ("w13", 2 * intermediate_size, hidden_size, 1),
            ("w2", hidden_size, intermediate_size, 16 // tp_size),
        ):
            shards = ("w1", "w3") if prefix == "w13" else ("w2",)
            # Offsets are checkpoint state in the resident scheme but are not
            # passed to W4A8 GMM. Keep them registered and budgeted to preserve
            # strict checkpoint completeness and avoid silently dropping keys.
            entries = [
                ("weight", (1, n // 2, k), torch.int8, "weight", 0),
                (
                    "weight_scale",
                    (1, n, 1),
                    torch.float32,
                    None if group_size else "weight_scale",
                    1,
                ),
                ("weight_offset", (1, n, 1), torch.float32, None, 0),
                ("scale_bias", (1, n, bias_width), torch.float32, "scale_bias", 0),
            ]
            new_outputs.extend((f"{prefix}_weight", f"{prefix}_scale_bias"))
            if group_size:
                entries.extend(
                    (
                        (
                            "weight_scale_second",
                            (1, n, k // group_size),
                            torch.float32,
                            "weight_scale",
                            1,
                        ),
                        (
                            "weight_offset_second",
                            (1, n, k // group_size),
                            torch.float32,
                            None,
                            0,
                        ),
                    )
                )
                new_outputs.append(f"{prefix}_weight_scale_second")
            else:
                new_outputs.append(f"{prefix}_weight_scale")
            groups.append(
                tuple(
                    CheckpointComponent(
                        f"{prefix}_{suffix}",
                        shape,
                        dtype,
                        shards,
                        f"{prefix}_{runtime_suffix}" if runtime_suffix else None,
                        () if suffix == "weight" else (("quant_method", "channel"),),
                        initial,
                    )
                    for suffix, shape, dtype, runtime_suffix, initial in entries
                )
            )
            # Raw carrier + contiguous transpose + NZ output, raw metadata,
            # bit-packed output and scale-rounding intermediates. The measured
            # peak additionally checks the deployed NPU implementation.
            minimum.append(
                3 * n * k // 2
                + 16 * n
                + 4 * n * bias_width
                + (28 * n * (k // group_size) if group_size else 0)
            )
        self._probe(
            f"ascend.w4a8.nz.group{group_size}.tp{tp_size}.v1",
            groups,
            minimum_workspace=minimum,
            new_outputs=tuple(new_outputs),
            dispatch_config={
                "normal_dispatcher_output_dtype": "bf16",
                "low_latency_dispatcher_output_dtype": "bf16",
            },
        )

    def write_checkpoint(self, name, target, loaded_weight, shard_id, metadata):
        if name not in self.layouts or shard_id not in self.layouts[name].shards:
            raise ValueError("Checkpoint shard does not match the W4A8 component")
        if name.startswith("w13_"):
            rows = (
                self.intermediate_size // 2
                if name.endswith("_weight")
                else self.intermediate_size
            )
            target = target.narrow(0, 0 if shard_id == "w1" else rows, rows)
        if name.endswith("_weight"):
            if loaded_weight.dtype != torch.int8 or loaded_weight.shape != target.shape:
                raise ValueError(
                    "Expected packed checkpoint INT8 [out/2, in] W4A8 weight"
                )
        else:
            if loaded_weight.dtype not in (
                torch.float32,
                torch.bfloat16,
                torch.float16,
            ):
                raise ValueError("Expected floating W4A8 scale, offset or scale bias")
            if target.shape[-1] == 1 and loaded_weight.shape == target.shape[:-1]:
                loaded_weight = loaded_weight.unsqueeze(-1)
            if loaded_weight.shape != target.shape:
                raise ValueError(
                    "W4A8 metadata shape differs from the expert component"
                )
            if not bool(torch.isfinite(loaded_weight).all()):
                raise ValueError("W4A8 scale, offset and scale bias must be finite")
            if "weight_scale" in name and bool(loaded_weight.lt(0).any()):
                raise ValueError("W4A8 expert scales must be nonnegative")
        target.copy_(loaded_weight)
