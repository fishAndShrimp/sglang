"""Bounded loading of the repository's native V4 MXFP4 expert format.

Checkpoint uint8 FP4 and E8M0 bytes are written directly to the registered host
master. Each expert is converted with the same prepare_w4a8_mxfp_weight used by
the resident NPU path, then its physical NZ bytes are copied back in place.
No Python dequantization or full-layer NPU staging is performed. The carrier
dtype is not a claim about numeric precision: the native GMM interprets FP4.
"""

from __future__ import annotations

from ..formats import CheckpointComponent
from .format import NpuExpertFormat, TorchNpuFormatAdapter

_WEIGHTS = ("w13_weight", "w2_weight")
_SCALES = ("w13_weight_scale_inv", "w2_weight_scale_inv")


class _TorchNpuFormatAdapter(TorchNpuFormatAdapter):
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


def _scale_view(raw):
    e, n, k32 = raw.shape
    if k32 % 2:
        raise ValueError("Native V4 scales require paired block-32 exponents")
    return raw.view(e, n, k32 // 2, 2).transpose(1, 2)


class NpuV4FP4Format(NpuExpertFormat):
    """One-expert native probe, fixed bank descriptors, bounded repacking.

    Uses the deployed torch_npu format adapter and rejects unsupported layouts.
    Probed metadata, never probe weights, remain resident in this object.
    """

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        device,
        *,
        native_acl=None,
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
        self.adapter = _TorchNpuFormatAdapter(device, native_acl)
        self.workspace_limit_bytes = workspace_limit_bytes
        groups = []
        minimum = []
        for weight_name, scale_name, n, k in (
            (_WEIGHTS[0], _SCALES[0], 2 * intermediate_size, hidden_size),
            (_WEIGHTS[1], _SCALES[1], hidden_size, intermediate_size),
        ):
            shards = ("w1", "w3") if weight_name == _WEIGHTS[0] else ("w2",)
            groups.append(
                (
                    CheckpointComponent(
                        weight_name, (1, n, k // 2), torch.uint8, shards, weight_name
                    ),
                    CheckpointComponent(
                        scale_name,
                        (1, n, k // 32),
                        torch.uint8,
                        shards,
                        scale_name.removesuffix("_inv"),
                        (("quant_method", "block"), ("format_ue8m0", False)),
                    ),
                )
            )
            minimum.append(n * (k // 2 + k // 32))
        self._probe(
            "ascend.v4.mxfp4.nz.v1",
            groups,
            minimum_workspace=minimum,
            new_outputs=_WEIGHTS,
            dispatch_config={
                "normal_dispatcher_output_dtype": "bf16",
                "low_latency_dispatcher_output_dtype": "mxfp8",
            },
        )

    def validate_master(self, raw_components):
        for name in _SCALES:
            if not bool(raw_components[name].any()):
                raise RuntimeError("FP4 expert weight scales are all zero")

    def write_checkpoint(self, name, target, loaded_weight, shard_id, metadata):
        import torch

        if (
            loaded_weight.dtype == getattr(torch, "float8_e8m0fnu", None)
            and name in _SCALES
        ):
            loaded_weight = loaded_weight.view(torch.uint8)
        if loaded_weight.dtype != torch.uint8 or loaded_weight.ndim != 2:
            raise ValueError(
                "Expected two-dimensional packed FP4/E8M0 checkpoint bytes"
            )
        h, intermediate = self.hidden_size, self.intermediate_size
        if shard_id in ("w1", "w3"):
            expected_columns = h // (32 if name in _SCALES else 2)
            original_i = loaded_weight.shape[0]
            if (
                loaded_weight.shape[1] != expected_columns
                or not 0 < original_i <= intermediate
            ):
                raise ValueError("Checkpoint gate/up shape differs from the V4 format")
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
        if metadata.get("intermediate_size", original_i) != original_i:
            raise ValueError("Checkpoint weight and scale padding dimensions disagree")
        target.copy_(loaded_weight)
        metadata["intermediate_size"] = original_i
