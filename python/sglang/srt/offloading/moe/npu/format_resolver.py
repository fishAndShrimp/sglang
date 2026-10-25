"""Resolve native expert layouts before the quant method allocates weights."""


def resolve_npu_format(layer, method, *, resident):
    from sglang.srt.hardware_backend.npu.quantization.fp4_moe_methods import (
        NPUW4A8MXFP4FusedMoEMethod,
    )

    if isinstance(method, NPUW4A8MXFP4FusedMoEMethod):
        from .v4_format import NpuV4FP4Format

        return NpuV4FP4Format

    from sglang.srt.layers.quantization.modelslim.modelslim import (
        ModelSlimFusedMoEMethod,
    )
    from sglang.srt.layers.quantization.modelslim.schemes import (
        ModelSlimW8A8Int8MoE,
    )

    if isinstance(method, ModelSlimFusedMoEMethod) and all(
        isinstance(getattr(layer, name, None), ModelSlimW8A8Int8MoE)
        for name in ("w13_scheme", "w2_scheme")
    ):
        from .w8a8_format import NpuV4W8A8Format

        # Resident layers use the same activation transport convention. GMM
        # input quantization stays owned by the W8A8 execution adapter.
        layer._moe_offload_bf16_dispatch = True
        return NpuV4W8A8Format
    if resident:
        return None
    raise ValueError(
        "Native offload supports MXFP4 or ModelSlim W8A8_DYNAMIC experts; "
        f"layer {layer.layer_id} resolved {type(method).__name__}"
    )
