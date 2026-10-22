"""DeepSeek clamped SwiGLU followed by INT8 per-token quantization.

This composable activation uses capture-safe tensor operations. It preserves
DeepSeek's clamp-before-SiLU semantics; it is not the OAI (up + 1) activation.
"""

import torch


class NPUSwigluInt8Quant:
    def __init__(self, limit):
        if not 0 < float(limit) < float("inf"):
            raise ValueError("SwiGLU limit must be positive and finite")
        self.limit = float(limit)

    def _apply_activation(self, hidden_states, group_list, group_list_type):
        if group_list_type == 1:
            valid_rows = group_list.sum()
        elif group_list_type == 0:
            valid_rows = group_list[-1]
        else:
            raise ValueError("Unsupported grouped activation count layout")
        # GMM padding may be unwritten NaN. Sanitize before activation/quant,
        # keeping fixed row capacity and device-only counts during graph replay.
        rows = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        x = torch.where((rows < valid_rows).unsqueeze(-1), hidden_states, 0)
        # Same native activation as DeepseekV2MLP: clamp gate/up first,
        # then SiLU(gate) * up. alpha=1/bias=0 avoids the OAI up+1 variant.
        activated = torch.ops.npu.npu_clipped_swiglu(
            x, alpha=1, limit=self.limit, bias=0, interleaved=False
        )
        return torch.ops.npu.npu_dynamic_quant(activated, dst_type=torch.int8)
