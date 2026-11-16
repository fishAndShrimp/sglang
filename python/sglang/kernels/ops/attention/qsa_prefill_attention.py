"""NPU prefill attention over unique physical token slots."""

import torch


def qsa_prefill_attention(q, k, v, slots, softmax_scale=None):
    """BF16 QSA prefill; valid slots in each row must be unique.

    q is [R,Hq,256], k/v are [N,Hkv,256], slots is int32 [R,S].
    Negative slots are padding. Empty rows return zero.
    The GBSA path covers arch22, R>=128, Hq=3, Hkv=1 and S=2051.
    Other supported QSA shapes use the existing operator.
    """
    from sgl_kernel_npu.operators.attention import qsa_sparse_attention
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    eligible = (
        q.ndim == k.ndim == v.ndim == 3
        and q.shape[0] >= 128
        and q.shape[1:] == (3, 256)
        and k.shape[1:] == (1, 256)
        and v.shape == k.shape
        and q.dtype == k.dtype == v.dtype == torch.bfloat16
        and slots.shape == (q.shape[0], 2051)
        and slots.dtype == torch.int32
        and 0 < k.shape[0] <= 2**24
        and k.is_contiguous()
        and v.is_contiguous()
        and get_hardware().generation == Generation.ARCH22
    )
    if not eligible:
        return qsa_sparse_attention(q, k, v, slots, softmax_scale)

    from cann_ops_transformer_qsa_probe.ops.attention.generic_block_sparse_attention import (
        generic_block_sparse_attention,
        generic_block_sparse_attention_metadata,
    )

    pool_rows = k.shape[0]
    valid = slots >= 0
    counts = valid.sum(-1, dtype=torch.int32).unsqueeze(0)
    # Slot IDs and the padding sentinel are exact integers in FP32 up to 2**24.
    indices = torch.where(valid, slots, pool_rows).float().sort(-1).values
    indices = torch.where(indices < pool_rows, indices, 0).int().unsqueeze(0)
    cu_base = torch.arange(2, dtype=torch.int64, device=q.device)
    options = dict(
        cu_seqlens_q=cu_base * q.shape[0],
        cu_seqlens_kv=cu_base * pool_rows,
        layout_q="TND",
        layout_kv="TND",
        layout_sparse_pattern=4,
        mask_mode=0,
        quant_mode=0,
        softmax_precision=0,
        residual_block_mode=0,
        is_consistent_topk=False,
    )
    metadata = generic_block_sparse_attention_metadata(
        indices,
        counts,
        3,
        1,
        256,
        [1, 1],
        max_seqlen_q=q.shape[0],
        max_seqlen_kv=pool_rows,
        **options,
    )
    output, _ = generic_block_sparse_attention(
        q.contiguous(),
        k,
        v,
        indices,
        counts,
        [1, 1],
        metadata=metadata,
        softmax_scale=1 / 16 if softmax_scale is None else softmax_scale,
        return_softmax_lse=False,
        **options,
    )
    return output
