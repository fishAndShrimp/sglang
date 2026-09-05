"""Temporary Ascend Triton kernels for Qwen3.8-Flash-Next QSA.

Migration target: ``sgl_kernel_npu.indexer.qsa``. See the adjacent README for
the ownership boundary and migration rules.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl


def can_run_qsa_mqa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
) -> bool:
    """Whether the staged Triton kernel covers this valid QSA configuration."""

    return (
        q.ndim == 3
        and k_cache.ndim == 4
        and page_table.ndim == 2
        and context_lens.ndim == 1
        and q.device.type == "npu"
        and k_cache.device == q.device
        and page_table.device == q.device
        and context_lens.device == q.device
        and q.dtype in (torch.bfloat16, torch.float16)
        and k_cache.dtype == q.dtype
        and q.shape[1:] == (4, 128)
        and k_cache.shape[1:] == (16, 1, 128)
        and page_table.shape[0] == q.shape[0]
        and context_lens.numel() == q.shape[0]
        and page_table.dtype in (torch.int32, torch.int64)
        and context_lens.dtype in (torch.int32, torch.int64)
        and page_table.is_contiguous()
        and context_lens.is_contiguous()
    )


@triton.jit
def _triton_qsa_mqa_decode_kernel(
    q,
    k_cache,
    page_table,
    context_lens,
    output,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    k_stride_page: tl.constexpr,
    k_stride_token: tl.constexpr,
    k_stride_d: tl.constexpr,
    page_table_stride: tl.constexpr,
    output_stride: tl.constexpr,
    SCALE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MAX_MODEL_LEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused paged MQA score kernel used by the Ascend QSA decode path."""

    batch = tl.program_id(0)
    token_block = tl.program_id(1)
    tokens = token_block * BLOCK_N + tl.arange(0, BLOCK_N)
    dims = tl.arange(0, HEAD_DIM)
    valid = tokens < tl.load(context_lens + batch)
    pages = tl.load(
        page_table + batch * page_table_stride + tokens // PAGE_SIZE,
        mask=valid,
        other=0,
    )
    k_offsets = (
        pages[:, None] * k_stride_page
        + (tokens % PAGE_SIZE)[:, None] * k_stride_token
        + dims[None, :] * k_stride_d
    )
    keys = tl.load(k_cache + k_offsets, mask=valid[:, None], other=0.0)
    score = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for head in range(NUM_HEADS):
        query = tl.load(
            q + batch * q_stride_b + head * q_stride_h + dims * q_stride_d
        )
        dot = tl.sum(keys.to(tl.float32) * query[None, :].to(tl.float32), axis=1)
        score += tl.maximum(dot, 0.0)
    tl.store(
        output + batch * output_stride + tokens,
        tl.where(valid, score / SCALE, -float("inf")),
        mask=tokens < MAX_MODEL_LEN,
    )


def triton_qsa_mqa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_model_len: int,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    """Compute paged QSA MQA logits without materializing gathered K or scores.

    This kernel is optimized for the 128-wide Qwen3.8 QSA indexer on Ascend.
    """

    if not can_run_qsa_mqa_decode(q, k_cache, page_table, context_lens):
        raise ValueError(
            "unsupported temporary QSA MQA Triton configuration: "
            f"q={tuple(q.shape)}/{q.dtype}/{q.device}, "
            f"k_cache={tuple(k_cache.shape)}/{k_cache.dtype}/{k_cache.device}, "
            f"page_table={tuple(page_table.shape)}/{page_table.dtype}/"
            f"{page_table.device}, context_lens={tuple(context_lens.shape)}/"
            f"{context_lens.dtype}/{context_lens.device}"
        )
    if max_model_len < 0:
        raise ValueError(f"max_model_len must be non-negative, got {max_model_len}")
    batch, heads, head_dim = q.shape
    output = torch.empty(
        (batch, max_model_len), dtype=torch.float32, device=q.device
    )
    if batch == 0 or max_model_len == 0:
        return output
    block_n = 128
    _triton_qsa_mqa_decode_kernel[(batch, triton.cdiv(max_model_len, block_n))](
        q,
        k_cache,
        page_table,
        context_lens,
        output,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(3),
        page_table.stride(0),
        output.stride(0),
        SCALE=float(score_scale or math.sqrt(head_dim)),
        NUM_HEADS=heads,
        HEAD_DIM=head_dim,
        PAGE_SIZE=k_cache.shape[1],
        MAX_MODEL_LEN=max_model_len,
        BLOCK_N=block_n,
    )
    return output


__all__ = ["can_run_qsa_mqa_decode", "triton_qsa_mqa_decode"]
