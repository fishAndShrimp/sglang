"""Experimental CANN 9.1 LightningIndexerV2 adapter for paged QSA rows.

This binding matches the locally validated 1024-entry CANN 9.1 metadata ABI.
It requires the same isolated binding support package as QSA prefill GBSA.
"""

from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def _binding():
    from cann_ops_transformer_qsa_probe.op_builder import OpBuilder

    class Builder(OpBuilder):
        def __init__(self):
            super().__init__("qsa_lightning_indexer_cann91")

        def sources(self):
            return [str(Path(__file__).parent / "csrc/qsa_lightning_indexer.cpp")]

        def schema(self):
            return None

        def register_meta(self):
            pass

    return Builder().load()


def qsa_lightning_block_indices(q, k_cache, page_table, compressed_lengths):
    """Return 512 logical compressed-group indices per independent query row.

    The caller already limits each row's compressed length to complete visible
    groups. Request duplication for MTP is represented by the row page table.
    Inputs and outputs remain on device, including zero-length padding rows.
    """
    rows, heads, dim = q.shape
    if dim != 128 or q.dtype != torch.bfloat16 or k_cache.dtype != q.dtype:
        raise ValueError("QSA LightningIndexerV2 requires BF16 Q/K and width 128")
    if k_cache.ndim != 4 or k_cache.shape[2:] != (1, 128):
        raise ValueError("QSA LightningIndexerV2 requires paged [P,S,1,128] keys")
    if page_table.shape[0] != rows or compressed_lengths.shape != (rows,):
        raise ValueError("QSA LightningIndexerV2 requires one page table and length per row")
    if not rows:
        return torch.empty((0, 512), dtype=torch.int32, device=q.device)
    lens = compressed_lengths.to(torch.int32).contiguous()
    cu_q = torch.arange(rows + 1, dtype=torch.int32, device=q.device)
    cu_k = torch.cat((lens.new_zeros(1), lens.cumsum(0, dtype=torch.int32)))
    q_lens = torch.ones_like(lens)
    residual = torch.zeros_like(lens)
    weights = torch.full((rows, heads), 128**-0.5, dtype=torch.float32, device=q.device)
    binding = _binding()
    metadata = binding.metadata(
        cu_q, lens, residual, cu_k, q_lens, heads, 512, 1,
        page_table.shape[1] * k_cache.shape[1],
    )
    indices, _ = binding.forward(
        q.contiguous(), k_cache, weights, cu_q, lens, residual, cu_k, q_lens,
        page_table.to(torch.int32).contiguous(), metadata, 512, 1, False,
    )
    return indices[:, 0]
