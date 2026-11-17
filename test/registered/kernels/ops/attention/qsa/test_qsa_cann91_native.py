"""Opt-in tests for the experimental CANN 9.1 native QSA adapters."""

import os

import pytest
import torch

from sglang.srt.utils import is_npu

pytestmark = pytest.mark.skipif(
    not is_npu() or os.getenv("SGLANG_TEST_CANN91_QSA_NATIVE") != "1",
    reason="Requires the isolated CANN 9.1 QSA native runtime",
)


def _assert_topk_scores(q, k, table, lengths, indices):
    q, k = q.float().cpu(), k.float().cpu()
    table, lengths, indices = table.cpu(), lengths.cpu(), indices.cpu()
    for row in range(q.shape[0]):
        count = int(lengths[row])
        selected = indices[row][indices[row] >= 0]
        assert selected.numel() == min(count, 512)
        assert selected.unique().numel() == selected.numel()
        if not count:
            continue
        assert selected.max() < count
        keys = k[table[row].long()].reshape(-1, 128)[:count]
        scores = torch.relu(q[row] @ keys.T).sum(0)
        # ReLU produces tied zero scores. Compare score multisets at the cutoff,
        # allowing any valid tied index, rather than requiring one top-k order.
        torch.testing.assert_close(
            scores[selected].sort().values,
            scores.topk(min(count, 512)).values.sort().values,
            atol=1e-5,
            rtol=1e-5,
        )


@pytest.mark.parametrize("rows,heads", [(32, 4), (128, 8)])
def test_lightning_boundary_lengths_and_graph_updates(rows, heads):
    from sglang.kernels.ops.attention.qsa_lightning_indexer import (
        qsa_lightning_block_indices,
    )

    torch.manual_seed(1234)
    q = torch.randn(rows, heads, 128, device="npu", dtype=torch.bfloat16) * 0.1
    k = torch.randn(128, 16, 1, 128, device="npu", dtype=torch.bfloat16) * 0.1
    table = torch.stack(
        [torch.randperm(128, device="npu")[:64] for _ in range(rows)]
    ).int()
    lengths = torch.tensor(
        [0, 1, 3, 511, 512, 513, 700, 1000] * (rows // 8),
        device="npu",
        dtype=torch.int32,
    )
    output = qsa_lightning_block_indices(q, k, table, lengths)
    _assert_topk_scores(q, k, table, lengths, output)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=False):
        captured = qsa_lightning_block_indices(q, k, table, lengths)
    q.copy_(torch.randn_like(q) * 0.1)
    lengths.copy_(lengths.roll(3))
    table.copy_(table.roll(1, 0))
    graph.replay()
    torch.npu.synchronize()
    _assert_topk_scores(q, k, table, lengths, captured)


@pytest.mark.parametrize(
    "rows,width", [(32, 2051), (32, 2054), (32, 2055), (128, 2051), (128, 2055)]
)
def test_gbsa_decode_graph_tracks_slots_and_values(rows, width):
    from sgl_kernel_npu.operators.attention import qsa_sparse_attention

    from sglang.kernels.ops.attention.qsa_prefill_attention import qsa_decode_attention

    torch.manual_seed(1234)
    pool = 16384
    q = torch.randn(rows, 3, 256, device="npu", dtype=torch.bfloat16) * 0.2
    k = torch.randn(pool, 1, 256, device="npu", dtype=torch.bfloat16) * 0.2
    v = torch.randn_like(k)
    slots = (
        torch.arange(width, device="npu")[None, :]
        + torch.arange(rows, device="npu")[:, None] * 37
    ).remainder(pool).int()
    slots[0].fill_(-1)
    slots[1, -23:] = -1
    reference = qsa_sparse_attention(q, k, v, slots)
    output = qsa_decode_attention(q, k, v, slots)
    torch.testing.assert_close(output, reference, atol=0.02, rtol=0.02)
    assert not output[0].count_nonzero()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=False):
        captured = qsa_decode_attention(q, k, v, slots)
    q.copy_(torch.randn_like(q) * 0.2)
    slots.copy_(slots.roll(2, 0))
    v.mul_(0.9)
    graph.replay()
    torch.npu.synchronize()
    reference = qsa_sparse_attention(q, k, v, slots)
    torch.testing.assert_close(captured, reference, atol=0.02, rtol=0.02)
    assert not captured[2].count_nonzero()
