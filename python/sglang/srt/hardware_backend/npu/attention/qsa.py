# SPDX-License-Identifier: Apache-2.0
# FIA page handling is adapted from vllm_ascend/models/qwen4_exp/fia.py.
# Prefill routing and the indexer's request layout follow attention_dispatch.py
# and lightning_indexer.py in the same directory.
"""Qwen sparse attention with logical block IDs throughout the NPU path."""

from functools import lru_cache
from itertools import accumulate, groupby

import torch
from sgl_kernel_npu.operators import attention
from sgl_kernel_npu.operators.indexer import lightning_indexer_v2
from sgl_kernel_npu.operators.norm import qsa_rmsnorm_mrope

from sglang.srt.hardware_backend.npu.utils import is_npu_arch35
from sglang.srt.layers.attention.qsa.metadata import compressed_decode_view
from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QSAMTPSharedSparseIndices,
)
from sglang.srt.runtime_context import get_parallel


@lru_cache(maxsize=1)
def _qsa_ops():
    """Load the packaged QSA operators on first use."""
    from sgl_kernel_npu._third_party.sgl_ops_transformer_qsa import ops

    return ops


def norm_rope(tensor, positions, norm, rotary):
    """Normalize and rotate indexer Q or K in one call, or return None to fall back."""
    if not (
        rotary.is_neox_style
        and tensor.dtype == norm.weight.dtype == torch.bfloat16
        and rotary.rotary_dim > 0
        and not (rotary.rotary_dim & (rotary.rotary_dim - 1))
        and (positions.ndim == 1 or getattr(rotary, "mrope_interleaved", False))
        and not getattr(rotary, "mrope_interleaved_glm", False)
    ):
        return None
    cache = rotary.cos_sin_cache
    lookup = positions.ndim == 2 and positions.size(-1) <= 128
    return qsa_rmsnorm_mrope(
        tensor,
        norm.weight,
        cache if lookup else cache[positions.long()].to(tensor.dtype),
        norm.variance_epsilon,
        getattr(rotary, "mrope_section", None) or (),
        rotary.rotary_dim,
        positions=positions if lookup else None,
    )


class GBSAMetadata:
    def __init__(self, pages, cu_q, lengths, visible, heads, max_query):
        """Prepare the page table, lengths and work schedule that QSA layers share."""
        self.pages = pages
        self.options = dict(
            cu_seqlens_q=cu_q,
            seqused_kv=lengths,
            layout_q="TND",
            layout_kv="PA_BBND",
            layout_sparse_pattern=4,
            mask_mode=1,
            residual_block_mode=1,
            is_consistent_topk=False,
            softmax_precision=0,
        )
        self.counts = (visible // 4).clamp_max(512).expand(heads[1], -1).contiguous()
        # Only the index shape and counts enter the schedule. IDs are layer-local.
        self.shape = self.counts.new_empty((*self.counts.shape, 512))
        self.heads = heads
        self.max_query = max_query
        self.schedule = self.build()

    def build(self):
        """Split GBSA work across cores based on the block count for each query."""
        return _qsa_ops().generic_block_sparse_attention_metadata(
            self.shape,
            self.counts,
            *self.heads,
            [1, 4],
            max_seqlen_q=self.max_query,
            max_seqlen_kv=self.pages.shape[1] * 128,
            **self.options
        )

    def refresh_graph(self, lengths):
        """Update lengths and the schedule at the addresses recorded in the graph."""
        self.options["seqused_kv"].copy_(lengths.clamp_min(1))
        self.counts.copy_((lengths // 4).clamp_max(512))
        self.schedule.copy_(self.build())

    def attend(self, q, k, v, blocks, scale):
        """Run GBSA with this layer's selected block IDs and the shared schedule."""
        ids, counts = attention.qsa_sort_count_groups(blocks.contiguous())
        num_kv_heads = k.size(2)
        ids = ids.expand(num_kv_heads, -1, -1).contiguous()
        counts = counts.expand(num_kv_heads, -1).contiguous()
        out, _ = _qsa_ops().generic_block_sparse_attention(
            q.contiguous(),
            k,
            v,
            ids,
            counts,
            [1, 4],
            metadata=self.schedule,
            block_table=self.pages,
            softmax_scale=scale,
            **self.options
        )
        return out


@lru_cache(maxsize=1)
def _fia_workspace(device, stream):
    """Reuse the buffer that holds FIA's attention mask on this stream."""
    return attention.QSAFIAMaskWorkspace(device, 4096, 64 << 20, 520)


def _fia(q, k, v, blocks, visible, pages, length, scale, out):
    """Turn the selected blocks into a token mask and write FIA's result into out."""
    n, h, d = q.shape
    width = ((length + 127) // 128) * 128
    pages = pages[:, : width // 128]
    # FIA expects dense physical pages. Descriptor-backed KV can have whole
    # unused pages between consecutive logical pages.
    page_elements = 128 * k.shape[2] * d
    stride, remainder = divmod(k.stride(0), page_elements)
    if remainder or k.stride() != v.stride() or k.stride(1) != k.shape[2] * d:
        raise ValueError("FIA requires dense KV pages with whole-page strides")
    shape = ((k.shape[0] - 1) * stride + 1, 128, k.shape[2] * d)
    strides = (page_elements, shape[-1], 1)
    num_kv_heads = k.shape[2]
    k, v = k.as_strided(shape, strides), v.as_strided(shape, strides)
    pages = (pages * stride if stride != 1 else pages).contiguous()
    tile = min(4096, max(1, (64 << 20) // width))
    for start in range(0, n, tile):
        rows = slice(start, min(start + tile, n))
        size = rows.stop - start
        mask, selected = attention.build_qsa_fia_mask(
            blocks[rows],
            visible[rows],
            length,
            width,
            _fia_workspace(q.device, torch.npu.current_stream(q.device)),
        )
        result, _ = attention.fused_infer_attention_score_v2(
            q[rows].reshape(1, size, h * d).contiguous(),
            k,
            v,
            atten_mask=mask.view(1, 1, size, width),
            actual_seq_qlen=[size],
            actual_seq_kvlen=[length],
            num_query_heads=h,
            num_key_value_heads=num_kv_heads,
            softmax_scale=scale,
            input_layout="BSH",
            sparse_mode=0,
            inner_precise=0,
            block_table=pages,
            block_size=128,
        )
        attention.copy_qsa_fia_output(result.reshape(size, h, d), selected, out[rows])


class LightningIndexerMetadata:
    """Request-level indexer inputs, shared by all layers in a batch."""

    def __init__(
        self,
        pages,
        lengths,
        visible,
        cu_q,
        heads,
        *,
        query_rows=None,
        short_blocks=None
    ):
        """Allocate the page, length and mask buffers used to score this batch."""
        self.cu_q = cu_q
        self.query_rows = query_rows
        self.short_blocks = short_blocks
        self.ranks = torch.arange(512, device=pages.device, dtype=torch.int32)
        self.weights = torch.full(
            (visible.numel(), heads),
            128**-0.5,
            dtype=torch.float32,
            device=pages.device,
        )
        self.pages = torch.empty_like(pages, memory_format=torch.contiguous_format)
        self.blocks = torch.empty_like(lengths)
        self.residual = torch.empty_like(lengths)
        self.valid = torch.empty(
            (visible.numel(), 512), dtype=torch.bool, device=pages.device
        )
        self.options = dict(
            cu_seqlens_q=self.cu_q,
            seqused_k=self.blocks,
            cmp_residual_k=self.residual,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
            cmp_ratio=4,
        )
        self.meta = None
        self.refresh(pages, lengths, visible)

    def refresh(self, pages, lengths, visible):
        """Split visible history into blocks and a tail (10 tokens = 2 blocks + 2 tokens)."""
        # A graph padding row may have no complete block. Score one safe key
        # on page zero and discard its result using the real visibility below.
        self.pages.copy_(torch.where(lengths[:, None] >= 4, pages, 0))
        self.blocks.copy_((lengths // 4).clamp_min(1))
        self.residual.copy_(lengths % 4)
        self.valid.copy_(self.ranks < (visible // 4)[:, None])
        if is_npu_arch35():
            meta = _qsa_ops().lightning_indexer_metadata(
                self.weights.size(1),
                1,
                128,
                512,
                batch_size=pages.size(0),
                **self.options
            )
            if self.meta is None:
                self.meta = meta
            else:
                self.meta.copy_(meta)

    def select(self, q, cache):
        """Score the rows that need top-k and return block IDs in query order."""
        if self.query_rows is not None:
            q = q.index_select(0, self.query_rows)
        # The base package supplies IndexerV2 on CANN 9.2 arch22.
        op = getattr(_qsa_ops(), "lightning_indexer", lightning_indexer_v2)
        blocks, _ = op(
            q.contiguous(),
            cache,
            self.weights,
            512,
            block_table=self.pages,
            metadata=self.meta,
            **self.options
        )
        blocks = torch.where(self.valid, blocks.squeeze(1), -1)
        if self.query_rows is not None:
            blocks = self.short_blocks.index_copy(0, self.query_rows, blocks)
        return blocks


class NPUQSAMetadata:
    """Batch-owned inputs for block selection and attention on NPU."""

    def __init__(self, backend, indexer_metadata, batch, mode, spec_info):
        """Prepare paged GBSA inputs or group prefill queries by request."""
        config = backend.runner.model_config
        tp = get_parallel().attn_tp_size
        heads = (
            config.get_num_attention_heads(tp),
            config.get_num_kv_heads(tp),
            config.head_dim,
        )
        self.index_heads = backend.qsa_profile.n_heads
        # Each live request writes distinct slots in the four-token pending ring.
        self.direct_ring_store = indexer_metadata.is_cuda_graph and (
            mode.is_decode()
            or (mode.is_target_verify() and spec_info.draft_token_num <= 4)
        )
        self.indexer = None
        self.short_blocks = None
        self.paged = None
        self.prefill = []
        lengths = indexer_metadata.sequence_lengths
        pages = (
            indexer_metadata.graph_compressed_page_table
            if indexer_metadata.is_cuda_graph
            else indexer_metadata.decode_page_table
        )
        if pages is None:
            pages, _ = compressed_decode_view(
                compressed_page_size=(
                    indexer_metadata.token_to_kv_pool.qsa_compressed_page_size
                ),
                compress_ratio=indexer_metadata.compress_ratio,
                sequence_lengths=lengths,
                token_slot_table=indexer_metadata.token_slot_table,
            )
        if indexer_metadata.decode_logical_positions is not None or indexer_metadata.is_cuda_graph:
            paged = True
        else:
            pos = batch.positions[0] if batch.positions.ndim == 2 else batch.positions
            self.visible = pos[: indexer_metadata.token_to_batch_idx.numel()].int() + 1
            paged = (
                getattr(spec_info, "positions", None) is not None
                or mode.is_dllm_extend()
                or batch.extend_seq_lens_cpu is None
                or indexer_metadata.prefill_sequence_lengths_cpu is None
                or (
                    batch._original_forward_mode is not None
                    and batch._original_forward_mode.is_decode()
                )
            )
            if paged:
                pages = pages.index_select(0, indexer_metadata.token_to_batch_idx.long())
                lengths = torch.minimum(
                    self.visible, indexer_metadata.get_seqlens_expanded()
                )
        if paged:
            cu = torch.arange(
                lengths.numel() + 1, device=lengths.device, dtype=torch.int64
            )
            self.paged = GBSAMetadata(
                pages, cu, lengths.clamp_min(1), lengths, heads, 1
            )
            if not (
                mode.is_decode() and backend._mtp_shared_sparse_indices is not None
            ):
                # In a verify graph each request owns a fixed-width contiguous
                # suffix. One table row scores all its queries with a causal mask.
                self.width = (
                    int(spec_info.draft_token_num)
                    if indexer_metadata.is_cuda_graph and mode.is_target_verify()
                    else 1
                )
                last = slice(self.width - 1, None, self.width)
                self.indexer = LightningIndexerMetadata(
                    pages[last],
                    lengths[last],
                    lengths,
                    cu[:: self.width].int(),
                    self.index_heads,
                )
            return
        self._prepare_prefill(indexer_metadata, batch.extend_seq_lens_cpu, pages, heads)

    def _prepare_prefill(self, indexer_metadata, sizes, pages, heads):
        """Choose which queries need top-k and which requests use FIA or GBSA."""
        ends = list(accumulate(sizes, initial=0))
        lengths = indexer_metadata.prefill_sequence_lengths_cpu
        scoring_rows, requests, scoring_sizes = [], [], []
        for i, (size, length) in enumerate(zip(sizes, lengths)):
            # Once 513 complete blocks are visible the indexer must rank them.
            count = min(size, max(0, length - (513 * 4 - 1)))
            if count:
                requests.append(i)
                scoring_sizes.append(count)
                scoring_rows.append(range(ends[i + 1] - count, ends[i + 1]))
        total = sum(scoring_sizes)
        if total < ends[-1]:
            ranks = torch.arange(512, device=pages.device, dtype=torch.int32)
            self.short_blocks = torch.where(
                ranks < (self.visible // 4)[:, None], ranks, -1
            )
        if total:
            rows = None
            visible = self.visible
            if total < ends[-1]:
                rows = torch.cat(
                    [
                        torch.arange(span.start, span.stop, device=pages.device)
                        for span in scoring_rows
                    ]
                )
                visible = visible.index_select(0, rows)
            req = torch.tensor(requests, device=pages.device)
            cu_q = torch.tensor(
                list(accumulate(scoring_sizes, initial=0)),
                device=pages.device,
                dtype=torch.int32,
            )
            self.indexer = LightningIndexerMetadata(
                pages.index_select(0, req),
                indexer_metadata.sequence_lengths.index_select(0, req),
                visible,
                cu_q,
                self.index_heads,
                query_rows=rows,
                short_blocks=self.short_blocks,
            )

        live = (i for i, size in enumerate(sizes) if size)
        for long_prefix, group in groupby(
            live, lambda i: lengths[i] - sizes[i] >= 24 * 1024
        ):
            group = list(group)
            if long_prefix:
                rows = slice(ends[group[0]], ends[group[-1] + 1])
                req = torch.tensor(group, device=pages.device)
                cu_q = torch.tensor(
                    list(accumulate((sizes[i] for i in group), initial=0)),
                    device=pages.device,
                    dtype=torch.int64,
                )
                plan = GBSAMetadata(
                    pages.index_select(0, req),
                    cu_q,
                    indexer_metadata.sequence_lengths.index_select(0, req),
                    self.visible[rows],
                    heads,
                    max(sizes[i] for i in group),
                )
                self.prefill.append((rows, None, None, plan))
            else:
                for i in group:
                    self.prefill.append(
                        (
                            slice(ends[i], ends[i + 1]),
                            pages[i : i + 1],
                            lengths[i],
                            None,
                        )
                    )

    def refresh_graph(self, indexer_metadata):
        """Refresh GBSA lengths and any indexer inputs for the next graph replay."""
        lengths = indexer_metadata.sequence_lengths
        self.paged.refresh_graph(lengths)
        if self.indexer is not None:
            last = slice(self.width - 1, None, self.width)
            self.indexer.refresh(
                indexer_metadata.graph_compressed_page_table[last], lengths[last], lengths
            )

    @staticmethod
    def store_ring(pool, layer_id, slots, keys, positions):
        """Keep token keys and positions until their four-token block is complete."""
        from sgl_kernel_npu.operators.mem_cache import qsa_store_cache_rows
        from sglang.srt.layers.attention.qsa.metadata import build_rope_position_matrix

        key_cache = pool.get_qsa_key_state_buffer(layer_id).unsqueeze(1)
        rope_cache = pool.qsa_rope_position_buffer[:, None, None, :]
        qsa_store_cache_rows(key_cache, slots, keys)
        qsa_store_cache_rows(
            rope_cache, slots, build_rope_position_matrix(positions, keys.size(0))
        )

    def select(self, q, pool, layer_id):
        """Keep all blocks when they fit, or let the indexer choose up to 512."""
        if self.indexer is None:
            return self.short_blocks
        cache = pool.get_qsa_compressed_k_buffer(layer_id).view(
            -1, pool.qsa_compressed_page_size, 1, 128
        )
        return self.indexer.select(q, cache)

    def attend(self, q, k, v, blocks, scale):
        """Run each group's attention kernel and return results in query order."""
        k, v = k.view(-1, 128, *k.shape[-2:]), v.view(-1, 128, *v.shape[-2:])
        if self.paged is not None:
            return self.paged.attend(q, k, v, blocks, scale)
        out = torch.empty_like(q, memory_format=torch.contiguous_format)
        for rows, pages, length, plan in self.prefill:
            if plan is None:
                _fia(
                    q[rows],
                    k,
                    v,
                    blocks[rows],
                    self.visible[rows],
                    pages,
                    length,
                    scale,
                    out[rows],
                )
            else:
                out[rows] = plan.attend(q[rows], k, v, blocks[rows], scale)
        return out


class DraftBlockIndices(QSAMTPSharedSparseIndices):
    """Reuse the captured block IDs during draft decode. GBSA reads the partial tail."""

    def __init__(self, *, layer_ids, num_requests, tail_width, device, token_topk):
        """The caller passes 2051 token slots (2048 + 3 for the tail).
        The draft cache only needs the 512 full block IDs (2051 // 4).
        """
        super().__init__(
            layer_ids=layer_ids,
            num_requests=num_requests,
            token_topk=token_topk // 4,
            tail_width=0,
            device=device,
        )
        self.indices.fill_(-1)
        self.max_new_blocks = (tail_width + 3) // 4

    def lookup(self, req_pool_indices, current_positions, layer_id):
        """Reuse saved blocks and add any completed by the latest draft tokens."""
        slot = self.layer_slots[int(layer_id)]
        rows = req_pool_indices.long()
        blocks = self.indices[slot, rows]
        first = self.captured_len[slot, rows] // 4
        end = (current_positions + 1) // 4
        for offset in range(self.max_new_blocks):
            block = first + offset
            # argmin picks an empty slot (-1) first, then the oldest block (lowest ID).
            dst = blocks.argmin(dim=1, keepdim=True)
            blocks.scatter_(
                1,
                dst,
                torch.where(
                    (block < end)[:, None], block[:, None], blocks.gather(1, dst)
                ),
            )
        return blocks
