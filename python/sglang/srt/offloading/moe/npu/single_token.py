"""Experimental global-single-token executor, independent of generic waves.

Retains the existing Hot pool, LFU policy, prefill, restore, and graph lifecycle.
Only active expert keys enter the decode planner. Six fixed execution slots
replace 64 expert-indexed slots. Device counts and pointers never visit the CPU.
"""

from types import SimpleNamespace

import torch

from ...transfer import EventCompletion, TransferTicket
from ..config import ExpertWave, Phase
from ..resources import tensor_allocations
from .experts import NPUExpertBank, NPUExpertRunner, NPUWaveInput, _compute_scope
from .transfer import _group_pack_copy


def active_expert_ids(counts, capacity):
    """Selected experts in receive-prefix order, followed by inactive fillers."""
    ids = torch.arange(counts.numel(), dtype=torch.int64, device=counts.device)
    return torch.argsort(torch.where(counts > 0, ids, ids + counts.numel()))[:capacity]


class SingleTokenProgram:
    def __init__(self, experts, *, dispatcher, layout, state, error,
                 catalog, address_catalog, source_table, execution, cache_stats=None):
        self.dispatcher, self.layout = dispatcher, layout
        self.state, self.error = state, error
        self.layer = experts.layer_id
        self.key_start = catalog.key(self.layer, 0)
        self.local_experts = experts.num_local_experts
        self.device = error.device
        self.capacity = 6
        if not dispatcher.global_bs1 or execution.capacity < self.capacity:
            raise ValueError("Single-token executor requires global BS=1 and six execution slots")
        self.source_table = source_table
        self.hot_count = address_catalog.hot_count
        self.host_count = address_catalog.host_count
        if not self.hot_count:
            raise ValueError("Initial single-token experiment requires a nonempty Hot pool")
        self.cache_stats = cache_stats
        self._owned = {}
        self.rows = self._own("rows", torch.arange(self.capacity, device=self.device))
        self.valid = self._own("valid", torch.zeros(self.capacity, dtype=torch.bool, device=self.device))
        self.groups = self._own("groups", torch.zeros(self.capacity, dtype=torch.int64, device=self.device))
        self.packed = self._own("packed", torch.zeros(
            len(address_catalog.component_nbytes), self.capacity, dtype=torch.int64, device=self.device))
        self.scan = self._own("scan", torch.tensor(self.capacity, dtype=torch.uint32, device=self.device))
        bank = execution.view(0, self.capacity)
        destinations = torch.tensor(
            [[region.address for region in bank.component_regions[name]]
             for name in address_catalog.component_nbytes], dtype=torch.int64, device=self.device)
        self.destinations = self._own("destinations", destinations)
        self.lengths = self._own("lengths", torch.tensor(
            [[size] * self.capacity for size in address_catalog.component_nbytes.values()],
            dtype=torch.uint32, device=self.device))
        self.bindings = SimpleNamespace(
            expert_bank=NPUExpertBank(bank.quant_info(), self.valid, error), error_status=error)
        self.runner = NPUExpertRunner(experts)
        self.wave = ExpertWave(0, 0, self.capacity, 0)

    def _own(self, name, value):
        self._owned[name] = value
        return value

    def allocations(self):
        return tensor_allocations(f"bs1.{self.layer}.", self._owned.items())

    def _prepare(self, layout, context):
        counts = layout.counts
        # With one global token and distinct top-k experts, each count is 0/1.
        # Synthetic capture can violate this; real replay must fault instead.
        duplicate = (counts > 1).any()
        self.error.bitwise_or_((duplicate & self.state.update_enabled).to(torch.int32) * 512)
        counts.masked_fill_(duplicate | (self.error != 0), 0)
        ids = active_expert_ids(counts, self.capacity)
        needed = counts.gather(0, ids) > 0
        keys = ids + self.key_start
        slots = self.state.key_to_slot.gather(0, keys)
        safe_slots = slots.clamp(0, self.hot_count - 1)
        hit = slots >= 0
        coherent = self.state.slot_valid.gather(0, safe_slots) & (
            self.state.slot_to_key.gather(0, safe_slots) == keys)
        bad = needed & ((slots < -1) | (slots >= self.hot_count) | (hit & ~coherent))
        self.error.bitwise_or_(bad.any().to(torch.int32) * 1024)
        needed = needed & (self.error == 0)
        # Cache maintenance cannot run between this lookup and the layer FFN:
        # it is serialized at graph tail, after all compute and weight readers.
        # Thus generation cannot change during this lease-free layer schedule.
        source_indices = torch.where(hit, safe_slots + self.host_count + 2, keys + 1)
        frequencies = self.state.step_access_count[
            self.key_start:self.key_start + self.local_experts]
        frequencies.add_(counts).clamp_(max=torch.iinfo(torch.int64).max // 4)
        if self.cache_stats is not None:
            self.cache_stats.record_counts(needed.to(torch.int64), hit,
                                           self.state.update_enabled & (self.error == 0))
        return source_indices, needed

    def _copy(self, source_indices, needed, context):
        # The compute frontier covers the preceding layer's FFN readers too.
        # Reusing the same six slots never races a previous reader.
        ready = context.record_compute_completion()
        context.wait_for(ready, context.copy_stream)
        with torch.npu.stream(context.copy_stream):
            src = self.source_table.gather(
                1, source_indices.unsqueeze(0).expand(self.source_table.shape[0], -1))
            src = torch.where(needed.unsqueeze(0), src, 0)
            self.groups.copy_(needed)
            self.packed.zero_()
            context.retain((self, src, source_indices, needed, ready))
            # needed is a PREFIX, so native packed destination j is fixed slot j.
            for component in range(self.source_table.shape[0]):
                status = _group_pack_copy(
                    src[component], self.destinations[component], self.lengths[component],
                    self.scan, self.groups, self.packed[component], self.device)
                if status is not None and status != 0:
                    raise RuntimeError(f"Single-token weight copy failed: {status}")
            event = torch.npu.Event()
            event.record(context.copy_stream)
            ticket = TransferTicket(EventCompletion(event), (self, src, event))
            context.retain_completion(ticket)
            return ticket

    def run(self, layer, hidden_states, routing, context):
        if context.phase is not Phase.DECODE or routing.topk_ids.shape[1] != self.capacity:
            raise ValueError("Single-token executor only supports decode with top-k=6")
        with _compute_scope(context, self.device):
            dispatched = self.dispatcher.dispatch(hidden_states, routing, context)
            layout = self.layout.bind(dispatched, context)
            if layout.compact_rows != self.capacity or dispatched.raw.hidden_states_scale is not None:
                raise ValueError("Single-token executor requires a six-row BF16 dispatch prefix")
            source_indices, needed = self._prepare(layout, context)
            copy = self._copy(source_indices, needed, context)
            x = dispatched.raw.hidden_states[:self.capacity].clone()
            count = needed.sum()
            x.masked_fill_((self.rows >= count).unsqueeze(1), 0)
            self.valid.copy_(needed)
            wave_input = NPUWaveInput(x, None, needed.to(torch.int64), layout.counts,
                                     count, layout.offsets[0], self.wave)
            context.wait_for(copy, context.compute_stream)
            execution = self.runner(layer, wave_input, self.bindings, context)
            context.retain(execution)
            context.wait_for(execution.read_done, context.compute_stream)
            layout.output[:self.capacity].copy_(execution.output)
            combined = self.dispatcher.combine(self.layout.combine_input(layout, context), context)
            return self.dispatcher.finish(combined, context)
