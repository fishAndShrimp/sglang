"""Tensor LFU backend. No routing or popularity decisions are read by Python.

The implementation uses the configured torch device. Its capture/operator support
must be validated by the bound CUDA/NPU runtime. Supplying this policy alone does
not assert graph support.
"""

from dataclasses import dataclass

from .interfaces import CachePolicy


@dataclass
class LFUState:
    frequency: object
    steps: object


class TensorLFUPolicy(CachePolicy):
    def __init__(self, aging_interval=0):
        if aging_interval < 0:
            raise ValueError("aging_interval must be nonnegative; zero disables aging")
        self.aging_interval = aging_interval

    def allocate_state(self, num_experts, device):
        import torch

        if num_experts < 0:
            raise ValueError("num_experts must be nonnegative")
        return LFUState(
            torch.zeros(num_experts, dtype=torch.int64, device=device),
            torch.zeros((), dtype=torch.int64, device=device),
        )

    def update_statistics(self, state, step_counts, update_enabled):
        import torch

        # Clamp before adding to prevent overflow. Every expert, including misses,
        # participates. update_enabled is a fixed-address device bool scalar.
        limit = torch.iinfo(torch.int64).max // 4
        counts = step_counts.clamp(min=0, max=limit)
        frequency = (state.frequency.clamp(max=limit) + counts).clamp(max=limit)
        steps = (state.steps + 1).remainder(max(self.aging_interval, 1))
        if self.aging_interval:
            frequency = torch.where(steps == 0, frequency // 2, frequency)
        state.frequency.copy_(torch.where(update_enabled, frequency, state.frequency))
        state.steps.copy_(torch.where(update_enabled, steps, state.steps))

    def select_hot_set(self, state, key_to_slot, target_keys):
        import torch

        # Stable sort gives deterministic ties: frequency, then existing resident,
        # then canonical key. Sorting workspace is metadata/workspace, not weights;
        # backend binding must include the allocator's peak in its resource budget.
        capacity = target_keys.numel()
        if capacity > state.frequency.numel():
            raise ValueError("Hot capacity exceeds the eligible expert count")
        resident_order = torch.argsort(
            (key_to_slot >= 0).to(torch.int32), descending=True, stable=True
        )
        order = torch.argsort(
            state.frequency[resident_order], descending=True, stable=True
        )
        selected = resident_order[order[:capacity]]
        target_keys.copy_(torch.where(state.frequency[selected] > 0, selected, -1))

    def copy_state(self, source, destination):
        destination.frequency.copy_(source.frequency)
        destination.steps.copy_(source.steps)
