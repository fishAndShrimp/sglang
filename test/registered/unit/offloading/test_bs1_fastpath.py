"""CPU semantic checks; real NPU serving/replay is validated separately."""

import unittest
import os
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.offloading.moe.config import ExpertWave, Phase
from sglang.srt.offloading.moe.npu import experts
from sglang.srt.offloading.moe.npu.transfer import _packed_destination_slots, MemFabricDeviceTransferOps
from sglang.srt.offloading.moe.runtime import _require_global_bs1
from sglang.srt.offloading.transfer import DeviceCopyPlan, CopySourceKind


class TestGlobalBS1Admission(unittest.TestCase):
    def test_padding_is_not_a_real_token(self):
        for counts in ([1, 0, 0, 0], [0, 0, 0, 1]):
            _require_global_bs1(SimpleNamespace(original_global_num_tokens_cpu=counts))
        for counts in (None, [1, 1, 1, 1], [2, 0, 0, 0], [0, 0, 0, 0], [-1, 2]):
            with self.assertRaises(RuntimeError):
                _require_global_bs1(SimpleNamespace(original_global_num_tokens_cpu=counts))


class TestPackedDestinations(unittest.TestCase):
    def test_fixed_destination_validation_rejects_aliases_and_bad_generation(self):
        for bad in (None, "duplicate", "out_of_range", "generation", "count"):
            n = 6
            plan = DeviceCopyPlan(
                torch.full((n,), int(CopySourceKind.HOT_SLOT)), torch.arange(n),
                torch.arange(n), torch.ones(n, dtype=torch.bool),
                torch.zeros(n, dtype=torch.int64), torch.tensor(n, dtype=torch.int32), n,
            )
            binding = SimpleNamespace(host_count=n, hot_count=n, destination_count=n,
                hot_generations=torch.zeros(n, dtype=torch.int64), hot_targets_alias=False,
                error_status=torch.tensor(0, dtype=torch.int32), mask=torch.zeros(n, dtype=torch.bool),
                fixed_destination_slots=torch.arange(n))
            if bad == "duplicate":
                plan.destination_slot[1] = 0
            elif bad == "out_of_range":
                plan.destination_slot[1] = n
            elif bad == "generation":
                plan.expected_generation[1] = 1
            elif bad == "count":
                plan.count.fill_(n - 1)
            MemFabricDeviceTransferOps._mask(plan, binding)
            self.assertEqual(binding.error_status.item() == 0, bad is None)
            self.assertEqual(binding.mask.sum().item(), n if bad is None else 0)

    def test_sparse_prefix_including_empty_and_all_active(self):
        for n in (1, 6, 8, 64, 256):
            generator = torch.Generator().manual_seed(n)
            destinations = torch.randperm(n, generator=generator) + 1
            masks = [torch.zeros(n, dtype=torch.bool), torch.ones(n, dtype=torch.bool)]
            if n <= 8:
                masks += [torch.tensor([(bits >> i) & 1 for i in range(n)], dtype=torch.bool)
                          for bits in range(1 << n)]
            else:
                masks += [torch.rand(n, generator=generator) < rate
                          for rate in (.01, .1, .5, .9)]
            for mask in masks:
                expected = torch.zeros(n, dtype=torch.int64)
                active = destinations[mask]
                expected[:active.numel()] = active
                self.assertTrue(torch.equal(_packed_destination_slots(mask, destinations), expected))


class TestCompactLayout(unittest.TestCase):
    def run_layout(self, counts, bound, *, partial=False):
        device = "cpu"
        layout = experts.NPUDispatchLayout(4, 256, 32, device, torch.zeros((), dtype=torch.int32))
        context = SimpleNamespace(phase=Phase.DECODE, capturing=True, retain=lambda x: None)
        x = torch.randn(32, 256, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16)
        total = sum(counts)
        x[total:] = float("nan")
        raw = SimpleNamespace(hidden_states=x, hidden_states_scale=None,
                              format="deepep_ll", masked_m=torch.tensor(counts))
        dispatched = experts.NPUDispatch(raw, None, 256, None, decode_row_bound=bound)
        wave = ExpertWave(0, 0, 2 if partial else 4, 0)
        with patch.object(experts, "_compute_scope", return_value=nullcontext()):
            view = layout.bind(dispatched, context)
            selected = layout.select_wave(view, wave, context)
            expected_rows = 32 if partial or bound is None else bound
            self.assertEqual(selected.hidden_states.shape, (expected_rows, 256))
            count = sum(counts[:2] if partial else counts)
            if bound is not None and total > bound:
                self.assertNotEqual(layout.error_status.item(), 0)
                self.assertEqual(selected.row_count.item(), 0)
            else:
                self.assertEqual(layout.error_status.item(), 0)
                self.assertTrue(torch.equal(selected.hidden_states[:count], x[:count]))
                self.assertTrue((selected.hidden_states[count:] == 0).all())
            output = selected.hidden_states * 2
            layout.write_wave(view, wave, output, context)
            result = layout.combine_input(view, context).output.clone()
        return result

    def test_matches_full_layout_and_clears_nan_padding(self):
        for counts in ([0, 0, 0, 0], [0, 2, 0, 1], [2, 2, 2, 2]):
            self.assertTrue(torch.equal(self.run_layout(counts, 8), self.run_layout(counts, None)))

    def test_overflow_latches_fault_without_truncating_silently(self):
        result = self.run_layout([3, 3, 3, 0], 8)
        self.assertTrue((result == 0).all())

    def test_partial_wave_uses_original_path(self):
        self.assertTrue(torch.equal(self.run_layout([1, 2, 0, 1], 8, partial=True),
                                    self.run_layout([1, 2, 0, 1], None, partial=True)))

    def test_capture_overflow_is_suppressed_but_replay_overflow_faults(self):
        enabled = torch.tensor(False)
        with patch.dict(os.environ, SGLANG_MOE_GLOBAL_BS1="1"):
            layout = experts.NPUDispatchLayout(
                4, 256, 32, "cpu", torch.zeros((), dtype=torch.int32),
                replay_enabled=enabled,
            )
        raw = SimpleNamespace(hidden_states=torch.zeros(32, 256, dtype=torch.bfloat16),
                              hidden_states_scale=None, format="deepep_ll",
                              masked_m=torch.tensor([3, 3, 3, 0]))
        context = SimpleNamespace(phase=Phase.DECODE, capturing=True, retain=lambda x: None)
        dispatched = experts.NPUDispatch(raw, None, 256, None, decode_row_bound=6)
        with patch.object(experts, "_compute_scope", return_value=nullcontext()):
            capture = layout.bind(dispatched, context)
            self.assertEqual(layout.error_status.item(), 0)
            self.assertEqual(capture.counts.sum().item(), 0)
            enabled.fill_(True)
            replay = layout.bind(dispatched, context)
            self.assertNotEqual(layout.error_status.item(), 0)
            self.assertEqual(replay.counts.sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
