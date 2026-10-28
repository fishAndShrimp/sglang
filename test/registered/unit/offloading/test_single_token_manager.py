"""Semantic invariants of the compact planner; NPU copy checked in serving."""
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.offloading.moe.npu.single_token import SingleTokenProgram, active_expert_ids


class TestSingleTokenPlanner(unittest.TestCase):
    def planner(self):
        program = SingleTokenProgram.__new__(SingleTokenProgram)
        program.capacity, program.local_experts = 6, 64
        program.key_start, program.hot_count, program.host_count = 64, 4, 128
        program.error = torch.tensor(0, dtype=torch.int32)
        program.cache_stats = None
        keys = torch.tensor([65, 70, 99, 127])
        inverse = torch.full((128,), -1, dtype=torch.int64)
        inverse[keys] = torch.arange(4)
        program.state = SimpleNamespace(
            key_to_slot=inverse, slot_to_key=keys,
            slot_valid=torch.ones(4, dtype=torch.bool),
            update_enabled=torch.tensor(True), step_access_count=torch.zeros(128, dtype=torch.int64))
        return program

    def test_active_order_matches_grouped_receive_prefix(self):
        for active in ([], [63], [0, 2, 17], [0, 1, 6, 35, 50, 63]):
            counts = torch.zeros(64, dtype=torch.int64)
            counts[active] = 1
            ids = active_expert_ids(counts, 6)
            self.assertEqual(ids[:len(active)].tolist(), active)
            self.assertEqual((counts[ids] > 0).tolist(), [True]*len(active)+[False]*(6-len(active)))

    def test_host_and_hot_addresses_and_lfu_demand(self):
        p = self.planner()
        counts = torch.zeros(64, dtype=torch.int64)
        active = [0, 1, 6, 35, 50, 63]
        counts[active] = 1
        indices, needed = p._prepare(SimpleNamespace(counts=counts), None)
        # Combined address table is [host sentinel, hosts..., hot sentinel, hot...].
        self.assertEqual(indices.tolist(), [65, 130, 131, 132, 115, 133])
        self.assertTrue(needed.all())
        self.assertEqual(p.error.item(), 0)
        self.assertTrue(torch.equal(p.state.step_access_count[64:], counts))

    def test_stale_active_cache_mapping_faults(self):
        for corrupt in ('key', 'valid', 'range'):
            p = self.planner()
            if corrupt == 'key':
                p.state.slot_to_key[0] = 66
            elif corrupt == 'valid':
                p.state.slot_valid[0] = False
            else:
                p.state.key_to_slot[65] = 9
            counts = torch.zeros(64, dtype=torch.int64)
            counts[1] = 1
            _, needed = p._prepare(SimpleNamespace(counts=counts), None)
            self.assertNotEqual(p.error.item(), 0)
            self.assertFalse(needed.any())

    def test_capture_duplicates_suppressed_replay_duplicates_fault(self):
        for replay in (False, True):
            p = self.planner()
            p.state.update_enabled.fill_(replay)
            counts = torch.zeros(64, dtype=torch.int64)
            counts[1] = 2
            _, needed = p._prepare(SimpleNamespace(counts=counts), None)
            self.assertEqual(p.error.item() != 0, replay)
            self.assertFalse(needed.any())


if __name__ == '__main__':
    unittest.main()
