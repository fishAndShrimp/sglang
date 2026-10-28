"""Semantic invariants of the compact planner; NPU copy checked in serving."""
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.offloading.moe.npu.single_token import SingleTokenProgram, active_expert_ids, split_hit_miss


class TestSingleTokenPlanner(unittest.TestCase):
    def planner(self):
        program = SingleTokenProgram.__new__(SingleTokenProgram)
        program.capacity, program.local_experts = 6, 64
        program.fast_meta = False
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
            self.assertTrue(torch.equal(ids, active_expert_ids(counts, 6, float_sort=True)))

    def test_host_and_hot_addresses_and_lfu_demand(self):
        p = self.planner()
        counts = torch.zeros(64, dtype=torch.int64)
        active = [0, 1, 6, 35, 50, 63]
        counts[active] = 1
        indices, needed, _, _ = p._prepare(SimpleNamespace(counts=counts), None)
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
            _, needed, _, _ = p._prepare(SimpleNamespace(counts=counts), None)
            self.assertNotEqual(p.error.item(), 0)
            self.assertFalse(needed.any())

    def test_capture_duplicates_suppressed_replay_duplicates_fault(self):
        for replay in (False, True):
            p = self.planner()
            p.state.update_enabled.fill_(replay)
            counts = torch.zeros(64, dtype=torch.int64)
            counts[1] = 2
            _, needed, _, _ = p._prepare(SimpleNamespace(counts=counts), None)
            self.assertEqual(p.error.item() != 0, replay)
            self.assertFalse(needed.any())

    def test_all_hit_miss_patterns_preserve_rows(self):
        # Exercise empty ranks, all-hit/all-miss, mixed sets, arbitrary Hot order,
        # and absent rows. This verifies GMM grouping and inverse permutations.
        for count in range(7):
            for bits in range(1 << count):
                needed = torch.arange(6) < count
                hit = torch.tensor([bool(bits & (1 << i)) for i in range(6)])
                slots = torch.tensor([4, 2, 5, 0, 3, 1])
                hits, misses, ho, mo = split_hit_miss(needed, hit, slots, 6)
                fast = split_hit_miss(needed, hit, slots, 6, float_sort=True)
                self.assertTrue(all(torch.equal(a, b) for a, b in zip((hits, misses, ho, mo), fast)))
                expected_groups = torch.zeros(6, dtype=torch.int64).scatter_add_(0, slots, hits.long())
                actual_groups = ((torch.arange(6).unsqueeze(1) == slots.unsqueeze(0)) & hits).sum(1)
                self.assertTrue(torch.equal(expected_groups, actual_groups))
                nh, nm = int(hits.sum()), int(misses.sum())
                self.assertEqual(slots[ho[:nh]].tolist(), sorted(slots[hits].tolist()))
                self.assertEqual(mo[:nm].tolist(), torch.where(misses)[0].tolist())
                values = torch.arange(1, 7, dtype=torch.float32)
                hot_output = values[ho].clone()
                miss_output = values[mo].clone()
                hot_output[nh:] = 0
                miss_output[nm:] = 0
                combined = hot_output[ho.argsort()] + miss_output[mo.argsort()]
                self.assertTrue(torch.equal(combined, torch.where(needed, values, 0)))

    def test_fast_meta_matches_original_planner(self):
        for active in ([], [1], [0, 1, 6, 35, 50, 63]):
            slow, fast = self.planner(), self.planner()
            fast.fast_meta = True
            counts = torch.zeros(64, dtype=torch.int64)
            counts[active] = 1
            a = slow._prepare(SimpleNamespace(counts=counts.clone()), None)
            b = fast._prepare(SimpleNamespace(counts=counts.clone()), None)
            self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)))
            self.assertTrue(torch.equal(slow.state.step_access_count, fast.state.step_access_count))

    def test_small_layout_preserves_native_counts_and_faults_real_overflow(self):
        for size, replay in ((0, True), (6, True), (7, True), (7, False)):
            p = self.planner()
            p.fast_meta, p.device = True, torch.device('cpu')
            p.state.update_enabled.fill_(replay)
            p.zero_offset = torch.zeros(1, dtype=torch.int64)
            p.layout = SimpleNamespace(hidden_size=4, max_rows=24, _owned={
                'counts': torch.empty(64, dtype=torch.int64),
                'output': torch.full((24, 4), float('nan'), dtype=torch.bfloat16)})
            source = torch.zeros(64, dtype=torch.int64)
            source[:size] = 1
            d = SimpleNamespace(decode_row_bound=6, raw=SimpleNamespace(
                hidden_states=torch.ones(24, 4, dtype=torch.bfloat16), masked_m=source))
            layout = p._bind(d, SimpleNamespace(retain=lambda x: None))
            self.assertEqual(int(source.sum()), size)
            self.assertEqual(int(layout.counts.sum()), size if size <= 6 else 0)
            self.assertEqual(bool(p.error), size > 6 and replay)
            self.assertTrue((layout.output == 0).all())


if __name__ == '__main__':
    unittest.main()
