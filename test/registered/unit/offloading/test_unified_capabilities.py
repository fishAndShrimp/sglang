"""CPU contracts for predictable offload capability selection."""
import importlib.util
from pathlib import Path
import sys
import unittest


source = Path(__file__).resolve().parents[4] / "python/sglang/srt/offloading/moe/npu/capabilities.py"
spec = importlib.util.spec_from_file_location("_offload_capabilities", source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
resolve = module.resolve_decode_capabilities


class TestUnifiedCapabilities(unittest.TestCase):
    def plan(self, ep, hot, **kwargs):
        e = 256 // ep
        return resolve(quant_kind=kwargs.pop("quant_kind", "w8a8"),
                       local_experts=e, execution_slots=kwargs.pop("execution_slots", e),
                       hot_slots=hot, **kwargs)

    def test_all_target_ep_sizes_keep_direct_hot_and_reuse(self):
        for ep in (1, 2, 4, 8, 16):
            with self.subTest(ep=ep):
                plan = self.plan(ep, 300)
                self.assertEqual(plan.direct_hot_slots, 300)
                self.assertTrue(plan.execution_reuse)

    def test_large_cache_keeps_partial_view_instead_of_disabling_direct(self):
        for ep in (1, 2, 4, 8, 16):
            with self.subTest(ep=ep):
                plan = self.plan(ep, 5000)
                self.assertEqual(plan.execution_slots + plan.direct_hot_slots, 1024)
                self.assertEqual(plan.to_dict()["direct_hot"], "partial")

    def test_format_speculation_and_diagnostic_have_explanations_without_losing_reuse(self):
        for options in ({"quant_kind": "w4a8"}, {"speculative": True},
                        {"disable_direct_hot": True}):
            plan = self.plan(8, 640, **options)
            self.assertEqual(plan.direct_hot_slots, 0)
            self.assertTrue(plan.direct_hot_reasons)
            self.assertTrue(plan.execution_reuse)
            self.assertTrue(plan.w2_overlap)
            self.assertTrue(plan.fused_metadata)

    def test_extra_execution_slots_change_view_offset_not_reuse(self):
        plan = self.plan(4, 900, execution_slots=128)
        self.assertEqual(plan.direct_hot_slots, 896)
        self.assertTrue(plan.execution_reuse)

    def test_model_expert_count_is_not_the_native_copy_batch_limit(self):
        plan = resolve(quant_kind="w8a8", local_experts=512, execution_slots=512, hot_slots=900)
        self.assertEqual(plan.direct_hot_slots, 512)
        self.assertEqual(plan.to_dict()["copy_descriptor_limit"], 256)

    def test_invalid_storage_is_rejected(self):
        for options in ({"hot_slots": 0}, {"execution_slots": 31}):
            args = dict(quant_kind="w8a8", local_experts=32, execution_slots=32, hot_slots=640)
            args.update(options)
            with self.assertRaises(ValueError):
                resolve(**args)


if __name__ == "__main__":
    unittest.main()
