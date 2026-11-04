"""Architecture/shape contracts without importing an accelerator runtime."""

import importlib.util
from pathlib import Path
import sys
import unittest


def load(name):
    path = Path(__file__).resolve().parents[4] / "python/sglang/srt/offloading/moe/npu" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


profiles, plans = load("model_profile"), load("capabilities")


class TestModelCapacityPlan(unittest.TestCase):
    def configs(self):
        for kind, e, k, layers, hidden, mid, first in (
            ("deepseek_v4", 256, 6, 43, 4096, 2048, 0),
            ("glm_moe_dsa", 256, 8, 78, 6144, 2048, 3),
            ("glm5_next", 288, 8, 45, 4096, 2048, 3),
            ("deepseek_v41", 384, 6, 40, 5120, 2304, 0),
            ("kimi_k3", 896, 16, 93, 7168, 3072, 1),
        ):
            text = dict(model_type=kind, n_routed_experts=e, num_experts_per_tok=k,
                        num_hidden_layers=layers, hidden_size=hidden, moe_intermediate_size=mid,
                        first_k_dense_replace=first, moe_layer_freq=1, hidden_act="silu")
            if kind == "glm_moe_dsa":
                text["architectures"] = ["GlmMoeDsaForCausalLM"]
            if kind == "kimi_k3":
                text.update(num_experts=e, num_experts_per_token=k, hidden_act="situ", routed_expert_hidden_size=3584)
                del text["n_routed_experts"], text["num_experts_per_tok"]
            yield dict(model_type=kind, text_config=text) if kind in ("kimi_k3", "glm5_next", "deepseek_v41") else text

    def test_all_five_architectures_all_ep_and_batch_shapes(self):
        for cfg in self.configs():
            p = profiles.inspect_model(cfg)
            for ep in (1, 2, 4, 8, 16):
                for bs in range(1, 9):
                    with self.subTest(model=p.model_id, ep=ep, bs=bs):
                        plan = plans.plan_capacity(global_experts=p.global_experts, ep_size=ep,
                            offloaded_layers=len(set(p.moe_layer_ids) - set(p.resident_layer_ids)), cache_ratio=.5)
                        self.assertEqual(plan.local_experts * ep, p.global_experts)
                        # More tokens increase work, not the fixed weight capacity.
                        self.assertLessEqual(min(p.global_experts, bs * p.top_k), plan.execution_slots * ep)
                        caps = plans.resolve_decode_capabilities(quant_kind="w8a8", local_experts=plan.local_experts,
                            execution_slots=plan.execution_slots, hot_slots=plan.hot_slots)
                        self.assertGreater(caps.direct_hot_slots, 0)
                        self.assertLessEqual(caps.direct_hot_slots + plan.execution_slots, 1024)
                        slices = plans.descriptor_slices(plan.local_experts, 3)
                        covered = [i for a, b in slices for i in range(a, b)]
                        self.assertEqual(covered, list(range(3 * plan.local_experts)))
                        self.assertTrue(all(0 < b-a <= 256 for a,b in slices))

    def test_capacity_recognition_does_not_enable_unvalidated_runtime(self):
        profiles_ = [profiles.inspect_model(c) for c in self.configs()]
        for p in profiles_[:2]:
            p.require_runtime()
        for p in profiles_[2:]:
            with self.assertRaisesRegex(ValueError, "capacity planning only"):
                p.require_runtime()
        self.assertEqual(profiles_[-1].expert_hidden_size, 3584)
        self.assertEqual(profiles_[-1].hidden_size, 7168)
        self.assertEqual(profiles_[-1].top_k, 16)
        self.assertEqual(profiles_[3].resident_layer_ids, ())

    def test_invalid_shapes_and_prefill_budget_fail_at_startup(self):
        base = dict(global_experts=288, ep_size=8, offloaded_layers=42, cache_ratio=.5)
        for change in (dict(ep_size=7), dict(execution_slots=35), dict(cache_ratio=.0001),
                       dict(cache_ratio=.001), dict(cache_ratio=True), dict(decode_mask_capacity=0)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                plans.plan_capacity(**(base | change))
        # An explicitly budgeted second execution bank can serve tiny caches.
        plan = plans.plan_capacity(**(base | dict(cache_ratio=.001, execution_slots=72)))
        self.assertEqual(plan.hot_slots, 1)
        for cfg in self.configs():
            text = cfg.get("text_config", cfg)
            text["hidden_act"] = "unknown"
            if cfg["model_type"] in ("deepseek_v4", "glm_moe_dsa"):
                with self.assertRaisesRegex(ValueError, "SiLU"):
                    profiles.inspect_model(cfg).require_runtime()

    def test_decode_rows_follow_token_and_padding_bounds_not_eight_requests(self):
        args = dict(request_bound=8, graph_bound=8, tokens_per_request=1, dispatch_padding=128)
        self.assertEqual(plans.plan_decode_mask(**args), 128)
        self.assertEqual(plans.plan_decode_mask(**(args | dict(request_bound=256))), 256)
        self.assertEqual(plans.plan_decode_mask(**(args | dict(graph_bound=64, tokens_per_request=3))), 192)
        with self.assertRaisesRegex(ValueError, "planned token bound"):
            plans.plan_decode_mask(**args, configured=8)


if __name__ == "__main__":
    unittest.main()
