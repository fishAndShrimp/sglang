"""Execute the real padding enablement function without serving-only imports."""

import ast
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

from sglang.srt.offloading.moe.offloader import MoEOffloaderV2

ROOT = Path(__file__).resolve().parents[4]


def load_enablement(namespace):
    path = ROOT / "python/sglang/srt/model_executor/forward_batch_info.py"
    tree = ast.parse(path.read_text())
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "enable_num_token_non_padded"
    )
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[node.name]


class OffloadPaddingTests(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(moe_offload_config=None)
        self.parallel = SimpleNamespace(moe_ep_size=1)
        self.backend = SimpleNamespace(is_customized=lambda: False)
        self.offloader = object()
        self.lookup = mock.Mock(side_effect=lambda: self.offloader)
        utils = ModuleType("sglang.srt.layers.moe.utils")
        utils.get_moe_a2a_backend = lambda: self.backend
        offload = ModuleType("sglang.srt.utils.offloader")
        offload.get_offloader = self.lookup
        self.modules = mock.patch.dict(
            sys.modules, {utils.__name__: utils, offload.__name__: offload}
        )
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.gather = mock.Mock(return_value=False)
        self.legacy_masks = mock.Mock(return_value=False)
        self.enabled = load_enablement(
            {
                "get_exec": lambda: SimpleNamespace(offload=self.config),
                "get_parallel": lambda: self.parallel,
                "world_dp_gather_enabled": self.gather,
                "_dp_tp_moe_masks_pad_rows": self.legacy_masks,
            }
        )

    def test_single_rank_json_enables_count_before_offloader_exists(self):
        self.config.moe_offload_config = "/model/offload.json"
        self.lookup.side_effect = AssertionError("offloader not installed yet")
        self.assertTrue(self.enabled())
        self.lookup.assert_not_called()

    def test_single_rank_injected_owner_is_detected_without_cached_state(self):
        self.assertFalse(self.enabled())
        # Constructor resources are unrelated: the exact production class is
        # used to verify the injection check, without instantiating an NPU.
        self.offloader = object.__new__(MoEOffloaderV2)
        self.assertTrue(self.enabled())
        self.offloader = object()
        self.assertFalse(self.enabled())

    def test_subclass_injection_is_accepted(self):
        class Binding(MoEOffloaderV2):
            pass

        self.offloader = object.__new__(Binding)
        self.assertTrue(self.enabled())

    def test_legacy_offloader_with_similar_fields_does_not_enable_count(self):
        self.offloader = SimpleNamespace(owner=object(), backend=object())
        self.assertFalse(self.enabled())
        self.legacy_masks.assert_called_once_with()

    def test_existing_parallel_modes_short_circuit_offloader_lookup(self):
        self.lookup.side_effect = AssertionError("unnecessary offloader lookup")
        self.parallel.moe_ep_size = 2
        self.assertTrue(self.enabled())
        self.parallel.moe_ep_size = 1
        self.gather.return_value = True
        self.assertTrue(self.enabled())
        self.gather.return_value = False
        self.backend.is_customized = lambda: True
        self.assertTrue(self.enabled())
        self.lookup.assert_not_called()

    def test_existing_hip_padding_rule_is_preserved(self):
        self.legacy_masks.return_value = True
        self.assertTrue(self.enabled())


if __name__ == "__main__":
    unittest.main()
