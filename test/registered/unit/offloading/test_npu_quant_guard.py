"""Run the constructor's real pre-allocation guard without serving imports."""

import ast
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import sglang
import torch
from sglang.srt.offloading.moe.npu.backend import NpuMoEBackend

ROOT = Path(__file__).resolve().parents[4]
FP4_MODULE = "sglang.srt.hardware_backend.npu.quantization.fp4_moe_methods"


def native_method_type():
    relative = "srt/hardware_backend/npu/quantization/fp4_moe_methods.py"
    roots = [ROOT / "python/sglang", *(Path(p) for p in sglang.__path__)]
    source = next(root / relative for root in roots if (root / relative).exists())
    tree = ast.parse(source.read_text())
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == "NPUW4A8MXFP4FusedMoEMethod"
    )
    # The real class is compiled, but its inherited registry/native kernel
    # constructors are unnecessary for a resolved-type check on a CPU host.
    namespace = {
        "__name__": FP4_MODULE,
        "torch": torch,
        "FusedMoEMethodBase": object,
    }
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace[node.name]


def constructor_allocation_block():
    source = ROOT / "python/sglang/srt/layers/moe/fused_moe_triton/layer.py"
    tree = ast.parse(source.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FusedMoE"
    )
    init = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    allocation = next(
        index
        for index, node in enumerate(init.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "create_weights"
    )
    nodes = init.body[allocation - 3 : allocation + 1]
    assert isinstance(nodes[0], ast.ImportFrom)
    assert nodes[0].module == "sglang.srt.utils.offloader"
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    return compile(module, str(source), "exec")


class QuantAllocationGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native_type = native_method_type()
        cls.allocation_block = constructor_allocation_block()

    def setUp(self):
        self.backend = object.__new__(NpuMoEBackend)
        self.backend.resident_layers = 3
        self.backend.layer_ids = (3, 4)
        self.offloader = SimpleNamespace(backend=self.backend)
        fp4 = ModuleType(FP4_MODULE)
        fp4.NPUW4A8MXFP4FusedMoEMethod = self.native_type
        offload = ModuleType("sglang.srt.utils.offloader")
        offload.get_offloader = lambda: self.offloader
        modules = mock.patch.dict(
            sys.modules, {FP4_MODULE: fp4, offload.__name__: offload}
        )
        modules.start()
        self.addCleanup(modules.stop)

    def allocate(self, method, layer_id=3):
        layer = SimpleNamespace(
            layer_id=layer_id,
            quant_method=method,
            num_local_experts=4,
            intermediate_size_per_partition=32,
            weight_loader=object(),
            weight_loader_fused=object(),
        )
        exec(
            self.allocation_block,
            {
                "self": layer,
                "hidden_size": 32,
                "intermediate_size": 32,
                "params_dtype": torch.bfloat16,
                "use_weight_loader_fused": False,
                "with_bias": False,
            },
        )
        return layer

    def test_native_resolved_type_reaches_original_allocation(self):
        method = object.__new__(self.native_type)
        method.create_weights = mock.Mock()
        layer = self.allocate(method)
        method.create_weights.assert_called_once()
        self.assertIs(method.create_weights.call_args.kwargs["layer"], layer)

    def test_fallback_type_is_rejected_before_allocating_any_weights(self):
        class Fp8MoEMethod:
            create_weights = mock.Mock()

        method = Fp8MoEMethod()
        with self.assertRaisesRegex(ValueError, "resolved Fp8MoEMethod"):
            self.allocate(method)
        method.create_weights.assert_not_called()

    def test_similar_class_name_does_not_bypass_type_check(self):
        fake = type("NPUW4A8MXFP4FusedMoEMethod", (), {})()
        fake.create_weights = mock.Mock()
        with self.assertRaises(ValueError):
            self.allocate(fake)
        fake.create_weights.assert_not_called()

    def test_resident_layers_preserve_original_quantization_method(self):
        method = SimpleNamespace(create_weights=mock.Mock())
        self.allocate(method, layer_id=2)
        method.create_weights.assert_called_once()

    def test_other_offloaders_preserve_original_allocation(self):
        for offloader in (object(), SimpleNamespace(backend=object())):
            with self.subTest(offloader=offloader):
                self.offloader = offloader
                method = SimpleNamespace(create_weights=mock.Mock())
                self.allocate(method)
                method.create_weights.assert_called_once()

    def test_out_of_catalog_layer_is_rejected_before_allocation(self):
        method = object.__new__(self.native_type)
        method.create_weights = mock.Mock()
        with self.assertRaisesRegex(ValueError, "outside the fixed catalog"):
            self.allocate(method, layer_id=5)
        method.create_weights.assert_not_called()


if __name__ == "__main__":
    unittest.main()
