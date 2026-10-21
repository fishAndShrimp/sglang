"""Execute the real hook functions without importing unrelated accelerator code."""

import ast
import contextlib
import pathlib
import sys
import types
import unittest
from unittest import mock

import torch


ROOT = pathlib.Path(__file__).resolve().parents[4]


def load_function(relative, name, namespace, class_name=None):
    tree = ast.parse((ROOT / relative).read_text())
    nodes = tree.body
    if class_name is not None:
        nodes = next(
            node.body
            for node in nodes
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
    node = next(
        node
        for node in nodes
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, relative, "exec"), namespace)
    return namespace[name]


class V4LoadingHookTests(unittest.TestCase):
    def context(self, stage):
        return load_function(
            "python/sglang/srt/model_loader/loader.py",
            "device_loading_context",
            {
                "torch": torch,
                "contextmanager": contextlib.contextmanager,
                "stage_module_for_post_load": stage,
                "is_pin_memory_available": lambda: True,
            },
        )

    def test_post_load_context_preserves_registered_host_pointer(self):
        layer = torch.nn.Module()
        layer.register_parameter(
            "weight",
            torch.nn.Parameter(torch.zeros(32, dtype=torch.uint8), requires_grad=False),
        )
        layer._moe_offload_storage = object()
        pointer = layer.weight.data_ptr()
        context = self.context(
            mock.Mock(side_effect=AssertionError("full layer staging"))
        )
        with context(layer, types.SimpleNamespace(type="npu")) as module:
            self.assertIs(module, layer)
            self.assertEqual(layer.weight.data_ptr(), pointer)

    def test_resident_context_still_calls_existing_stager(self):
        stage = mock.Mock(
            side_effect=lambda module, *args, **kwargs: contextlib.nullcontext(module)
        )
        context = self.context(stage)
        layer = torch.nn.Module()
        with context(layer, types.SimpleNamespace(type="npu")):
            pass
        stage.assert_called_once()

    def method(self, name):
        return load_function(
            "python/sglang/srt/hardware_backend/npu/quantization/fp4_moe_methods.py",
            name,
            {"torch": torch},
            "NPUW4A8MXFP4FusedMoEMethod",
        )

    def test_create_hook_runs_before_any_expert_allocation(self):
        hook = mock.Mock(return_value=True)
        offloader = types.ModuleType("sglang.srt.utils.offloader")
        offloader.get_offloader = lambda: types.SimpleNamespace(
            backend=types.SimpleNamespace(prepare_npu_fp4_weights=hook)
        )
        fused = types.ModuleType("sglang.srt.layers.moe.fused_moe_triton")
        fused.FusedMoeWeightScaleSupported = object()
        modules = {offloader.__name__: offloader, fused.__name__: fused}
        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch.object(
                torch, "empty", side_effect=AssertionError("eager allocation")
            ),
        ):
            layer = torch.nn.Module()
            method = object()
            self.method("create_weights")(
                method, layer, 128, 7168, 2048, torch.bfloat16, weight_loader=object()
            )
        self.assertIs(hook.call_args.args[0], layer)
        self.assertIs(hook.call_args.kwargs["method"], method)
        self.assertEqual(len(tuple(layer.parameters())), 0)

    def test_post_load_hook_delegates_without_touching_full_layer(self):
        npu = types.ModuleType("sglang.srt.hardware_backend.npu.utils")
        npu.NPUACLFormat = object()
        storage = types.SimpleNamespace(finalize=mock.Mock())
        with mock.patch.dict(sys.modules, {npu.__name__: npu}):
            self.method("process_weights_after_loading")(
                object(), types.SimpleNamespace(_moe_offload_storage=storage)
            )
        storage.finalize.assert_called_once()


if __name__ == "__main__":
    unittest.main()
