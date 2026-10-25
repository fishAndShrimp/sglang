"""Run NPU graph backend lifecycle methods without importing the NPU stack."""

import ast
import sys
import unittest
from contextlib import nullcontext
from functools import wraps
from threading import RLock
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
from test_moe_graph_leases import Owner

from sglang.srt.offloading.moe.graph import GraphLeaseBackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestNPUMoEGraphLifetime(unittest.TestCase):
    def test_backend_without_moe_preserves_existing_constructor(self):
        source = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/hardware_backend/npu/graph_runner/npu_cudagraph_backend.py"
        )
        tree = ast.parse(source.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        init = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        )
        namespace = {
            "RLock": RLock,
            "ThreadPoolExecutor": lambda **kwargs: None,
            "get_parallel": lambda: SimpleNamespace(tp_group=None),
            "TorchMemorySaverAdapter": SimpleNamespace(create=lambda **kwargs: None),
        }
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[future, init], type_ignores=[])
                ),
                str(source),
                "exec",
            ),
            namespace,
        )
        device = SimpleNamespace(
            current_device=lambda: 0, set_device=lambda device: None
        )
        for owner in (None, object()):
            runner = SimpleNamespace(device_module=device)
            if owner is not None:
                runner.model_runner = SimpleNamespace(
                    offloader=SimpleNamespace(owner=owner)
                )
            backend = SimpleNamespace()
            namespace["__init__"](backend, runner)
            self.assertEqual(backend._pending_graphs, {})
            self.assertFalse(hasattr(backend, "_moe_graph_leases"))

    def _graph_backend(self):
        source = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/hardware_backend/npu/graph_runner/npu_cudagraph_backend.py"
        )
        tree = ast.parse(source.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        methods = [
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef)
            and n.name in ("capture_one", "can_run", "cleanup", "stop_submissions")
        ]
        log = []
        graphs = {}

        class Graph:
            def reset(self):
                log.append("graph_reset")

            def __del__(self):
                log.append("graph_destroy")

        npu = SimpleNamespace(
            NPUGraph=Graph, graph=lambda *args, **kwargs: nullcontext()
        )
        namespace = {
            "torch": SimpleNamespace(npu=npu),
            "empty_context": nullcontext,
            "wraps": wraps,
        }
        submission = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_submission"
        )
        # Postponed annotations avoid importing graph runner/typing dependencies.
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[future, submission] + methods, type_ignores=[])
                ),
                str(source),
                "exec",
            ),
            namespace,
        )
        backend = SimpleNamespace(
            _submission_lock=RLock(),
            _submissions_closed=False,
            _graphs=graphs,
            _pending_graphs={},
            _outputs={},
            _pool=None,
            _capture_stream=None,
            _device_module=SimpleNamespace(synchronize=lambda: log.append("sync")),
            _tp_group=SimpleNamespace(barrier=lambda: None),
            _enable_torch_compile=False,
            _memory_saver_adapter=None,
            _update_executor=SimpleNamespace(
                shutdown=lambda **kwargs: log.append("shutdown")
            ),
        )
        for name, method in namespace.items():
            if name in ("capture_one", "can_run", "cleanup", "stop_submissions"):
                setattr(backend, name, MethodType(method, backend))
        backend.capture_session = lambda stream: nullcontext()
        owner = Owner(log)
        owner.runtime.synchronize = lambda: log.append("owner_drain")
        original_release = owner._release_graph

        def release(key):
            self.assertFalse(graphs)
            self.assertFalse(backend._pending_graphs)
            self.assertIn("graph_reset", log)
            log.append("lease_close")
            original_release(key)

        owner._release_graph = release
        wrapper = GraphLeaseBackend(backend, owner, object())
        return wrapper, backend, owner, log

    def test_graph_lease_lives_until_cleanup_after_device_drain(self):
        wrapper, backend, owner, log = self._graph_backend()
        # Only this key is patched; avoid reverting unrelated torch module imports.
        with patch.dict(sys.modules, {"torch_npu": ModuleType("torch_npu")}):
            with wrapper.capture_session("capture"):
                wrapper.capture_one("bucket", lambda: torch.ones(1))
        self.assertEqual(len(owner.graphs), 1)
        self.assertEqual(log.count("lease_close"), 0)
        wrapper.cleanup()
        self.assertLess(log.index("shutdown"), log.index("owner_drain"))
        self.assertLess(log.index("shutdown"), log.index("graph_destroy"))
        self.assertLess(log.index("owner_drain"), log.index("graph_reset"))
        self.assertLess(log.index("graph_destroy"), log.index("lease_close"))
        self.assertFalse(owner.graphs)

    def test_failed_capture_is_retained_but_not_replayable_until_cleanup(self):
        wrapper, backend, owner, log = self._graph_backend()
        calls = []

        def forward():
            calls.append(True)
            if len(calls) == 3:  # Warmups succeed; fail inside actual capture.
                raise RuntimeError("capture failed")

        with patch.dict(sys.modules, {"torch_npu": ModuleType("torch_npu")}):
            with self.assertRaisesRegex(RuntimeError, "capture failed"):
                with wrapper.capture_session("capture"):
                    try:
                        wrapper.capture_one("bucket", forward)
                    except RuntimeError:
                        self.assertIn("bucket", backend._pending_graphs)
                        self.assertFalse(backend.can_run(None, "bucket"))
                        self.assertNotIn("lease_close", log)
                        raise
        self.assertTrue(owner.failed)
        self.assertFalse(owner.graphs)
        self.assertLess(log.index("graph_reset"), log.index("lease_close"))

    def test_duplicate_capture_is_rejected_before_warmup(self):
        wrapper, backend, owner, log = self._graph_backend()
        with patch.dict(sys.modules, {"torch_npu": ModuleType("torch_npu")}):
            with wrapper.capture_session("capture"):
                wrapper.capture_one("bucket", lambda: None)
                original_graph = backend._graphs["bucket"]
                original_keys = set(wrapper._leases)
                with self.assertRaisesRegex(ValueError, "existing MoE graph"):
                    wrapper.capture_one(
                        "bucket", lambda: self.fail("Unexpected warmup")
                    )
                self.assertIs(backend._graphs["bucket"], original_graph)
                self.assertEqual(wrapper._leases, original_keys)
        del original_graph
        wrapper.cleanup()
        self.assertFalse(owner.graphs)


if __name__ == "__main__":
    unittest.main()
