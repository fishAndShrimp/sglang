"""CPU graph ownership tests; do not certify CUDA/NPU graph execution."""

import contextlib
import gc
import unittest
import weakref
from types import SimpleNamespace

from sglang.srt.offloading.base import BaseOffloader
from sglang.srt.offloading.moe.graph import GraphLeaseBackend
from sglang.srt.offloading.moe.interfaces import BoundMoERuntime
from sglang.srt.offloading.moe.offloader import MoEOffloaderV2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class CaptureResources:
    pass


class Runtime:
    def __init__(self, calls):
        self.calls = calls
        self.update_enabled = True
        self.resource_ref = None
        self.fail_drain = False

    @contextlib.contextmanager
    def capture_scope(self, owner, runner, stream):
        self.calls.append(("scope_enter", stream))
        self.update_enabled = False
        resources = CaptureResources()
        self.resource_ref = weakref.ref(resources)
        try:
            yield resources
        finally:
            self.calls.append(("scope_exit",))
            self.update_enabled = True

    def synchronize(self):
        self.calls.append(("drain",))
        if self.fail_drain:
            raise RuntimeError("drain failure")


class Owner:
    def __init__(self, calls):
        self.calls = calls
        self.runtime = Runtime(calls)
        self.graphs = set()
        self.failed = False

    def switch_phase(self, phase):
        self.calls.append(("phase", phase))

    def wait_idle(self):
        self.calls.append(("idle",))

    def fail_graph_capture(self):
        self.failed = True
        self.calls.append(("poison",))

    def acquire_graph(self, key):
        self.graphs.add(key)
        self.calls.append(("lease", key[1]))

        def close():
            self.calls.append(("release", key[1]))
            self.graphs.remove(key)

        return SimpleNamespace(close=close)


class Backend:
    def __init__(self, calls):
        self.calls = calls
        self.graphs = {}
        self.fail_cleanup = False
        self.extra_diagnostic = "preserved"

    @contextlib.contextmanager
    def capture_session(self, stream):
        self.calls.append(("backend_capture_enter", stream))
        try:
            yield
        finally:
            self.calls.append(("backend_capture_exit",))

    def capture_one(self, key, forward_fn, capture_inputs=None, post_warmup_hook=None):
        self.graphs[key] = object()  # Partial captures are tracked before executing.
        for stage in ("warmup0", "warmup1", "capture"):
            self.calls.append((stage, key))
            forward_fn()
            if stage != "capture" and post_warmup_hook is not None:
                post_warmup_hook()

    def can_run(self, batch, key):
        return key in self.graphs

    def replay_session(self):
        return contextlib.nullcontext()

    def replay(self, key, batch, **kwargs):
        self.calls.append(("replay", key, kwargs))
        return (key, batch)

    def replay_with_input_update(self, key, *args, **kwargs):
        self.calls.append(("rebind", key, args, kwargs))
        return key

    def cleanup(self):
        self.calls.append(("destroy",))
        if self.fail_cleanup:
            raise RuntimeError("graph reset failure")
        self.graphs.clear()


class TestGraphLeaseBackend(unittest.TestCase):
    def make_wrapper(self):
        calls = []
        owner = Owner(calls)
        backend = Backend(calls)
        wrapper = GraphLeaseBackend(backend, owner, object())
        return wrapper, owner, backend, calls

    def capture(self, wrapper, owner, key=1):
        with wrapper.capture_session("capture_stream"):
            self.assertFalse(owner.runtime.update_enabled)
            # Runner-side autotune happens before backend.capture_one.
            wrapper.capture_one(
                key, lambda: self.assertFalse(owner.runtime.update_enabled)
            )
        self.assertTrue(owner.runtime.update_enabled)

    def test_capture_guards_warmup_and_retains_resources_until_destroy(self):
        wrapper, owner, backend, calls = self.make_wrapper()
        self.capture(wrapper, owner)
        self.assertTrue(owner.graphs)
        gc.collect()
        self.assertIsNotNone(owner.runtime.resource_ref())
        self.assertEqual(calls.count(("warmup0", 1)), 1)
        self.assertEqual(calls.count(("warmup1", 1)), 1)
        self.assertEqual(calls.count(("capture", 1)), 1)
        wrapper.cleanup()
        self.assertLess(calls.index(("drain",)), calls.index(("destroy",)))
        self.assertLess(calls.index(("destroy",)), calls.index(("release", 1)))
        self.assertEqual(owner.graphs, set())
        gc.collect()
        self.assertIsNone(owner.runtime.resource_ref())
        count = len(calls)
        wrapper.cleanup()
        self.assertEqual(len(calls), count)

    def test_each_bucket_and_each_wrapper_get_a_distinct_owner_lease(self):
        wrapper, owner, backend, calls = self.make_wrapper()
        with wrapper.capture_session("stream"):
            wrapper.capture_one(1, lambda: None)
            wrapper.capture_one(2, lambda: None)
        other = GraphLeaseBackend(Backend(calls), owner, object())
        self.capture(other, owner, key=1)
        self.assertEqual(len(owner.graphs), 3)
        wrapper.cleanup()
        self.assertEqual(len(owner.graphs), 1)
        other.cleanup()
        self.assertEqual(owner.graphs, set())

    def test_capture_failure_unwinds_scopes_before_draining_and_releasing(self):
        wrapper, owner, backend, calls = self.make_wrapper()

        def failed_forward():
            raise RuntimeError("capture op failed")

        with self.assertRaisesRegex(RuntimeError, "capture op failed"):
            with wrapper.capture_session("stream"):
                wrapper.capture_one(7, failed_forward)
        self.assertTrue(owner.failed)
        self.assertEqual(owner.graphs, set())
        self.assertEqual(backend.graphs, {})
        self.assertLess(calls.index(("backend_capture_exit",)), calls.index(("drain",)))
        self.assertLess(calls.index(("scope_exit",)), calls.index(("drain",)))
        self.assertLess(calls.index(("destroy",)), calls.index(("release", 7)))
        with self.assertRaises(RuntimeError):
            wrapper.replay(7, None)

    def test_drain_or_destroy_failure_keeps_graph_lease_for_retry(self):
        for fail_drain in (False, True):
            with self.subTest(fail_drain=fail_drain):
                wrapper, owner, backend, calls = self.make_wrapper()
                self.capture(wrapper, owner)
                owner.runtime.fail_drain = fail_drain
                backend.fail_cleanup = not fail_drain
                with self.assertRaises(RuntimeError):
                    wrapper.cleanup()
                self.assertTrue(owner.graphs)
                self.assertTrue(owner.failed)
                with self.assertRaises(RuntimeError):
                    wrapper.replay(1, None)
                self.assertIsNotNone(owner.runtime.resource_ref())
                self.assertNotIn(("release", 1), calls)
                if fail_drain:
                    self.assertNotIn(("destroy",), calls)
                owner.runtime.fail_drain = backend.fail_cleanup = False
                wrapper.cleanup()
                self.assertEqual(owner.graphs, set())

    def test_failed_capture_without_reset_preserves_lease(self):
        wrapper, owner, backend, calls = self.make_wrapper()

        def unsafe_cleanup():
            raise RuntimeError("No explicit failed graph reset")

        backend.cleanup_failed_capture = unsafe_cleanup
        with self.assertRaisesRegex(RuntimeError, "No explicit failed graph reset"):
            with wrapper.capture_session("stream"):
                wrapper.capture_one(
                    1, lambda: (_ for _ in ()).throw(ValueError("capture"))
                )
        self.assertTrue(owner.failed)
        self.assertTrue(owner.graphs)
        self.assertNotIn(("destroy",), calls)
        del backend.cleanup_failed_capture
        wrapper.cleanup()
        self.assertEqual(owner.graphs, set())

    def test_replay_and_npu_rebind_delegate_without_capture_ticket_reuse(self):
        wrapper, owner, backend, calls = self.make_wrapper()
        self.capture(wrapper, owner)
        before = len(calls)
        with wrapper.replay_session():
            self.assertEqual(wrapper.replay(1, "batch", flag=True), (1, "batch"))
            self.assertEqual(
                wrapper.replay_with_input_update(1, [1, 2], attr_name="seq"), 1
            )
        self.assertEqual(
            calls[before:],
            [
                ("replay", 1, {"flag": True}),
                ("rebind", 1, ([1, 2],), {"attr_name": "seq"}),
            ],
        )
        self.assertEqual(wrapper.extra_diagnostic, "preserved")
        with self.assertRaises(ValueError):
            wrapper.replay(2, None)
        with self.assertRaises(ValueError):
            wrapper.replay_with_input_update(2, [])
        wrapper.cleanup()

    def test_capture_requires_scope_and_duplicate_bucket_cannot_replace_graph(self):
        wrapper, owner, backend, calls = self.make_wrapper()
        with self.assertRaises(RuntimeError):
            wrapper.capture_one(1, lambda: None)
        with wrapper.capture_session("stream"):
            wrapper.capture_one(1, lambda: None)
            with self.assertRaises(ValueError):
                wrapper.capture_one(1, lambda: None)
        self.assertEqual(len(owner.graphs), 1)
        wrapper.cleanup()

    def test_runtime_must_explicitly_bind_capture_scope(self):
        wrapper, owner, backend, calls = self.make_wrapper()
        owner.runtime.capture_scope = lambda *args: BoundMoERuntime.capture_scope(
            owner.runtime, *args
        )
        with self.assertRaises(NotImplementedError):
            with wrapper.capture_session("stream"):
                self.fail("Unbound runtime entered capture")
        self.assertTrue(owner.failed)
        self.assertEqual(owner.graphs, set())

    def test_missing_capture_keepalive_is_rejected(self):
        wrapper, owner, backend, calls = self.make_wrapper()
        owner.runtime.capture_scope = lambda *args: contextlib.nullcontext()
        with self.assertRaisesRegex(TypeError, "keepalive"):
            with wrapper.capture_session("stream"):
                self.fail("Missing resources entered capture")
        self.assertTrue(owner.failed)

    def test_default_offloader_is_unchanged_and_moe_binding_is_opt_in(self):
        wrapper, owner, backend, calls = self.make_wrapper()
        self.assertIs(BaseOffloader().bind_graph_backend(backend, None), backend)
        facade = MoEOffloaderV2(None, None)
        with self.assertRaises(RuntimeError):
            facade.bind_graph_backend(backend, None)
        facade.owner, facade._installed = owner, True
        self.assertIsInstance(
            facade.bind_graph_backend(backend, None), GraphLeaseBackend
        )


if __name__ == "__main__":
    unittest.main()
