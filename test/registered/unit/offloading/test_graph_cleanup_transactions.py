"""Graph retirement gates shared runtime submissions without closing the model."""

import unittest
import weakref
from contextlib import contextmanager
from threading import Event, Thread
from types import MethodType

from test_moe_framework import FakeRuntime, catalog, completed_step, config
from test_moe_graph_leases import Backend

from sglang.srt.offloading.moe.config import Phase
from sglang.srt.offloading.moe.graph import GraphLeaseBackend
from sglang.srt.offloading.moe.owner import MoEOffloadOwner, OwnerState
from sglang.srt.offloading.moe.runtime import InvocationRuntime, ManagedExecutionContext
from sglang.srt.offloading.transfer import ImmediateCompletion, TransferTicket
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class GraphCleanupTransactionTests(unittest.TestCase):
    def binding(self):
        runtime = FakeRuntime()
        owner = MoEOffloadOwner(config(), catalog(), runtime)
        backend = Backend([])
        wrapper = GraphLeaseBackend(backend, owner, object())
        wrapper._leases[1] = owner.acquire_graph((wrapper._identity, 1))
        backend.graphs[1] = object()
        return owner, backend, wrapper

    def test_successful_reset_allows_same_model_to_create_another_backend(self):
        owner, backend, wrapper = self.binding()
        wrapper.cleanup()
        self.assertEqual(owner.state, OwnerState.READY)
        token = owner.begin_forward(Phase.DECODE, "compute")
        owner.finish_forward(token, completed_step())
        next_lease = owner.acquire_graph((object(), 2))
        next_lease.close()
        owner.close()

    def test_active_forward_rejection_does_not_poison_its_graph(self):
        owner, backend, wrapper = self.binding()
        token = owner.begin_forward(Phase.DECODE, "compute")
        with self.assertRaisesRegex(RuntimeError, "Concurrent"):
            wrapper.cleanup()
        self.assertFalse(wrapper._closing)
        self.assertTrue(wrapper.can_run(None, 1))
        owner.finish_forward(token, completed_step())
        wrapper.cleanup()
        owner.close()

    def test_failed_reset_keeps_gate_and_leases_until_successful_retry(self):
        owner, backend, wrapper = self.binding()
        backend.fail_cleanup = True
        with self.assertRaisesRegex(RuntimeError, "graph reset"):
            wrapper.cleanup()
        self.assertTrue(owner._graphs)
        self.assertTrue(owner._graph_cleanups)
        self.assertEqual(owner.state, OwnerState.FAILED)
        backend.fail_cleanup = False
        wrapper.cleanup()
        self.assertFalse(owner._graphs)
        self.assertFalse(owner._graph_cleanups)
        owner.close()

    def test_overlapping_graph_cleanup_transactions_keep_the_gate_until_last(self):
        owner, backend, wrapper = self.binding()
        first, second = object(), object()
        owner.begin_graph_cleanup(first)
        owner.begin_graph_cleanup(second)
        owner.finish_graph_cleanup(first)
        with self.assertRaises(RuntimeError):
            owner.begin_forward(Phase.DECODE, "compute")
        owner.finish_graph_cleanup(second)
        token = owner.begin_forward(Phase.DECODE, "compute")
        owner.finish_forward(token, completed_step())
        wrapper.cleanup()
        owner.close()

    def test_reset_of_one_backend_preserves_another_failed_capture(self):
        owner, backend, wrapper = self.binding()
        runtime = owner.runtime
        runtime.retire_graph_resources = MethodType(
            InvocationRuntime.retire_graph_resources, runtime
        )
        other_backend = Backend([])
        other = GraphLeaseBackend(other_backend, owner, object())
        other._leases[2] = owner.acquire_graph((other._identity, 2))
        other_backend.graphs[2] = object()

        def captured_context():
            return ManagedExecutionContext(
                Phase.DECODE,
                compute_stream=object(),
                copy_stream=object(),
                record_completion=lambda stream: TransferTicket(ImmediateCompletion()),
                failure_drain=lambda: None,
                capturing=True,
            )

        class Resource:
            pass

        failed = captured_context()
        resource = Resource()
        resource_ref = weakref.ref(resource)
        failed.retain(resource)
        del resource
        unknown = captured_context()
        runtime._failed_contexts = [failed, unknown]
        other._capture_resources.append([failed])
        other._failed = True
        other_backend.fail_cleanup = True
        with self.assertRaisesRegex(RuntimeError, "graph reset"):
            other.cleanup()

        wrapper.cleanup()
        self.assertTrue(other_backend.graphs)
        self.assertTrue(other._leases)
        self.assertFalse(failed._disposed)
        self.assertIsNotNone(resource_ref())
        self.assertEqual(runtime._failed_contexts, [failed, unknown])
        self.assertFalse(unknown._disposed)

        other_backend.fail_cleanup = False
        other.cleanup()
        self.assertFalse(other_backend.graphs)
        self.assertTrue(failed._disposed)
        self.assertIsNone(resource_ref())
        self.assertEqual(runtime._failed_contexts, [unknown])
        self.assertFalse(unknown._disposed)

    def test_capture_before_first_lease_blocks_concurrent_close_and_cleanup(self):
        runtime = FakeRuntime()
        owner = MoEOffloadOwner(config(), catalog(), runtime)
        wrapper = GraphLeaseBackend(Backend([]), owner, object())
        other = GraphLeaseBackend(Backend([]), owner, object())
        entered, capture_bucket, unwinding, finish_exit = (Event() for _ in range(4))
        errors = []

        @contextmanager
        def capture_scope(owner, runner, stream):
            try:
                yield []
            finally:
                unwinding.set()
                if not finish_exit.wait(5):
                    raise RuntimeError("Capture exit was not released")

        runtime.capture_scope = capture_scope

        def capture():
            try:
                with wrapper.capture_session("capture"):
                    entered.set()
                    if not capture_bucket.wait(5):
                        raise RuntimeError("Capture body was not released")
                    wrapper.capture_one(1, lambda: None)
            except BaseException as error:
                errors.append(error)

        worker = Thread(target=capture)
        worker.start()
        try:
            self.assertTrue(entered.wait(5))
            self.assertFalse(owner._graphs)  # Runner autotuning has no lease yet.
            self.assertIs(owner._capture_session, wrapper._identity)
            for operation in (
                owner.close,
                other.cleanup,
                lambda: owner.begin_forward(Phase.DECODE, "compute"),
                lambda: owner.acquire_graph((other._identity, 2)),
            ):
                with self.assertRaisesRegex(RuntimeError, "Concurrent"):
                    operation()
            with self.assertRaisesRegex(RuntimeError, "Concurrent"):
                with other.capture_session("other"):
                    self.fail("A second capture entered the shared runtime")
            self.assertFalse(other._closing)
            self.assertFalse(other._failed)
            self.assertEqual(owner.state, OwnerState.READY)
            self.assertNotIn("close", runtime.actions)

            capture_bucket.set()
            self.assertTrue(unwinding.wait(5))
            self.assertTrue(owner._graphs)  # The owning session can bind a bucket.
            with self.assertRaisesRegex(RuntimeError, "Concurrent"):
                owner.close()
            with self.assertRaisesRegex(RuntimeError, "Concurrent"):
                other.cleanup()
            self.assertIs(owner._capture_session, wrapper._identity)
        finally:
            capture_bucket.set()
            finish_exit.set()
            worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertIsNone(owner._capture_session)
        wrapper.cleanup()
        other.cleanup()
        owner.close()
        self.assertEqual(owner.state, OwnerState.CLOSED)

    def test_failed_capture_releases_session_gate_without_reopening_submissions(self):
        runtime = FakeRuntime()
        owner = MoEOffloadOwner(config(), catalog(), runtime)
        wrapper = GraphLeaseBackend(Backend([]), owner, object())

        @contextmanager
        def capture_scope(owner, runner, stream):
            yield []

        runtime.capture_scope = capture_scope
        with self.assertRaisesRegex(RuntimeError, "autotuning failed"):
            with wrapper.capture_session("capture"):
                raise RuntimeError("autotuning failed")
        self.assertIsNone(owner._capture_session)
        self.assertTrue(wrapper._closed)
        self.assertEqual(owner.state, OwnerState.FAILED)
        with self.assertRaises(RuntimeError):
            owner.begin_forward(Phase.DECODE, "compute")
        owner.close()


if __name__ == "__main__":
    unittest.main()
