"""CPU lifecycle regressions; no accelerator graph or memory-size certification."""

import unittest
import weakref
from contextlib import contextmanager, nullcontext
from threading import RLock, Thread
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from test_moe_graph_leases import Backend, Owner

from sglang.srt.offloading.moe.config import Phase
from sglang.srt.offloading.moe.graph import GraphLeaseBackend
from sglang.srt.offloading.moe.owner import MoEOffloadOwner, OwnerState
from sglang.srt.offloading.moe.runtime import BoundMoERuntime, ExecutionContext
from sglang.srt.offloading.transfer import ImmediateCompletion, TransferTicket
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class Payload:
    pass


class TestMoEWarmupLifetime(unittest.TestCase):
    def make_runtime(self):
        calls = []
        device = SimpleNamespace(
            copy_stream="copy",
            stream_scope=lambda stream: nullcontext(),
            current_stream=lambda: "compute",
            record=lambda stream: TransferTicket(ImmediateCompletion()),
            allocated=lambda: 0,
            reset_peak=lambda: None,
        )
        device.synchronize = Mock(side_effect=lambda: calls.append("device_drain"))
        runtime = BoundMoERuntime(
            device_runtime=device,
            residency=SimpleNamespace(update_enabled=torch.tensor(False)),
            residency_ops=SimpleNamespace(
                begin_step=lambda context: None, error_status=torch.tensor(0)
            ),
            cache_updates=SimpleNamespace(
                emit_step_end=lambda context: TransferTicket(ImmediateCompletion())
            ),
            snapshot=None,
            reload_weights=None,
            layer_programs={3: SimpleNamespace(prefill=object(), decode=object())},
            resources=(),
            expert_nbytes=1,
            storage=(),
        )
        # This fixture tests object lifetimes, not allocator peak accounting.
        runtime._account_capture_workspace = lambda owner: None
        owner = Owner(calls)
        owner.runtime = runtime
        wrapper = GraphLeaseBackend(Backend(calls), owner, object())
        return wrapper, owner, runtime, device, calls

    @staticmethod
    def real_owner(runtime):
        owner = MoEOffloadOwner.__new__(MoEOffloadOwner)
        owner._lock = RLock()
        owner.runtime = runtime
        owner.config = SimpleNamespace(ep_rank=0)
        owner.phase, owner.state = Phase.DECODE, OwnerState.READY
        owner._sequence = 0
        owner._active = owner._capture_session = None
        owner._pending = []
        owner._graphs, owner._graph_cleanups = set(), set()
        owner._closing = False
        owner.allocation_ledger = None
        runtime._owner = owner
        return owner

    @staticmethod
    def invocation(runtime, references):
        batch = object()
        runtime._capture_head(None, batch, 1)
        context = runtime.context
        payload = Payload()
        references.append(weakref.ref(payload))
        context.retain(payload)
        context.retain(SimpleNamespace(invocation_id=context.invocation_id))
        runtime._capture_tail(None, None, batch, 1)

    def test_each_warmup_releases_resources_without_collecting_captured_buckets(self):
        wrapper, owner, runtime, device, calls = self.make_runtime()
        references = []
        resets = []

        def reset():
            # Existing reset hooks run while warmup resources are still alive.
            self.assertIsNotNone(references[-1]())
            resets.append(True)

        with wrapper.capture_session("capture"):
            for bucket in (1, 2):
                wrapper.capture_one(
                    bucket,
                    lambda: self.invocation(runtime, references),
                    post_warmup_hook=reset,
                )
                self.assertEqual(len(runtime._capture), bucket)
                # No gc.collect(): warmup lease/context cycles must be broken.
                self.assertEqual(
                    [ref() is not None for ref in references],
                    [False, False, True] * bucket,
                )
        self.assertEqual(len(resets), 4)
        wrapper.cleanup()
        self.assertTrue(all(ref() is None for ref in references))

    def test_failed_drain_keeps_warmup_and_previously_captured_resources(self):
        wrapper, owner, runtime, device, calls = self.make_runtime()
        references = []

        def fail_reset():
            device.synchronize.side_effect = RuntimeError("warmup drain failed")

        with self.assertRaisesRegex(RuntimeError, "warmup drain failed"):
            with wrapper.capture_session("capture"):
                wrapper.capture_one(1, lambda: self.invocation(runtime, references))
                wrapper.capture_one(
                    2,
                    lambda: self.invocation(runtime, references),
                    post_warmup_hook=fail_reset,
                )
        self.assertIsNotNone(references[2]())  # First bucket's actual capture.
        self.assertIsNotNone(references[3]())  # Failed second-bucket warmup.
        self.assertTrue(owner.graphs)
        self.assertTrue(owner.failed)
        device.synchronize.side_effect = None
        wrapper.cleanup()
        self.assertFalse(owner.graphs)

    def test_completion_failure_does_not_release_keepalives(self):
        completion = ImmediateCompletion()
        completion.synchronize = Mock(side_effect=RuntimeError("reader unfinished"))
        context = ExecutionContext(
            Phase.DECODE,
            compute_stream="compute",
            copy_stream="copy",
            record_completion=lambda stream: TransferTicket(ImmediateCompletion()),
        )
        payload = Payload()
        reference = weakref.ref(payload)
        context.retain(payload)
        context.forward_batch = payload
        context.retain(SimpleNamespace(context=context))
        ticket = TransferTicket(completion)
        context.retain(ticket)
        context.register_completion(ticket)
        del payload
        with self.assertRaisesRegex(RuntimeError, "Finish the invocation"):
            context.dispose_completed()
        owner = MoEOffloadOwner.__new__(MoEOffloadOwner)
        owner._lock = RLock()
        owner._active = owner._capture_session = None
        owner.state = OwnerState.READY
        result = context.finish()
        owner._pending = [result]
        with self.assertRaisesRegex(RuntimeError, "reader unfinished"):
            owner.wait_idle()
        self.assertIsNotNone(reference())
        self.assertIs(owner.state, OwnerState.FAILED)
        completion.synchronize.side_effect = None
        owner.wait_idle()
        self.assertIsNone(reference())
        self.assertEqual(result.resources, [])
        self.assertFalse(hasattr(context, "forward_batch"))

    def test_capture_gate_covers_first_lease_and_runtime_scope_exit(self):
        wrapper, _, runtime, _, calls = self.make_runtime()
        owner = self.real_owner(runtime)
        wrapper.owner = owner
        other = GraphLeaseBackend(Backend(calls), owner, object())
        observations = []

        def probe_from_other_thread():
            errors = []

            def probe():
                for operation in (
                    owner.close,
                    runtime.close,
                    lambda: owner.begin_forward(Phase.DECODE, "compute"),
                    other.cleanup,
                ):
                    try:
                        operation()
                    except RuntimeError:
                        errors.append(True)

            worker = Thread(target=probe)
            worker.start()
            worker.join(timeout=2)
            self.assertFalse(
                worker.is_alive(), "Capture gate must reject without blocking"
            )
            self.assertEqual(errors, [True, True, True, True])
            observations.append(True)

        original_scope = runtime.capture_scope

        @contextmanager
        def scope(*args):
            with original_scope(*args) as resources:
                try:
                    yield resources
                finally:
                    probe_from_other_thread()

        runtime.capture_scope = scope
        with wrapper.capture_session("capture"):
            self.assertFalse(owner._graphs)
            probe_from_other_thread()
        self.assertEqual(observations, [True, True])
        self.assertIsNone(owner._capture_session)
        wrapper.cleanup()
        other.cleanup()
        owner.close()

    def test_failed_reset_retires_only_its_wrappers_contexts(self):
        first, _, runtime, _, calls = self.make_runtime()
        owner = self.real_owner(runtime)
        first.owner = owner
        second = GraphLeaseBackend(Backend(calls), owner, object())
        references = []
        for wrapper in (first, second):
            with wrapper.capture_session("capture"):
                wrapper.capture_one(1, lambda: self.invocation(runtime, references))
        captured = [first._capture_resources[0][0], second._capture_resources[0][0]]
        unknown = runtime._new_context(Phase.DECODE, capturing=True)
        payload = Payload()
        unknown_ref = weakref.ref(payload)
        unknown.retain(payload)
        del payload
        runtime._failed_contexts = captured + [unknown]
        first.backend.fail_cleanup = True
        with self.assertRaisesRegex(RuntimeError, "graph reset failure"):
            first.cleanup()
        self.assertIsNotNone(references[2]())
        self.assertIsNotNone(references[5]())
        second.cleanup()
        self.assertIsNotNone(references[2]())
        self.assertIsNone(references[5]())
        self.assertEqual(runtime._failed_contexts, [captured[0], unknown])
        first.backend.fail_cleanup = False
        first.cleanup()
        self.assertEqual(runtime._failed_contexts, [unknown])
        self.assertIsNotNone(unknown_ref())
        owner.close()
        self.assertIsNone(unknown_ref())

    def test_transition_owns_wait_and_retires_or_pins_on_failure(self):
        _, _, runtime, device, _ = self.make_runtime()
        references = []
        completion = ImmediateCompletion()
        completion.synchronize = Mock()

        def operation(context):
            payload = Payload()
            references.append(weakref.ref(payload))
            context.retain(payload)
            return TransferTicket(completion)

        runtime._transition(operation)
        self.assertIsNone(references[-1]())
        completion.synchronize.assert_called_once_with()
        device.synchronize.assert_not_called()
        completion.synchronize.side_effect = RuntimeError("transition wait failed")
        with self.assertRaisesRegex(RuntimeError, "transition wait failed"):
            runtime._transition(operation)
        self.assertIsNotNone(references[-1]())
        self.assertIsNone(runtime._context)
        device.synchronize.side_effect = RuntimeError("drain failed")
        with self.assertRaisesRegex(RuntimeError, "drain failed"):
            runtime.close()
        self.assertIsNotNone(references[-1]())
        device.synchronize.side_effect = None
        runtime.close()
        self.assertIsNone(references[-1]())


if __name__ == "__main__":
    unittest.main()
