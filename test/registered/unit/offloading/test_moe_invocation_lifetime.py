"""Completion proofs retire resources without GC or per-step host waits."""

import gc
import unittest
import weakref
from types import SimpleNamespace

import torch

from sglang.srt.offloading.moe.config import Phase
from sglang.srt.offloading.moe.execution import WeightLease, invocation_id
from sglang.srt.offloading.moe.owner import MoEOffloadOwner
from sglang.srt.offloading.moe.npu.runtime import NpuBoundMoERuntime
from sglang.srt.offloading.moe.policy import TensorLFUPolicy
from sglang.srt.offloading.moe.runtime import (
    DecodeStateSnapshot,
    ManagedExecutionContext,
)
from sglang.srt.offloading.moe.state import ResidencyState
from sglang.srt.offloading.transfer import (
    ImmediateCompletion,
    JoinedCompletion,
    TransferTicket,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class Probe(ImmediateCompletion):
    def __init__(self, ready=True):
        self.ready = ready
        self.queries = 0
        self.waits = 0
        self.syncs = 0

    def done(self):
        self.queries += 1
        return self.ready

    def wait_on(self, stream):
        self.waits += 1

    def synchronize(self):
        self.syncs += 1
        self.ready = True


class Payload:
    pass


class TestInvocationLifetime(unittest.TestCase):
    def context(self, *, phase=Phase.PREFILL, capturing=False, drain=lambda: None):
        events = []

        def record(stream):
            event = Probe()
            events.append(event)
            return TransferTicket(event)

        return (
            ManagedExecutionContext(
                phase,
                compute_stream=object(),
                copy_stream=object(),
                record_completion=record,
                failure_drain=drain,
                capturing=capturing,
            ),
            events,
        )

    def test_retention_does_not_add_an_implicit_dependency(self):
        context, events = self.context()
        unrelated = Probe(ready=False)
        context.retain(TransferTicket(unrelated))
        finished = context.finish()
        self.assertTrue(finished.resources_idle.completion.done())
        self.assertEqual(unrelated.queries, 0)
        self.assertEqual(len(events), 2)
        self.assertIs(finished.output_ready, finished.residency_ready)

    def test_registered_external_dependencies_are_deduplicated_and_preserved(self):
        context, events = self.context()
        external = Probe(ready=False)
        ticket = TransferTicket(JoinedCompletion((external, external)))
        context.retain(ticket)
        context.register_completion(ticket)
        context.register_completion(external)
        finished = context.finish()
        self.assertFalse(finished.resources_idle.completion.done())
        self.assertEqual(external.queries, 1)
        external.ready = True
        finished.resources_idle.completion.wait_on(object())
        self.assertEqual(external.waits, 1)
        self.assertEqual([event.waits for event in events], [1, 1])

    def test_latest_record_covers_prior_events_on_the_same_stream(self):
        context, events = self.context()
        context.record_compute_completion()
        context.record_compute_completion()
        finished = context.finish()
        self.assertTrue(finished.resources_idle.completion.done())
        self.assertEqual([event.queries for event in events], [0, 0, 1, 1])

    def test_owner_retires_completed_cycle_without_gc_or_host_sync(self):
        # Construct only the real owner's submission state; allocation budgeting
        # is tested elsewhere. This exercises begin_forward's actual retirement.
        owner = object.__new__(MoEOffloadOwner)
        from threading import RLock

        from sglang.srt.offloading.moe.owner import OwnerState

        owner._lock = RLock()
        owner.state = OwnerState.READY
        owner._closing = False
        owner._graph_cleanups = set()
        owner._active = None
        owner._capture_session = None
        owner.phase = Phase.PREFILL
        owner._sequence = 0
        owner._token = object()
        context, events = self.context()
        payload = Payload()
        payload_ref, context_ref = weakref.ref(payload), weakref.ref(context)
        context.retain(payload)
        # Defensive regression: disposal also breaks an accidental third-party
        # context backreference; production reservations use invocation IDs.
        context.retain(SimpleNamespace(context=context))
        finished = context.finish()
        owner._pending = [finished]
        enabled = gc.isenabled()
        gc.disable()
        try:
            del payload, context
            owner.begin_forward(Phase.PREFILL, object())
            self.assertIsNone(payload_ref())
            self.assertIsNone(context_ref())
            self.assertFalse(finished.resources)
            self.assertEqual([event.syncs for event in events], [0, 0])
        finally:
            if enabled:
                gc.enable()

    def test_lease_identity_does_not_keep_the_context_alive(self):
        context, _ = self.context()
        reference = weakref.ref(context)
        lease = WeightLease(
            (),
            None,
            TransferTicket(ImmediateCompletion()),
            object(),
            invocation_id(context),
        )
        del context
        self.assertIsNone(reference())
        self.assertIsNotNone(lease.context_token)

    def test_worker_must_exit_before_seal_or_dispose(self):
        context, _ = self.context()
        context.worker_started()
        with self.assertRaisesRegex(RuntimeError, "join workers"):
            context.finish()
        with self.assertRaisesRegex(RuntimeError, "Join workers"):
            context.dispose_after_drain()
        context.worker_finished()
        context.finish()
        context.dispose_completed()

    def test_capture_requires_destruction_even_after_successful_drain(self):
        context, _ = self.context(phase=Phase.DECODE, capturing=True)
        payload = Payload()
        reference = weakref.ref(payload)
        context.retain(payload)
        del payload
        with self.assertRaisesRegex(RuntimeError, "Destroy the captured graph"):
            context.dispose_after_drain()
        self.assertIsNotNone(reference())
        context.dispose_after_drain(graph_destroyed=True)
        self.assertIsNone(reference())

    def test_closing_keeps_existing_context_available_to_exiting_workers(self):
        runtime = object.__new__(NpuBoundMoERuntime)
        runtime._initialize_lifecycle()
        context, _ = self.context()
        runtime._context = context
        runtime._closing = True
        payload = Payload()
        runtime.context.retain(payload)
        self.assertIn(payload, context._resources)
        with self.assertRaisesRegex(RuntimeError, "closed or closing"):
            runtime._require_open()
        with self.assertRaisesRegex(RuntimeError, "active host submissions"):
            runtime.close()

    def test_snapshot_restores_hot_state_and_resets_transient_state(self):
        state = ResidencyState.allocate(4, 2, 2, "cpu")
        state.key_to_slot.copy_(torch.tensor([0, -1, 1, -1]))
        state.slot_to_key.copy_(torch.tensor([0, 2]))
        state.slot_valid.fill_(True)
        state.slot_generation.copy_(torch.tensor([3, 4]))
        state.execution_key.fill_(2)
        state.execution_valid.fill_(True)
        state.step_access_count.fill_(7)
        state.update_enabled.fill_(True)
        policy = TensorLFUPolicy()
        policy_state = policy.allocate_state(4, "cpu")
        snapshot = DecodeStateSnapshot(
            state, policy, policy_state, error_status=torch.tensor(0, dtype=torch.int32)
        )
        self.assertEqual(
            set(vars(snapshot.saved)),
            {"key_to_slot", "slot_to_key", "slot_valid", "slot_generation"},
        )
        context, _ = self.context(phase=Phase.DECODE)
        snapshot.save(context)
        state.slot_generation.zero_()
        state.update_enabled.fill_(False)
        snapshot.restore(
            context, lambda saved, ctx: TransferTicket(ImmediateCompletion())
        )
        torch.testing.assert_close(state.slot_generation, torch.tensor([3, 4]))
        self.assertTrue(bool((state.execution_key == -1).all()))
        self.assertFalse(bool(state.execution_valid.any()))
        self.assertFalse(bool(state.step_access_count.any()))
        self.assertFalse(bool(state.update_enabled))


if __name__ == "__main__":
    unittest.main()
