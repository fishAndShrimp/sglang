"""CPU lifetime, mapped-host and exact mode restoration regressions."""

import gc
import threading
import unittest
import weakref
from dataclasses import fields

import torch
from sglang.srt.offloading.moe.config import Phase
from sglang.srt.offloading.moe.policy import TensorLFUPolicy
from sglang.srt.offloading.moe.resources import ResourceKind
from sglang.srt.offloading.moe.runtime import (
    DecodeStateSnapshot,
    ManagedExecutionContext,
)
from sglang.srt.offloading.moe.state import ResidencyState
from sglang.srt.offloading.transfer import Completion, TransferTicket
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class Fence(Completion):
    def __init__(self, name, log, ready=False):
        self.name, self.log, self.ready = name, log, ready

    def wait_on(self, stream):
        self.log.append(("wait", self.name, stream))

    def synchronize(self):
        self.log.append(("sync", self.name))
        self.ready = True

    def done(self):
        return self.ready


def context(phase=Phase.DECODE, *, drain=None):
    log = []

    def record(stream):
        name = f"{stream}-{len(log)}"
        log.append(("record", name))
        return TransferTicket(Fence(name, log))

    ctx = ManagedExecutionContext(
        phase,
        compute_stream="compute",
        copy_stream="copy",
        record_completion=record,
        failure_drain=drain or (lambda: log.append(("drain",))),
    )
    return ctx, log


class TestManagedContext(unittest.TestCase):
    def test_thread_retention_and_idle_include_communication_and_copy(self):
        ctx, log = context(Phase.PREFILL)
        comm = Fence("communication", log)
        worker = threading.Thread(target=lambda: ctx.retain(TransferTicket(comm)))
        worker.start()
        worker.join()
        completion = ctx.finish()
        self.assertFalse(completion.resources_idle.completion.done())
        completion.resources_idle.completion.synchronize()
        self.assertTrue(comm.ready)
        self.assertTrue(completion.resources_idle.completion.done())
        self.assertTrue(
            any(name.startswith("copy") for op, name in log if op == "sync")
        )
        with self.assertRaises(RuntimeError):
            ctx.retain(object())
        with self.assertRaises(RuntimeError):
            ctx.finish()

    def test_replay_contexts_record_new_events(self):
        first, _ = context()
        second, _ = context()
        a, b = first.finish(), second.finish()
        a.resources_idle.completion.synchronize()
        self.assertTrue(a.resources_idle.completion.done())
        self.assertFalse(b.resources_idle.completion.done())
        self.assertIsNot(a.output_ready.completion, b.output_ready.completion)

    def test_failed_drain_keeps_objects_until_retry_succeeds(self):
        class Resource:
            pass

        attempt = []

        def drain():
            attempt.append(1)
            if len(attempt) == 1:
                raise RuntimeError("native queue still live")

        ctx, _ = context(drain=drain)
        resource = Resource()
        weak = weakref.ref(resource)
        ctx.retain(resource)
        del resource
        with self.assertRaises(RuntimeError):
            ctx.abort()
        gc.collect()
        self.assertIsNotNone(weak())
        ctx.abort()
        gc.collect()
        self.assertIsNone(weak())

    def test_completion_ticket_keeps_storage_after_context_goes_away(self):
        class Resource:
            pass

        ctx, _ = context()
        resource = Resource()
        weak = weakref.ref(resource)
        ctx.retain(resource)
        ticket = ctx.finish()
        del ctx, resource
        gc.collect()
        self.assertIsNotNone(weak())
        del ticket
        gc.collect()
        self.assertIsNone(weak())


class TestExactSnapshot(unittest.TestCase):
    def make_snapshot(self):
        state = ResidencyState.allocate(4, 2, 2, "cpu")
        state.key_to_slot.copy_(torch.tensor([-1, 1, -1, 0]))
        state.slot_to_key.copy_(torch.tensor([3, 1]))
        state.slot_valid.fill_(True)
        state.slot_generation.copy_(torch.tensor([7, 9]))
        policy = TensorLFUPolicy(aging_interval=8)
        policy_state = policy.allocate_state(4, "cpu")
        policy_state.frequency.copy_(torch.tensor([0, 4, 0, 8]))
        policy_state.steps.fill_(5)
        return (
            state,
            policy_state,
            DecodeStateSnapshot(
                state,
                policy,
                policy_state,
                error_status=torch.zeros((), dtype=torch.int32),
            ),
        )

    def test_reload_original_slots_then_restore_maps_policy_and_addresses(self):
        state, policy, snapshot = self.make_snapshot()
        original = {f.name: getattr(state, f.name).clone() for f in fields(state)}
        addresses = {f.name: getattr(state, f.name).data_ptr() for f in fields(state)}
        ctx, log = context()
        saved = snapshot.save(ctx)
        saved.completion.synchronize()
        state.key_to_slot.fill_(-1)
        state.slot_to_key.fill_(-1)
        state.slot_generation.fill_(99)
        policy.frequency.zero_()
        policy.steps.zero_()
        weights = [-1, -1]

        def reload(saved_state, current):
            self.assertEqual(state.slot_to_key.tolist(), [-1, -1])
            weights[:] = saved_state.slot_to_key.tolist()
            log.append(("reload", tuple(weights)))
            return TransferTicket(Fence("weights", log))

        ticket = snapshot.restore(ctx, reload)
        self.assertEqual(weights, [3, 1])
        self.assertIn(("wait", "weights", "compute"), log)
        for f in fields(state):
            self.assertTrue(torch.equal(getattr(state, f.name), original[f.name]))
            self.assertEqual(getattr(state, f.name).data_ptr(), addresses[f.name])
        self.assertEqual(policy.frequency.tolist(), [0, 4, 0, 8])
        self.assertEqual(policy.steps.item(), 5)
        self.assertFalse(ticket.completion.done())
        allocations = snapshot.allocations()
        self.assertEqual(len({a.storage_key for a in allocations}), len(allocations))
        self.assertTrue(all(a.kind is ResourceKind.METADATA for a in allocations))

    def test_reload_failure_does_not_publish_saved_metadata(self):
        state, _, snapshot = self.make_snapshot()
        ctx, _ = context()
        snapshot.save(ctx)
        state.slot_to_key.fill_(-1)

        def fail(saved, current):
            raise RuntimeError("incomplete bundle")

        with self.assertRaises(RuntimeError):
            snapshot.restore(ctx, fail)
        self.assertEqual(state.slot_to_key.tolist(), [-1, -1])
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            snapshot.save(ctx)

    def test_device_fault_with_successful_submission_does_not_publish(self):
        state, policy, snapshot = self.make_snapshot()
        ctx, log = context()
        snapshot.save(ctx)
        state.slot_to_key.fill_(-1)
        state.slot_valid.zero_()
        policy.frequency.zero_()

        def faulty_reload(saved, current):
            snapshot.error_status.fill_(1)
            return TransferTicket(Fence("masked_native_copy", log))

        with self.assertRaisesRegex(RuntimeError, "Device residency/transfer fault"):
            snapshot.restore(ctx, faulty_reload)
        self.assertEqual(state.slot_to_key.tolist(), [-1, -1])
        self.assertFalse(state.slot_valid.any())
        self.assertEqual(policy.frequency.tolist(), [0, 0, 0, 0])
        self.assertIn(("sync", "masked_native_copy"), log)


if __name__ == "__main__":
    unittest.main()
