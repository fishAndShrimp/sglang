"""Deterministic CPU tests for bounded prefill submission and bank lifetimes."""

import contextlib
import threading
import time
import unittest
from types import SimpleNamespace

from sglang.srt.offloading.moe.config import ExpertWave, Phase
from sglang.srt.offloading.moe.execution import (
    ExecutionRequest,
    SourcePolicy,
    WeightLease,
)
from sglang.srt.offloading.moe.prefill import (
    BoundedPrefillWeightPipeline,
    PrefillPipelineConfig,
)
from sglang.srt.offloading.transfer import Completion, JoinedCompletion, TransferTicket
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class Fence(Completion):
    def __init__(self, name, context, *, ready=True):
        self.name, self.context = name, context
        self.ready = threading.Event()
        self.queried = threading.Event()
        if ready:
            self.ready.set()

    def wait_on(self, stream):
        self.context.append(("wait", self.name, stream))

    def done(self):
        self.queried.set()
        return self.ready.is_set()

    def synchronize(self):
        raise AssertionError("Prefill scheduling must not synchronize the device")


class Context:
    phase = Phase.PREFILL
    capturing = False
    compute_stream = "compute"
    copy_stream = "copy"

    def __init__(self):
        self.lock = threading.RLock()
        self.calls, self.retained = [], []

    def retain(self, value):
        with self.lock:
            self.retained.append(value)

    def append(self, value):
        with self.lock:
            self.calls.append(value)

    def ticket(self, name, *, ready=True):
        return TransferTicket(Fence(name, self, ready=ready))


class Access:
    """Controlled copies, with real WeightLease ownership and reader tickets."""

    def __init__(self):
        self.condition = threading.Condition()
        self.requests, self.leases = [], []
        self.initial_ready = True
        self.fail_at = None

    def stage(self, request, context):
        with self.condition:
            index = len(self.requests)
            self.requests.append(request)
            context.append(("copy", request.wave, request.local_demand))
            self.condition.notify_all()
            if index == self.fail_at:
                raise RuntimeError("partial driver submission")
            lease = WeightLease(
                (request,),
                request.wave,
                context.ticket(f"copy{index}", ready=self.initial_ready),
                self,
                context,
            )
            context.retain(lease)
            self.leases.append(lease)
            self.condition.notify_all()
            return lease

    def combine_prefill(self, leases, bindings, context):
        for lease in leases:
            lease.transferred = True
        grouped = WeightLease(
            tuple(r for lease in leases for r in lease.reservations),
            bindings,
            TransferTicket(
                JoinedCompletion(lease.ticket.completion for lease in leases)
            ),
            self,
            context,
        )
        context.retain(grouped)
        return grouped

    def release(self, lease, context, *, ready=True):
        lease.read_done = context.ticket(f"read{lease.bindings.index}", ready=ready)
        context.retain(lease.read_done)
        return lease.read_done

    def wait_copies(self, count):
        with self.condition:
            return self.condition.wait_for(lambda: len(self.leases) >= count, timeout=2)


def config(**changes):
    values = dict(
        local_experts=9,
        initial_capacity=5,
        group_size=2,
        secondary_capacity=2,
        microbatch_size=1,
        max_inflight=1,
    )
    values.update(changes)
    return PrefillPipelineConfig(**values)


class TestBoundedPrefillPipeline(unittest.TestCase):
    def make_pipeline(self, *, settings=None, access=None, scope=None, fence=None):
        context = Context()
        access = access or Access()

        @contextlib.contextmanager
        def worker_scope(ctx):
            ctx.append(("worker_enter", threading.current_thread().name))
            try:
                yield
            finally:
                ctx.append(("worker_exit",))

        def record(ctx):
            ctx.append(("copy_fence",))
            return ctx.ticket("copy_frontier")

        pipeline = BoundedPrefillWeightPipeline(
            access,
            settings or config(),
            make_request=lambda layer, wave, offset, ctx: ExecutionRequest(
                layer, offset, wave, Phase.PREFILL, SourcePolicy.HOST_ONLY
            ),
            combine_bindings=lambda layer, wave, leases, ctx: wave,
            worker_scope=scope or worker_scope,
            record_copy_completion=fence or record,
        )
        self.addCleanup(lambda: pipeline.abort(context))
        return pipeline, access, context

    def test_reject_bad_capacities_and_truncate_single_small_layer(self):
        for changes in (
            {"microbatch_size": 3},
            {"group_size": 6},
            {"max_inflight": 0},
            {"group_size": 1.5},
            {"secondary_capacity": 1},
            {"secondary_bank": 0},
            {"poll_interval": float("nan")},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                config(**changes)
        small = config(local_experts=1, microbatch_size=3, secondary_capacity=0)
        self.assertEqual(
            (small.microbatch_size, small.group_size, small.initial_capacity), (1, 1, 1)
        )

    def test_dispatch_feedback_seals_without_waiting_for_copy(self):
        access = Access()
        access.initial_ready = False
        pipeline, access, context = self.make_pipeline(access=access)
        pipeline.start(3, context)
        self.assertTrue(access.wait_copies(1))
        waves = pipeline.seal(3, Fence("dispatch", context), context)
        self.assertEqual(waves[0], ExpertWave(0, 0, 1, 0))
        self.assertEqual([wave.bank for wave in waves], [0, 1, 0, 1, 0])
        self.assertFalse(access.leases[0].ticket.completion.done())
        self.assertEqual(len(access.requests), 1)
        self.assertIn(("worker_exit",), context.calls)

    def test_initial_capacity_and_offsets_include_partial_microbatch(self):
        pipeline, access, context = self.make_pipeline(
            settings=config(microbatch_size=2)
        )
        pipeline.start(3, context)
        self.assertTrue(access.wait_copies(3))
        waves = pipeline.seal(3, Fence("dispatch", context, ready=False), context)
        self.assertEqual(waves[0].local_end, 5)
        self.assertEqual(
            [request.local_demand for request in access.requests], [0, 2, 4]
        )
        self.assertEqual(
            [request.wave.local_end for request in access.requests], [2, 4, 5]
        )
        self.assertEqual(
            [(w.local_start, w.local_end) for w in waves], [(0, 5), (5, 7), (7, 9)]
        )

    def test_inflight_bound_while_dispatch_is_pending(self):
        access = Access()
        access.initial_ready = False
        pipeline, access, context = self.make_pipeline(
            access=access, settings=config(max_inflight=2)
        )
        pipeline.start(3, context)
        self.assertTrue(access.wait_copies(2))
        self.assertTrue(access.leases[1].ticket.completion.queried.wait(2))
        self.assertEqual(len(access.requests), 2)
        access.leases[0].ticket.completion.ready.set()
        self.assertTrue(access.wait_copies(3))
        self.assertTrue(access.leases[2].ticket.completion.queried.wait(2))
        self.assertEqual(len(access.requests), 3)
        waves = pipeline.seal(3, Fence("dispatch", context), context)
        self.assertEqual(waves[0].local_end, 3)

    def test_ready_dispatch_still_loads_one_microbatch(self):
        entered, release = threading.Event(), threading.Event()

        @contextlib.contextmanager
        def gated_scope(ctx):
            entered.set()
            if not release.wait(2):
                raise RuntimeError("test gate timed out")
            yield

        pipeline, access, context = self.make_pipeline(scope=gated_scope)
        pipeline.start(3, context)
        self.assertTrue(entered.wait(2))
        result, errors = [], []

        def seal():
            try:
                result.append(pipeline.seal(3, Fence("dispatch", context), context))
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=seal)
        thread.start()
        # Wait until feedback has actually been handed to the submitter.
        deadline = time.monotonic() + 2
        while (
            not pipeline._session(3, context).dispatch_ready
            and time.monotonic() < deadline
        ):
            time.sleep(0.001)
        release.set()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(result[0][0].local_end, 1)
        self.assertEqual(len(access.requests), 1)

    def test_bank_reuse_waits_old_reader_and_copies_every_expert_once(self):
        access = Access()
        access.initial_ready = False
        pipeline, access, context = self.make_pipeline(access=access)
        pipeline.start(3, context)
        self.assertTrue(access.wait_copies(1))
        waves = pipeline.seal(3, Fence("dispatch", context), context)
        readers = []
        for wave in waves:
            current = pipeline.acquire(3, wave, context)
            self.assertEqual(current.bindings, wave)
            readers.append(access.release(current, context, ready=False))
        ticket = pipeline.finish(3, context)
        self.assertFalse(ticket.completion.done())
        copied_ids = [
            local_id
            for request in access.requests
            for local_id in range(request.wave.local_start, request.wave.local_end)
        ]
        self.assertEqual(copied_ids, list(range(9)))
        for index in range(2, len(waves)):
            copied = context.calls.index(("copy", waves[index], 0))
            waited = context.calls.index(("wait", f"read{index - 2}", "copy"))
            self.assertLess(waited, copied)
        for lease in access.leases:
            lease.ticket.completion.ready.set()
        for reader in readers:
            reader.completion.ready.set()
        self.assertTrue(ticket.completion.done())
        self.assertTrue(any(item is ticket for item in context.retained))

    def test_single_group_never_stages_second_bank(self):
        pipeline, access, context = self.make_pipeline(
            settings=config(local_experts=1, secondary_capacity=0)
        )
        pipeline.start(3, context)
        waves = pipeline.seal(3, Fence("dispatch", context), context)
        self.assertEqual(len(waves), 1)
        lease = pipeline.acquire(3, waves[0], context)
        access.release(lease, context)
        pipeline.finish(3, context)
        self.assertEqual(len(access.requests), 1)

    def test_k_is_recomputed_for_each_invocation(self):
        access = Access()
        access.initial_ready = False
        pipeline, access, first = self.make_pipeline(access=access)
        pipeline.start(3, first)
        self.assertTrue(access.wait_copies(1))
        waves = pipeline.seal(3, Fence("dispatch", first), first)
        self.assertEqual(waves[0].local_end, 1)
        for wave in waves:
            access.release(pipeline.acquire(3, wave, first), first)
        pipeline.finish(3, first)
        second = Context()
        self.addCleanup(lambda: pipeline.abort(second))
        access.initial_ready = True
        previous = len(access.requests)
        pipeline.start(3, second)
        self.assertTrue(access.wait_copies(previous + 5))
        waves = pipeline.seal(3, Fence("dispatch", second), second)
        self.assertEqual(waves[0].local_end, 5)

    def test_missing_stale_duplicate_or_unretired_wave_rejected(self):
        pipeline, access, context = self.make_pipeline()
        with self.assertRaisesRegex(RuntimeError, "layer-start"):
            pipeline.validate_session(3, context)
        pipeline.start(3, context)
        with self.assertRaises(RuntimeError):
            pipeline.start(3, context)
        with self.assertRaises(RuntimeError):
            pipeline.validate_session(3, Context())
        waves = pipeline.seal(3, Fence("dispatch", context), context)
        with self.assertRaises(ValueError):
            pipeline.acquire(3, waves[-1], context)
        current = pipeline.acquire(3, waves[0], context)
        with self.assertRaises(RuntimeError):
            pipeline.acquire(3, waves[1], context)
        with self.assertRaises(RuntimeError):
            pipeline.finish(3, context)
        access.release(current, context)

    def test_failed_submission_is_joined_then_fenced_and_abort_is_idempotent(self):
        access = Access()
        access.fail_at = 0
        pipeline, access, context = self.make_pipeline(access=access)
        pipeline.start(3, context)
        # Sealing observes and propagates asynchronous worker failures.
        with self.assertRaisesRegex(RuntimeError, "submitter failed"):
            pipeline.seal(3, Fence("dispatch", context), context)
        ticket = pipeline.abort(context)
        self.assertIs(pipeline.abort(context), ticket)
        self.assertLess(
            context.calls.index(("worker_exit",)), context.calls.index(("copy_fence",))
        )
        self.assertTrue(ticket.keepalive)
        with self.assertRaises(RuntimeError):
            pipeline.start(4, context)

    def test_abort_stops_bounded_worker_without_waiting_for_device(self):
        access = Access()
        access.initial_ready = False
        pipeline, access, context = self.make_pipeline(access=access)
        pipeline.start(3, context)
        self.assertTrue(access.wait_copies(1))
        ticket = pipeline.abort(context)
        self.assertFalse(ticket.completion.done())
        self.assertEqual(len(access.requests), 1)
        self.assertIn(("worker_exit",), context.calls)

    def test_fence_failure_keeps_session_for_runtime_drain(self):
        fail = [True]

        def fence(context):
            if fail[0]:
                raise RuntimeError("event record failed")
            return context.ticket("retry_frontier")

        pipeline, access, context = self.make_pipeline(fence=fence)
        pipeline.start(3, context)
        with self.assertRaisesRegex(RuntimeError, "event record failed"):
            pipeline.abort(context)
        self.assertTrue(pipeline._sessions)
        self.assertTrue(context.retained)
        fail[0] = False
        pipeline.abort(context)
        self.assertEqual(pipeline._sessions, {})

    def test_worker_device_context_failure_propagates_and_is_retained(self):
        @contextlib.contextmanager
        def fail_scope(context):
            raise RuntimeError("device context failure")
            yield

        pipeline, access, context = self.make_pipeline(scope=fail_scope)
        pipeline.start(3, context)
        with self.assertRaisesRegex(RuntimeError, "submitter failed") as error:
            pipeline.seal(3, Fence("dispatch", context), context)
        self.assertIn("device context failure", str(error.exception.__cause__))
        self.assertEqual(access.requests, [])
        self.assertTrue(pipeline.abort(context).keepalive)

    def test_new_layer_waits_for_previous_final_reader_without_host_sync(self):
        pipeline, access, context = self.make_pipeline(
            settings=config(local_experts=1, secondary_capacity=0)
        )
        pipeline.start(3, context)
        waves = pipeline.seal(3, Fence("dispatch", context), context)
        lease = pipeline.acquire(3, waves[0], context)
        reader = access.release(lease, context, ready=False)
        pipeline.finish(3, context)
        context.calls.clear()
        pipeline.start(4, context)
        self.assertTrue(access.wait_copies(2))
        self.assertFalse(reader.completion.done())
        wait = context.calls.index(("wait", "read0", "copy"))
        copied = next(i for i, call in enumerate(context.calls) if call[0] == "copy")
        self.assertLess(wait, copied)
        pipeline.abort(context)

    def test_sealed_session_still_owns_shared_banks(self):
        pipeline, access, context = self.make_pipeline()
        pipeline.start(3, context)
        self.assertTrue(access.wait_copies(5))
        pipeline.seal(3, Fence("dispatch", context), context)
        with self.assertRaisesRegex(RuntimeError, "Finish the previous"):
            pipeline.start(4, context)
        with self.assertRaisesRegex(RuntimeError, "Finish the previous"):
            pipeline.start(3, Context())

    def test_malformed_request_fails_before_copy_submission(self):
        pipeline, access, context = self.make_pipeline()
        pipeline.make_request = lambda *args: ExecutionRequest(
            3, None, ExpertWave(0, 0, 1, 0), Phase.DECODE
        )
        pipeline.start(3, context)
        with self.assertRaisesRegex(RuntimeError, "submitter failed") as error:
            pipeline.seal(3, Fence("dispatch", context), context)
        self.assertIn("HOST_ONLY", str(error.exception.__cause__))
        self.assertEqual(access.requests, [])

    def test_nonprefill_or_capture_rejected_before_start(self):
        pipeline, access, context = self.make_pipeline()
        for phase, capturing in ((Phase.DECODE, False), (Phase.PREFILL, True)):
            invalid = SimpleNamespace(phase=phase, capturing=capturing)
            with self.assertRaises(RuntimeError):
                pipeline.start(3, invalid)
        self.assertEqual(access.requests, [])


if __name__ == "__main__":
    unittest.main()
