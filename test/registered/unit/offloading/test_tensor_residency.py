"""CPU differential checks for the graph-oriented fixed tensor residency planner."""

import random
from threading import Event, Thread, get_ident
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch
from weakref import ref

import torch

from sglang.srt.offloading.moe.config import (
    ExpertCatalog,
    ExpertLayerSpec,
    ExpertWave,
    Phase,
    fixed_waves,
)
from sglang.srt.offloading.moe.execution import (
    ExecutionRequest,
    ExpertWeightAccess,
    MaterializationPlan,
)
from sglang.srt.offloading.moe.prefill import (
    BoundedPrefillWeightPipeline,
    PrefillPipelineConfig,
)
from sglang.srt.offloading.moe.residency import BoundExecutionWave, TensorResidencyOps
from sglang.srt.offloading.moe.resources import (
    AllocationLedger,
    ResourceKind,
    tensor_allocations,
)
from sglang.srt.offloading.moe.state import ResidencyState
from sglang.srt.offloading.moe.runtime import ExecutionContext
from sglang.srt.offloading.transfer import (
    Completion,
    CopySourceKind,
    HostCopyPlan,
    HostTransferOps,
    TransferTicket,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class Fence(Completion):
    def __init__(self, calls, name):
        self.calls, self.name = calls, name

    def wait_on(self, stream):
        self.calls.append(("wait", self.name, stream))

    def synchronize(self):
        raise AssertionError("Tensor planning must not synchronize the host")

    def done(self):
        raise AssertionError("Tensor planning must not query device completion")


class Context:
    phase = Phase.DECODE
    capturing = True
    compute_stream = "compute"
    copy_stream = "copy"

    def __init__(self):
        self.calls, self.retained = [], []
        self.invocation_id = object()

    def retain(self, resource):
        self.retained.append(resource)

    def record_compute_completion(self):
        name = "event" + str(len(self.calls))
        self.calls.append(("record", name))
        return TransferTicket(Fence(self.calls, name))

    def copied(self):
        return TransferTicket(Fence(self.calls, "bundle"))


def setup(hot=3, experts=7, slots=3):
    cat = ExpertCatalog(
        (ExpertLayerSpec(3, experts, 1, 0), ExpertLayerSpec(4, experts, 1, 0))
    )
    state = ResidencyState.allocate(cat.num_experts, hot, slots, "cpu")
    error = torch.zeros((), dtype=torch.int32)
    binding = SimpleNamespace(error_status=error, hot_generations=state.slot_generation)
    waves = fixed_waves(experts, slots)
    records = [
        BoundExecutionWave(layer.layer_id, wave, 0, binding)
        for layer in cat.layers
        for wave in waves
    ]
    ops = TensorResidencyOps(
        state, cat, execution_waves=records, update_bindings=binding, error_status=error
    )
    return ops, state, waves, Context()


def load_hot(state, keys):
    state.key_to_slot.fill_(-1)
    state.slot_to_key.copy_(torch.tensor(keys))
    state.slot_valid.copy_(state.slot_to_key >= 0)
    for slot, key in enumerate(keys):
        if key >= 0:
            state.key_to_slot[key] = slot
    state.slot_generation.copy_(torch.arange(len(keys)) + 5)


class ManualCompletion(Completion):
    def __init__(self, complete, calls):
        self.complete, self.calls = complete, calls

    def wait_on(self, stream):
        self.calls.append(("wait", stream))

    def synchronize(self):
        raise AssertionError("Pipeline abort must fence without a device host wait")

    def done(self):
        return self.complete


class TestPrefillPipelineFailures(unittest.TestCase):
    def make_pipeline(
        self,
        *,
        fail_copy_at=None,
        fail_combine=False,
        config=None,
        on_copy=None,
        ordered_copy_stream=False,
    ):
        calls, prepared = [], []
        ops, _, _, _ = setup(hot=0, experts=4, slots=2)

        def prepare(request, context):
            plan = MaterializationPlan(
                HostCopyPlan(()),
                SimpleNamespace(wave=request.wave, offset=request.local_demand),
            )
            prepared.append(ref(plan))
            return plan

        ops.prepare_prefill = prepare

        class HostOps(HostTransferOps):
            def materialize_into(self, plan, bindings, stream):
                calls.append(("submit", bindings.wave.local_start))
                if on_copy is not None:
                    on_copy(bindings.wave)
                if bindings.wave.local_start + 1 == fail_copy_at:
                    # The driver can queue writes before it reports an error.
                    raise RuntimeError("copy failed after partial submission")
                return TransferTicket(ManualCompletion(True, calls))

        def record(stream):
            calls.append(("record", stream))
            return TransferTicket(ManualCompletion(True, calls))

        context = ExecutionContext(
            Phase.PREFILL,
            compute_stream="compute",
            copy_stream="copy",
            record_completion=record,
        )

        def copy_fence(context):
            session = pipeline._active_session
            calls.append(
                (
                    "fence",
                    (session.worker is not None and not session.worker.done()),
                    context._workers,
                )
            )
            return TransferTicket(ManualCompletion(False, calls))

        def combine_bindings(layer, wave, leases, context):
            if fail_combine:
                raise ValueError("initial group binding validation failed")
            return SimpleNamespace(wave=wave)

        pipeline = BoundedPrefillWeightPipeline(
            ExpertWeightAccess(ops, host_ops=HostOps()),
            config or PrefillPipelineConfig(4, 2, 2, 2),
            make_request=lambda layer, wave, offset, context: ExecutionRequest(
                layer, offset, wave, Phase.PREFILL
            ),
            combine_bindings=combine_bindings,
            worker_scope=lambda context: nullcontext(),
            record_copy_completion=copy_fence,
            ordered_copy_stream=ordered_copy_stream,
        )

        def cleanup():
            if pipeline._active_session is not None:
                pipeline.abort(context)
            pipeline.close()

        self.addCleanup(cleanup)
        return SimpleNamespace(
            pipeline=pipeline, context=context, calls=calls, prepared=prepared
        )

    def assert_aborted_and_retained(self, fixture, session):
        pipeline, context = fixture.pipeline, fixture.context
        ticket = pipeline.abort(context)
        self.assertIs(ticket, pipeline.abort(context))
        self.assertIsNone(pipeline._active_session)
        self.assertTrue(session.closed)
        self.assertIn(session, ticket.keepalive)
        self.assertFalse(ticket.completion.done())
        self.assertEqual(
            [call for call in fixture.calls if call[0] == "fence"],
            [("fence", False, 0)],
        )
        # The partial-copy fence must remain an independent dependency even
        # after the invocation's compute/copy recorders report completion.
        completion = context.finish()
        self.assertFalse(completion.resources_idle.completion.done())
        self.assertTrue(completion.resources)
        with self.assertRaisesRegex(RuntimeError, "aborted|closed"):
            pipeline.start(3, context)
        return ticket

    def test_worker_start_failure_keeps_session_until_idempotent_abort(self):
        fixture = self.make_pipeline()
        with patch(
            "threading.Thread.start",
            side_effect=RuntimeError("thread start failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "thread start failed"):
                fixture.pipeline.start(3, fixture.context)
        session = fixture.pipeline._active_session
        self.assertFalse(session.worker_started)
        self.assertTrue(session.worker_done)
        self.assertEqual(fixture.context._workers, 0)
        self.assertFalse(any(call[0] == "submit" for call in fixture.calls))
        self.assert_aborted_and_retained(fixture, session)

    def test_partial_copy_failure_joins_worker_before_fencing_every_plan(self):
        fixture = self.make_pipeline(fail_copy_at=2)
        fixture.pipeline.start(3, fixture.context)
        session = fixture.pipeline._active_session
        session.worker.result(timeout=2)
        self.assertFalse((session.worker is not None and not session.worker.done()))
        self.assertIsInstance(session.error, RuntimeError)
        self.assertEqual(len(session.leases), 1)
        self.assertEqual(len(fixture.prepared), 2)
        # No ticket was returned for the second copy. Its reservation/bindings
        # must still be pinned by the real access/residency submission path.
        failed_plan = fixture.prepared[1]()
        self.assertIsNotNone(failed_plan)
        self.assertTrue(any(item is failed_plan for item in fixture.context._resources))
        self.assert_aborted_and_retained(fixture, session)
        submitted = fixture.calls.index(("submit", 1))
        self.assertLess(submitted, fixture.calls.index(("fence", False, 0)))

    def test_initial_group_failure_preserves_all_microbatch_leases(self):
        fixture = self.make_pipeline(fail_combine=True)
        fixture.pipeline.start(3, fixture.context)
        session = fixture.pipeline._active_session
        session.worker.result(timeout=2)
        self.assertFalse((session.worker is not None and not session.worker.done()))
        initial = tuple(session.leases)
        self.assertEqual(len(initial), 2)
        with self.assertRaisesRegex(ValueError, "binding validation"):
            fixture.pipeline.seal(
                3, ManualCompletion(True, fixture.calls), fixture.context
            )
        self.assertEqual(tuple(session.leases), initial)
        self.assertTrue(all(not lease.transferred for lease in initial))
        self.assertEqual(session.waves, ())
        ticket = self.assert_aborted_and_retained(fixture, session)
        for lease in initial:
            self.assertIn(lease.ticket, ticket.keepalive)

    def test_worker_is_reused_across_layers_and_closed_before_restart(self):
        workers = []
        fixture = self.make_pipeline(on_copy=lambda wave: workers.append(get_ident()))
        pipeline, context = fixture.pipeline, fixture.context
        thread_starts = []
        original_start = Thread.start

        def start(thread):
            thread_starts.append(thread)
            original_start(thread)

        with patch.object(Thread, "start", start):
            for layer in (3, 4, 3, 4):
                pipeline.start(layer, context)
                pipeline._active_session.worker.result(timeout=2)
                waves = pipeline.seal(layer, ManualCompletion(True, []), context)
                for wave in waves:
                    lease = pipeline.acquire(layer, wave, context)
                    pipeline.access.release_after(
                        lease, context.record_compute_completion(), context
                    )
                pipeline.finish(layer, context)
                self.assertEqual(context._workers, 0)
        self.assertEqual(len(thread_starts), 1)
        self.assertEqual(len(set(workers)), 1)
        self.assertNotEqual(workers[0], get_ident())
        pipeline.close()
        self.assertFalse(thread_starts[0].is_alive())
        with self.assertRaisesRegex(RuntimeError, "closed"):
            pipeline.start(3, context)

    def test_next_wave_submission_does_not_block_current_ffn_and_orders_reuse(self):
        entered, resume = Event(), Event()

        def on_copy(wave):
            if wave.local_start == 2:
                entered.set()
                if not resume.wait(2):
                    raise TimeoutError("test copy gate was not released")

        fixture = self.make_pipeline(
            config=PrefillPipelineConfig(6, 2, 2, 2), on_copy=on_copy
        )
        self.addCleanup(resume.set)
        pipeline, context = fixture.pipeline, fixture.context
        pipeline.start(3, context)
        pipeline._active_session.worker.result(timeout=2)
        waves = pipeline.seal(3, ManualCompletion(True, []), context)
        first = pipeline.acquire(3, waves[0], context)
        self.assertTrue(entered.wait(2))
        self.assertFalse(pipeline._active_session.worker.done())
        self.assertIsNone(pipeline._active_session.error)
        pipeline.access.wait_ready(first, context.compute_stream, context=context)
        # A's final reader must be queued before the background A overwrite.
        reader_calls = []
        read_done = TransferTicket(ManualCompletion(False, reader_calls))
        pipeline.access.release_after(first, read_done, context)
        resume.set()
        second = pipeline.acquire(3, waves[1], context)
        pipeline._active_session.worker.result(timeout=2)
        self.assertEqual(reader_calls, [("wait", "copy")])
        pipeline.access.release_after(
            second, context.record_compute_completion(), context
        )
        third = pipeline.acquire(3, waves[2], context)
        pipeline.access.release_after(
            third, context.record_compute_completion(), context
        )
        pipeline.finish(3, context)
        self.assertEqual(context._workers, 0)

    def test_background_wave_failure_is_reported_and_fenced_before_release(self):
        fixture = self.make_pipeline(fail_copy_at=3)
        pipeline, context = fixture.pipeline, fixture.context
        pipeline.start(3, context)
        session = pipeline._active_session
        session.worker.result(timeout=2)
        waves = pipeline.seal(3, ManualCompletion(True, []), context)
        first = pipeline.acquire(3, waves[0], context)
        session.worker.result(timeout=2)
        pipeline.access.release_after(
            first, context.record_compute_completion(), context
        )
        with self.assertRaisesRegex(RuntimeError, "submitter failed"):
            pipeline.acquire(3, waves[1], context)
        self.assertEqual(len(fixture.prepared), 3)
        self.assertIsNotNone(fixture.prepared[-1]())
        self.assert_aborted_and_retained(fixture, session)
        self.assertLess(
            fixture.calls.index(("submit", 2)),
            fixture.calls.index(("fence", False, 0)),
        )

    def test_abort_joins_background_submission_before_recording_fence(self):
        entered, resume, aborting, aborted = Event(), Event(), Event(), Event()

        def on_copy(wave):
            if wave.local_start == 2:
                entered.set()
                if not resume.wait(2):
                    raise TimeoutError("test copy gate was not released")

        fixture = self.make_pipeline(on_copy=on_copy)
        self.addCleanup(resume.set)
        pipeline, context = fixture.pipeline, fixture.context
        pipeline.start(3, context)
        pipeline._active_session.worker.result(timeout=2)
        waves = pipeline.seal(3, ManualCompletion(True, []), context)
        pipeline.acquire(3, waves[0], context)
        self.assertTrue(entered.wait(2))
        errors = []

        def abort():
            try:
                aborting.set()
                pipeline.abort(context)
            except BaseException as error:
                errors.append(error)
            finally:
                aborted.set()

        thread = Thread(target=abort)
        thread.start()
        try:
            self.assertTrue(aborting.wait(2))
            self.assertFalse(aborted.is_set())
            self.assertFalse(any(call[0] == "fence" for call in fixture.calls))
        finally:
            resume.set()
            thread.join(2)
        self.assertTrue(aborted.is_set())
        self.assertEqual(errors, [])
        self.assertEqual(fixture.calls[-1], ("fence", False, 0))
        self.assertEqual(context._workers, 0)

    def test_ordered_initial_fence_replaces_waits_but_keeps_child_leases(self):
        for ordered in (False, True):
            with self.subTest(ordered=ordered):
                fixture = self.make_pipeline(ordered_copy_stream=ordered)
                pipeline, context = fixture.pipeline, fixture.context
                pipeline.start(3, context)
                session = pipeline._active_session
                session.worker.result(timeout=2)
                children = tuple(session.leases)
                waves = pipeline.seal(3, ManualCompletion(True, []), context)
                # The fence must precede even the host submission of B.
                self.assertNotIn(("submit", 2), fixture.calls)
                if ordered:
                    self.assertEqual(fixture.calls[-1], ("fence", False, 0))
                first = pipeline.acquire(3, waves[0], context)
                pipeline.access.wait_ready(first, "compute", context=context)
                self.assertEqual(
                    fixture.calls.count(("wait", "compute")), 1 if ordered else 2
                )
                self.assertTrue(all(child.transferred for child in children))
                for child in children:
                    self.assertIn(child, first.ticket.keepalive)
                pipeline.abort(context)

    def test_close_rejects_an_active_session(self):
        fixture = self.make_pipeline()
        fixture.pipeline.start(3, fixture.context)
        with self.assertRaisesRegex(RuntimeError, "Finish or abort"):
            fixture.pipeline.close()


class TestTensorResidency(unittest.TestCase):
    def test_allocation_reports_physical_storage_without_hiding_aliases(self):
        storage = torch.empty(8, dtype=torch.int64)
        view = storage[2:4]
        records = tensor_allocations(
            "scratch.",
            (("view", view), ("empty", torch.empty(0))),
            ResourceKind.WORKSPACE,
        )
        self.assertEqual(len(records), 1)
        allocation = records[0]
        self.assertEqual(allocation.name, "scratch.view")
        self.assertEqual(allocation.nbytes, storage.untyped_storage().nbytes())
        self.assertEqual(allocation.storage_key, ("cpu", storage.data_ptr()))
        self.assertIs(allocation.keepalive, view)
        empty_view = tensor_allocations("view.", (("empty", storage[:0]),))
        self.assertEqual(empty_view[0].nbytes, allocation.nbytes)
        ledger = AllocationLedger({ResourceKind.WORKSPACE: 64}, records)
        self.assertEqual(ledger.usage[ResourceKind.WORKSPACE], 64)
        aliases = tensor_allocations("alias.", (("base", storage), ("view", view)))
        with self.assertRaisesRegex(ValueError, "physical allocations once"):
            AllocationLedger({ResourceKind.METADATA: 128}, aliases)

    def test_policy_gate_is_latched_before_policy_reuses_due(self):
        ops, state, _, ctx = setup()
        state.update_enabled.fill_(True)
        load_hot(state, [1, 5, 9])
        due = torch.tensor(True)
        update = ops.prepare_update(torch.tensor([1, 2, 9]), ctx, update_enabled=due)
        due.zero_()
        ops.commit_update(update, ctx.copied(), ctx)
        self.assertEqual(state.slot_to_key.tolist(), [1, 2, 9])
        self.assertEqual(state.slot_generation.tolist(), [5, 7, 7])

    def test_false_policy_gate_ignores_stale_targets_and_preserves_generation(self):
        ops, state, _, ctx = setup()
        state.update_enabled.fill_(True)
        load_hot(state, [1, 5, 9])
        before = {name: value.clone() for name, value in vars(state).items()}
        update = ops.prepare_update(
            torch.tensor([99, 99, -2]), ctx, update_enabled=torch.tensor(False)
        )
        self.assertEqual(int(update.copy_plan.count), 0)
        self.assertFalse(bool(update.copy_plan.valid.any()))
        ops.commit_update(update, ctx.copied(), ctx)
        for name, value in before.items():
            torch.testing.assert_close(getattr(state, name), value)
        self.assertEqual(int(ops.error_status), 0)
        self.assertNotIn("tail.target", ops._owned)

    def test_live_execution_bank_cannot_be_replanned_or_evicted(self):
        # Migrated from the retired cache integration stack: native production
        # residency must enforce exactly the same reader ownership contract.
        ops, state, waves, ctx = setup()
        request = ExecutionRequest(
            3, torch.ones(7, dtype=torch.int64), waves[0], Phase.DECODE
        )
        reservation = ops.resolve_execution(request, ctx)
        with self.assertRaisesRegex(RuntimeError, "Retire"):
            ops.resolve_execution(request, ctx)
        with self.assertRaisesRegex(RuntimeError, "Retire"):
            ops.prepare_update(state.slot_to_key, ctx)
        with self.assertRaisesRegex(RuntimeError, "Retire"):
            ops.begin_step(ctx)
        ops.publish_execution(reservation, ctx.copied(), ctx)
        ops.finish_execution(reservation, ctx.copied(), ctx)
        ops.begin_step(ctx)

    def test_hot_update_joins_auxiliary_readers_before_invalidating_sources(self):
        ops, state, waves, ctx = setup()
        state.update_enabled.fill_(True)
        load_hot(state, [1, 5, 9])
        reservation = ops.resolve_execution(
            ExecutionRequest(
                3, torch.ones(7, dtype=torch.int64), waves[0], Phase.DECODE
            ),
            ctx,
        )
        ops.publish_execution(reservation, ctx.copied(), ctx)
        ops.finish_execution(
            reservation, TransferTicket(Fence(ctx.calls, "aux-reader")), ctx
        )
        before = len(ctx.calls)
        update = ops.prepare_update(torch.tensor([1, 2, 9]), ctx)
        self.assertEqual(ctx.calls[before], ("wait", "aux-reader", "compute"))
        self.assertEqual(len(update.copy_dependencies), 1)
        self.assertEqual(ctx.calls[-1], ("record", update.copy_dependencies[0].name))
        self.assertEqual(len(reservation.copy_dependencies), 1)
        ops.commit_update(update, ctx.copied(), ctx)

    def test_mixed_sources_and_counts_come_from_post_dispatch(self):
        ops, state, waves, ctx = setup()
        load_hot(state, [1, 5, 9])
        counts = torch.tensor([2, 4, 0, 0, 1, 3, 8])
        ops.begin_step(ctx)
        expected = [0] * 14
        for wave in waves:
            request = ExecutionRequest(3, counts, wave, Phase.DECODE)
            reservation = ops.resolve_execution(request, ctx)
            plan = reservation.copy_plan
            for i, key in enumerate(range(wave.local_start, wave.local_end)):
                expected[key] = counts[key].item()
                self.assertEqual(plan.valid[i].item(), bool(counts[key]))
                if counts[key]:
                    self.assertEqual(
                        plan.source_kind[i].item(),
                        int(
                            CopySourceKind.HOT_SLOT
                            if key in (1, 5)
                            else CopySourceKind.HOST
                        ),
                    )
                    self.assertEqual(
                        plan.source_index[i].item(),
                        [1, 5, 9].index(key) if key in (1, 5) else key,
                    )
            ops.publish_execution(reservation, ctx.copied(), ctx)
            self.assertEqual(
                state.execution_valid[: plan.capacity].tolist(), plan.valid.tolist()
            )
            ops.finish_execution(reservation, ctx.copied(), ctx)
        self.assertEqual(state.step_access_count.tolist(), expected)
        self.assertEqual(state.slot_to_key.tolist(), [1, 5, 9])
        self.assertEqual(state.slot_generation.tolist(), [5, 6, 7])

    def test_empty_demand_zero_hot_and_tail_waves(self):
        ops, state, waves, ctx = setup(hot=0)
        counts = torch.zeros(7, dtype=torch.int32)
        for wave in waves:
            reservation = ops.resolve_execution(
                ExecutionRequest(3, counts, wave, Phase.DECODE), ctx
            )
            self.assertEqual(reservation.copy_plan.count.item(), 0)
            ops.publish_execution(reservation, ctx.copied(), ctx)
            self.assertFalse(
                state.execution_valid[: reservation.copy_plan.capacity].any()
            )
            ops.finish_execution(reservation, ctx.copied(), ctx)
        update = ops.prepare_update(torch.empty(0, dtype=torch.int64), ctx)
        ops.commit_update(update, ctx.copied(), ctx)
        self.assertEqual(ops.error_status.item(), 0)

    def test_tail_preserves_slots_invalidates_victims_before_commit(self):
        ops, state, _, ctx = setup()
        state.update_enabled.fill_(True)
        load_hot(state, [1, 5, 9])
        update = ops.prepare_update(torch.tensor([9, 2, 1]), ctx)
        self.assertEqual(state.slot_to_key.tolist(), [1, 5, 9])
        self.assertEqual(state.slot_valid.tolist(), [True, False, True])
        self.assertEqual(state.key_to_slot[5].item(), -1)
        self.assertEqual(state.key_to_slot[2].item(), -1)
        self.assertEqual(update.copy_plan.valid.tolist(), [False, True, False])
        self.assertEqual(update.copy_plan.source_index.tolist(), [1, 2, 9])
        self.assertEqual(update.copy_plan.count.item(), 1)
        ops.commit_update(update, ctx.copied(), ctx)
        self.assertEqual(state.slot_to_key.tolist(), [1, 2, 9])
        self.assertEqual(state.slot_generation.tolist(), [5, 7, 7])
        self.assertEqual(state.key_to_slot[2].item(), 1)
        self.assertIn(("wait", "bundle", "compute"), ctx.calls)
        self.assertEqual(ctx.calls[-1][0], "record")

    def test_disabled_updates_preserve_every_metadata_tensor(self):
        ops, state, _, ctx = setup()
        load_hot(state, [1, 5, -1])
        old = {name: tensor.clone() for name, tensor in vars(state).items()}
        update = ops.prepare_update(torch.tensor([3, 3, 888]), ctx)
        self.assertEqual(update.copy_plan.count.item(), 0)
        ops.commit_update(update, ctx.copied(), ctx)
        for name, tensor in old.items():
            self.assertTrue(torch.equal(getattr(state, name), tensor), name)
        self.assertEqual(ops.error_status.item(), 0)

    def test_randomized_tail_matches_reference_and_stable_allocations(self):
        rng = random.Random(1021)
        for hot in (1, 3, 7, 14):
            ops, state, _, ctx = setup(hot=hot)
            state.update_enabled.fill_(True)
            keys = [-1] * hot
            generation = [0] * hot
            pointers = [a.storage_key for a in ops.allocations()]
            for _ in range(80):
                chosen = rng.sample(range(14), rng.randrange(hot + 1))
                targets = chosen + [-1] * (hot - len(chosen))
                retained = set(keys) & set(chosen)
                newcomers = iter(key for key in chosen if key not in retained)
                proposed = [
                    key if key in retained else next(newcomers, -1) for key in keys
                ]
                generation = [
                    g + int(a != b) for a, b, g in zip(keys, proposed, generation)
                ]
                update = ops.prepare_update(
                    torch.tensor(targets, dtype=torch.int64), ctx
                )
                self.assertEqual(
                    update.copy_plan.count.item(),
                    sum(a != b and b >= 0 for a, b in zip(keys, proposed)),
                )
                ops.commit_update(update, ctx.copied(), ctx)
                self.assertEqual(state.slot_to_key.tolist(), proposed)
                self.assertEqual(state.slot_generation.tolist(), generation)
                self.assertEqual(
                    state.slot_valid.tolist(), [key >= 0 for key in proposed]
                )
                inverse = [-1] * 14
                for slot, key in enumerate(proposed):
                    if key >= 0:
                        inverse[key] = slot
                self.assertEqual(state.key_to_slot.tolist(), inverse)
                self.assertEqual(ops.error_status.item(), 0)
                keys = proposed
            self.assertEqual([a.storage_key for a in ops.allocations()], pointers)
            self.assertEqual(len(set(pointers)), len(pointers))
            self.assertGreater(sum(item.nbytes for item in ops.allocations()), 0)

    def test_copy_error_suppresses_execution_and_hot_publication(self):
        ops, state, waves, ctx = setup()
        state.update_enabled.fill_(True)
        load_hot(state, [1, 5, 9])
        reservation = ops.resolve_execution(
            ExecutionRequest(
                3, torch.ones(7, dtype=torch.int64), waves[0], Phase.DECODE
            ),
            ctx,
        )
        ops.error_status.fill_(1)
        ops.publish_execution(reservation, ctx.copied(), ctx)
        self.assertFalse(state.execution_valid.any())
        self.assertFalse(bool(state.execution_valid.any()))
        ops.finish_execution(reservation, ctx.copied(), ctx)
        ops.error_status.zero_()
        update = ops.prepare_update(torch.tensor([1, 2, 9]), ctx)
        ops.error_status.fill_(1)
        ops.commit_update(update, ctx.copied(), ctx)
        self.assertEqual(state.key_to_slot[2].item(), -1)
        self.assertFalse(state.slot_valid[1])

    def test_generation_change_during_copy_poisoned(self):
        ops, state, _, ctx = setup()
        state.update_enabled.fill_(True)
        load_hot(state, [1, 5, 9])
        update = ops.prepare_update(torch.tensor([1, 2, 9]), ctx)
        state.slot_generation[1] += 1
        ops.commit_update(update, ctx.copied(), ctx)
        self.assertNotEqual(ops.error_status.item(), 0)
        self.assertFalse(state.slot_valid[1])
        self.assertEqual(state.key_to_slot[2].item(), -1)

    def test_bad_targets_latch_error_and_never_copy_or_publish(self):
        for target in ([1, 1, -1], [-2, -1, -1], [14, -1, -1]):
            ops, state, _, ctx = setup()
            load_hot(state, [1, 5, 9])
            state.update_enabled.fill_(True)
            old = state.slot_to_key.clone()
            update = ops.prepare_update(torch.tensor(target), ctx)
            self.assertEqual(update.copy_plan.count.item(), 0)
            ops.commit_update(update, ctx.copied(), ctx)
            self.assertNotEqual(ops.error_status.item(), 0)
            self.assertTrue(torch.equal(old, state.slot_to_key))

    def test_rejects_foreign_double_and_unbound_operations(self):
        ops, _, waves, ctx = setup()
        req = ExecutionRequest(
            3, torch.ones(7, dtype=torch.int64), waves[0], Phase.DECODE
        )
        reservation = ops.resolve_execution(req, ctx)
        with self.assertRaises(ValueError):
            ops.publish_execution(reservation, ctx.copied(), Context())
        ops.publish_execution(reservation, ctx.copied(), ctx)
        with self.assertRaises(RuntimeError):
            ops.publish_execution(reservation, ctx.copied(), ctx)
        ops.finish_execution(reservation, ctx.copied(), ctx)
        with self.assertRaises(RuntimeError):
            ops.finish_execution(reservation, ctx.copied(), ctx)
        with self.assertRaises(ValueError):
            ops.resolve_execution(
                ExecutionRequest(5, req.local_demand, waves[0], Phase.DECODE), ctx
            )
        update = ops.prepare_update(torch.tensor([-1, -1, -1]), ctx)
        with self.assertRaises(RuntimeError):
            ops.prepare_update(torch.tensor([-1, -1, -1]), ctx)
        ops.commit_update(update, ctx.copied(), ctx)
        with self.assertRaises(RuntimeError):
            ops.commit_update(update, ctx.copied(), ctx)

    def test_prefill_injected_host_plan_does_not_mutate_decode_state(self):
        ops, state, waves, ctx = setup()
        ctx.phase, ctx.capturing = Phase.PREFILL, False
        old = {name: tensor.clone() for name, tensor in vars(state).items()}
        ops.prepare_prefill = lambda request, context: MaterializationPlan(
            HostCopyPlan(()), "borrowed"
        )
        request = ExecutionRequest(3, None, waves[0], Phase.PREFILL)
        reservation = ops.resolve_execution(request, ctx)
        ops.publish_execution(reservation, ctx.copied(), ctx)
        ops.finish_execution(reservation, ctx.copied(), ctx)
        for name, tensor in old.items():
            self.assertTrue(torch.equal(getattr(state, name), tensor), name)

    def test_planning_does_not_extract_tensor_values(self):
        from unittest.mock import patch

        ops, state, waves, ctx = setup()
        state.update_enabled.fill_(True)
        demand, targets = torch.ones(7, dtype=torch.int64), torch.tensor([1, 3, 5])
        with (
            patch.object(torch.Tensor, "item", side_effect=AssertionError("host item")),
            patch.object(
                torch.Tensor, "tolist", side_effect=AssertionError("host list")
            ),
        ):
            ops.begin_step(ctx)
            reservation = ops.resolve_execution(
                ExecutionRequest(3, demand, waves[0], Phase.DECODE), ctx
            )
            ops.publish_execution(reservation, ctx.copied(), ctx)
            ops.finish_execution(reservation, ctx.copied(), ctx)
            update = ops.prepare_update(targets, ctx)
            ops.commit_update(update, ctx.copied(), ctx)

    def test_invalid_slot_with_same_old_key_is_reloaded(self):
        ops, state, _, ctx = setup()
        load_hot(state, [1, 5, 9])
        state.update_enabled.fill_(True)
        state.slot_valid[1] = False
        state.key_to_slot[5] = -1
        update = ops.prepare_update(torch.tensor([1, 5, 9]), ctx)
        self.assertEqual(update.copy_plan.valid.tolist(), [False, True, False])
        self.assertEqual(update.copy_plan.count.item(), 1)
        self.assertEqual(state.slot_generation.tolist(), [5, 7, 7])
        self.assertFalse(state.slot_valid[1])
        ops.commit_update(update, ctx.copied(), ctx)
        self.assertTrue(state.slot_valid[1])
        self.assertEqual(state.key_to_slot[5].item(), 1)

    def test_corrupt_inverse_is_not_silently_treated_as_host_miss(self):
        for kind in ("invalid", "wrong_key", "too_large", "negative"):
            ops, state, waves, ctx = setup()
            load_hot(state, [1, 5, 9])
            if kind == "invalid":
                state.slot_valid[0] = False
            elif kind == "wrong_key":
                state.slot_to_key[0] = 7
            elif kind == "too_large":
                state.key_to_slot[1] = 100
            else:
                state.key_to_slot[1] = -2
            request = ExecutionRequest(
                3, torch.ones(7, dtype=torch.int64), waves[0], Phase.DECODE
            )
            reservation = ops.resolve_execution(request, ctx)
            self.assertNotEqual(ops.error_status.item(), 0, kind)
            self.assertEqual(reservation.copy_plan.count.item(), 0, kind)
            ops.publish_execution(reservation, ctx.copied(), ctx)
            self.assertFalse(state.execution_valid.any())

    def test_tail_rejects_corrupt_canonical_maps_without_out_of_bounds(self):
        for kind in (
            "missing_inverse",
            "duplicate",
            "key_out_of_bounds",
            "duplicate_inverse_sum",
        ):
            ops, state, _, ctx = setup()
            load_hot(state, [1, 5, 9])
            state.update_enabled.fill_(True)
            if kind == "missing_inverse":
                state.key_to_slot[5] = -1
            elif kind == "duplicate":
                state.slot_to_key[1] = 1
            elif kind == "duplicate_inverse_sum":
                state.slot_to_key.copy_(torch.tensor([1, 1, -1]))
                state.slot_valid.copy_(torch.tensor([True, True, False]))
                state.key_to_slot.fill_(-1)
                state.key_to_slot[1] = 2  # Sum of slots+1, minus 1, mimics an inverse.
            else:
                state.slot_to_key[1] = 100
            update = ops.prepare_update(torch.tensor([1, 5, 9]), ctx)
            self.assertEqual(update.copy_plan.count.item(), 0, kind)
            ops.commit_update(update, ctx.copied(), ctx)
            self.assertNotEqual(ops.error_status.item(), 0, kind)

    def test_binding_rejects_aliasing_banks_and_duplicate_metadata_names(self):
        cat = ExpertCatalog((ExpertLayerSpec(3, 8, 1, 0),))
        state = ResidencyState.allocate(8, 2, 5, "cpu")
        cases = (
            (
                BoundExecutionWave(3, ExpertWave(0, 0, 3, 0), 0, None),
                BoundExecutionWave(3, ExpertWave(1, 3, 6, 1), 2, None),
            ),
            (
                BoundExecutionWave(3, ExpertWave(0, 0, 3, 0), 0, None),
                BoundExecutionWave(3, ExpertWave(1, 3, 6, 0), 2, None),
            ),
            (
                BoundExecutionWave(3, ExpertWave(0, 0, 3, 0), 0, None),
                BoundExecutionWave(3, ExpertWave(0, 3, 6, 0), 0, None),
            ),
        )
        for records in cases:
            with self.subTest(records=records), self.assertRaises(ValueError):
                TensorResidencyOps(
                    state, cat, execution_waves=records, update_bindings=None
                )

    def test_retired_bank_reader_waits_before_next_overwrite(self):
        ops, _, waves, ctx = setup()
        demand = torch.ones(7, dtype=torch.int64)
        first = ops.resolve_execution(
            ExecutionRequest(3, demand, waves[0], Phase.DECODE), ctx
        )
        ops.publish_execution(first, ctx.copied(), ctx)
        reader = TransferTicket(Fence(ctx.calls, "reader"))
        ops.finish_execution(first, reader, ctx)
        before = len(ctx.calls)
        ops.resolve_execution(ExecutionRequest(3, demand, waves[1], Phase.DECODE), ctx)
        self.assertEqual(ctx.calls[before], ("wait", "reader", "compute"))

    def test_prefill_publication_keeps_worker_off_compute_stream(self):
        ops, _, waves, ctx = setup()
        ctx.phase, ctx.capturing = Phase.PREFILL, False
        ops.prepare_prefill = lambda request, context: MaterializationPlan(
            HostCopyPlan(()), "borrowed"
        )
        request = ExecutionRequest(3, None, waves[0], Phase.PREFILL)
        reservation = ops.resolve_execution(request, ctx)
        ready = ops.publish_execution(reservation, ctx.copied(), ctx)
        self.assertEqual(ctx.calls, [])
        ready.completion.wait_on(ctx.compute_stream)
        self.assertEqual(ctx.calls, [("wait", "bundle", "compute")])
