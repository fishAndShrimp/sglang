"""Unique completion facts preserve external readers and failure semantics."""

import unittest
import gc
import weakref

from sglang.srt.offloading.moe.config import Phase
from sglang.srt.offloading.moe.runtime import ManagedExecutionContext
from sglang.srt.offloading.transfer import (
    EventCompletion,
    JoinedCompletion,
    TransferTicket,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class Event:
    def __init__(self, ready=True):
        self.ready = ready
        self.queries = self.syncs = 0

    def query(self):
        self.queries += 1
        return self.ready

    def synchronize(self):
        self.syncs += 1


class Stream:
    def __init__(self):
        self.waits = []

    def wait_event(self, event):
        self.waits.append(event)


class TransferFrontierTests(unittest.TestCase):
    def test_nested_join_deduplicates_same_event_across_ticket_wrappers(self):
        first, second = Event(), Event()
        a, b = EventCompletion(first), EventCompletion(second)
        joined = JoinedCompletion(
            (a, JoinedCompletion((a, b)), EventCompletion(first), b)
        )
        self.assertTrue(joined.done())
        self.assertEqual((first.queries, second.queries), (1, 1))
        stream = Stream()
        joined.wait_on(stream)
        self.assertEqual(stream.waits, [first, second])
        joined.synchronize()
        self.assertEqual((first.syncs, second.syncs), (1, 1))

    def test_distinct_events_are_not_discarded_based_on_assumed_stream_order(self):
        compute, external_reader = Event(), Event(ready=False)
        joined = JoinedCompletion(
            (EventCompletion(compute), EventCompletion(external_reader))
        )
        self.assertFalse(joined.done())
        external_reader.ready = True
        self.assertTrue(joined.done())
        self.assertEqual(external_reader.queries, 2)

    def test_join_creation_does_not_query_wait_or_synchronize(self):
        event = Event()
        completion = EventCompletion(event)
        JoinedCompletion((JoinedCompletion((completion,)), completion))
        self.assertEqual((event.queries, event.syncs), (0, 0))

    def test_failure_is_not_converted_to_success(self):
        class FailedEvent(Event):
            def query(self):
                raise RuntimeError("device query failed")

        joined = JoinedCompletion((EventCompletion(FailedEvent()),))
        with self.assertRaisesRegex(RuntimeError, "device query failed"):
            joined.done()

    def test_context_wait_skips_only_proven_same_stream_events(self):
        compute, copy = Stream(), Stream()
        context = ManagedExecutionContext(
            Phase.DECODE,
            compute_stream=compute,
            copy_stream=copy,
            record_completion=lambda stream: TransferTicket(EventCompletion(Event())),
            failure_drain=lambda: None,
        )
        compute_done = context.record_compute_completion()
        copy_done = context._record_on(copy)
        external = EventCompletion(Event())
        joined = JoinedCompletion(
            (compute_done.completion, copy_done.completion, external)
        )
        context.wait_for(joined, compute)
        self.assertEqual(compute.waits, [copy_done.completion.event, external.event])
        self.assertEqual(compute_done.completion.event.queries, 0)
        self.assertEqual(compute_done.completion.event.syncs, 0)

    def test_custom_aggregate_completion_keeps_its_additional_gate(self):
        class GatedCompletion(JoinedCompletion):
            def done(self):
                return False

        context = ManagedExecutionContext(
            Phase.DECODE,
            compute_stream=Stream(),
            copy_stream=Stream(),
            record_completion=lambda stream: TransferTicket(EventCompletion(Event())),
            failure_drain=lambda: None,
        )
        external = GatedCompletion((EventCompletion(Event()),))
        context.register_completion(external)
        completion = context.finish()
        self.assertFalse(completion.resources_idle.completion.done())

    def test_retirement_clears_batch_even_when_caller_keeps_context(self):
        class Batch:
            pass

        was_enabled = gc.isenabled()
        gc.disable()
        try:
            context = ManagedExecutionContext(
                Phase.DECODE,
                compute_stream=Stream(),
                copy_stream=Stream(),
                record_completion=lambda stream: TransferTicket(
                    EventCompletion(Event())
                ),
                failure_drain=lambda: None,
            )
            context.forward_batch = Batch()
            batch = weakref.ref(context.forward_batch)
            completion = context.finish()
            self.assertTrue(completion.resources_idle.completion.done())
            completion.dispose_completed()
            self.assertIsNone(batch())
            self.assertTrue(context._disposed)
            self.assertIsNone(context._record)
            self.assertIsNone(context._failure_drain)
        finally:
            if was_enabled:
                gc.enable()


if __name__ == "__main__":
    unittest.main()
