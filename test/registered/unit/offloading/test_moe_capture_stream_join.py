"""CPU stream-dependency regressions; actual NPU capture needs hardware validation."""

import unittest
from types import SimpleNamespace

from sglang.srt.offloading.moe.config import Phase
from sglang.srt.offloading.moe.npu.runtime import NpuBoundMoERuntime
from sglang.srt.offloading.moe.runtime import ManagedExecutionContext
from sglang.srt.offloading.transfer import EventCompletion, TransferTicket
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class CaptureStream:
    def __init__(self, graph):
        self.graph = graph
        self.frontier = None

    def wait_event(self, event):
        self.graph.waits.append((self, event))
        self.graph.submit(self, event.node)


class CaptureGraph:
    """Require every recorded node to reach the capture origin before ending."""

    def __init__(self):
        self.compute = CaptureStream(self)
        self.copy = CaptureStream(self)
        self.parents = {}
        self.waits = []

    def submit(self, stream, *dependencies):
        parents = set(dependencies)
        if stream.frontier is not None:
            parents.add(stream.frontier)
        node = len(self.parents)
        self.parents[node] = parents
        stream.frontier = node
        return node

    @staticmethod
    def forbidden_host_wait():
        raise AssertionError("Capture must not synchronize or query the host")

    def record(self, stream):
        event = SimpleNamespace(
            node=self.submit(stream),
            synchronize=self.forbidden_host_wait,
            query=self.forbidden_host_wait,
        )
        return TransferTicket(EventCompletion(event), (event, stream))

    def end_capture(self):
        reachable = set()
        pending = [self.compute.frontier]
        while pending:
            node = pending.pop()
            if node is not None and node not in reachable:
                reachable.add(node)
                pending.extend(self.parents[node])
        if reachable != set(self.parents):
            raise RuntimeError("CaptureEnd: auxiliary stream has unjoined tasks")


class TestMoECaptureStreamJoin(unittest.TestCase):
    def context(self, graph, *, phase=Phase.DECODE, capturing=True):
        return ManagedExecutionContext(
            phase,
            compute_stream=graph.compute,
            copy_stream=graph.copy,
            record_completion=graph.record,
            failure_drain=graph.forbidden_host_wait,
            capturing=capturing,
        )

    @staticmethod
    def materialize_and_publish(context, graph):
        context.record_compute_completion().completion.wait_on(graph.copy)
        copied = graph.record(graph.copy)
        context.retain(copied)
        copied.completion.wait_on(graph.compute)
        return context.record_compute_completion()

    def test_npu_capture_tail_closes_copy_stream_after_cache_publication(self):
        graph = CaptureGraph()
        runtime = NpuBoundMoERuntime(
            device_runtime=SimpleNamespace(
                current_stream=lambda: graph.compute,
                copy_stream=graph.copy,
                record=graph.record,
            ),
            residency=SimpleNamespace(
                update_enabled=SimpleNamespace(fill_=lambda value: None)
            ),
            residency_ops=SimpleNamespace(begin_step=lambda context: None),
            cache_updates=SimpleNamespace(
                emit_step_end=lambda context: self.materialize_and_publish(
                    context, graph
                )
            ),
            snapshot=None,
            reload_weights=None,
            layer_programs={3: object()},
            resources=(),
            expert_nbytes=1,
            storage=(),
        )
        runtime._capture = []
        batch = object()
        runtime._capture_head(None, batch, 1)
        runtime._capture_tail(None, None, batch, 1)
        graph.end_capture()
        self.assertIsNone(runtime._context)
        self.assertEqual(len(runtime._capture), 1)

    def test_explicit_completion_tickets_still_close_capture(self):
        graph = CaptureGraph()
        context = self.context(graph)
        output_ready = context.record_compute_completion()
        residency_ready = self.materialize_and_publish(context, graph)
        completion = context.finish(
            output_ready=output_ready, residency_ready=residency_ready
        )
        graph.end_capture()
        self.assertIs(completion.output_ready, output_ready)
        self.assertIs(completion.residency_ready, residency_ready)

    def test_eager_finish_does_not_add_stream_dependencies(self):
        for phase in (Phase.DECODE, Phase.PREFILL):
            with self.subTest(phase=phase):
                graph = CaptureGraph()
                context = self.context(graph, phase=phase, capturing=False)
                completion = context.finish()
                self.assertEqual(graph.waits, [])
                self.assertIsNot(completion.output_ready, completion.residency_ready)
                self.assertIsNotNone(graph.copy.frontier)

    def test_failed_capture_join_preserves_context_and_keepalives(self):
        graph = CaptureGraph()
        context = self.context(graph)
        resource = object()
        context.retain(resource)

        def fail(event):
            raise RuntimeError("stream wait failed")

        graph.compute.wait_event = fail
        with self.assertRaisesRegex(RuntimeError, "stream wait failed"):
            context.finish()
        self.assertFalse(context._closed)
        self.assertIsNone(context._completion)
        self.assertIn(resource, context._resources)
        self.assertTrue(
            any(
                isinstance(resource, TransferTicket)
                and resource.keepalive[1] is graph.copy
                for resource in context._resources
            )
        )


if __name__ == "__main__":
    unittest.main()
