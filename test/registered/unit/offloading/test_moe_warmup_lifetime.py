"""CPU lifecycle regressions; no accelerator graph or memory-size certification."""

import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from test_moe_graph_leases import Backend, Owner

from sglang.srt.offloading.moe.config import Phase
from sglang.srt.offloading.moe.graph import GraphLeaseBackend
from sglang.srt.offloading.moe.npu.runtime import NpuBoundMoERuntime
from sglang.srt.offloading.moe.runtime import ManagedExecutionContext
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
            current_stream=lambda: "compute",
            record=lambda stream: TransferTicket(ImmediateCompletion()),
            allocated=lambda: 0,
            reset_peak=lambda: None,
        )
        device.synchronize = Mock(side_effect=lambda: calls.append("device_drain"))
        runtime = NpuBoundMoERuntime(
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
            layer_programs={3: object()},
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
    def invocation(runtime, references):
        batch = object()
        runtime._capture_head(None, batch, 1)
        context = runtime.context
        payload = Payload()
        references.append(weakref.ref(payload))
        context.retain(payload)
        # Production weight leases refer back to the invocation context.
        context.retain(SimpleNamespace(context=context))
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
        gc.collect()
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
        context = ManagedExecutionContext(
            Phase.DECODE,
            compute_stream="compute",
            copy_stream="copy",
            record_completion=lambda stream: TransferTicket(ImmediateCompletion()),
            failure_drain=lambda: None,
        )
        payload = Payload()
        reference = weakref.ref(payload)
        context.retain(payload)
        context.retain(SimpleNamespace(context=context))
        context.retain(TransferTicket(completion))
        del payload
        with self.assertRaisesRegex(RuntimeError, "Finish the invocation"):
            context.release_completed()
        context.finish()
        with self.assertRaisesRegex(RuntimeError, "reader unfinished"):
            context.release_completed()
        self.assertIsNotNone(reference())
        completion.synchronize.side_effect = None
        context.release_completed()
        self.assertIsNone(reference())


if __name__ == "__main__":
    unittest.main()
