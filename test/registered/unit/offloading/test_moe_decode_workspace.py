"""CPU tests of production decode lifetimes/layout; native EP/GMM are simulated."""

import unittest
import weakref
from collections import namedtuple
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.offloading.moe.config import Phase, fixed_waves
from sglang.srt.offloading.moe.npu import experts as native
from sglang.srt.offloading.moe.npu.runtime import NpuBoundMoERuntime
from sglang.srt.offloading.moe.programs import (
    DecodeLocalExpertProgram,
    DispatchOnceProgram,
)
from sglang.srt.offloading.moe.runtime import ManagedExecutionContext
from sglang.srt.offloading.transfer import ImmediateCompletion, TransferTicket
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

Routes = namedtuple("Routes", "topk_ids topk_weights")


@dataclass
class QuantInfo:
    w13_weight: object
    w2_weight: object


class TestDecodeWorkspace(unittest.TestCase):
    def fixture(self, ids=(0, 2), experts=4, slots=2):
        ids = torch.tensor(ids, dtype=torch.int64).unsqueeze(1)
        rows, hidden = max(8, ids.numel()), 4
        raw_refs, output_refs, waits = [], [], []
        error = torch.zeros((), dtype=torch.int32)
        layout = native.NPUDispatchLayout(experts, hidden, rows, "cpu", error)
        quant = QuantInfo(torch.ones(slots, 1, 1), torch.ones(slots, 1, 1))
        bank = native.NPUExpertBank(quant, torch.ones(slots, dtype=torch.bool), error)
        bindings = SimpleNamespace(expert_bank=bank, error_status=error)
        test = self

        class Dispatch:
            handle = None

            def dispatch(self, hidden_states, topk_output):
                expert_ids = topk_output.topk_ids[:, 0]
                selected = (expert_ids >= 0).nonzero().flatten()
                order = selected[expert_ids[selected].argsort()]
                receive = torch.full((rows, hidden), float("nan"), dtype=torch.bfloat16)
                receive[: order.numel()].copy_(hidden_states[order])
                raw_refs.append(weakref.ref(receive))
                self.handle = order
                self.token_rows = hidden_states.shape[0]
                return SimpleNamespace(
                    format=SimpleNamespace(value="deepep_ll"),
                    hidden_states=receive,
                    hidden_states_scale=None,
                    masked_m=torch.bincount(expert_ids[order], minlength=experts),
                    topk_ids=topk_output.topk_ids,
                    topk_weights=topk_output.topk_weights,
                )

            def combine(self, combine_input):
                result = torch.zeros(self.token_rows, hidden, dtype=torch.bfloat16)
                result.index_copy_(
                    0, self.handle, combine_input.hidden_states[: self.handle.numel()]
                )
                self.handle = None
                return result

        class Core:
            def run(self, runner_input, quant_info, running_state):
                # Previous waves/layers must not remain alive during a new F.
                test.assertTrue(all(ref() is None for ref in output_refs))
                output = torch.full((rows, hidden), float("nan"), dtype=torch.bfloat16)
                offset = 0
                for lane, count in enumerate(runner_input.expert_tokens.tolist()):
                    output[offset : offset + count].copy_(
                        runner_input.hidden_states[offset : offset + count]
                        + quant_info.w13_weight[lane, 0, 0]
                    )
                    offset += count
                output_refs.append(weakref.ref(output))
                return SimpleNamespace(hidden_states=output)

        class Access:
            def stage(self, request, context):
                wave = request.wave
                size = wave.local_end - wave.local_start
                quant.w13_weight[:size, 0, 0].copy_(
                    torch.arange(wave.local_start + 1, wave.local_end + 1)
                )
                return bindings

            def wait_ready(self, lease, stream):
                waits.append("ready")
                return lease

            def release_after(self, lease, ticket, context):
                waits.append("read_done")

        dispatcher = native.NPUOffloadDispatcher(
            SimpleNamespace(dispatcher=Dispatch(), reduce_results=False),
            error_status=error,
        )
        # Communication results are simulated; the packing/runner code is real.
        dispatcher._combine_input = lambda raw, hidden_states: SimpleNamespace(
            hidden_states=hidden_states
        )
        runner = native.NPUExpertRunner(
            None, _core=Core(), _input_type=SimpleNamespace
        )
        local = DecodeLocalExpertProgram(
            Access(),
            lambda layer, wave_input, wave, context: SimpleNamespace(wave=wave),
            runner,
        )
        program = DispatchOnceProgram(
            fixed_waves(experts, slots),
            dispatcher,
            local,
            layout,
            local_experts=experts,
        )
        device = SimpleNamespace(
            copy_stream="copy",
            current_stream=lambda: "compute",
            record=lambda stream: TransferTicket(ImmediateCompletion()),
            synchronize=lambda: waits.append("drain"),
        )
        runtime = NpuBoundMoERuntime(
            device_runtime=device,
            residency=SimpleNamespace(update_enabled=torch.tensor(True)),
            residency_ops=SimpleNamespace(begin_step=lambda context: None),
            cache_updates=None,
            snapshot=None,
            reload_weights=None,
            layer_programs={i: SimpleNamespace(decode=program) for i in range(3, 43)},
            resources=(),
            expert_nbytes=1,
            storage=(),
        )
        context = runtime._new_context(Phase.DECODE, capturing=True)
        context.forward_batch = SimpleNamespace(
            moe_num_token_non_padded=lambda: torch.tensor(ids.shape[0])
        )
        runtime._context, runtime._capture = context, []
        return SimpleNamespace(
            runtime=runtime,
            context=context,
            layout=layout,
            routes=Routes(ids, torch.ones_like(ids, dtype=torch.float32)),
            input=torch.ones(ids.shape[0], hidden, dtype=torch.bfloat16),
            raw_refs=raw_refs,
            output_refs=output_refs,
            bank=bank,
            waits=waits,
        )

    def test_forty_layers_reuse_workspace_without_pinning_receive_or_ffn_outputs(self):
        with patch.dict("os.environ", {"MOE_ENABLE_TOPK_NEG_ONE": "1"}), patch.object(
            native, "_compute_scope", lambda *args: nullcontext()
        ):
            fixture = self.fixture()
            pointers = {k: v.data_ptr() for k, v in fixture.layout._owned.items()}
            value = fixture.input
            for step, layer in enumerate(range(3, 43), 1):
                value = fixture.runtime.run_layer(layer, value, fixture.routes)
                self.assertTrue(all(ref() is None for ref in fixture.raw_refs))
                self.assertTrue(all(ref() is None for ref in fixture.output_refs))
                torch.testing.assert_close(
                    value,
                    fixture.input + step * (fixture.routes.topk_ids + 1),
                )
            self.assertEqual(
                pointers, {k: v.data_ptr() for k, v in fixture.layout._owned.items()}
            )
            self.assertEqual(fixture.waits, ["ready", "read_done"] * 80)
            # Normal decode submission/capture never adds a host synchronization.
            self.assertNotIn("drain", fixture.waits)
            completion = fixture.context.finish()
            self.assertTrue(completion.resources_idle.completion.done())

    def test_padding_is_sanitized_in_place_including_empty_and_partial_waves(self):
        cases = [((0, 2), 4, 2), ((2, 2), 4, 2), ((-1, -1), 4, 2), ((4, 0), 5, 2)]
        with patch.dict("os.environ", {"MOE_ENABLE_TOPK_NEG_ONE": "1"}), patch.object(
            native, "_compute_scope", lambda *args: nullcontext()
        ):
            for ids, experts, slots in cases:
                with self.subTest(ids=ids, experts=experts):
                    fixture = self.fixture(ids, experts, slots)
                    write = fixture.layout.write_wave

                    def checked_write(layout, wave, output, context):
                        # Padding cleanup must reuse the native GMM allocation.
                        self.assertIs(output, fixture.output_refs[-1]())
                        self.assertTrue(bool(torch.isfinite(output).all()))
                        count = int(layout.selected.row_count)
                        self.assertTrue(bool((output[count:] == 0).all()))
                        write(layout, wave, output, context)

                    fixture.layout.write_wave = checked_write
                    result = fixture.runtime.run_layer(3, fixture.input, fixture.routes)
                    torch.testing.assert_close(
                        result,
                        torch.where(
                            fixture.routes.topk_ids >= 0,
                            fixture.input + fixture.routes.topk_ids + 1,
                            0,
                        ),
                    )

    def test_scopes_keep_transfer_storage_and_events_until_completion(self):
        class Completion(ImmediateCompletion):
            ready = False

            def done(self):
                return self.ready

            def synchronize(self):
                self.ready = True

        completion = Completion()
        context = ManagedExecutionContext(
            Phase.DECODE,
            compute_stream="compute",
            copy_stream="copy",
            record_completion=lambda stream: TransferTicket(ImmediateCompletion()),
            failure_drain=lambda: None,
            capturing=True,
        )
        with context.temporary_scope():
            temporary = torch.ones(4)
            protected = torch.ones(4)
            temporary_ref = weakref.ref(temporary)
            protected_ref = weakref.ref(protected)
            context.retain(temporary)
            context.retain(TransferTicket(completion, (protected,)))
            del temporary, protected
        self.assertIsNone(temporary_ref())
        self.assertIsNotNone(protected_ref())
        self.assertFalse(completion.done())
        finished = context.finish()
        self.assertFalse(finished.resources_idle.completion.done())
        finished.resources_idle.completion.synchronize()
        self.assertTrue(completion.done())
        # Captured event/transfer storage stays pinned until explicit teardown.
        self.assertIsNotNone(protected_ref())

    def test_prefill_keeps_invocation_lifetimes_for_worker_submissions(self):
        context = ManagedExecutionContext(
            Phase.PREFILL,
            compute_stream="compute",
            copy_stream="copy",
            record_completion=lambda stream: TransferTicket(ImmediateCompletion()),
            failure_drain=lambda: None,
        )
        with context.temporary_scope():
            tensor = torch.ones(4)
            reference = weakref.ref(tensor)
            context.retain(tensor)
            del tensor
        self.assertIsNotNone(reference())
        context.finish()
        context.release_completed()
        self.assertIsNone(reference())

    def test_failed_wave_keeps_tensors_until_failure_drain(self):
        with patch.dict("os.environ", {"MOE_ENABLE_TOPK_NEG_ONE": "1"}), patch.object(
            native, "_compute_scope", lambda *args: nullcontext()
        ):
            fixture = self.fixture()

            def fail(*args):
                raise RuntimeError("write failed after F")

            fixture.layout.write_wave = fail
            with self.assertRaisesRegex(RuntimeError, "write failed after F"):
                fixture.runtime.run_layer(3, fixture.input, fixture.routes)
            self.assertIsNotNone(fixture.raw_refs[-1]())
            self.assertIsNotNone(fixture.output_refs[-1]())
            self.assertFalse(fixture.context._temporary_resources)
            fixture.context.abort()
            self.assertIn("drain", fixture.waits)
            self.assertIsNone(fixture.raw_refs[-1]())
            self.assertIsNone(fixture.output_refs[-1]())


if __name__ == "__main__":
    unittest.main()
