"""CPU contracts for native Ascend bank binding and EP packing, not NPU proof."""

import importlib.util
import os
import pathlib
import random
import sys
import types
import unittest
from collections import namedtuple
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.offloading.moe.config import ExpertWave, Phase, fixed_waves
from sglang.srt.offloading.moe.npu_experts import (
    NPUDispatch,
    NPUDispatchLayout,
    NPUExpertBank,
    NPUExpertRunner,
    NPUOffloadDispatcher,
    build_npu_layer_programs,
)
from sglang.srt.offloading.transfer import Completion, TransferTicket
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class Fence(Completion):
    def __init__(self, calls, name):
        self.calls, self.name = calls, name

    def wait_on(self, stream):
        self.calls.append(("wait", self.name, stream))

    def done(self):
        raise AssertionError("EP/FFN must not poll on CPU")

    def synchronize(self):
        raise AssertionError("EP/FFN must not synchronize on CPU")


class Context:
    compute_stream = "compute"
    copy_stream = "copy"

    def __init__(self, phase=Phase.DECODE):
        self.phase, self.capturing = phase, phase is Phase.DECODE
        self.calls, self.retained = [], []

    def retain(self, resource):
        self.retained.append(resource)

    def record_compute_completion(self):
        name = "event" + str(len(self.calls))
        self.calls.append(("record", name))
        return TransferTicket(Fence(self.calls, name))


@dataclass
class QuantInfo:
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    w13_weight_scale: object = None
    w2_weight_scale: object = None


def bank(width, error, valid=None):
    weights = torch.arange(1, width + 1, dtype=torch.bfloat16).view(width, 1, 1)
    quant = QuantInfo(weights, weights.clone(), weights.clone(), weights.clone())
    return NPUExpertBank(
        quant, torch.ones(width, dtype=torch.bool) if valid is None else valid, error
    )


def dispatch(x, counts, context, *, scale=None, format="deepep_ll"):
    raw = SimpleNamespace(
        format=format,
        hidden_states=x,
        hidden_states_scale=scale,
        topk_ids=object(),
        topk_weights=object(),
        masked_m=torch.tensor(counts, dtype=torch.int64),
        num_recv_tokens_per_expert=list(counts),
        expert_tokens=torch.tensor(counts, dtype=torch.int64),
        group_list_type=1,
    )
    return NPUDispatch(raw, context.record_compute_completion(), x.shape[1], object())


class Core:
    """Reference expert multiplication, with intentionally undefined padding."""

    def __init__(self):
        self.calls = []

    def run(self, runner_input, quant, state):
        self.calls.append((runner_input, quant, state))
        output = torch.full_like(
            runner_input.hidden_states, float("nan"), dtype=torch.bfloat16
        )
        offset = 0
        for index, count in enumerate(runner_input.expert_tokens.tolist()):
            output[offset : offset + count] = (
                runner_input.hidden_states[offset : offset + count].to(torch.bfloat16)
                * quant.w13_weight[index, 0, 0]
            )
            offset += count
        return SimpleNamespace(hidden_states=output)


class LayoutTest(unittest.TestCase):
    def setup_layout(self, counts, rows=11, hidden=4, phase=Phase.DECODE):
        context = Context(phase)
        error = torch.zeros((), dtype=torch.int32)
        adapter = NPUDispatchLayout(len(counts), hidden, max(rows, 1), "cpu", error)
        x = (
            torch.arange(rows * hidden, dtype=torch.float32)
            .view(rows, hidden)
            .to(torch.bfloat16)
        )
        dispatched = dispatch(x, counts, context)
        layout = adapter.bind(dispatched, context)
        return adapter, layout, error, context, x

    def test_random_waves_reassemble_original_rows_without_weights(self):
        random.seed(1021)
        for _ in range(40):
            n = random.randint(1, 9)
            counts = [random.randint(0, 3) for _ in range(n)]
            rows = sum(counts) + 3
            adapter, layout, error, context, x = self.setup_layout(counts, rows)
            expected = torch.zeros_like(x)
            offset = 0
            for expert, count in enumerate(counts):
                expected[offset : offset + count] = x[offset : offset + count] * (
                    expert + 1
                )
                offset += count
            for wave in fixed_waves(n, random.randint(1, n)):
                local = adapter.select_wave(layout, wave, context)
                width = wave.local_end - wave.local_start
                bound = bank(width, error)
                bound.quant_info.w13_weight.add_(wave.local_start)
                runner = NPUExpertRunner(
                    None, _core=Core(), _input_type=SimpleNamespace
                )
                result = runner(1, local, SimpleNamespace(expert_bank=bound), context)
                result.read_done.completion.wait_on(context.compute_stream)
                adapter.write_wave(layout, wave, result.output, context)
            output = adapter.combine_input(layout, context).hidden_states
            torch.testing.assert_close(output, expected)

    def test_device_only_offsets_and_inverse_no_item_or_tolist(self):
        adapter, layout, _, context, _ = self.setup_layout([1, 0, 2])
        with (
            patch.object(torch.Tensor, "item", side_effect=AssertionError("host read")),
            patch.object(
                torch.Tensor, "tolist", side_effect=AssertionError("host read")
            ),
        ):
            for wave in fixed_waves(3, 2):
                local = adapter.select_wave(layout, wave, context)
                adapter.write_wave(
                    layout, wave, local.hidden_states.to(torch.bfloat16), context
                )
            adapter.combine_input(layout, context)

    def test_dynamic_demand_reuses_fixed_workspace(self):
        adapter, layout, error, context, x = self.setup_layout([2, 1, 0])
        addresses = [a.storage_key for a in adapter.allocations()]
        for counts in ([0, 0, 3], [0, 0, 0], [1, 1, 1]):
            layout = adapter.bind(dispatch(x, counts, context), context)
            for wave in fixed_waves(3, 2):
                local = adapter.select_wave(layout, wave, context)
                adapter.write_wave(layout, wave, local.hidden_states, context)
            output = adapter.combine_input(layout, context).hidden_states
            torch.testing.assert_close(output[: sum(counts)], x[: sum(counts)])
            self.assertTrue(
                torch.equal(
                    output[sum(counts) :], torch.zeros_like(output[sum(counts) :])
                )
            )
            self.assertEqual(addresses, [a.storage_key for a in adapter.allocations()])
            self.assertEqual(error.item(), 0)

    def test_mxfp8_and_flat_scale_bits_follow_identical_row_mapping(self):
        context = Context()
        error = torch.zeros((), dtype=torch.int32)
        adapter = NPUDispatchLayout(3, 64, 8, "cpu", error)
        x = torch.arange(8 * 64).reshape(8, 64).to(torch.float8_e4m3fn)
        scale = torch.arange(16, dtype=torch.uint8)
        layout = adapter.bind(dispatch(x, [2, 0, 3], context, scale=scale), context)
        wave = ExpertWave(1, 1, 3, 0)
        local = adapter.select_wave(layout, wave, context)
        self.assertEqual(local.hidden_states.dtype, x.dtype)
        torch.testing.assert_close(
            local.hidden_states.view(torch.uint8)[:3], x.view(torch.uint8)[2:5]
        )
        torch.testing.assert_close(
            local.hidden_states_scale[:3], scale.reshape(8, 2)[2:5]
        )
        self.assertEqual(torch.count_nonzero(local.hidden_states_scale[3:]).item(), 0)

    def test_invalid_counts_are_latched_and_never_index_outside_buffer(self):
        for counts in ([-1, 2], [100, 0]):
            adapter, layout, error, context, _ = self.setup_layout(counts)
            self.assertEqual(error.item(), 64)
            self.assertEqual(layout.counts.sum().item(), 0)
            for wave in fixed_waves(2, 1):
                local = adapter.select_wave(layout, wave, context)
                adapter.write_wave(layout, wave, local.hidden_states, context)
            self.assertEqual(
                torch.count_nonzero(
                    adapter.combine_input(layout, context).hidden_states
                ).item(),
                0,
            )

    def test_empty_receive_and_empty_wave(self):
        adapter, layout, error, context, _ = self.setup_layout([0, 0], rows=0)
        for wave in fixed_waves(2, 1):
            local = adapter.select_wave(layout, wave, context)
            runner = NPUExpertRunner(None, _core=Core(), _input_type=SimpleNamespace)
            result = runner(
                1, local, SimpleNamespace(expert_bank=bank(1, error)), context
            )
            adapter.write_wave(layout, wave, result.output, context)
        self.assertEqual(
            adapter.combine_input(layout, context).hidden_states.shape, (0, 4)
        )

    def test_normal_is_eager_only_and_tp_requires_count_convention(self):
        adapter, _, _, context, x = self.setup_layout([1, 1])
        with self.assertRaisesRegex(ValueError, "Decode requires"):
            adapter.bind(dispatch(x, [1, 1], context, format="deepep_normal"), context)
        context = Context(Phase.PREFILL)
        layout = adapter.bind(
            dispatch(x, [1, 1], context, format="deepep_normal"), context
        )
        torch.testing.assert_close(layout.counts, torch.tensor([1, 1]))
        raw = dispatch(x, [1, 1], context, format="ascend_tp")
        adapter.bind(raw, context)
        raw.raw.group_list_type = 0
        with self.assertRaisesRegex(ValueError, "count convention"):
            adapter.bind(raw, context)

    def test_workspace_and_wave_bounds_fail_before_native_ffn(self):
        adapter, layout, _, context, x = self.setup_layout([1, 1])
        with self.assertRaisesRegex(ValueError, "capacity"):
            adapter.bind(dispatch(torch.zeros(12, 4), [1, 1], context), context)
        wave = ExpertWave(0, 0, 1, 0)
        adapter.select_wave(layout, wave, context)
        with self.assertRaisesRegex(RuntimeError, "current wave"):
            adapter.select_wave(layout, wave, context)
        with self.assertRaisesRegex(RuntimeError, "unfinished"):
            adapter.combine_input(layout, context)
        with self.assertRaisesRegex(RuntimeError, "match"):
            adapter.write_wave(layout, ExpertWave(1, 1, 2, 0), x, context)


class RunnerTest(unittest.TestCase):
    def prepare(self, valid=(True, True), counts=(1, 2)):
        context, error = Context(), torch.zeros((), dtype=torch.int32)
        adapter = NPUDispatchLayout(2, 4, 8, "cpu", error)
        layout = adapter.bind(
            dispatch(torch.ones(8, 4, dtype=torch.bfloat16), counts, context), context
        )
        local = adapter.select_wave(layout, ExpertWave(0, 0, 2, 0), context)
        core = Core()
        runner = NPUExpertRunner(None, _core=core, _input_type=SimpleNamespace)
        bound = bank(3, error, torch.tensor([*valid, False]))
        return context, error, local, core, runner, bound

    def test_tail_quant_info_uses_fixed_native_views_and_padding_zero(self):
        context, error, local, core, runner, bound = self.prepare()
        result = runner(3, local, SimpleNamespace(expert_bank=bound), context)
        quant = core.calls[0][1]
        self.assertEqual(quant.w13_weight.shape[0], 2)
        self.assertEqual(
            quant.w13_weight.data_ptr(), bound.quant_info.w13_weight.data_ptr()
        )
        self.assertTrue(
            torch.equal(result.output[3:], torch.zeros_like(result.output[3:]))
        )
        self.assertFalse(torch.isnan(result.output).any())
        self.assertEqual(core.calls[0][0].group_list_type, 1)
        self.assertEqual(error.item(), 0)

    def test_invalid_required_lane_never_passes_count_to_native_gmm(self):
        context, error, local, core, runner, bound = self.prepare(valid=(True, False))
        result = runner(3, local, SimpleNamespace(expert_bank=bound), context)
        self.assertEqual(error.item(), 128)
        self.assertEqual(core.calls[0][0].expert_tokens.sum().item(), 0)
        self.assertEqual(torch.count_nonzero(result.output).item(), 0)

    def test_unused_invalid_lane_is_safe_and_existing_error_gates_all(self):
        context, error, local, core, runner, bound = self.prepare(
            valid=(True, False), counts=(1, 0)
        )
        runner(3, local, SimpleNamespace(expert_bank=bound), context)
        self.assertEqual(error.item(), 0)
        error.fill_(1)
        output = runner(3, local, SimpleNamespace(expert_bank=bound), context).output
        self.assertEqual(core.calls[-1][0].expert_tokens.sum().item(), 0)
        self.assertEqual(torch.count_nonzero(output).item(), 0)

    def test_quant_info_validation_and_canonical_error_identity(self):
        context, error, local, _, runner, bound = self.prepare()
        with self.assertRaisesRegex(ValueError, "canonical error_status"):
            runner(
                3,
                local,
                SimpleNamespace(expert_bank=bound, error_status=error.clone()),
                context,
            )
        with self.assertRaisesRegex(ValueError, "validity"):
            NPUExpertBank(bound.quant_info, torch.ones(3), error)
        with self.assertRaisesRegex(ValueError, "smaller"):
            bank(1, error, torch.ones(2, dtype=torch.bool)).for_width(2)

    def test_partial_native_failure_keeps_bank_and_input_alive(self):
        context, _, local, core, runner, bound = self.prepare()

        def fail(*args):
            self.assertTrue(
                any(
                    isinstance(item, tuple) and bound in item
                    for item in context.retained
                    if isinstance(item, tuple) and any(value is bound for value in item)
                )
            )
            raise RuntimeError("native partial submit")

        core.run = fail
        with self.assertRaisesRegex(RuntimeError, "partial submit"):
            runner(3, local, SimpleNamespace(expert_bank=bound), context)
        self.assertTrue(
            any(
                isinstance(item, tuple) and any(value is bound for value in item)
                for item in context.retained
            )
        )

    def test_all_registered_reader_streams_join_before_bank_retirement(self):
        context, _, local, _, runner, bound = self.prepare()
        runner.reader_streams = ("aux0", "aux1")
        events = []

        class Event:
            def __init__(self):
                events.append(self)

            def record(self, stream):
                self.stream = stream

        class Stream:
            def wait_event(self, event):
                context.calls.append(("aux_wait", event.stream, "copy"))

        context.copy_stream = Stream()

        with patch.object(torch, "npu", SimpleNamespace(Event=Event), create=True):
            result = runner(3, local, SimpleNamespace(expert_bank=bound), context)
            result.read_done.completion.wait_on(context.copy_stream)
        self.assertEqual([event.stream for event in events], ["aux0", "aux1"])
        self.assertIn(("aux_wait", "aux0", "copy"), context.calls)
        self.assertIn(("aux_wait", "aux1", "copy"), context.calls)

    def test_default_constructor_imports_real_native_core_and_preserves_config(self):
        native = types.ModuleType("sglang.srt.layers.moe.moe_runner.ascend")
        configs, calls = [], []

        class NativeCore(Core):
            def __init__(self, config):
                super().__init__()
                configs.append(config)

            def run(self, runner_input, quant_info, running_state):
                calls.append((runner_input, quant_info, running_state))
                return super().run(runner_input, quant_info, running_state)

        native.AscendRunnerCore, native.AscendRunnerInput = NativeCore, SimpleNamespace
        experts = SimpleNamespace(moe_runner_config=SimpleNamespace(swiglu_limit=7.0))
        with patch.dict(sys.modules, {native.__name__: native}):
            runner = NPUExpertRunner(experts)
        context, _, local, _, _, bound = self.prepare()
        runner(3, local, SimpleNamespace(expert_bank=bound), context)
        self.assertIs(configs[0], experts.moe_runner_config)
        self.assertEqual(configs[0].swiglu_limit, 7.0)
        self.assertIs(calls[0][0].hidden_states, local.hidden_states)
        self.assertEqual(calls[0][1].w13_weight.shape[0], 2)
        self.assertEqual(calls[0][0].group_list_type, 1)
        self.assertEqual(calls[0][2], {})

    def test_actual_ascend_core_calls_native_gmms_and_v4_clamp_with_bank_views(self):
        # Load the real runner source, replacing only unavailable NPU operators
        # and serving registries. This exercises its real config/activation/GMM
        # flow, in addition to the numerical packing tests above.
        import sglang

        paths = [
            pathlib.Path(root) / "srt/layers/moe/moe_runner/ascend.py"
            for root in sglang.__path__
        ]
        path = next(path for path in paths if path.is_file())
        calls = []
        scales = object()

        def module(name, **attributes):
            result = types.ModuleType(name)
            result.__dict__.update(attributes)
            return result

        class Activation:
            def __init__(self, limit):
                calls.append(("clamp", limit))

            def _apply_activation(self, x, *, group_list, group_list_type):
                calls.append(("activation", group_list.clone(), group_list_type))
                return x * 2, scales

        activation_names = (
            "AllGatherActivationWrapper",
            "NPUGeluAndMul",
            "NPUSitu",
            "NPUSituMXFP8Quant",
            "NPUSwiglu",
            "NPUSwigluDeepEPKernel",
            "NPUSwigluMxfp8Quant",
            "NPUSwigluOAI",
            "NPUSwigluQuant",
            "NPUSwigluStepAndMul",
        )
        activation = module(
            "sglang.srt.hardware_backend.npu.moe.activation",
            **{
                name: (
                    Activation if name == "NPUSwigluMxfp8Quant" else type(name, (), {})
                )
                for name in activation_names
            },
        )

        class W4A8:
            use_fused_gmm1 = True

            def apply(self, quant, x, counts, **kwargs):
                calls.append((kwargs["weight_prefix"], quant, counts.clone(), kwargs))
                return x + (1 if kwargs["weight_prefix"] == "w13" else 3)

            def apply_fused_gmm1_swiglu(self, *args, **kwargs):
                raise AssertionError(
                    "V4 clamp must disable the fused plain-SiLU kernel"
                )

        methods = module(
            "sglang.srt.hardware_backend.npu.quantization.moe_methods",
            NPUW4A8MXFP4MoEMethod=W4A8,
            NPUMXFP8MoEMethod=type("MXFP8", (), {}),
            NPUW4A8Int8MoEMethod=type("W4A8Int8", (), {}),
            NPUW8A8Int8MoEMethod=type("W8A8Int8", (), {}),
        )

        class CoreBase:
            def __init__(self, config):
                self.config = config

        base = module(
            "sglang.srt.layers.moe.moe_runner.base",
            MoeQuantInfo=object,
            MoeRunnerConfig=SimpleNamespace,
            MoeRunnerCore=CoreBase,
            RunnerInput=object,
            RunnerOutput=object,
            register_pre_permute=lambda *args: lambda function: function,
            register_post_permute=lambda *args: lambda function: function,
        )
        utils = module(
            "sglang.srt.layers.moe.utils",
            MoeRunnerBackend=SimpleNamespace(ASCEND="ascend"),
            get_moe_a2a_backend=lambda: SimpleNamespace(is_deepep=lambda: True),
        )
        environ = module("sglang.srt.environ", envs=SimpleNamespace())
        target = "sglang.srt.layers.moe.moe_runner.ascend"
        spec = importlib.util.spec_from_file_location(target, path)
        native = importlib.util.module_from_spec(spec)
        modules = {
            value.__name__: value
            for value in (activation, methods, base, utils, environ)
        }
        modules[target] = native
        experts = SimpleNamespace(
            moe_runner_config=SimpleNamespace(
                layer=SimpleNamespace(w13_kernel=W4A8(), w2_kernel=W4A8()),
                activation="silu",
                swiglu_limit=7.0,
                num_local_experts=256,
            )
        )
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(native)
            runner = NPUExpertRunner(experts)
            context, _, local, _, _, bound = self.prepare()
            result = runner(3, local, SimpleNamespace(expert_bank=bound), context)
        self.assertEqual(
            [call[0] for call in calls], ["clamp", "w13", "activation", "w2"]
        )
        self.assertEqual(calls[0][1], 7.0)
        self.assertEqual(calls[1][1].w13_weight.shape[0], 2)
        self.assertEqual(
            calls[1][1].w13_weight.data_ptr(), bound.quant_info.w13_weight.data_ptr()
        )
        torch.testing.assert_close(calls[1][2], torch.tensor([1, 2]))
        self.assertEqual(calls[1][3]["group_list_type"], 1)
        self.assertIs(calls[3][3]["pertoken_scale"], scales)
        self.assertEqual(calls[3][3]["output_dtype"], torch.bfloat16)
        torch.testing.assert_close(
            result.output[:3], torch.full((3, 4), 7, dtype=torch.bfloat16)
        )
        self.assertEqual(torch.count_nonzero(result.output[3:]).item(), 0)


class DispatcherTest(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(
            os.environ,
            {
                "MOE_ENABLE_TOPK_NEG_ONE": "1",
                "DEEP_USE_MODE": "default",
                "MOE_EXPERT_TOKEN_NUMS_TYPE": "1",
            },
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.error = torch.zeros((), dtype=torch.int32)

    def test_real_dispatch_wait_frontier_handle_and_combine_ids(self):
        context = Context(Phase.PREFILL)
        x = torch.ones(4, 8, dtype=torch.bfloat16)
        original = torch.ones(4, 6, dtype=torch.bfloat16)
        raw = dispatch(x, [1, 1], context).raw
        handle = object()

        class NativeDispatcher:
            def dispatch(self, **kwargs):
                self.handle = handle
                context.calls.append(("native_receive_wait",))
                return raw

            def combine(self, combine_input):
                self.input = combine_input
                self.handle = None
                return combine_input.hidden_states

            def _get_impl(self):
                return self

        native = NativeDispatcher()
        experts = SimpleNamespace(dispatcher=native, reduce_results=False)
        wrapper = NPUOffloadDispatcher(experts, error_status=self.error)
        result = wrapper.dispatch(original, object(), context)
        self.assertIs(result.handle, handle)
        self.assertEqual(context.calls[-2][0], "native_receive_wait")
        self.assertEqual(context.calls[-1][0], "record")
        module = types.ModuleType("sglang.srt.layers.moe.token_dispatcher.deepep")

        def constructor(hidden, ids, weights):
            return SimpleNamespace(
                hidden_states=hidden, topk_ids=ids, topk_weights=weights
            )

        module.DeepEPLLCombineInput = module.DeepEPNormalCombineInput = constructor
        adapter = NPUDispatchLayout(2, 8, 4, "cpu", torch.zeros((), dtype=torch.int32))
        layout = adapter.bind(result, context)
        layout.output.fill_(3)
        with patch.dict(sys.modules, {module.__name__: module}):
            combined = wrapper.combine(adapter.combine_input(layout, context), context)
            output = wrapper.finish(combined, context)
        self.assertIs(native.input.topk_ids, raw.topk_ids)
        self.assertIs(native.input.topk_weights, raw.topk_weights)
        self.assertEqual(output.shape, (4, 6))
        self.assertTrue((output == 3).all())
        self.assertIsNone(native.handle)
        self.assertTrue(any(value is handle for value in context.retained))

    def test_unsupported_overlap_and_environment_rejected_before_collective(self):
        experts = SimpleNamespace(
            dispatcher=SimpleNamespace(_inners=[object(), object()])
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            NPUOffloadDispatcher(experts, error_status=self.error)
        with patch.dict(os.environ, {"MOE_EXPERT_TOKEN_NUMS_TYPE": "0"}):
            with self.assertRaisesRegex(ValueError, "type 1"):
                NPUOffloadDispatcher(
                    SimpleNamespace(dispatcher=object()), error_status=self.error
                )
        with patch.dict(os.environ, {"DEEP_USE_MODE": "alltoall"}):
            with self.assertRaisesRegex(ValueError, "default DeepEP"):
                NPUOffloadDispatcher(
                    SimpleNamespace(dispatcher=object()), error_status=self.error
                )
        with patch.dict(os.environ, {"MOE_ENABLE_TOPK_NEG_ONE": "0"}):
            with self.assertRaisesRegex(ValueError, "NEG_ONE=1"):
                NPUOffloadDispatcher(
                    SimpleNamespace(dispatcher=object()), error_status=self.error
                )

    def test_builder_binds_decode_request_to_full_demand_and_unique_budget_names(self):
        error = torch.zeros((), dtype=torch.int32)
        experts = SimpleNamespace(
            dispatcher=object(),
            num_local_experts=3,
            layer_id=5,
            moe_runner_config=SimpleNamespace(hidden_size=4),
        )
        access = object()
        pipeline = SimpleNamespace(access=access)
        native = types.ModuleType("sglang.srt.layers.moe.token_dispatcher.deepep")
        native.DeepEPDispatcher = object
        native_patcher = patch.dict(sys.modules, {native.__name__: native})
        native_patcher.start()
        self.addCleanup(native_patcher.stop)
        with patch(
            "sglang.srt.offloading.moe.npu_experts.NPUExpertRunner", return_value=Core()
        ):
            programs = build_npu_layer_programs(
                experts,
                access=access,
                residency=SimpleNamespace(error_status=error),
                decode_waves=fixed_waves(3, 2),
                pipeline=pipeline,
                max_dispatch_rows=10,
            )
        counts = torch.tensor([1, 2, 3])
        wave = fixed_waves(3, 2)[0]
        request = programs.decode.local_program.make_request(
            5, SimpleNamespace(local_demand=counts), wave, Context()
        )
        self.assertIs(request.local_demand, counts)
        self.assertIs(programs.prefill.pipeline, pipeline)
        allocations = programs.allocations()
        self.assertTrue(
            all(item.name.startswith("npu_dispatch.5.") for item in allocations)
        )
        self.assertEqual(
            len({item.storage_key for item in allocations}), len(allocations)
        )
        with patch(
            "sglang.srt.offloading.moe.npu_experts.NPUExpertRunner", return_value=Core()
        ):
            shared = build_npu_layer_programs(
                experts,
                access=access,
                residency=SimpleNamespace(error_status=error),
                decode_waves=fixed_waves(3, 2),
                pipeline=pipeline,
                max_dispatch_rows=10,
                shared_layout=programs.layout,
            )
            self.assertIs(shared.layout, programs.layout)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                build_npu_layer_programs(
                    experts,
                    access=access,
                    residency=SimpleNamespace(error_status=error.clone()),
                    decode_waves=fixed_waves(3, 2),
                    pipeline=pipeline,
                    max_dispatch_rows=10,
                    shared_layout=programs.layout,
                )

    def test_device_padding_mask_preserves_input_and_excludes_all_invalid_rows(self):
        topk_type = namedtuple("TopKOutput", "topk_weights topk_ids router_logits")
        ids = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=torch.int32)
        weights = torch.full((4, 2), 0.5)
        logits = object()
        routing = topk_type(weights, ids, logits)
        context = Context()
        wrapper = NPUOffloadDispatcher(
            SimpleNamespace(dispatcher=object()), error_status=self.error
        )
        x = torch.zeros(4, 8, dtype=torch.bfloat16)
        for real_rows in (2, 0, 4):
            context.num_token_non_padded = torch.tensor([real_rows], dtype=torch.int32)
            with (
                patch.object(
                    torch.Tensor, "item", side_effect=AssertionError("host read")
                ),
                patch.object(
                    torch.Tensor, "tolist", side_effect=AssertionError("host read")
                ),
            ):
                masked, count = wrapper._mask_routing(x, routing, context)
            torch.testing.assert_close(masked.topk_ids[:real_rows], ids[:real_rows])
            self.assertTrue((masked.topk_ids[real_rows:] == -1).all())
            self.assertTrue((masked.topk_weights[real_rows:] == 0).all())
            self.assertIs(masked.router_logits, logits)
            self.assertEqual(count.shape, torch.Size([]))
        self.assertTrue((routing.topk_ids >= 0).all())
        self.assertTrue((routing.topk_weights == 0.5).all())
        for bad in (-1, 5):
            self.error.zero_()
            context.num_token_non_padded = torch.tensor(bad, dtype=torch.int32)
            masked, _ = wrapper._mask_routing(x, routing, context)
            self.assertEqual(self.error.item(), 256)
            self.assertTrue((masked.topk_ids == -1).all())
            self.assertTrue((masked.topk_weights == 0).all())

    def test_missing_or_invalid_padding_boundary_fails_before_dispatch(self):
        wrapper = NPUOffloadDispatcher(
            SimpleNamespace(dispatcher=object()), error_status=self.error
        )
        context = Context()
        x = torch.zeros(4, 8, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "requires device"):
            wrapper.dispatch(x, object(), context)
        for invalid in (4, torch.tensor(4.0), torch.ones(2, dtype=torch.int32)):
            context.num_token_non_padded = invalid
            with self.assertRaisesRegex(ValueError, "integer scalar"):
                wrapper.dispatch(x, object(), context)


if __name__ == "__main__":
    unittest.main()
