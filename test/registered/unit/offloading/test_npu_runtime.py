"""Lifecycle tests use CPU tensors and explicit event/graph simulators."""

import json
import os
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from sglang.srt.offloading.moe.config import (
    ExpertCatalog,
    ExpertLayerSpec,
    MoEOffloadConfig,
)
from sglang.srt.offloading.moe.npu.config import (
    NpuOffloadSettings,
    validate_native_environment,
)
from sglang.srt.offloading.moe.npu.runtime import LayerPrograms, NpuBoundMoERuntime
from sglang.srt.offloading.moe.owner import MoEOffloadOwner, OwnerState
from sglang.srt.offloading.moe.policy import TensorLFUPolicy
from sglang.srt.offloading.moe.resources import Allocation, ResourceKind
from sglang.srt.offloading.moe.runtime import DecodeStateSnapshot
from sglang.srt.offloading.moe.state import ResidencyState
from sglang.srt.offloading.transfer import ImmediateCompletion, TransferTicket


class Device:
    def __init__(self):
        self.compute, self.copy_stream = object(), object()
        self.events, self.drains = [], 0
        self.memory_peak = 100

    def current_stream(self):
        return self.compute

    def stream_scope(self, stream):
        return nullcontext()

    def record(self, stream):
        ticket = TransferTicket(ImmediateCompletion(), (object(), stream))
        self.events.append(ticket)
        return ticket

    def synchronize(self):
        self.drains += 1

    def allocated(self):
        return 100

    def reset_peak(self):
        pass

    def peak(self):
        return self.memory_peak


class Ops:
    def __init__(self, state):
        self.state = state
        self.error_status = torch.zeros((), dtype=torch.int32)
        self.begins = 0

    def begin_step(self, context):
        self.begins += 1
        self.state.step_access_count.zero_()


class Program:
    def __init__(self):
        self.starts, self.runs = [], 0

    def layer_start(self, layer_id, context):
        self.starts.append(layer_id)

    def run(self, layer, hidden, routing, context):
        self.runs += 1
        return hidden + routing


class Expert(torch.nn.Module):
    pass


class MLP(torch.nn.Module):
    is_deepseek_v4 = True
    layer_id = 3

    def __init__(self):
        super().__init__()
        self.experts = Expert()

    def forward(self, hidden, routing):
        return self.experts._moe_offload_runtime.run_layer(3, hidden, routing)


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MLP()

    def forward(self, hidden, routing):
        return self.forward_hc_pre_from_prev(hidden, routing)

    def forward_hc_pre_from_prev(self, hidden, routing):
        return self.mlp(hidden, routing)


def batch(phase):
    return SimpleNamespace(
        moe_num_token_non_padded=lambda: torch.tensor(1, dtype=torch.int32),
        forward_mode=SimpleNamespace(
            is_decode_or_idle=lambda: phase == "decode",
            is_extend_without_speculative=lambda: phase == "prefill",
        ),
    )


def setup_runtime():
    state = ResidencyState.allocate(2, 1, 1, "cpu")
    device, ops = Device(), Ops(state)
    policy = TensorLFUPolicy()
    policy_state = policy.allocate_state(2, "cpu")
    snapshot = DecodeStateSnapshot(
        state, policy, policy_state, error_status=ops.error_status
    )
    prefill, decode = Program(), Program()
    tails, reloads, releases = [], [], []

    def tail(ctx):
        tails.append(bool(state.update_enabled))
        return ctx.record_compute_completion()

    def reload(saved, ctx):
        reloads.append(saved.slot_to_key.clone())
        return ctx.record_compute_completion()

    resources = tuple(
        Allocation(name, kind, size, name, object())
        for name, kind, size in (
            ("host", ResourceKind.HOST_WEIGHTS, 2),
            ("hot", ResourceKind.HOT_WEIGHTS, 1),
            ("execution", ResourceKind.EXECUTION_WEIGHTS, 1),
        )
    )
    runtime = NpuBoundMoERuntime(
        device_runtime=device,
        residency=state,
        residency_ops=ops,
        cache_updates=SimpleNamespace(emit_step_end=tail),
        snapshot=snapshot,
        reload_weights=reload,
        layer_programs={3: LayerPrograms(prefill, decode)},
        resources=resources,
        expert_nbytes=1,
        storage=(),
        release=lambda: releases.append(True),
    )
    owner = MoEOffloadOwner(
        MoEOffloadConfig(0.5, 1, 1, 0, workspace_limit_bytes=20),
        ExpertCatalog((ExpertLayerSpec(3, 2, 1, 0),)),
        runtime,
    )
    runner = SimpleNamespace(
        model=Decoder(), capture_head_hooks=[], capture_tail_hooks=[]
    )
    runtime.install_model_hooks(SimpleNamespace(owner=owner), runner)
    return SimpleNamespace(**locals())


class RuntimeTests(unittest.TestCase):
    def test_warmup_capture_updates_disabled_and_replay_uses_fresh_events(self):
        s = setup_runtime()
        with s.runtime.capture_scope(s.owner, None, s.device.compute):
            for _ in range(2):
                s.runner.capture_head_hooks[0](None, batch("decode"), 1)
                self.assertEqual(s.runner.model(3, 4), 7)
                s.runner.capture_tail_hooks[0](None, None, batch("decode"), 1)
        self.assertEqual(s.tails, [False, False])
        self.assertTrue(s.state.update_enabled)
        captured_event_ids = {id(x) for x in s.device.events}
        for _ in range(3):
            with s.runtime.forward_scope(s.owner, batch("decode")):
                pass  # full graph replay: no Python model or policy callbacks
        self.assertEqual(s.tails, [False, False])
        self.assertEqual(s.ops.begins, 2)
        last = s.owner._pending[-1].output_ready
        self.assertNotIn(id(last), captured_event_ids)
        s.owner.close()
        self.assertEqual(s.releases, [True])
        self.assertFalse(hasattr(s.runner.model.mlp.experts, "_moe_offload_runtime"))
        self.assertEqual(s.runner.capture_head_hooks, [])

    def test_eager_prefill_saves_then_restores_exact_slot_and_policy(self):
        s = setup_runtime()
        s.state.slot_to_key.fill_(1)
        s.state.key_to_slot[1] = 0
        s.state.slot_valid.fill_(True)
        s.policy_state.frequency.copy_(torch.tensor([8, 3]))
        with s.runtime.forward_scope(s.owner, batch("prefill")):
            self.assertEqual(s.runner.model(2, 5), 7)
        self.assertEqual(s.prefill.starts, [3])
        self.assertEqual(s.tails, [])
        s.state.slot_to_key.fill_(-1)
        s.policy_state.frequency.zero_()
        with s.runtime.forward_scope(s.owner, batch("decode")):
            pass
        self.assertEqual(s.reloads[0].tolist(), [1])
        self.assertEqual(s.state.slot_to_key.tolist(), [1])
        self.assertEqual(s.policy_state.frequency.tolist(), [8, 3])
        s.owner.close()

    def test_eager_decode_fails_and_pins_failed_context_until_teardown(self):
        s = setup_runtime()
        with self.assertRaisesRegex(RuntimeError, "captured full graph"):
            with s.runtime.forward_scope(s.owner, batch("decode")):
                s.runner.model(2, 4)
        self.assertEqual(s.owner.state, OwnerState.FAILED)
        self.assertEqual(len(s.runtime._failed_contexts), 1)
        s.owner.close()
        self.assertEqual(s.runtime._failed_contexts, [])

    def test_native_direct_decoder_method_starts_prefill_before_attention(self):
        s = setup_runtime()
        self.assertIn("forward_hc_pre_from_prev", s.runner.model.__dict__)
        with s.runtime.forward_scope(s.owner, batch("prefill")):
            self.assertEqual(s.runner.model.forward_hc_pre_from_prev(4, 5), 9)
        self.assertEqual(s.prefill.starts, [3])
        s.owner.close()
        self.assertNotIn("forward_hc_pre_from_prev", s.runner.model.__dict__)

    def test_prefill_reads_padding_bound_after_eager_preprocessing(self):
        s = setup_runtime()
        forward_batch = batch("prefill")
        value = [None]
        forward_batch.moe_num_token_non_padded = lambda: value[0]
        with s.runtime.forward_scope(s.owner, forward_batch) as context:
            value[0] = torch.tensor(5, dtype=torch.int32)
            s.runner.model.forward_hc_pre_from_prev(1, 2)
            self.assertIs(context.num_token_non_padded, value[0])
        s.owner.close()

    def test_decode_failure_does_not_call_eager_pipeline_abort(self):
        s = setup_runtime()

        def reject_abort(context):
            raise AssertionError("Decode must never enter eager pipeline.abort")

        s.runtime.pipelines = (SimpleNamespace(abort=reject_abort),)
        with self.assertRaisesRegex(RuntimeError, "failure"):
            with s.runtime.forward_scope(s.owner, batch("decode")):
                raise RuntimeError("failure")
        s.owner.close()
        self.assertGreater(s.device.drains, 0)

    def test_failed_abort_still_stops_other_workers_and_drains_before_retry(self):
        s = setup_runtime()
        calls = []
        fail = [True]

        def abort_first(context):
            calls.append("first")
            if fail[0]:
                raise RuntimeError("copy event failed")
            return TransferTicket(ImmediateCompletion())

        def abort_second(context):
            calls.append("second")
            return TransferTicket(ImmediateCompletion())

        s.runtime.pipelines = (
            SimpleNamespace(abort=abort_first),
            SimpleNamespace(abort=abort_second),
        )
        with self.assertRaisesRegex(RuntimeError, "copy event failed"):
            with s.runtime.forward_scope(s.owner, batch("prefill")):
                raise RuntimeError("partial submission")
        self.assertEqual(calls, ["first", "second"])
        self.assertGreater(s.device.drains, 0)
        self.assertEqual(s.releases, [])
        self.assertEqual(s.owner.state, OwnerState.FAILED)
        self.assertEqual(len(s.runtime._failed_contexts), 1)
        fail[0] = False
        s.owner.close()
        self.assertEqual(s.releases, [True])

    def test_native_restore_fault_does_not_publish_saved_validity(self):
        s = setup_runtime()
        with s.runtime.forward_scope(s.owner, batch("prefill")):
            s.runner.model(0, 0)
        s.state.key_to_slot.fill_(-1)
        s.ops.error_status.fill_(1)
        with self.assertRaisesRegex(RuntimeError, "Device residency"):
            with s.runtime.forward_scope(s.owner, batch("decode")):
                pass
        self.assertEqual(s.owner.state, OwnerState.FAILED)
        s.owner.close()

    def test_graph_workspace_budget_rejects_measured_overflow(self):
        s = setup_runtime()
        s.device.memory_peak = 121
        with self.assertRaisesRegex(ValueError, "workspace budget exceeded"):
            with s.runtime.capture_scope(s.owner, None, s.device.compute):
                pass
        s.owner.close()

    def test_graph_lease_prevents_resource_release(self):
        s = setup_runtime()
        lease = s.owner.acquire_graph("one")
        with self.assertRaisesRegex(RuntimeError, "Destroy graph"):
            s.owner.close()
        self.assertEqual(s.releases, [])
        lease.close()
        s.owner.close()
        s.owner.close()
        self.assertEqual(s.releases, [True])

    def test_unsupported_mode_rejected_before_owner_begin(self):
        s = setup_runtime()
        with self.assertRaises(ValueError):
            with s.runtime.forward_scope(s.owner, batch("mixed-speculative")):
                pass
        self.assertIsNone(s.owner._active)
        s.owner.close()


class ConfigurationTests(unittest.TestCase):
    def test_native_ep_environment_rejected_before_weight_loading(self):
        with mock.patch.dict(os.environ, {"MOE_ENABLE_TOPK_NEG_ONE": "1"}, clear=True):
            validate_native_environment()
            for key, value in (
                ("MOE_ENABLE_TOPK_NEG_ONE", "0"),
                ("DEEP_USE_MODE", "ops"),
                ("DEEP_USE_MODE", "alltoall"),
                ("MOE_EXPERT_TOKEN_NUMS_TYPE", "0"),
                ("MOE_EXPERT_TOKEN_NUMS_TYPE", "invalid"),
                ("SGLANG_DSV4_FP4_DEQUANT", "1"),
                ("SGLANG_DSV4_FP4_DEQUANT", "TRUE"),
            ):
                with self.subTest(key=key, value=value):
                    with mock.patch.dict(os.environ, {key: value}):
                        with self.assertRaisesRegex(ValueError, key):
                            validate_native_environment()

    def settings(self, **kwargs):
        values = dict(
            cache_ratio=0.25,
            execution_slots=16,
            host_pool_gib=4,
            workspace_limit_bytes=1024,
            metadata_limit_bytes=1024,
            max_dispatch_rows=32,
        )
        values.update(kwargs)
        return NpuOffloadSettings(**values)

    def test_strict_json_and_topology_budget(self):
        values = self.settings().to_dict()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "offload.json"
            path.write_text(json.dumps(values))
            cfg = NpuOffloadSettings.from_file(path).owner_config(
                ep_size=8, ep_rank=3, tp_size=8, moe_tp_size=1
            )
            self.assertEqual(cfg.host_allocation_limit_bytes, 4 * 1024**3)
            values["typo"] = 1
            path.write_text(json.dumps(values))
            with self.assertRaisesRegex(ValueError, "Unknown"):
                NpuOffloadSettings.from_file(path)
            path.write_text('{"version":1,"version":1}')
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                NpuOffloadSettings.from_file(path)

    def test_invalid_sizes_and_nan_do_not_silently_coerce(self):
        for change in (
            dict(cache_ratio=float("nan")),
            dict(execution_slots=True),
            dict(host_pool_gib=0),
            dict(prefill_microbatch_size=9),
            dict(version=2),
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.settings(**change)


if __name__ == "__main__":
    unittest.main()
