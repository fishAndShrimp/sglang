"""Whole-layer bank reuse and cache recovery contracts; no NPU kernels."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.offloading.moe.npu.prefill import (
    AscendPrefillWeights,
    LayerPrefetchPipeline,
    whole_layer_bank_layout,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Command:
    def __init__(self, dependencies, action):
        self.dependencies, self.action = dependencies, action
        self.done = self.running = False

    def run(self):
        if self.done:
            return
        if self.running:
            raise AssertionError("Cyclic stream dependency")
        self.running = True
        for dependency in self.dependencies:
            dependency.run()
        self.action()
        self.done, self.running = True, False


class _Stream:
    def __init__(self):
        self.tail = None
        self.waits = []

    def enqueue(self, action, extra=()):
        dependencies = tuple(extra) + (() if self.tail is None else (self.tail,))
        self.tail = _Command(dependencies, action)

    def wait_stream(self, other):
        if other.tail is not None:
            self.enqueue(lambda: None, (other.tail,))

    def drain(self):
        if self.tail is not None:
            self.tail.run()


class _Completion:
    def __init__(self, command):
        self.command = command

    def wait_on(self, stream):
        stream.waits.append(self.command)
        if self.command is not None:
            stream.enqueue(lambda: None, (self.command,))


class _Context:
    def __init__(self):
        self.invocation_id = object()
        self.compute_stream = _Stream()
        self.resources = []

    def retain(self, value):
        self.resources.append(value)

    def record_compute_completion(self):
        return SimpleNamespace(completion=_Completion(self.compute_stream.tail))


class TestLayerPrefetchPipeline(CustomTestCase):
    def test_capacity_never_silently_shrinks_hot_or_overlaps_banks(self):
        self.assertEqual(
            whole_layer_bank_layout(4, 6, 4), (("execution", 0, 4), ("hot", 0, 4))
        )
        self.assertEqual(
            whole_layer_bank_layout(4, 2, 8), (("execution", 0, 4), ("execution", 4, 4))
        )
        for capacity in ((4, 2, 4), (4, 6, 3), (0, 6, 4)):
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                whole_layer_bank_layout(*capacity)

    def test_delayed_copy_and_reader_preserve_complete_layer_results(self):
        """Draining copies first must not overwrite a still-unread bank."""
        for count in (1, 2, 3, 6):
            with self.subTest(layers=count):
                context, copy = _Context(), _Stream()
                banks, output, submitted = [None, None], [], []
                layers = tuple(range(3, 3 + count))

                def load(layer, bank, ctx):
                    submitted.append(layer)
                    # Both projections and their scales share the same lifetime.
                    copy.enqueue(
                        lambda: banks.__setitem__(bank, (layer, layer + 1, 0.5, 2.0))
                    )
                    return SimpleNamespace(completion=_Completion(copy.tail))

                pipeline = LayerPrefetchPipeline(layers, load=load, copy_stream=copy)
                for index, layer in enumerate(layers):
                    pipeline.start(layer, context)
                    if index == 0:
                        self.assertEqual(submitted, list(layers[:2]))
                    bank = pipeline.acquire(layer, context)

                    def read(layer=layer, bank=bank):
                        w13, w2, s13, s2 = banks[bank]
                        output.append((w13 * s13) * (w2 * s2))
                        self.assertEqual((w13, w2), (layer, layer + 1))

                    context.compute_stream.enqueue(read)
                    reader = context.record_compute_completion()
                    pipeline.retire(layer, reader, context)
                pipeline.finish_forward(context)
                # Forces bank-reuse waits to protect pending compute readers.
                copy.drain()
                context.compute_stream.drain()
                self.assertEqual(output, [layer * (layer + 1) for layer in layers])
                self.assertEqual(submitted, list(layers))
                self.assertEqual(len(copy.waits), 1 + max(0, count - 2))
                self.assertEqual(pipeline.ready, {})

    def test_partial_forward_and_wrong_layer_are_rejected(self):
        context, copy = _Context(), _Stream()
        pipeline = LayerPrefetchPipeline(
            (3, 4),
            load=lambda *args: SimpleNamespace(completion=_Completion(None)),
            copy_stream=copy,
        )
        with self.assertRaisesRegex(RuntimeError, "first offloaded"):
            pipeline.start(4, context)
        pipeline.start(3, context)
        with self.assertRaisesRegex(RuntimeError, "already consumed"):
            pipeline.start(3, context)
        with self.assertRaisesRegex(RuntimeError, "model order"):
            pipeline.acquire(4, context)
        with self.assertRaisesRegex(RuntimeError, "every offloaded"):
            pipeline.finish_forward(context)
        with self.assertRaisesRegex(RuntimeError, "model order"):
            pipeline.acquire(3, _Context())

    def test_next_forward_cannot_reuse_previous_ready_event(self):
        copy, submitted = _Stream(), []

        def load(layer, bank, context):
            submitted.append(context.invocation_id)
            return SimpleNamespace(completion=_Completion(None))

        pipeline = LayerPrefetchPipeline((3,), load=load, copy_stream=copy)
        for _ in range(2):
            context = _Context()
            pipeline.start(3, context)
            pipeline.acquire(3, context)
            pipeline.retire(3, context.record_compute_completion(), context)
            pipeline.finish_forward(context)
        self.assertEqual(len(submitted), 2)
        self.assertIsNot(submitted[0], submitted[1])


class TestAscendPrefillWeights(CustomTestCase):
    def test_cumulative_counts_keep_zero_groups_and_cache_usage_in_layer_order(self):
        weights = object.__new__(AscendPrefillWeights)
        context = _Context()
        usage = torch.zeros(8, dtype=torch.int64)
        weights.owner = SimpleNamespace(
            local_experts=4,
            error=torch.zeros((), dtype=torch.int32),
            state=SimpleNamespace(
                step_access_count=usage, update_enabled=torch.tensor(True)
            ),
            catalog=SimpleNamespace(key=lambda layer, expert: 4 + expert),
            context=context,
        )
        weights.quant_info = (object(), object())
        weights.pipeline = SimpleNamespace(acquire=lambda layer, ctx: 1)
        inputs = SimpleNamespace(
            expert_tokens=torch.tensor([0, 3, 3, 5]), group_list_type=0
        )
        quant = weights.weights(4, inputs)
        self.assertIs(quant, weights.quant_info[1])
        self.assertEqual(inputs.expert_tokens.tolist(), [0, 3, 0, 2])
        self.assertEqual(inputs.group_list_type, 1)
        self.assertEqual(usage.tolist(), [0, 0, 0, 0, 0, 3, 0, 2])
        weights.owner.state.update_enabled.fill_(False)
        weights.weights(4, inputs)
        self.assertEqual(usage.tolist(), [0, 0, 0, 0, 0, 3, 0, 2])
        weights.owner.error.fill_(1)
        weights.weights(4, inputs)
        self.assertEqual(inputs.expert_tokens.tolist(), [0, 0, 0, 0])

    def test_borrowed_hot_restores_current_slot_owners_after_partial_submission(self):
        """Cache metadata keeps its owners when a prefill H2D fails midway."""
        weights = object.__new__(AscendPrefillWeights)
        copy, compute = _Stream(), _Stream()
        host = torch.tensor([[11, 12], [21, 22], [31, 32]])
        hot = torch.tensor([[31, 32], [11, 12], [91, 92]])
        keys, valid = torch.tensor([2, 0]), torch.tensor([True, True])
        weights.owner = SimpleNamespace(weight_copy_stream=copy)
        weights.borrows_hot, weights.hot_dirty = True, False
        weights.banks, weights.plans = (object(), object()), {4: object()}
        weights.restore_plan = SimpleNamespace(
            source_index=keys, valid=valid, count=torch.zeros((), dtype=torch.int32)
        )
        weights.restore_bindings = object()

        def partial_copy(*args):
            copy.enqueue(lambda: hot[0].fill_(-1))
            raise RuntimeError("partial H2D")

        def restore(plan, bindings, stream):
            stream.enqueue(lambda: hot[:2].copy_(host[plan.source_index]))

        weights.host_ops = SimpleNamespace(materialize_into=partial_copy)
        weights.restore_ops = SimpleNamespace(materialize_into=restore)
        with self.assertRaisesRegex(RuntimeError, "partial H2D"):
            weights._load(4, 1, _Context())
        self.assertTrue(weights.hot_dirty)
        with patch.object(
            torch, "npu", SimpleNamespace(current_stream=lambda: compute), create=True
        ):
            weights.restore_hot()
        compute.drain()
        copy.drain()
        self.assertEqual(hot.tolist(), [[31, 32], [11, 12], [91, 92]])
        self.assertEqual(keys.tolist(), [2, 0])
        self.assertEqual(weights.restore_plan.count.item(), 2)
        self.assertFalse(weights.hot_dirty)


class TestPrefillAclDescriptors(CustomTestCase):
    def test_preparation_covers_complete_bundle_and_reuses_native_descriptors(self):
        from sglang.srt.offloading.moe.npu.native_acl import AclHostTransferOps
        from sglang.srt.offloading.moe.npu.transfer import DeviceMemoryRegion
        from sglang.srt.offloading.moe.storage import HostMemoryCapability
        from sglang.srt.offloading.transfer import HostCopyPlan, HostCopyRegion

        batches = []
        device = SimpleNamespace(type="npu", index=0)

        def batch(*args):
            batches.append(args)
            return object()

        native = SimpleNamespace(
            api_info=lambda: {"queue_ordering": "torch_npu_stream_flush"},
            Batch=batch,
        )
        ops = AclHostTransferOps(native, lambda value: None, max_batch_entries=3)
        regions = tuple(
            HostCopyRegion(
                HostMemoryCapability(4096 + index * 64, 64, True, None, native),
                DeviceMemoryRegion(16384 + index * 64, 64, device, native),
                0,
                0,
                64,
            )
            for index in range(12)
        )
        plan = HostCopyPlan(regions)
        ops.prepare(plan, device)
        self.assertEqual(len(batches), 4)
        self.assertEqual(
            [address for b in batches for address in b[0]],
            [r.source.host_address for r in regions],
        )
        self.assertEqual(
            [address for b in batches for address in b[1]],
            [r.destination.address for r in regions],
        )
        self.assertEqual([size for b in batches for size in b[2]], [64] * 12)
        ops.prepare(plan, device)
        self.assertEqual(len(batches), 4)


class TestAscendPrefillIntegration(CustomTestCase):
    def test_allgather_slices_payload_and_block_scales_together(self):
        from sglang.srt.layers.moe.moe_runner.ascend import _pre_permute_allgather

        payload = torch.arange(16, dtype=torch.int8).reshape(4, 4)
        scale = torch.arange(16, dtype=torch.uint8).reshape(4, 2, 2)
        routed = SimpleNamespace(
            hidden_states=payload,
            scale=scale,
            expert_tokens=torch.tensor([1, 1]),
            expanded_row_idx=torch.arange(4, dtype=torch.int32),
        )
        output = SimpleNamespace(
            hidden_states=payload,
            hidden_states_scale=scale,
            topk_ids=torch.tensor([[0, 2], [3, 1]]),
            topk_weights=torch.ones((2, 2)),
            quant_mode=-1,
            group_list_type=1,
            forward_state=SimpleNamespace(),
        )
        config = SimpleNamespace(
            num_local_experts=2,
            moe_ep_rank=0,
            layer=SimpleNamespace(_ascend_ops_offloader=object()),
        )
        with patch(
            "sglang.srt.layers.moe.moe_runner.ascend.npu_moe_init_routing",
            return_value=routed,
        ):
            inputs = _pre_permute_allgather(output, config, {})
        self.assertTrue(torch.equal(inputs.hidden_states, payload[:2]))
        self.assertTrue(torch.equal(inputs.hidden_states_scale, scale[:2]))
        self.assertEqual(inputs.hidden_states.data_ptr(), payload.data_ptr())
        self.assertEqual(output.forward_state.num_routes, 4)
        self.assertEqual(inputs.expert_tokens.tolist(), [1, 1])

    def test_empty_receive_retires_bank_without_launching_ffn(self):
        from sglang.srt.layers.moe.moe_runner.ascend import AscendRunnerCore

        retired = []
        owner = SimpleNamespace(
            prepare_weights=lambda *args: object(),
            prefill=SimpleNamespace(finish_layer=retired.append),
        )
        core = object.__new__(AscendRunnerCore)
        core.config = SimpleNamespace(
            layer=SimpleNamespace(layer_id=3, _ascend_ops_offloader=owner)
        )
        inputs = SimpleNamespace(
            hidden_states=torch.empty((0, 4), dtype=torch.int8),
            hidden_states_scale=torch.empty(0),
            expert_tokens=torch.zeros(2, dtype=torch.int64),
            group_list_type=1,
        )
        with patch(
            "sglang.srt.layers.moe.moe_runner.ascend.get_is_extend_in_batch",
            return_value=True,
        ):
            result = core.run(inputs, None, {})
        self.assertEqual(tuple(result.hidden_states.shape), (0, 4))
        self.assertEqual(result.hidden_states.dtype, torch.bfloat16)
        self.assertEqual(retired, [3])

    def test_ffn_retires_weights_only_after_both_projections(self):
        from sglang.srt.layers.moe.moe_runner.ascend import AscendRunnerCore

        sequence = []

        def projection(value, label, factor):
            sequence.append(label)
            return value * factor

        owner = SimpleNamespace(
            prepare_weights=lambda *args: object(),
            wait_for_w2=lambda layer: sequence.append("w2-ready"),
            prefill=SimpleNamespace(
                finish_layer=lambda layer: sequence.append("retired")
            ),
        )
        core = object.__new__(AscendRunnerCore)
        core.config = SimpleNamespace(
            layer=SimpleNamespace(
                layer_id=3,
                _ascend_ops_offloader=owner,
                w13_kernel=SimpleNamespace(
                    apply=lambda quant, x, counts, **kw: projection(x, "w13", 2)
                ),
                w2_kernel=SimpleNamespace(
                    apply=lambda quant, x, counts, **kw: projection(x, "w2", 3)
                ),
            )
        )
        core.activation = SimpleNamespace(_apply_activation=lambda x: (x.relu(), None))
        inputs = SimpleNamespace(
            hidden_states=torch.tensor([[1.0, -1.0, 2.0, 3.0]]),
            hidden_states_scale=None,
            expert_tokens=torch.tensor([0, 1]),
            group_list_type=1,
        )
        with (
            patch(
                "sglang.srt.layers.moe.moe_runner.ascend.get_is_extend_in_batch",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.moe.moe_runner.ascend._uses_fused_gmm1",
                return_value=False,
            ),
        ):
            result = core.run(inputs, None, {})
        self.assertTrue(
            torch.equal(result.hidden_states, inputs.hidden_states.relu() * 6)
        )
        self.assertEqual(sequence, ["w13", "w2-ready", "w2", "retired"])

    def test_prefetch_starts_before_prefill_communication_on_idle_rank(self):
        from sglang.srt.layers.moe.token_dispatcher.ascend_ops import (
            AscendOpsDispatcher,
        )

        sequence = []
        dispatcher = object.__new__(AscendOpsDispatcher)
        dispatcher._offload_layer = SimpleNamespace(
            layer_id=3,
            _ascend_ops_offloader=SimpleNamespace(
                start_prefill_layer=lambda layer: sequence.append("prefetch")
            ),
        )
        dispatcher.moe_ascend_ops_config = SimpleNamespace(prefill="alltoallv")
        dispatcher.tp_group = object()

        def dispatch(*args):
            sequence.append("dispatch")
            return "received peer tokens"

        dispatcher._dispatch_alltoallv = dispatch
        with patch(
            "sglang.srt.layers.moe.token_dispatcher.ascend_ops.get_is_extend_in_batch",
            return_value=True,
        ):
            result = dispatcher.dispatch(
                torch.empty((0, 4)), (torch.empty((0, 2)), torch.empty((0, 2)), None)
            )
        self.assertEqual(result, "received peer tokens")
        self.assertEqual(sequence, ["prefetch", "dispatch"])


if __name__ == "__main__":
    unittest.main()
