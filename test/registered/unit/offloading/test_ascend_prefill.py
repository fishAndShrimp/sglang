"""CPU stream contracts and interpreted Triton metadata; no NPU required."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import triton.language as tl
from triton.runtime.interpreter import InterpretedFunction

from sglang.srt.offloading.moe.npu.prefill import (
    AscendPrefillWeights,
    LayerPrefetchPipeline,
    whole_layer_bank_layout,
)
from sglang.srt.offloading.moe.npu.small_batch_kernels import prepare_prefill_counts
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

    def synchronize(self):
        if self.command is not None:
            self.command.run()


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
                pipeline.begin_forward(context)
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

    def test_dispatch_rendezvous_waits_for_lookahead_on_every_rank(self):
        """A ready current bank must not let a slower future copy enter HCCL."""
        contexts, copies, banks = [], [], []
        for rank in range(2):
            context, copy, bank_values = _Context(), _Stream(), [None, None]

            def load(layer, bank, ctx, copy=copy, values=bank_values):
                copy.enqueue(lambda: values.__setitem__(bank, layer))
                return SimpleNamespace(completion=_Completion(copy.tail))

            pipeline = LayerPrefetchPipeline((3, 4), load=load, copy_stream=copy)
            pipeline.begin_forward(context)
            # Both ranks have current weights. Only rank 1 has lookahead ready.
            pipeline.ready[rank].completion.synchronize()
            self.assertEqual(bank_values[0], 3)
            pipeline.start(3, context)
            if rank == 0:
                self.assertIsNone(bank_values[1])  # The host must not wait.
            contexts.append(context)
            copies.append(copy)
            banks.append(bank_values)

        # The existing count AllGather joins the callers' device frontiers.
        # Executing it first must pull in all earlier copies on both ranks.
        communication, observations = _Stream(), []
        communication.enqueue(
            lambda: observations.append([list(values) for values in banks]),
            tuple(
                context.compute_stream.tail
                for context in contexts
                if context.compute_stream.tail is not None
            ),
        )
        self.assertEqual(observations, [])  # No CPU synchronization at start.
        communication.drain()
        self.assertEqual(observations, [[[3, 4], [3, 4]]])
        self.assertTrue(all(copy.tail.done for copy in copies))

    def test_partial_forward_and_wrong_layer_are_rejected(self):
        context, copy = _Context(), _Stream()
        pipeline = LayerPrefetchPipeline(
            (3, 4),
            load=lambda *args: SimpleNamespace(completion=_Completion(None)),
            copy_stream=copy,
        )
        with self.assertRaisesRegex(RuntimeError, "forward entry"):
            pipeline.start(3, context)
        pipeline.begin_forward(context)
        with self.assertRaisesRegex(RuntimeError, "model order"):
            pipeline.start(4, context)
        with self.assertRaisesRegex(RuntimeError, "previous prefill"):
            pipeline.begin_forward(_Context())
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
            pipeline.begin_forward(context)
            pipeline.start(3, context)
            pipeline.acquire(3, context)
            pipeline.retire(3, context.record_compute_completion(), context)
            pipeline.finish_forward(context)
            pipeline.reset()
        self.assertEqual(len(submitted), 2)
        self.assertIsNot(submitted[0], submitted[1])


class TestAscendPrefillWeights(CustomTestCase):
    def setUp(self):
        super().setUp()
        # Execute the production kernel on CPU, not a second implementation.
        kernel = InterpretedFunction(prepare_prefill_counts.fn)
        self.kernel_patch = patch(
            "sglang.srt.offloading.moe.npu.small_batch_kernels.prepare_prefill_counts",
            kernel,
        )
        self.kernel_patch.start()
        self.addCleanup(self.kernel_patch.stop)
        # tl.sum is itself a jitted helper when imported outside interpreter mode.
        sum_patch = patch.object(tl, "sum", InterpretedFunction(tl.sum.fn))
        sum_patch.start()
        self.addCleanup(sum_patch.stop)

    def test_cumulative_counts_keep_zero_groups_and_cache_usage_in_layer_order(self):
        weights = object.__new__(AscendPrefillWeights)
        weights.counts = torch.empty(4, dtype=torch.int64)
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

    def test_count_kernel_preserves_padding_and_latches_invalid_groups(self):
        # Three experts exercise the masked fourth lane. Invalid counts must
        # stop every group, including after a later valid layer is submitted.
        from sglang.srt.offloading.moe.npu.small_batch_kernels import (
            prepare_prefill_counts as kernel,
        )

        for raw, cumulative, expected, failed in (
            ([2, 0, 3], False, [2, 0, 3], False),
            ([2, 2, 5], True, [2, 0, 3], False),
            ([0, 0, 0], True, [0, 0, 0], False),
            ([2, -1, 3], False, [0, 0, 0], True),
            ([2, 1, 5], True, [0, 0, 0], True),
        ):
            with self.subTest(raw=raw, cumulative=cumulative):
                counts = torch.tensor(raw, dtype=torch.int64)
                error = torch.zeros((), dtype=torch.int32)
                enabled = torch.tensor(True)
                usage = torch.tensor([100, 1, 2, 3, 100], dtype=torch.int64)
                out = torch.full((5,), -9, dtype=torch.int64)

                def launch():
                    kernel[(1,)](
                        counts,
                        error,
                        enabled,
                        usage[1:4],
                        out[1:4],
                        EXPERTS=3,
                        CUMULATIVE=cumulative,
                        BLOCK=4,
                    )

                launch()
                self.assertEqual(counts.tolist(), raw)
                self.assertEqual(out.tolist(), [-9, *expected, -9])
                self.assertEqual(
                    usage.tolist(),
                    [100, *[a + b for a, b in zip((1, 2, 3), expected)], 100],
                )
                self.assertEqual(error.item(), int(failed))
                previous_usage = usage.clone()
                counts.copy_(torch.tensor([1, 2, 3]))
                enabled.fill_(False)
                launch()
                self.assertTrue(torch.equal(usage, previous_usage))
                if failed:
                    self.assertEqual(out.tolist(), [-9, 0, 0, 0, -9])
                    self.assertEqual(error.item(), 1)

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
    def test_old_extension_requires_rebuild_instead_of_silent_flush(self):
        from sglang.srt.offloading.moe.npu.native_acl import AclHostTransferOps

        old_native = SimpleNamespace(
            api_info=lambda: {"queue_ordering": "torch_npu_stream_flush"}
        )
        with self.assertRaisesRegex(ValueError, "[Rr]ebuild"):
            AclHostTransferOps(old_native, lambda value: None)

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
            api_info=lambda: {"queue_ordering": "torch_npu_op_queue"},
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
    def _combine_call(self, owner, layer, mode, values):
        from sglang.srt.layers.moe.token_dispatcher.ascend_ops import (
            AscendOpsCombineInput,
            AscendOpsDispatcher,
            _AllGatherState,
            _AllToAllVState,
            _MC2State,
        )

        dispatcher = object.__new__(AscendOpsDispatcher)
        dispatcher._offload_layer = SimpleNamespace(layer_id=layer)
        dispatcher.hidden_size = values.shape[1]
        if owner is not None:
            dispatcher._offload_layer._ascend_ops_offloader = owner
        rows = values.shape[0]
        if mode == "allgather":
            state = _AllGatherState(object(), [rows], values.dtype)
        elif mode == "alltoallv":
            state = _AllToAllVState(
                object(),
                [rows],
                [rows],
                torch.ones((rows, 1)),
                torch.zeros((rows, 1), dtype=torch.int32),
                torch.arange(rows, dtype=torch.int32),
                values.shape,
            )
        else:
            dispatcher._ep_comm_name = "test"
            dispatcher.ep_size = dispatcher.num_experts = 1
            dispatcher.ep_rank = 0
            dispatcher.moe_ascend_ops_config = SimpleNamespace(ll_combine_alg="")
            state = _MC2State(max(0, rows - 1), None, None, None, None, None, None)
        return dispatcher, AscendOpsCombineInput(values, state)

    @contextmanager
    def _combine_device(self, context, communication, trace):
        """Model a collective's stream handoff, not the dispatcher's decisions."""

        def collective(output, values, **kwargs):
            trace.append("combine-submit")
            communication.wait_stream(context.compute_stream)

            def complete():
                output.copy_(values)
                trace.append("combine-done")

            communication.enqueue(complete)
            # Synchronous dist calls make their output available to subsequent
            # work on the caller stream without waiting on the Python thread.
            context.compute_stream.wait_stream(communication)

        def finalize(values, *args, **kwargs):
            context.compute_stream.enqueue(lambda: trace.append("finalize-done"))
            return values

        with (
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.dist.get_rank",
                return_value=0,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.dist.all_to_all_single",
                side_effect=collective,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.dist.reduce_scatter_tensor",
                side_effect=collective,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.npu_moe_finalize_routing",
                side_effect=finalize,
            ),
        ):
            yield

    def _forward_owner(self, *, fail_second_load=False):
        from sglang.srt.offloading.moe.npu.ascend_ops import AscendOpsOffloader

        context, copy = _Context(), _Stream()
        banks, submitted = [torch.tensor([0]), torch.tensor([99])], []
        owner = object.__new__(AscendOpsOffloader)
        owner.ready, owner.step, owner.decode_active_rows = True, 0, None
        owner.weight_copy_stream = copy
        owner.state = SimpleNamespace(
            step_access_count=torch.zeros(3, dtype=torch.int64),
            update_enabled=torch.tensor(False),
        )
        owner.policy = SimpleNamespace(config=SimpleNamespace(update_interval=4))
        owner._check_error = lambda: None
        owner.cache_tail_graphs = {
            due: SimpleNamespace(replay=lambda: None) for due in (False, True)
        }
        weights = object.__new__(AscendPrefillWeights)
        owner.prefill, weights.owner = weights, owner
        weights.borrows_hot, weights.hot_dirty = True, False
        weights.banks, weights.plans = banks, {layer: layer for layer in (3, 4, 5)}
        weights.restore_plan = SimpleNamespace(
            valid=torch.tensor([True]), count=torch.zeros((), dtype=torch.int32)
        )
        weights.restore_bindings = object()

        def load(layer, bank, stream):
            submitted.append(layer)
            stream.enqueue(lambda: bank.fill_(layer))
            if fail_second_load and layer == 4:
                raise RuntimeError("partial initial H2D")
            return SimpleNamespace(completion=_Completion(stream.tail))

        def restore(plan, bindings, stream):
            stream.enqueue(lambda: banks[1].fill_(99))

        weights.host_ops = SimpleNamespace(materialize_into=load)
        weights.restore_ops = SimpleNamespace(materialize_into=restore)
        weights.pipeline = LayerPrefetchPipeline(
            (3, 4, 5), load=weights._load, copy_stream=copy
        )
        return owner, context, banks, submitted

    def test_forward_entry_prefetch_precedes_model_on_all_prefill_ranks(self):
        """Initial copies must be submitted before even resident-layer attention."""
        for global_extend, local_extend, idle in (
            (True, True, False),
            (True, False, True),
            (True, False, False),
        ):
            with self.subTest(
                global_extend=global_extend, local_extend=local_extend, idle=idle
            ):
                owner, context, banks, submitted = self._forward_owner()
                batch = SimpleNamespace(
                    is_extend_in_batch=global_extend,
                    forward_mode=SimpleNamespace(
                        is_extend_without_speculative=lambda: local_extend,
                        is_idle=lambda: idle,
                    ),
                )
                observed = []
                with (
                    patch(
                        "sglang.srt.offloading.moe.npu.ascend_ops._Context",
                        return_value=context,
                    ),
                    patch(
                        "sglang.srt.layers.dp_attention.get_is_extend_in_batch",
                        return_value=False,
                    ),
                    patch.object(
                        torch,
                        "npu",
                        SimpleNamespace(current_stream=lambda: context.compute_stream),
                        create=True,
                    ),
                ):
                    with owner.forward_scope(batch):
                        self.assertEqual(submitted, [3, 4])
                        # Represents attention / resident MoE before the first
                        # offloaded dispatcher can consume a layer entry.
                        context.compute_stream.enqueue(
                            lambda: observed.append("resident")
                        )
                        owner.weight_copy_stream.drain()
                        self.assertEqual(observed, [])
                        self.assertEqual([bank.item() for bank in banks], [3, 4])
                        for layer in (3, 4, 5):
                            owner.start_prefill_layer(layer)
                            bank = owner.prefill.pipeline.acquire(layer, context)
                            context.compute_stream.enqueue(
                                lambda bank=bank: observed.append(banks[bank].item())
                            )
                            owner.prefill.finish_layer(layer)
                self.assertEqual(submitted, [3, 4, 5])
                self.assertEqual(observed, ["resident", 3, 4, 5])
                self.assertEqual(banks[1].item(), 99)
                self.assertIsNone(owner.prefill.pipeline.invocation_id)

    def test_decode_verify_and_global_idle_ignore_stale_prefill_flag(self):
        for name, idle in (
            ("decode", False),
            ("target_verify", False),
            ("idle", True),
            ("local_extend_label", False),
        ):
            with self.subTest(mode=name):
                owner, context, _, submitted = self._forward_owner()
                batch = SimpleNamespace(
                    is_extend_in_batch=False,
                    forward_mode=SimpleNamespace(
                        is_extend_without_speculative=lambda: name
                        == "local_extend_label",
                        is_idle=lambda: idle,
                    ),
                )
                with (
                    patch(
                        "sglang.srt.offloading.moe.npu.ascend_ops._Context",
                        return_value=context,
                    ),
                    patch(
                        "sglang.srt.layers.dp_attention.get_is_extend_in_batch",
                        return_value=True,
                    ),
                ):
                    with owner.forward_scope(batch):
                        self.assertEqual(submitted, [])
                self.assertEqual(submitted, [])
                self.assertEqual(owner.step, 1)

    def test_initial_prefetch_failure_restores_hot_before_model_entry(self):
        owner, context, banks, submitted = self._forward_owner(fail_second_load=True)
        batch = SimpleNamespace(is_extend_in_batch=True)
        entered = False
        with (
            patch(
                "sglang.srt.offloading.moe.npu.ascend_ops._Context",
                return_value=context,
            ),
            patch.object(
                torch,
                "npu",
                SimpleNamespace(current_stream=lambda: context.compute_stream),
                create=True,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "partial initial H2D"):
                with owner.forward_scope(batch):
                    entered = True
        self.assertFalse(entered)
        self.assertEqual(submitted, [3, 4])
        self.assertEqual(banks[1].item(), 99)
        self.assertFalse(owner.prefill.hot_dirty)
        self.assertIsNone(owner.prefill.pipeline.invocation_id)
        self.assertFalse(owner.state.update_enabled.item())
        self.assertEqual(owner.step, 0)

    def test_decode_keeps_packed_preparation_and_w2_event(self):
        from sglang.srt.offloading.moe.npu.ascend_ops import AscendOpsOffloader

        owner = object.__new__(AscendOpsOffloader)
        owner.ready = True
        owner._prepare_fused = Mock(return_value="packed decode weights")
        owner.prefill = SimpleNamespace(
            weights=Mock(side_effect=AssertionError("prefill"))
        )
        event, stream, inputs = object(), Mock(), object()
        owner.copy_events = {4: (None, None, event)}
        with (
            patch(
                "sglang.srt.layers.dp_attention.get_is_extend_in_batch",
                return_value=False,
            ),
            patch.object(
                torch,
                "npu",
                SimpleNamespace(current_stream=lambda: stream),
                create=True,
            ),
        ):
            self.assertEqual(owner.prepare_weights(4, inputs), "packed decode weights")
            owner.wait_for_w2(4)
        owner._prepare_fused.assert_called_once_with(4, inputs)
        stream.wait_event.assert_called_once_with(event)
        owner.prefill.weights.assert_not_called()

    def test_dispatch_joins_copy_before_collectives_even_for_empty_ranks(self):
        """The dispatch entry must gate both prefill backends before HCCL."""
        from sglang.srt.layers.moe.token_dispatcher.ascend_ops import (
            AscendOpsDispatcher,
        )

        for mode in ("allgather", "alltoallv"):
            for rows in (0, 2):
                with self.subTest(mode=mode, local_rows=rows):
                    owner, context, banks, _ = self._forward_owner()
                    owner.context = context
                    owner.prefill.pipeline.begin_forward(context)
                    owner.prefill.pipeline.ready[0].completion.synchronize()
                    self.assertEqual([b.item() for b in banks], [3, 99])
                    communication, observed = _Stream(), []
                    dispatcher = object.__new__(AscendOpsDispatcher)
                    dispatcher._offload_layer = SimpleNamespace(
                        layer_id=3, _ascend_ops_offloader=owner
                    )
                    dispatcher.moe_ascend_ops_config = SimpleNamespace(prefill=mode)
                    dispatcher.tp_group = object()
                    payload = torch.zeros((rows, 2))

                    def dispatch_device(values, weights, ids, group):
                        communication.wait_stream(context.compute_stream)
                        communication.enqueue(
                            lambda: observed.append([b.item() for b in banks])
                        )
                        context.compute_stream.wait_stream(communication)
                        return values

                    dispatcher._dispatch_allgather = dispatch_device
                    dispatcher._dispatch_alltoallv = dispatch_device
                    with patch(
                        "sglang.srt.layers.moe.token_dispatcher.ascend_ops.get_is_extend_in_batch",
                        return_value=True,
                    ):
                        result = dispatcher.dispatch(payload, (None, None, None))
                    self.assertIs(result, payload)
                    self.assertEqual([b.item() for b in banks], [3, 99])
                    self.assertEqual(observed, [])
                    communication.drain()
                    self.assertEqual(observed, [[3, 4]])

    def test_refill_waits_for_combine_submission_and_device_completion(self):
        """Refilling a consumed bank must neither delay nor race its return collective."""
        for mode in ("allgather", "alltoallv"):
            for rows in (0, 2):
                with self.subTest(mode=mode, received_rows=rows):
                    owner, context, banks, submitted = self._forward_owner()
                    owner.context = context
                    communication, trace = _Stream(), []
                    load = owner.prefill.host_ops.materialize_into

                    def tracked_load(layer, bank, stream):
                        trace.append(f"submit-{layer}")
                        load(layer, bank, stream)
                        stream.enqueue(lambda: trace.append(f"copy-{layer}"))
                        return SimpleNamespace(completion=_Completion(stream.tail))

                    owner.prefill.host_ops.materialize_into = tracked_load
                    owner.prefill.pipeline.begin_forward(context)
                    owner.start_prefill_layer(3)
                    bank = owner.prefill.pipeline.acquire(3, context)
                    values = torch.zeros((rows, 2))
                    context.compute_stream.enqueue(
                        lambda: values.fill_(banks[bank].item())
                    )
                    trace.clear()
                    dispatcher, inputs = self._combine_call(owner, 3, mode, values)
                    with self._combine_device(context, communication, trace):
                        result = dispatcher.combine(inputs)

                    self.assertEqual(submitted, [3, 4, 5])
                    self.assertLess(
                        trace.index("combine-submit"), trace.index("submit-5")
                    )
                    self.assertFalse(communication.tail.done)
                    self.assertEqual(owner.prefill.pipeline.current, 1)
                    # Adversarial scheduling: run the copy stream first. Its new
                    # H2D must still execute the queued return communication
                    # before overwriting bank A, including on an empty rank.
                    owner.weight_copy_stream.drain()
                    self.assertTrue(communication.tail.done)
                    self.assertLess(trace.index("combine-done"), trace.index("copy-5"))
                    if mode == "alltoallv":
                        self.assertLess(
                            trace.index("finalize-done"), trace.index("copy-5")
                        )
                    self.assertTrue(torch.equal(result, torch.full_like(values, 3)))
                    self.assertEqual(banks[0].item(), 5)

    def test_combine_failure_leaves_bank_owned_by_current_layer(self):
        """A failed return collective must not advance or overwrite its layer bank."""
        owner, context, _, submitted = self._forward_owner()
        owner.context = context
        owner.prefill.pipeline.begin_forward(context)
        owner.start_prefill_layer(3)
        owner.prefill.pipeline.acquire(3, context)
        copy_tail = owner.weight_copy_stream.tail
        dispatcher, inputs = self._combine_call(
            owner, 3, "alltoallv", torch.ones((2, 2))
        )
        with patch(
            "sglang.srt.layers.moe.token_dispatcher.ascend_ops.dist.all_to_all_single",
            side_effect=RuntimeError("collective enqueue failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "collective enqueue failed"):
                dispatcher.combine(inputs)
        self.assertEqual(owner.prefill.pipeline.current, 0)
        self.assertTrue(owner.prefill.pipeline.started)
        self.assertEqual(submitted, [3, 4])
        self.assertIs(owner.weight_copy_stream.tail, copy_tail)

    def test_decode_and_resident_combine_do_not_require_prefill_session(self):
        from sglang.srt.layers.moe.token_dispatcher import ascend_ops

        def unexpected_retire(layer):
            raise AssertionError("decode cannot consume a prefill bank")

        owner = SimpleNamespace(prefill=SimpleNamespace(finish_layer=unexpected_retire))
        values = torch.tensor([[1.0, 2.0], [99.0, 99.0]])
        dispatcher, inputs = self._combine_call(owner, 3, "mc2", values)
        with patch.object(
            ascend_ops,
            "torch_npu",
            SimpleNamespace(npu_moe_distribute_combine_v2=lambda x, *a, **kw: x * 2),
            create=True,
        ):
            result = dispatcher.combine(inputs)
        self.assertTrue(torch.equal(result, torch.tensor([[2.0, 4.0]])))

        for mode in ("allgather", "alltoallv"):
            with self.subTest(resident_mode=mode):
                context, communication, trace = _Context(), _Stream(), []
                dispatcher, inputs = self._combine_call(None, 0, mode, values)
                with self._combine_device(context, communication, trace):
                    result = dispatcher.combine(inputs)
                context.compute_stream.drain()
                self.assertTrue(torch.equal(result, values))

    def test_hot_restore_follows_combine_and_precedes_maintenance_and_decode(self):
        """Borrowed Hot must survive return communication, then be ready for cache users."""
        owner, context, banks, _ = self._forward_owner()
        owner.step = 3
        weights, communication, trace, observed = owner.prefill, _Stream(), [], []

        def restore(plan, bindings, stream):
            def complete():
                banks[1].fill_(99)
                trace.append("restore")

            stream.enqueue(complete)

        weights.restore_ops = SimpleNamespace(materialize_into=restore)

        def check_hot():
            self.assertEqual(banks[1].item(), 99)
            trace.append("cache")

        owner.cache_tail_graphs = {
            due: SimpleNamespace(
                replay=lambda: context.compute_stream.enqueue(check_hot)
            )
            for due in (False, True)
        }
        with (
            patch(
                "sglang.srt.offloading.moe.npu.ascend_ops._Context",
                side_effect=lambda: context,
            ),
            patch.object(
                torch,
                "npu",
                SimpleNamespace(current_stream=lambda: context.compute_stream),
                create=True,
            ),
            self._combine_device(context, communication, trace),
        ):
            with owner.forward_scope(SimpleNamespace(is_extend_in_batch=True)):
                for layer in (3, 4, 5):
                    owner.start_prefill_layer(layer)
                    bank = weights.pipeline.acquire(layer, context)
                    values = torch.zeros((1, 2))
                    context.compute_stream.enqueue(
                        lambda bank=bank, values=values: values.fill_(
                            banks[bank].item()
                        )
                    )
                    dispatcher, inputs = self._combine_call(
                        owner, layer, "alltoallv", values
                    )
                    result = dispatcher.combine(inputs)
                    context.compute_stream.enqueue(
                        lambda result=result: observed.append(result[0, 0].item())
                    )
            self.assertEqual(observed, [3, 4, 5])
            self.assertEqual(trace.count("combine-done"), 3)
            self.assertGreater(
                trace.index("restore"),
                max(i for i, item in enumerate(trace) if item == "finalize-done"),
            )
            self.assertLess(trace.index("restore"), trace.index("cache"))
            self.assertFalse(weights.hot_dirty)
            self.assertIsNone(weights.pipeline.invocation_id)
            self.assertFalse(owner.state.update_enabled.item())
            self.assertEqual(context.resources, [])

            # No prefill hook runs in the next decode forward.
            context = _Context()
            with owner.forward_scope():
                context.compute_stream.enqueue(check_hot)
            self.assertEqual(trace[-3:], ["cache", "cache", "cache"])

    def test_allgather_keeps_native_capacity_without_host_count_read(self):
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
        with (
            patch(
                "sglang.srt.layers.moe.moe_runner.ascend.npu_moe_init_routing",
                return_value=routed,
            ),
            patch.object(torch.Tensor, "item", side_effect=AssertionError("host sync")),
        ):
            inputs = _pre_permute_allgather(output, config, {})
        self.assertIs(inputs.hidden_states, payload)
        self.assertIs(inputs.hidden_states_scale, scale)
        self.assertEqual(output.forward_state.num_routes, 4)
        self.assertEqual(inputs.expert_tokens.tolist(), [1, 1])

    def test_empty_receive_leaves_bank_for_combine_without_launching_ffn(self):
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
        self.assertEqual(retired, [])

    def test_ffn_preserves_results_without_refilling_before_combine(self):
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
        self.assertEqual(sequence, ["w13", "w2-ready", "w2"])

        # The same runner must not retire/overwrite banks during Decode.
        sequence.clear()
        with (
            patch(
                "sglang.srt.layers.moe.moe_runner.ascend.get_is_extend_in_batch",
                return_value=False,
            ),
            patch(
                "sglang.srt.layers.moe.moe_runner.ascend._uses_fused_gmm1",
                return_value=False,
            ),
        ):
            decoded = core.run(inputs, None, {})
        self.assertTrue(torch.equal(decoded.hidden_states, result.hidden_states))
        self.assertEqual(sequence, ["w13", "w2-ready", "w2"])

    def test_prefill_dispatch_consumes_layer_entry_on_idle_rank(self):
        from sglang.srt.layers.moe.token_dispatcher.ascend_ops import (
            AscendOpsDispatcher,
        )

        sequence = []
        dispatcher = object.__new__(AscendOpsDispatcher)
        dispatcher._offload_layer = SimpleNamespace(
            layer_id=3,
            _ascend_ops_offloader=SimpleNamespace(
                start_prefill_layer=lambda layer: sequence.append("enter-prefill")
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
        self.assertEqual(sequence, ["enter-prefill", "dispatch"])

        sequence.clear()
        dispatcher._dispatch_low_latency = dispatch
        with patch(
            "sglang.srt.layers.moe.token_dispatcher.ascend_ops.get_is_extend_in_batch",
            return_value=False,
        ):
            dispatcher.dispatch(
                torch.ones((1, 4)), (torch.ones((1, 2)), torch.zeros((1, 2)), None)
            )
        self.assertEqual(sequence, ["dispatch"])


if __name__ == "__main__":
    unittest.main()
