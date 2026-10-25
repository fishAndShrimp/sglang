"""Cache contracts migrated to the production NPU assembly and tensor planner.

Only hardware/transport seams are simulated. The retired cache_* runtime is not
used as an oracle or kept as a second implementation in these tests.
"""

import unittest
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
import test_native_periodic_cache as native_cache_fixtures
from test_npu_backend import Assembly
from test_tensor_residency import load_hot, setup

from sglang.srt.offloading.moe.cache_metadata import CacheRouteStats
from sglang.srt.offloading.moe.config import Phase, fixed_waves
from sglang.srt.offloading.moe.execution import ExecutionRequest
from sglang.srt.offloading.moe.interfaces import CachePolicy
from sglang.srt.offloading.moe.npu import experts as native_experts
from sglang.srt.offloading.moe.owner import MoEOffloadOwner
from sglang.srt.offloading.moe.policy import TensorLFUPolicy
from sglang.srt.offloading.moe.resources import AllocationLedger, ResourceKind
from sglang.srt.offloading.transfer import (
    Completion,
    ImmediateCompletion,
    TransferTicket,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class OpaquePolicy(CachePolicy):
    """Prove the assembly uses the protocol, not catalog/config implementation details."""

    def __init__(self):
        self._algorithm = TensorLFUPolicy()

    def allocate_state(self, num_experts, device):
        return self._algorithm.allocate_state(num_experts, device)

    def observe(self, state, counts, enabled, due_out):
        self._algorithm.observe(state, counts, enabled, due_out)

    def select_hot_set(self, state, key_to_slot, target_keys):
        self._algorithm.select_hot_set(state, key_to_slot, target_keys)

    def copy_state(self, source, destination):
        self._algorithm.copy_state(source, destination)


class TestProductionCacheContracts(unittest.TestCase):
    def setUp(self):
        self.fixture = native_cache_fixtures.TestNativePeriodicCache()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.runtime = self.fixture.assembly.finalize()

    def test_document_sequence_matches_usage_and_actual_hot_weight_bytes(self):
        routes = (
            ([1, 1, 0], [1, 1, 3]),
            ([1, 2, 2], [1, 3, 2]),
            ([1, 1, 3, 2], [1, 1, 3]),
            ([1, 1, 1, 0, 2, 2, 3, 3], [1, 1, 0, 2, 3]),
        )
        policy = self.runtime.cache_updates.policy
        with patch.object(policy, "finish_step", wraps=policy.finish_step) as finish:
            for left, right in routes:
                self.fixture.run_step(
                    self.runtime,
                    {
                        3: torch.bincount(torch.tensor(left), minlength=4).tolist(),
                        4: torch.bincount(torch.tensor(right), minlength=4).tolist(),
                    },
                )
        self.assertEqual(finish.call_count, 4)
        self.assertEqual(self.runtime.residency.slot_to_key.tolist(), [1, 2, 5, 7])
        state = self.runtime.cache_updates.policy_state
        torch.testing.assert_close(
            state.usage, torch.tensor([2, 8, 5, 3, 1, 7, 2, 4]) * 0.7
        )
        self.assertFalse(bool(state.update_due))
        for tensor in self.fixture.assembly.backend.format.banks[0].tensors.values():
            self.assertEqual(tensor[:, 0].tolist(), [31, 32, 41, 43])

    def test_disabled_capture_still_materializes_actual_demand(self):
        runtime = self.runtime
        self.fixture.run_step(runtime, {3: [0, 2, 0, 0]}, enabled=False)
        self.assertEqual(float(runtime.cache_updates.policy_state.usage.sum()), 0)
        self.assertEqual(int(runtime.cache_stats.accesses), 0)
        # The last layer has empty demand; inspect the first layer's actual
        # physical execution slot, whose bytes survive invalidation by empties.
        execution = self.fixture.assembly.backend.format.banks[1]
        for tensor in execution.tensors.values():
            self.assertEqual(int(tensor[1, 0]), 31)
        self.assertEqual(runtime.residency.slot_to_key.tolist(), [0, 2, 4, 7])

    def test_statistics_reset_on_capture_and_global_idle_without_losing_addresses(self):
        stats = self.runtime.cache_stats
        addresses = tuple(t.data_ptr() for t in vars(stats).values())
        self.fixture.run_step(self.runtime, {3: [2, 3, 0, 0], 4: [0, 0, 5, 7]})
        self.assertEqual(
            (int(stats.accesses), int(stats.hits), int(stats.misses)), (17, 9, 8)
        )
        self.fixture.run_step(self.runtime, {}, enabled=False)
        self.assertEqual(tuple(int(t) for t in vars(stats).values()), (0, 0, 0))
        self.assertEqual(tuple(t.data_ptr() for t in vars(stats).values()), addresses)

    def test_snapshot_restores_hot_state_in_place_but_invalidates_execution(self):
        self.fixture.run_step(self.runtime, {3: [0, 5, 0, 0]})
        state = self.runtime.residency
        names = ("key_to_slot", "slot_to_key", "slot_valid", "slot_generation")
        saved = {name: getattr(state, name).clone() for name in names}
        addresses = {name: getattr(state, name).data_ptr() for name in names}
        owner = MoEOffloadOwner(
            self.fixture.assembly.config, self.fixture.assembly.catalog, self.runtime
        )
        self.addCleanup(owner.close)
        owner.switch_phase(Phase.PREFILL)
        state.execution_key.fill_(123)
        state.execution_valid.fill_(True)
        owner.switch_phase(Phase.DECODE)
        for name, value in saved.items():
            torch.testing.assert_close(getattr(state, name), value)
            self.assertEqual(getattr(state, name).data_ptr(), addresses[name])
        self.assertFalse(bool(state.execution_valid.any()))
        self.assertEqual(int(self.runtime.cache_updates.policy_state.step_mod), 1)

    def test_nonperiodic_policy_without_private_attributes_and_cold_start(self):
        assembly = Assembly(ratio=0.5, layer_count=1, local_experts=4)
        self.addCleanup(assembly.close)
        assembly.policy = OpaquePolicy()
        runtime = assembly.finalize()
        self.assertEqual(runtime.residency.slot_to_key.tolist(), [-1, -1])
        context = runtime._new_context(Phase.DECODE, capturing=True)
        runtime._context = context
        try:
            runtime.ops.begin_step(context)
            runtime.residency.update_enabled.fill_(True)
            runtime.residency.step_access_count.copy_(torch.tensor([0, 2, 3, 0]))
            runtime.emit_step_end(context)
            self.assertEqual(set(runtime.residency.slot_to_key.tolist()), {1, 2})
            context.finish()
        finally:
            runtime._context = None


class TestCountMetadataContracts(unittest.TestCase):
    def test_native_layout_rejects_host_counts_during_decode(self):
        layout = native_experts.NPUDispatchLayout(
            4, 4, 8, "cpu", torch.zeros((), dtype=torch.int32)
        )
        raw = SimpleNamespace(
            hidden_states=torch.zeros(4, 4, dtype=torch.bfloat16),
            format=SimpleNamespace(value="deepep_normal"),
            num_recv_tokens_per_expert=[1, 1, 1, 1],
        )
        for capturing in (False, True):
            context = SimpleNamespace(phase=Phase.DECODE, capturing=capturing)
            with self.assertRaisesRegex(ValueError, "device-count"):
                layout.bind(SimpleNamespace(raw=raw), context)

    def test_zero_receive_rows_preserve_native_geometry_for_every_wave(self):
        layout = native_experts.NPUDispatchLayout(
            4, 4, 8, "cpu", torch.zeros((), dtype=torch.int32)
        )
        raw = SimpleNamespace(
            hidden_states=torch.empty(0, 4, dtype=torch.bfloat16),
            hidden_states_scale=None,
            format=SimpleNamespace(value="deepep_ll"),
            masked_m=torch.zeros(4, dtype=torch.int64),
        )
        context = SimpleNamespace(
            phase=Phase.DECODE, capturing=True, retain=lambda resource: None
        )
        with patch.object(
            native_experts, "_compute_scope", lambda *args: nullcontext()
        ):
            bound = layout.bind(SimpleNamespace(raw=raw), context)
            for wave in fixed_waves(4, 2):
                selected = layout.select_wave(bound, wave, context)
                self.assertEqual(tuple(selected.hidden_states.shape), (0, 4))
                layout.write_wave(
                    bound, wave, torch.empty(0, 4, dtype=torch.bfloat16), context
                )
            self.assertEqual(
                tuple(layout.combine_input(bound, context).hidden_states.shape), (0, 4)
            )

    def test_count_representation_conserves_assignments_across_disjoint_waves(self):
        ops, state, waves, context = setup()
        load_hot(state, [1, 5, 9])
        state.update_enabled.fill_(True)
        stats = ops.cache_stats = CacheRouteStats.allocate("cpu")
        counts = torch.tensor([2, 4, 0, 0, 1, 3, 8])
        ops.begin_step(context)
        for layer in (3, 4):
            for wave in waves:
                reservation = ops.resolve_execution(
                    ExecutionRequest(layer, counts, wave, Phase.DECODE), context
                )
                ops.publish_execution(reservation, context.copied(), context)
                ops.finish_execution(reservation, context.copied(), context)
        torch.testing.assert_close(state.step_access_count, torch.cat((counts, counts)))
        self.assertEqual(int(stats.accesses), 36)
        self.assertEqual(int(stats.hits), 7)
        self.assertEqual(int(stats.misses), 29)

    def test_metadata_allocations_are_unique_and_report_physical_bytes(self):
        stats = CacheRouteStats.allocate("cpu")
        allocations = stats.allocations()
        ledger = AllocationLedger({ResourceKind.METADATA: 24}, allocations)
        self.assertEqual(ledger.usage[ResourceKind.METADATA], 24)
        self.assertEqual(len({a.storage_key for a in allocations}), 3)

    def test_counting_and_planning_never_read_device_values_on_host(self):
        ops, state, waves, context = setup()
        load_hot(state, [1, 5, 9])
        state.update_enabled.fill_(True)
        ops.cache_stats = CacheRouteStats.allocate("cpu")
        demand = torch.ones(7, dtype=torch.int64)
        target = torch.tensor([1, 2, 9])
        with ExitStack() as stack:
            for method in ("item", "tolist", "cpu", "__bool__"):
                stack.enter_context(
                    patch.object(
                        torch.Tensor, method, side_effect=AssertionError(method)
                    )
                )
            ops.begin_step(context)
            for wave in waves:
                reservation = ops.resolve_execution(
                    ExecutionRequest(3, demand, wave, Phase.DECODE), context
                )
                ops.publish_execution(reservation, context.copied(), context)
                ops.finish_execution(reservation, context.copied(), context)
            update = ops.prepare_update(target, context)
            ops.commit_update(update, context.copied(), context)


class TestSharedNativePrefillPipeline(unittest.TestCase):
    def test_one_pipeline_protects_shared_banks_across_layers_and_invocations(self):
        assembly = Assembly(ratio=0.0, slots=4, group=2, layer_count=3, local_experts=7)
        self.addCleanup(assembly.close)
        runtime = assembly.finalize()
        self.assertEqual(len(runtime.pipelines), 1)
        pipeline = runtime.pipelines[0]
        self.assertTrue(
            all(program.pipeline is pipeline for _, program, _ in assembly.builds)
        )
        guarded_readers = []
        waits = []
        test = self

        class Reader(Completion):
            def __init__(self, layer, wave, bank):
                self.layer, self.wave = layer, wave
                self.tensors = tuple(bank.tensors.values())
                self.expected = tuple(tensor.clone() for tensor in self.tensors)
                self.ready = False

            def wait_on(self, stream):
                if not self.ready:
                    for actual, expected in zip(self.tensors, self.expected):
                        torch.testing.assert_close(actual, expected)
                    waits.append((self.layer, self.wave.bank, stream))
                    self.ready = True

            def done(self):
                return self.ready

            def synchronize(self):
                test.fail("Ordinary prefill bank reuse must use device waits")

        def run_layer(layer, context):
            previous = tuple(guarded_readers)
            pipeline.start(layer, context)
            waves = pipeline.seal(layer, ImmediateCompletion(), context)
            # The new worker must join every previous layer reader before its
            # first write, including readers on the final A/B waves.
            self.assertTrue(all(reader.ready for reader in previous))
            self.assertGreaterEqual(len(waves), 3)
            for wave in waves:
                lease = pipeline.acquire(layer, wave, context)
                pipeline.access.wait_ready(lease, context.compute_stream)
                expected = [
                    layer * 10 + expert
                    for expert in range(wave.local_start, wave.local_end)
                ]
                for tensor in lease.bindings.bank.tensors.values():
                    self.assertEqual(tensor[:, 0].tolist(), expected)
                reader = Reader(layer, wave, lease.bindings.bank)
                guarded_readers.append(reader)
                pipeline.access.release_after(lease, TransferTicket(reader), context)
            return pipeline.finish(layer, context)

        first = runtime._new_context(Phase.PREFILL)
        runtime._context = first
        try:
            run_layer(3, first)
            run_layer(4, first)
            first_completion = first.finish()
        finally:
            runtime._context = None
        self.assertFalse(first_completion.resources_idle.completion.done())
        second = runtime._new_context(Phase.PREFILL)
        self.assertIsNot(first.invocation_id, second.invocation_id)
        runtime._context = second
        try:
            final = run_layer(5, second)
            # Complete remaining readers using the same device dependency path.
            final.completion.wait_on(second.compute_stream)
            second.finish()
        finally:
            runtime._context = None
        self.assertTrue(first_completion.resources_idle.completion.done())
        self.assertTrue(all(reader.ready for reader in guarded_readers))
        self.assertEqual({bank for _, bank, _ in waits}, {0, 1})
        self.assertFalse(pipeline._sessions)


if __name__ == "__main__":
    unittest.main()
