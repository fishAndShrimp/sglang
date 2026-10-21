"""Real CPU tensor planner + byte-copy emulator + policy + exact restore.

The only transport emulation is the native MemFabric kernel; this is deliberately
not a graph/EP/FP4 numerical certification.
"""

import random
import unittest

import torch
from sglang.srt.offloading.moe.config import (
    ExpertCatalog,
    ExpertLayerSpec,
    Phase,
    fixed_waves,
)
from sglang.srt.offloading.moe.execution import (
    CacheUpdateCoordinator,
    ExecutionRequest,
    ExpertWeightAccess,
)
from sglang.srt.offloading.moe.npu_transfer import (
    DeviceMemoryRegion,
    MemFabricCopyComponent,
    MemFabricDeviceTransferOps,
    MemFabricTransferBindings,
)
from sglang.srt.offloading.moe.policy import TensorLFUPolicy
from sglang.srt.offloading.moe.residency import BoundExecutionWave, TensorResidencyOps
from sglang.srt.offloading.moe.runtime import (
    DecodeStateSnapshot,
    ManagedExecutionContext,
)
from sglang.srt.offloading.moe.state import ResidencyState
from sglang.srt.offloading.moe.storage import HostMemoryCapability
from sglang.srt.offloading.transfer import (
    ComposedTransferBackend,
    DeviceCopyPlan,
    ImmediateCompletion,
    TransferTicket,
)
from sglang.test.ci.ci_register import register_cpu_ci
from test_npu_transfer import FakeGroupPackCopy

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestCacheTransferIntegration(unittest.TestCase):
    def test_changing_routes_multilayer_hits_tail_updates_and_exact_restore(self):
        catalog = ExpertCatalog(
            (ExpertLayerSpec(3, 4, 1, 0), ExpertLayerSpec(4, 4, 1, 0))
        )
        state = ResidencyState.allocate(8, 3, 2, "cpu")
        state.update_enabled.fill_(True)
        error = torch.zeros((), dtype=torch.int32)
        registry, host, hot, execution = {}, [], [], []
        # Two physical components, neither a multiple of common vector widths.
        for component, width in enumerate((17, 3)):
            host.append(
                [
                    torch.full((width,), 10 * component + key + 1, dtype=torch.uint8)
                    for key in range(8)
                ]
            )
            hot.append([torch.zeros(width, dtype=torch.uint8) for _ in range(3)])
            execution.append([torch.zeros(width, dtype=torch.uint8) for _ in range(2)])
            for tensor in host[-1] + hot[-1] + execution[-1]:
                registry[tensor.data_ptr()] = tensor

        def binding(destinations, capacity):
            components = []
            for ci, values in enumerate(destinations):
                components.append(
                    MemFabricCopyComponent(
                        tuple(
                            HostMemoryCapability(
                                t.data_ptr(), t.numel(), True, t.data_ptr(), t
                            )
                            for t in host[ci]
                        ),
                        tuple(DeviceMemoryRegion.from_tensor(t) for t in hot[ci]),
                        tuple(DeviceMemoryRegion.from_tensor(t) for t in values),
                        host[ci][0].numel(),
                        name=f"part{ci}",
                    )
                )
            return MemFabricTransferBindings(
                components,
                capacity=capacity,
                hot_generations=state.slot_generation,
                error_status=error,
                device="cpu",
                allow_cpu=True,
                max_batch_entries=2,
            )

        waves = fixed_waves(4, 2)
        exec_bindings = {
            (layer, wave): binding(execution, 2) for layer in (3, 4) for wave in waves
        }
        hot_binding = binding(hot, 3)
        ops = TensorResidencyOps(
            state,
            catalog,
            execution_waves=tuple(
                BoundExecutionWave(layer, wave, 0, exec_bindings[(layer, wave)])
                for layer in (3, 4)
                for wave in waves
            ),
            update_bindings=hot_binding,
            error_status=error,
        )
        policy = TensorLFUPolicy(aging_interval=4)
        policy_state = policy.allocate_state(8, "cpu")
        targets = torch.empty(3, dtype=torch.int64)
        snapshot = DecodeStateSnapshot(state, policy, policy_state, error_status=error)
        kernel = FakeGroupPackCopy(registry)
        current = [None]
        transfer = ComposedTransferBackend(
            device_ops=MemFabricDeviceTransferOps(
                retain=lambda value: current[0].retain(value),
                group_pack_copy=kernel,
                record_completion=lambda stream: ImmediateCompletion(),
                allow_cpu=True,
            )
        )
        access = ExpertWeightAccess(ops, transfer)
        updates = CacheUpdateCoordinator(policy, policy_state, state, targets, access)
        rng = random.Random(1021)
        hits = misses = 0
        for step in range(16):
            ctx = ManagedExecutionContext(
                Phase.DECODE,
                compute_stream="compute",
                copy_stream="copy",
                record_completion=lambda stream: TransferTicket(ImmediateCompletion()),
                failure_drain=lambda: None,
            )
            current[0] = ctx
            ops.begin_step(ctx)
            before = state.slot_to_key.clone()
            for layer in (3, 4):
                counts = torch.tensor([rng.randrange(4) for _ in range(4)])
                if step == 5:
                    counts.zero_()  # Empty rank-local demand does not read stale lanes.
                for wave in waves:
                    lease = access.stage(
                        ExecutionRequest(layer, counts, wave, Phase.DECODE), ctx
                    )
                    access.wait_ready(lease, ctx.compute_stream)
                    plan = lease.reservations[0].copy_plan
                    hits += int(((plan.source_kind == 2) & plan.valid).sum())
                    misses += int(((plan.source_kind == 1) & plan.valid).sum())
                    for lane, local in enumerate(
                        range(wave.local_start, wave.local_end)
                    ):
                        key = catalog.key(layer, local)
                        self.assertEqual(
                            bool(state.execution_valid[lane]), bool(counts[local])
                        )
                        if counts[local]:
                            for ci in range(2):
                                torch.testing.assert_close(
                                    execution[ci][lane], host[ci][key]
                                )
                    access.release_after(
                        lease, TransferTicket(ImmediateCompletion()), ctx
                    )
            torch.testing.assert_close(
                state.slot_to_key, before
            )  # Misses have not admitted themselves.
            tail = updates.emit_step_end(ctx)
            ctx.finish(residency_ready=tail)
            for slot, key in enumerate(state.slot_to_key.tolist()):
                if key >= 0:
                    self.assertEqual(int(state.key_to_slot[key]), slot)
                    for ci in range(2):
                        torch.testing.assert_close(hot[ci][slot], host[ci][key])
            self.assertEqual(int(error), 0)
            if step in (3, 9):
                transition = ManagedExecutionContext(
                    Phase.PREFILL,
                    compute_stream="compute",
                    copy_stream="copy",
                    record_completion=lambda stream: TransferTicket(
                        ImmediateCompletion()
                    ),
                    failure_drain=lambda: None,
                )
                current[0] = transition
                old_slots, old_heat = (
                    state.slot_to_key.clone(),
                    policy_state.frequency.clone(),
                )
                snapshot.save(transition)
                for group in hot:
                    for tensor in group:
                        tensor.fill_(255)  # Prefill borrowed the physical hot arena.
                state.slot_valid.zero_()
                state.key_to_slot.fill_(-1)
                policy_state.frequency.zero_()

                def reload(saved, context):
                    valid = saved.slot_valid.clone()
                    plan = DeviceCopyPlan(
                        torch.where(valid, 1, 0),
                        saved.slot_to_key,
                        torch.arange(3),
                        valid,
                        saved.slot_generation,
                        valid.sum(dtype=torch.int32),
                        3,
                    )
                    return transfer.materialize_into(
                        plan, hot_binding, context.copy_stream
                    )

                restored = snapshot.restore(transition, reload)
                transition.finish(residency_ready=restored)
                torch.testing.assert_close(state.slot_to_key, old_slots)
                torch.testing.assert_close(policy_state.frequency, old_heat)
                for slot, key in enumerate(old_slots.tolist()):
                    if key >= 0:
                        for ci in range(2):
                            torch.testing.assert_close(hot[ci][slot], host[ci][key])
        self.assertGreater(hits, 0)
        self.assertGreater(misses, 0)


if __name__ == "__main__":
    unittest.main()
