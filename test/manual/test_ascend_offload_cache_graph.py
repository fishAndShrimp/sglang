"""NPU oracle: captured cache maintenance must match eager state and Hot bytes."""

import os

import torch

from sglang.srt.offloading.moe.config import ExpertCatalog, ExpertLayerSpec
from sglang.srt.offloading.moe.execution import (
    CacheUpdateCoordinator,
    ExpertWeightAccess,
)
from sglang.srt.offloading.moe.npu.ascend_ops import AscendOpsOffloader, _Context
from sglang.srt.offloading.moe.npu.native_acl import get_native_acl
from sglang.srt.offloading.moe.npu.storage import MemFabricHostPool
from sglang.srt.offloading.moe.npu.transfer import (
    DeviceMemoryRegion,
    MemFabricAddressCatalog,
    MemFabricDeviceTransferOps,
    MemFabricTransferBindings,
)
from sglang.srt.offloading.moe.periodic_lfu import PeriodicLFUConfig, PeriodicLFUPolicy
from sglang.srt.offloading.moe.residency import TensorResidencyOps
from sglang.srt.offloading.moe.state import ResidencyState
from sglang.srt.offloading.moe.storage import HostMemoryCapability


def main():
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    n = int(os.environ.get("MOE_TEST_CACHE_EXPERTS", 32))
    capacity = int(os.environ.get("MOE_TEST_CACHE_SLOTS", 8))
    size = 1024
    native = get_native_acl(os.environ["MOE_NATIVE_ACL_DIR"])
    pool = MemFabricHostPool(6 * n * size, 0, physical_bytes=1024**3, native_acl=native)
    host = pool.allocate(6 * n * size).tensor.view(6, n, size)
    names = tuple(str(c) for c in range(6))
    initial_keys = torch.randperm(n, generator=torch.Generator().manual_seed(1234))[
        :capacity
    ]
    # Large catalogs must not alias every 251 experts. Give each component/key
    # a unique eight-byte identity and varying payload to detect partial copies.
    host.copy_(
        torch.randint(
            0,
            256,
            host.shape,
            dtype=torch.uint8,
            generator=torch.Generator().manual_seed(317),
        )
    )
    host[:, :, :8].copy_(
        torch.arange(6 * n, dtype=torch.int64).view(torch.uint8).reshape(6, n, 8)
    )
    hosts = {
        name: tuple(
            HostMemoryCapability(
                host[c, e].data_ptr(),
                size,
                True,
                pool.capability.device_pointer(host[c, e].data_ptr(), size),
                pool,
            )
            for e in range(n)
        )
        for c, name in enumerate(names)
    }

    def make(fused=False):
        owner = object.__new__(AscendOpsOffloader)
        owner.device, owner.cfg, owner.hot_slots = (
            device,
            {},
            capacity,
        )
        owner.catalog = ExpertCatalog((ExpertLayerSpec(0, n, 1, 0),))
        owner.error = torch.zeros((), dtype=torch.int32, device=device)
        owner.context = _Context()
        hot = torch.zeros((6, capacity, size), dtype=torch.uint8, device=device)
        regions = {
            name: tuple(
                DeviceMemoryRegion.from_tensor(hot[c, e]) for e in range(capacity)
            )
            for c, name in enumerate(names)
        }
        owner.addresses = MemFabricAddressCatalog(
            hosts, regions, {name: size for name in names}, device=device
        )
        owner.sources = torch.stack(
            [torch.cat(pair) for pair in owner.addresses.source_tables]
        )
        owner.state = ResidencyState.allocate(n, capacity, n, device)
        bindings = MemFabricTransferBindings(
            owner.addresses,
            regions,
            capacity=capacity,
            hot_generations=owner.state.slot_generation,
            error_status=owner.error,
        )
        owner.residency = TensorResidencyOps(
            owner.state,
            owner.catalog,
            execution_waves=[],
            update_bindings=bindings,
            error_status=owner.error,
        )
        device_ops = MemFabricDeviceTransferOps(retain=owner._retain)
        if fused:
            from sglang.srt.offloading.moe.npu.cache_transfer import (
                FixedSlotCacheTransfer,
            )

            device_ops = FixedSlotCacheTransfer(bindings, retain=owner._retain)
        owner.access = ExpertWeightAccess(owner.residency, device_ops=device_ops)
        owner.policy = PeriodicLFUPolicy(
            PeriodicLFUConfig(None, max_swaps=4), owner.catalog
        )
        owner.policy_state = owner.policy.allocate_state(n, device)
        owner.target_keys = initial_keys.to(device)
        owner.due = torch.zeros((), dtype=torch.bool, device=device)
        owner.coordinator = CacheUpdateCoordinator(
            owner.policy,
            owner.policy_state,
            owner.state,
            owner.target_keys,
            owner.access,
            due_out=owner.due,
        )
        owner.state.update_enabled.fill_(True)
        owner.access.apply_update(
            owner.target_keys, owner.context
        ).completion.synchronize()
        owner._publish_sources()
        owner.state.update_enabled.fill_(False)
        if fused:
            from sglang.srt.offloading.moe.npu.cache_update import FusedCacheUpdate

            owner.cache_update = FusedCacheUpdate(owner)

        def check():
            assert owner.error.item() == 0

        owner._check_error = check
        return owner, hot

    eager, eager_hot = make()
    captured, captured_hot = make(fused=True)
    captured._capture_cache_tails()
    assert captured.policy_state.step_mod.item() == 0
    assert not captured.policy_state.usage.any().item()
    fields = ("key_to_slot", "slot_to_key", "slot_valid", "slot_generation")
    initial = captured.state.slot_to_key.cpu().clone()
    for step in range(1, 25):
        counts = torch.zeros(n, dtype=torch.int64, device=device)
        start = (capacity + step * 17) % (n - 4)
        counts[start : start + 4] = step + 3
        maintenance = step % 4 == 0
        for owner in (eager, captured):
            owner.context = _Context()
            owner.state.step_access_count.copy_(counts)
            owner.state.update_enabled.fill_(True)
        eager.coordinator.emit_step_end(eager.context, maintenance=maintenance)
        if maintenance:
            eager._publish_sources()
        captured.cache_tail_graphs[maintenance].replay()
        torch.npu.synchronize()
        for field in fields:
            assert torch.equal(
                getattr(eager.state, field), getattr(captured.state, field)
            ), (step, field)
        for field in ("usage", "step_mod", "update_due"):
            assert torch.equal(
                getattr(eager.policy_state, field),
                getattr(captured.policy_state, field),
            ), (step, field)
        assert torch.equal(eager_hot, captured_hot), step
        keys = captured.state.slot_to_key.cpu()
        assert torch.equal(captured_hot.cpu(), host[:, keys]), step
        assert eager.error.item() == captured.error.item() == 0
        eager.context.resources.clear()
    assert not torch.equal(initial, captured.state.slot_to_key.cpu())
    saved_hot = captured_hot.clone()
    saved_state = {field: getattr(captured.state, field).clone() for field in fields}
    saved_target = captured.target_keys.clone()
    captured.due.fill_(True)
    captured.state.update_enabled.fill_(True)
    for fault in ("duplicate_target", "target_range", "stale_inverse", "stale_slot"):
        captured.error.zero_()
        for field, value in saved_state.items():
            getattr(captured.state, field).copy_(value)
        captured.target_keys.copy_(saved_target)
        if fault == "duplicate_target":
            captured.target_keys[1] = captured.target_keys[0]
        elif fault == "target_range":
            captured.target_keys[-1] = n + 1
        elif fault == "stale_inverse":
            captured.state.key_to_slot[0] = capacity + 1
        else:
            captured.state.slot_to_key[0] = n + 1
        captured.cache_update.apply()
        torch.npu.synchronize()
        assert captured.error.item() != 0, fault
        assert torch.equal(saved_hot, captured_hot), fault
    for field, value in saved_state.items():
        getattr(captured.state, field).copy_(value)
    captured.target_keys.copy_(saved_target)
    captured.error.zero_()
    # Test transport faults separately from the residency planner. The
    # native copy must see zero lengths before any invalid pointer is used.
    plan = captured.residency._tail["plan"]
    saved = captured_hot.clone()
    for fault in ("source", "destination", "kind", "count", "latched"):
        captured.error.zero_()
        plan.valid.fill_(True)
        plan.source_kind.fill_(1)
        plan.source_index.copy_(captured.state.slot_to_key)
        plan.destination_slot.copy_(torch.arange(capacity, device=device))
        plan.count.fill_(capacity)
        if fault == "source":
            plan.source_index[0] = n + 20
        elif fault == "destination":
            plan.destination_slot[1] = 0
        elif fault == "kind":
            plan.source_kind[0] = 2
        elif fault == "count":
            plan.count.fill_(capacity - 1)
        else:
            captured.error.fill_(1)
        captured.access.device_ops.materialize_into(
            plan, captured.residency.update_bindings, torch.npu.current_stream()
        ).completion.synchronize()
        assert captured.error.item() != 0, fault
        assert torch.equal(saved, captured_hot), fault
    print(
        "PASS: 24 dynamic steps, six maintenance rounds, graph/eager LFU state, mapping, generations and all Hot bytes identical",
        flush=True,
    )


if __name__ == "__main__":
    main()
