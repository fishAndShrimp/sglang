"""NPU integration check for packed expert copies and dynamic graph replay.

Run on an idle NPU with the same MemFabric / native ACL environment as serving.
Set MOE_NATIVE_ACL_DIR to the already-built native extension directory.
"""

import os
from types import SimpleNamespace

import torch

from sglang.srt.offloading.moe.npu.ascend_ops import AscendOpsOffloader, _Context
from sglang.srt.offloading.moe.npu.native_acl import get_native_acl
from sglang.srt.offloading.moe.npu.storage import MemFabricHostPool
from sglang.srt.offloading.moe.state import ResidencyState


def main():
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    from sglang.srt.model_loader.loader import device_loading_context

    module = torch.nn.Module()
    module.register_parameter(
        "weight",
        torch.nn.Parameter(torch.arange(16, dtype=torch.float32), requires_grad=False),
    )
    module._moe_offload_storage = object()
    original = module.weight.data_ptr()
    with device_loading_context(module, device):
        assert module.weight.device.type == "cpu"
        module.weight.add_(100)
    assert module.weight.data_ptr() == original
    assert torch.equal(module.weight, torch.arange(16, dtype=torch.float32) + 100)
    n, h, components, size = int(os.environ.get("MOE_TEST_EXPERTS", "64")), 2, 6, 1024
    if not 8 <= n <= 256:
        raise ValueError("MOE_TEST_EXPERTS must be in [8, 256]")
    names = tuple(
        f"{prefix}_{suffix}"
        for prefix in ("w13", "w2")
        for suffix in ("weight", "weight_scale", "weight_offset")
    )
    native = get_native_acl(os.environ["MOE_NATIVE_ACL_DIR"])
    pool = MemFabricHostPool(
        components * n * size, 0, physical_bytes=1024**3, native_acl=native
    )
    host = pool.allocate(components * n * size).tensor.view(components, n, size)
    for c in range(components):
        for e in range(n):
            host[c, e].fill_(
                0 if names[c].endswith("_offset") else (10 * c + e + 1) % 255
            )
    hot = host[:, [2, 6]].to(device)
    destination = torch.zeros_like(host, device=device)
    owner = object.__new__(AscendOpsOffloader)
    owner.device, owner.ready, owner.local_experts, owner.hot_slots = device, True, n, h
    owner.copy_capacity = n
    owner.direct_hot, owner.execution_keys = False, None
    owner.context = _Context()
    owner.catalog = SimpleNamespace(
        num_experts=n,
        key=lambda layer, e: e,
        offloaded_layers=[SimpleNamespace(layer_id=0)],
    )
    owner.state = ResidencyState.allocate(n, h, n, device)
    owner.state.key_to_slot[2] = 0
    owner.state.key_to_slot[6] = 1
    owner.state.slot_to_key.copy_(torch.tensor([2, 6], device=device))
    owner.state.slot_valid.fill_(True)
    owner.state.update_enabled.fill_(True)
    owner.error = torch.zeros((), dtype=torch.int32, device=device)
    owner.sources = torch.tensor(
        [
            [0]
            + [
                pool.capability.device_pointer(host[c, e].data_ptr(), size)
                for e in range(n)
            ]
            + [0]
            + [hot[c, e].data_ptr() for e in range(h)]
            for c in range(components)
        ],
        dtype=torch.int64,
        device=device,
    )
    owner.destinations = torch.tensor(
        [[destination[c, e].data_ptr() for e in range(n)] for c in range(components)],
        dtype=torch.int64,
        device=device,
    )
    owner.lengths = torch.tensor(
        [[size] * n for _ in range(components)], dtype=torch.int32, device=device
    )
    owner.quant_info = object()
    owner.execution = SimpleNamespace(
        tensors={name: destination[c] for c, name in enumerate(names)}
    )
    owner._init_copy_plan(names)
    owner._publish_sources()
    descriptors = [d for phase in owner.fused_descriptors for d in phase]
    assert len(descriptors) == 4
    assert len({d[-1].data_ptr() for d in descriptors}) == 4
    assert all(d[0].numel() == n and d[3].item() == n for d in descriptors)
    assert all(d[2].dtype == d[3].dtype == torch.int32 for d in descriptors)
    # Read GMM's counts on the compute stream before the W2 wait. W2's packed
    # output buffers must not overwrite the count vector used by GMM1.
    observed_counts = torch.empty(n, dtype=torch.int64, device=device)

    def run():
        owner.prepare_weights(0, inp)
        observed_counts.copy_(inp.expert_tokens)
        owner.wait_for_w2(0)

    def reset_destination():
        destination.fill_(255)
        destination[2].zero_()
        destination[5].zero_()

    counts = torch.zeros(n, dtype=torch.int64, device=device)
    inp = SimpleNamespace(expert_tokens=counts, group_list_type=1)

    def verify(values, run, *, cumulative=False):
        inp.expert_tokens = counts
        inp.group_list_type = 0 if cumulative else 1
        reset_destination()
        raw = torch.tensor(values, device=device)
        counts.copy_(raw.cumsum(0) if cumulative else raw)
        before = owner.state.step_access_count.cpu().clone()
        run()
        torch.npu.synchronize()
        assert torch.equal(
            owner.state.step_access_count.cpu(), before + torch.tensor(values)
        )
        active = [e for e, count in enumerate(values) if count]
        packed = torch.tensor(
            [values[e] for e in active] + [0] * (n - len(active)),
            dtype=torch.int64,
        )
        assert torch.equal(owner.packed_counts.cpu(), packed)
        assert torch.equal(observed_counts.cpu(), packed)
        assert inp.group_list_type == 1
        actual = destination.cpu()
        for slot in range(n):
            expected = (
                host[:, active[slot]]
                if slot < len(active)
                else torch.full_like(host[:, 0], 255)
            )
            expected = expected.clone()
            expected[2].zero_()
            expected[5].zero_()
            assert torch.equal(actual[:, slot], expected), (values, slot)
        assert owner.error.item() == 0
        if active:
            # Exercise the real grouped-matmul consumer with packed weights and
            # counts. Token rows retain original expert order; zero groups at
            # the tail must never make GMM read stale execution weights.
            x = torch.ones((sum(values), 32), dtype=torch.bfloat16, device=device)
            for component in (0, 3):
                weight = destination[component].view(n, 32, 32).to(torch.bfloat16)
                weight.mul_(1 / 256)
                output = torch.ops.npu.npu_grouped_matmul(
                    x=[x],
                    weight=[weight],
                    split_item=2,
                    group_list_type=1,
                    group_type=0,
                    group_list=owner.packed_counts,
                    output_dtype=torch.bfloat16,
                )[0]
                expected = (
                    torch.tensor(
                        [
                            float(host[component, expert, 0]) / 8
                            for expert, count in enumerate(values)
                            for _ in range(count)
                        ],
                        dtype=torch.bfloat16,
                    )
                    .unsqueeze(1)
                    .expand(-1, 32)
                )
                torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)

    cases = [
        [0] * n,
        [0, 3, 1, 0, 0, 0, 7, 0] + [0] * (n - 8),
        [1] * n,
        [0] * n,  # Dense -> empty must clear all packed count tails.
        [9, 0, 0, 0, 1, 0, 0, 1] + [0] * (n - 8),
        [0] * (n - 1) + [7],
    ]
    for values in cases:
        verify(values, run)
    for values in cases:
        verify(values, run, cumulative=True)
    inp.group_list_type = 1
    inp.expert_tokens = counts
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
    for values in cases:
        verify(values, graph.replay)
    stable_address = owner.published_sources.data_ptr()
    # Publish a real cache replacement after graph capture. Replay must
    # observe the new mapping through the same table storage.
    hot[:, 0].copy_(host[:, 4].to(device))
    owner.state.key_to_slot[2] = -1
    owner.state.key_to_slot[4] = 0
    owner.state.slot_to_key[0] = 4
    owner._publish_sources()
    assert owner.published_sources.data_ptr() == stable_address
    verify([1] * n, graph.replay)
    owner.state.slot_to_key[0] = 5  # stale reverse map must latch a fault
    owner._publish_sources()
    assert owner.error.item() != 0
    owner.state.slot_to_key[0] = 4
    owner.error.zero_()
    owner._publish_sources()
    inp.expert_tokens = torch.tensor(
        [-1] + [0] * (n - 1), dtype=torch.int64, device=device
    )
    reset_destination()
    run()
    torch.npu.synchronize()
    assert owner.error.item() != 0
    assert not inp.expert_tokens.any().item()
    assert not observed_counts.any().item()
    for c, name in enumerate(names):
        assert destination[c].eq(0 if name.endswith("_offset") else 255).all().item()
    print(
        "PASS: load-context ownership, packed H2D/Hot D2D weights/scales, GMM row order, zero offsets, empty rank, dense-to-empty tails, last expert, cumulative counts, changing graph replay routes, invalid-count masking",
        flush=True,
    )
    import triton

    from sglang.srt.offloading.moe.npu.small_batch_kernels import (
        prepare_fixed_copies,
        prepare_packed_copies,
    )

    width = n * 3
    table = torch.arange(6 * width, dtype=torch.int64, device=device).reshape(6, width)
    access = torch.zeros(width, dtype=torch.int64, device=device)
    raw = torch.arange(n, dtype=torch.int64, device=device) % 4
    out_counts = torch.empty_like(raw)
    out_sources = torch.empty(4 * n, dtype=torch.int64, device=device)
    fault = torch.zeros_like(owner.error)
    enabled = torch.ones((), dtype=torch.bool, device=device)
    prepare_packed_copies[(1,)](
        raw,
        fault,
        enabled,
        access[n : 2 * n],
        table[:, n : 2 * n],
        out_counts,
        out_sources,
        EXPERTS=n,
        SOURCE_STRIDE=width,
        START=0,
        COMPONENT_COUNT=4,
        SKIP_OFFSETS=True,
        CUMULATIVE=False,
        BLOCK=triton.next_power_of_2(n),
    )
    torch.npu.synchronize()
    assert torch.equal(
        out_sources.reshape(4, n).cpu(), table[[0, 1, 3, 4], n : 2 * n].cpu()
    )
    assert torch.equal(access[n : 2 * n], raw)
    assert torch.equal(out_counts, raw)
    assert not access[:n].any().item() and not access[2 * n :].any().item()
    print("PASS: nonzero layer offset and strided published source table", flush=True)
    # Physical Hot reads must not overwrite another layer's execution key.
    raw.fill_(1)
    fault.zero_()
    slots = torch.full((width,), -1, dtype=torch.int64, device=device)
    slots[2], slots[6] = 0, 1
    keys = torch.full((n,), -1, dtype=torch.int64, device=device)
    groups = torch.zeros((n + h, 2), dtype=torch.int64, device=device)
    fixed_counts = torch.empty_like(raw)
    fixed_sources = torch.empty(4 * n, dtype=torch.int64, device=device)
    fixed_lengths = torch.empty(4 * n, dtype=torch.uint32, device=device)
    graphs = {}
    for layer in (0, 1):
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            prepare_fixed_copies[(1,)](
                raw,
                fault,
                enabled,
                access[layer * n : (layer + 1) * n],
                table[:, layer * n : (layer + 1) * n],
                owner.lengths,
                fixed_counts,
                fixed_sources,
                fixed_lengths,
                EXPERTS=n,
                SOURCE_STRIDE=width,
                START=0,
                COMPONENT_COUNT=4,
                SKIP_OFFSETS=True,
                CUMULATIVE=False,
                BLOCK=triton.next_power_of_2(n),
                DIRECT_HOT=True,
                HotSlots=slots,
                HOT_CAPACITY=h,
                Groups=groups,
                ExecutionKeys=keys,
                KEY_START=layer * n,
            )
        graphs[layer] = graph
    torch.npu.synchronize()
    keys.fill_(-1)
    # Initial miss, same-layer reuse, cross-layer miss, Hot reads preserving
    # layer one's slots, then reuse of those two preserved slots by layer one.
    for layer, copies in ((0, n - 2), (0, 0), (1, n), (0, n - 2), (1, n - 2)):
        graphs[layer].replay()
        torch.npu.synchronize()
        lengths = fixed_lengths.reshape(4, n).cpu().to(torch.int64)
        assert int((lengths[0] > 0).sum()) == copies
        assert torch.equal(lengths, lengths[:1].expand_as(lengths))
        assert torch.equal(fixed_counts, raw)
        assert torch.equal(
            fixed_sources.reshape(4, n).cpu(),
            table[[0, 1, 3, 4], layer * n : (layer + 1) * n].cpu(),
        )
        if layer == 0:
            assert groups[2, 0].item() == n and groups[6, 0].item() == n + 1
            assert lengths[:, [2, 6]].eq(0).all()
        else:
            assert torch.equal(keys.cpu(), torch.arange(n) + n)
        assert fault.item() == 0
    print("PASS: Hot slot addressing and cross-layer execution reuse", flush=True)
    # Pool lifetime deliberately extends beyond graph lifetime until process exit.


if __name__ == "__main__":
    main()
