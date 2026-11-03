"""Optional fixed-slot Host-to-Hot maintenance; preserve residency transactions."""

import torch
import triton
import triton.language as tl

from .transfer import (
    _group_pack_copy,
    _submission_failure,
    _ticket,
    _npu_completion,
)


@triton.jit
def prepare_cache_transfer(
    Kind,
    Index,
    Destination,
    Valid,
    Count,
    Error,
    HostPointers,
    Sizes,
    Sources,
    Lengths,
    CAPACITY: tl.constexpr,
    HOST_COUNT: tl.constexpr,
    COMPONENTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    slot = tl.arange(0, BLOCK)
    inside = slot < CAPACITY
    valid = tl.load(Valid + slot, inside, 0) != 0
    kind = tl.load(Kind + slot, inside, 0)
    index = tl.load(Index + slot, inside, 0)
    destination = tl.load(Destination + slot, inside, 0)
    # This adapter accepts only Host sources and an injective fixed destination
    # plan. It cannot copy from Hot into aliased Hot storage.
    bad_source = valid & ((kind != 1) | (index < 0) | (index >= HOST_COUNT))
    bad = tl.sum((inside & (bad_source | (destination != slot))).to(tl.int32), 0) > 0
    bad = bad | (tl.load(Count) != tl.sum(valid.to(tl.int32), 0))
    error = tl.load(Error) | bad.to(tl.int32)
    tl.store(Error, error)
    selected = valid & (error == 0)
    safe_index = tl.where(selected, index + 1, 0)
    for component in tl.static_range(COMPONENTS):
        source = tl.load(
            HostPointers + component * (HOST_COUNT + 1) + safe_index, inside, 0
        )
        length = tl.load(Sizes + component)
        tl.store(Sources + component * CAPACITY + slot, source, inside)
        tl.store(
            Lengths + component * CAPACITY + slot, tl.where(selected, length, 0), inside
        )


class FixedSlotCacheTransfer:
    """A private descriptor arena for one immutable maintenance binding.

    Fixed all-one groups map native source i to destination i. Inactive entries
    have zero byte length. Every component retains its own buffers; subsequent
    transactions reuse them only after the owner's stream completion boundary.
    """

    def __init__(self, bindings, *, retain):
        if (
            bindings.capacity != bindings.destination_count
            or not bindings.hot_targets_alias
        ):
            raise ValueError("Cache transfer requires a complete fixed-slot Hot bank")
        if not 0 < bindings.capacity <= 4096:
            raise ValueError("Fused cache transfer supports 1..4096 Hot slots")
        self.bindings, self.retain = bindings, retain
        capacity = bindings.capacity
        device = bindings.device
        components = len(bindings.tables)
        self.hosts = torch.stack([table[0] for table in bindings.tables])
        self.sizes = torch.tensor(
            list(bindings.address_catalog.component_nbytes.values()),
            dtype=torch.uint32,
            device=device,
        )
        self.sources = torch.empty(
            (components, capacity), dtype=torch.int64, device=device
        )
        self.lengths = torch.empty(
            (components, capacity), dtype=torch.uint32, device=device
        )
        self.descriptors = []
        for component, tables in enumerate(bindings.tables):
            for begin in range(0, capacity, bindings.max_batch_entries):
                end = min(capacity, begin + bindings.max_batch_entries)
                n = end - begin
                self.descriptors.append(
                    (
                        self.sources[component, begin:end],
                        tables[2][begin + 1 : end + 1],
                        self.lengths[component, begin:end],
                        torch.tensor(n, dtype=torch.uint32, device=device),
                        torch.ones(n, dtype=torch.int64, device=device),
                        torch.zeros(n, dtype=torch.int64, device=device),
                    )
                )

    def materialize_into(self, plan, bindings, stream):
        if bindings is not self.bindings:
            raise ValueError("Cache transfer cannot change its fixed binding")
        if torch.device(stream.device) != bindings.device:
            raise ValueError("Cache transfer stream belongs to another device")
        bindings.validate_plan(plan)
        resources = (self, plan, bindings)
        self.retain(resources)
        with torch.npu.stream(stream):
            try:
                prepare_cache_transfer[(1,)](
                    plan.source_kind,
                    plan.source_index,
                    plan.destination_slot,
                    plan.valid,
                    plan.count,
                    bindings.error_status,
                    self.hosts,
                    self.sizes,
                    self.sources,
                    self.lengths,
                    CAPACITY=bindings.capacity,
                    HOST_COUNT=bindings.host_count,
                    COMPONENTS=len(bindings.tables),
                    BLOCK=triton.next_power_of_2(bindings.capacity),
                )
                for descriptor in self.descriptors:
                    status = _group_pack_copy(*descriptor, bindings.device)
                    if status is not None and status != 0:
                        raise RuntimeError(f"Cache copy submission failed: {status}")
                return _ticket(_npu_completion, stream, resources, self.retain)
            except Exception as error:
                raise _submission_failure(
                    error, _npu_completion, stream, resources, self.retain
                ) from error
