"""Fixed-slot Host-to-Hot maintenance; preserve residency transactions."""

import torch
import triton
import triton.language as tl

from .transfer import (
    _group_pack_copy,
    _npu_completion,
    _submission_failure,
    _ticket,
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


@triton.jit
def validate_transfer_tiles(Kind, Index, Destination, Valid, Bad, Counts,
                            H: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = tl.load(Valid + i, i < H, 0) != 0
    kind = tl.load(Kind + i, i < H, 0)
    index = tl.load(Index + i, i < H, 0)
    dst = tl.load(Destination + i, i < H, 0)
    bad = (dst != i) | (valid & ((kind != 1) | (index < 0) | (index >= N)))
    tl.store(Bad + tl.program_id(0), tl.sum(((i < H) & bad).to(tl.int32), 0))
    tl.store(Counts + tl.program_id(0), tl.sum(valid.to(tl.int32), 0))


@triton.jit
def validate_transfer_total(Bad, Counts, Count, Error, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    bad = tl.sum(tl.load(Bad + i, i < PARTS, 0), 0) > 0
    count = tl.sum(tl.load(Counts + i, i < PARTS, 0), 0)
    tl.store(Error, tl.load(Error) | (bad | (count != tl.load(Count))).to(tl.int32))


@triton.jit
def fill_transfer_tiles(Index, Valid, Error, HostPointers, Sizes, Sources, Lengths,
                        H: tl.constexpr, N: tl.constexpr, COMPONENTS: tl.constexpr,
                        BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    selected = (tl.load(Valid + i, i < H, 0) != 0) & (tl.load(Error) == 0)
    index = tl.load(Index + i, i < H, 0)
    safe = tl.where(selected, index + 1, 0)
    for component in tl.static_range(COMPONENTS):
        source = tl.load(HostPointers + component * (N + 1) + safe, i < H, 0)
        length = tl.load(Sizes + component)
        tl.store(Sources + component * H + i, source, i < H)
        tl.store(Lengths + component * H + i, tl.where(selected, length, 0), i < H)


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
        if bindings.capacity <= 0:
            raise ValueError("Cache transfer requires positive capacity")
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
        self.parts = triton.cdiv(capacity, 256)
        from .capabilities import SINGLE_PROGRAM_HOT_LIMIT
        self.tiled = capacity > SINGLE_PROGRAM_HOT_LIMIT
        if self.tiled:
            self.partial_bad = torch.empty(self.parts, dtype=torch.int32, device=device)
            self.partial_count = torch.empty_like(self.partial_bad)
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
                if self.tiled:
                    validate_transfer_tiles[(self.parts,)](
                        plan.source_kind, plan.source_index, plan.destination_slot,
                        plan.valid, self.partial_bad, self.partial_count,
                        H=bindings.capacity, N=bindings.host_count, BLOCK=256,
                    )
                    validate_transfer_total[(1,)](
                        self.partial_bad, self.partial_count, plan.count, bindings.error_status,
                        PARTS=self.parts, BLOCK=triton.next_power_of_2(self.parts),
                    )
                    fill_transfer_tiles[(self.parts,)](
                        plan.source_index, plan.valid, bindings.error_status,
                        self.hosts, self.sizes, self.sources, self.lengths,
                        H=bindings.capacity, N=bindings.host_count,
                        COMPONENTS=len(bindings.tables), BLOCK=256,
                    )
                else:
                    self._prepare_small(plan, bindings)
                for descriptor in self.descriptors:
                    status = _group_pack_copy(*descriptor, bindings.device)
                    if status is not None and status != 0:
                        raise RuntimeError(f"Cache copy submission failed: {status}")
                return _ticket(_npu_completion, stream, resources, self.retain)
            except Exception as error:
                raise _submission_failure(
                    error, _npu_completion, stream, resources, self.retain
                ) from error

    def _prepare_small(self, plan, bindings):
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
