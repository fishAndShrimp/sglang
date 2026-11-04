"""NPU metadata for packed decode, physical Hot slots, and whole-layer prefill."""

import triton
import triton.language as tl


@triton.jit
def prepare_packed_copies(
    Counts,
    Error,
    Enabled,
    Access,
    Sources,
    Groups,
    OutSources,
    EXPERTS: tl.constexpr,
    SOURCE_STRIDE: tl.constexpr,
    START: tl.constexpr,
    COMPONENT_COUNT: tl.constexpr,
    SKIP_OFFSETS: tl.constexpr,
    CUMULATIVE: tl.constexpr,
    BLOCK: tl.constexpr,
    COMPONENT_MAP: tl.constexpr = -1,
):
    # Exactly one program owns the fault latch and this layer's access counters.
    e = tl.arange(0, BLOCK)
    raw = tl.load(Counts + e, e < EXPERTS, 0)
    count = raw
    if CUMULATIVE:
        previous = tl.load(Counts + e - 1, (e > 0) & (e < EXPERTS), 0)
        count = raw - previous
    bad = tl.sum(((count < 0) & (e < EXPERTS)).to(tl.int32), 0) > 0
    error = tl.load(Error) | bad.to(tl.int32)
    tl.store(Error, error)
    # group_pack_copy expects per-expert counts, never cumulative boundaries.
    # Its packed output becomes GMM's count vector after the copy completes.
    tl.store(Groups + e, tl.where(error == 0, count, 0), e < EXPERTS)
    enabled = tl.load(Enabled) != 0
    accesses = tl.load(Access + START + e, e < EXPERTS, 0)
    tl.store(
        Access + START + e,
        accesses + tl.where(enabled & (error == 0), count, 0),
        e < EXPERTS,
    )
    for position in tl.static_range(COMPONENT_COUNT):
        if COMPONENT_MAP >= 0:
            component = (COMPONENT_MAP >> (4 * position)) & 15
        else:
            component = (position // 2) * 3 + position % 2 if SKIP_OFFSETS else position
        source = tl.load(
            Sources + component * SOURCE_STRIDE + START + e, e < EXPERTS, 0
        )
        tl.store(OutSources + position * EXPERTS + e, source, e < EXPERTS)


@triton.jit
def prepare_prefill_counts(
    Counts,
    Error,
    Enabled,
    Access,
    OutCounts,
    EXPERTS: tl.constexpr,
    CUMULATIVE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Keep full-layer expert order; normalize counts and observe usage once."""
    e = tl.arange(0, BLOCK)
    inside = e < EXPERTS
    count = tl.load(Counts + e, inside, 0)
    if CUMULATIVE:
        previous = tl.load(Counts + e - 1, (e > 0) & inside, 0)
        count = count - previous
    bad = tl.sum(((count < 0) & inside).to(tl.int32), 0) > 0
    error = tl.load(Error) | bad.to(tl.int32)
    tl.store(Error, error)
    count = tl.where(error == 0, count, 0)
    tl.store(OutCounts + e, count, inside)
    accesses = tl.load(Access + e, inside, 0)
    enabled = tl.load(Enabled) != 0
    tl.store(Access + e, accesses + tl.where(enabled, count, 0), inside)


@triton.jit
def prepare_fixed_copies(
    Counts,
    Error,
    Enabled,
    Access,
    Sources,
    Lengths,
    OutCounts,
    OutSources,
    OutLengths,
    EXPERTS: tl.constexpr,
    SOURCE_STRIDE: tl.constexpr,
    START: tl.constexpr,
    COMPONENT_COUNT: tl.constexpr,
    SKIP_OFFSETS: tl.constexpr,
    CUMULATIVE: tl.constexpr,
    BLOCK: tl.constexpr,
    DIRECT_HOT: tl.constexpr = False,
    DIRECT_HOT_CAPACITY: tl.constexpr = 0,
    HOT_OFFSET: tl.constexpr = 0,
    HotSlots=None,
    HOT_CAPACITY: tl.constexpr = 0,
    Groups=None,
    ExecutionKeys=None,
    KEY_START: tl.constexpr = 0,
    COMPONENT_MAP: tl.constexpr = -1,
):
    # Exactly one program owns the fault latch and this layer's access counters.
    e = tl.arange(0, BLOCK)
    raw = tl.load(Counts + e, e < EXPERTS, 0)
    count = raw
    if CUMULATIVE:
        previous = tl.load(Counts + e - 1, (e > 0) & (e < EXPERTS), 0)
        count = raw - previous
    bad = tl.sum(((count < 0) & (e < EXPERTS)).to(tl.int32), 0) > 0
    if DIRECT_HOT:
        # Counts are packed through float32 below; reject inexact integers.
        bad = bad | (tl.sum((count > 16777216).to(tl.int32), 0) > 0)
        hot_slot = tl.load(HotSlots + KEY_START + e, e < EXPERTS, -1)
        bad = bad | (
            tl.sum(((hot_slot < -1) | (hot_slot >= HOT_CAPACITY)).to(tl.int32), 0) > 0
        )
    error = tl.load(Error) | bad.to(tl.int32)
    tl.store(Error, error)
    tl.store(OutCounts + e, tl.where(error == 0, raw, 0), e < EXPERTS)
    enabled = tl.load(Enabled) != 0
    accesses = tl.load(Access + START + e, e < EXPERTS, 0)
    tl.store(
        Access + START + e,
        accesses + tl.where(enabled & (error == 0), count, 0),
        e < EXPERTS,
    )
    needed = (count > 0) & (error == 0)
    copy_needed = needed
    if DIRECT_HOT:
        hit = (hot_slot >= 0) & (hot_slot < DIRECT_HOT_CAPACITY)
        physical = tl.where(hit, HOT_OFFSET + hot_slot, e)
        position = tl.cumsum(needed.to(tl.int32), 0) - 1
        if EXPERTS <= 64:
            pair = tl.arange(0, 2 * BLOCK)
            match = (pair[:, None] // 2 == position[None, :]) & needed[None, :]
            values = tl.where(
                pair[:, None] % 2 == 0,
                physical[None, :].to(tl.float32),
                count[None, :].to(tl.float32),
            )
            packed = tl.sum(tl.where(match, values, 0), 1).to(tl.int64)
            # The remaining descriptors stay zero from initialization. Clear the
            # entire mutable prefix on every replay, including the empty case.
            tl.store(Groups + pair, packed, pair < 2 * EXPERTS)
        else:
            # The caller clears Groups on the same stream before this kernel.
            # Unique prefix positions make these stores race-free; keep UB O(E).
            tl.store(Groups + 2 * position, physical.to(tl.int64), needed)
            tl.store(Groups + 2 * position + 1, count.to(tl.int64), needed)
        copy_needed = needed & ~hit
    previous = tl.load(ExecutionKeys + e, e < EXPERTS, -1)
    copy_needed = copy_needed & (previous != KEY_START + e)
    # Hot reads leave execution identity untouched; both phases finish before reuse.
    tl.store(ExecutionKeys + e, KEY_START + e, (e < EXPERTS) & copy_needed)
    for position in tl.static_range(COMPONENT_COUNT):
        if COMPONENT_MAP >= 0:
            component = (COMPONENT_MAP >> (4 * position)) & 15
        else:
            component = (position // 2) * 3 + position % 2 if SKIP_OFFSETS else position
        source = tl.load(
            Sources + component * SOURCE_STRIDE + START + e, e < EXPERTS, 0
        )
        length = tl.load(Lengths + component * EXPERTS + e, e < EXPERTS, 0)
        tl.store(OutSources + position * EXPERTS + e, source, e < EXPERTS)
        tl.store(
            OutLengths + position * EXPERTS + e,
            tl.where(copy_needed, length, 0),
            e < EXPERTS,
        )
