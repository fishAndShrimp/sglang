"""Optional NPU kernels for the fixed-slot offload path."""

import triton
import triton.language as tl


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
    ExecutionKeys=None,
    KEY_START: tl.constexpr = 0,
    TRACK_EXECUTION: tl.constexpr = False,
    SKIP_REUSED: tl.constexpr = False,
    HOT_DIRECT: tl.constexpr = False,
    HotSlots=None,
    Groups=None,
    HOT_KEY_START: tl.constexpr = 0,
    HOT_CAPACITY: tl.constexpr = 0,
    DIRECT_HOT_CAPACITY: tl.constexpr = 0,
    COMPACT_GROUPS: tl.constexpr = 0,
):
    # Exactly one program owns the fault latch and this layer's access counters.
    e = tl.arange(0, BLOCK)
    raw = tl.load(Counts + e, e < EXPERTS, 0)
    count = raw
    if CUMULATIVE:
        previous = tl.load(Counts + e - 1, (e > 0) & (e < EXPERTS), 0)
        count = raw - previous
    bad = tl.sum(((count < 0) & (e < EXPERTS)).to(tl.int32), 0) > 0
    if HOT_DIRECT:
        bad = bad | (tl.sum((count > 16777216).to(tl.int32), 0) > 0)
        hot_slot = tl.load(HotSlots + HOT_KEY_START + e, e < EXPERTS, -1)
        bad = bad | (tl.sum(((hot_slot < -1) | (hot_slot >= HOT_CAPACITY)).to(tl.int32), 0) > 0)
    error = tl.load(Error) | bad.to(tl.int32)
    tl.store(Error, error)
    tl.store(OutCounts + e, tl.where(error == 0, raw, 0), e < EXPERTS)
    enabled = tl.load(Enabled) != 0
    accesses = tl.load(Access + START + e, e < EXPERTS, 0)
    tl.store(Access + START + e, accesses + tl.where(enabled, count, 0), e < EXPERTS)
    needed = (count > 0) & (error == 0)
    copy_needed = needed
    if HOT_DIRECT:
        hit = hot_slot >= 0
        if DIRECT_HOT_CAPACITY > 0 and DIRECT_HOT_CAPACITY < HOT_CAPACITY:
            hit = hit & (hot_slot < DIRECT_HOT_CAPACITY)
        copy_needed = needed & ~hit
        physical = tl.where(hit, EXPERTS + hot_slot, e)
        # Fixed small reduction packs descriptors without integer gather/scatter.
        position = tl.cumsum(needed.to(tl.int32), 0) - 1
        if COMPACT_GROUPS > 0:
            # Device-side choice depends on active experts, not request BS.
            # Inactive descriptors must be cleared on *every* graph replay.
            if tl.sum(needed.to(tl.int32), 0) <= COMPACT_GROUPS:
                compact_pair = tl.arange(0, 2 * COMPACT_GROUPS)
                compact_match = (compact_pair[:, None] // 2 == position[None, :]) & needed[None, :]
                compact_values = tl.where(compact_pair[:, None] % 2 == 0,
                                  physical[None, :].to(tl.float32), count[None, :].to(tl.float32))
                compact_packed = tl.sum(tl.where(compact_match, compact_values, 0), 1).to(tl.int64)
                tl.store(Groups + compact_pair, compact_packed, compact_pair < 2 * EXPERTS)
                tail = tl.arange(0, 2 * BLOCK)
                tl.store(Groups + tail, 0, (tail >= 2 * COMPACT_GROUPS) & (tail < 2 * EXPERTS))
            else:
                pair = tl.arange(0, 2 * BLOCK)
                match = (pair[:, None] // 2 == position[None, :]) & needed[None, :]
                values = tl.where(pair[:, None] % 2 == 0,
                                  physical[None, :].to(tl.float32), count[None, :].to(tl.float32))
                packed_values = tl.sum(tl.where(match, values, 0), 1).to(tl.int64)
                tl.store(Groups + pair, packed_values, pair < 2 * EXPERTS)
        else:
            pair = tl.arange(0, 2 * BLOCK)
            match = (pair[:, None] // 2 == position[None, :]) & needed[None, :]
            values = tl.where(pair[:, None] % 2 == 0,
                              physical[None, :].to(tl.float32), count[None, :].to(tl.float32))
            packed_values = tl.sum(tl.where(match, values, 0), 1).to(tl.int64)
            tl.store(Groups + pair, packed_values, pair < 2 * EXPERTS)
    if TRACK_EXECUTION:
        if SKIP_REUSED:
            previous_key = tl.load(ExecutionKeys + e, e < EXPERTS, -1)
            copy_needed = copy_needed & (previous_key != KEY_START + e)
        # Consumers run after both weight phases complete on the compute stream.
        # Direct Hot reads leave execution storage and its identity untouched.
        tl.store(ExecutionKeys + e, KEY_START + e, (e < EXPERTS) & copy_needed)
    for position in tl.static_range(COMPONENT_COUNT):
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

@triton.jit
def weighted_local_sum(
    Y,
    RowMap,
    Weights,
    Ids,
    Out,
    HIDDEN: tl.constexpr,
    TOPK: tl.constexpr,
    EXPERTS: tl.constexpr,
    Y_ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    h = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    total = tl.full((BLOCK,), 0, tl.float32)
    for k in tl.static_range(TOPK):
        route = token * TOPK + k
        expert = tl.load(Ids + route)
        row = tl.load(RowMap + route)
        weight = tl.load(Weights + route).to(tl.float32)
        valid = (expert >= 0) & (expert < EXPERTS) & (row >= 0) & (row < Y_ROWS)
        value = tl.load(Y + row * HIDDEN + h, valid & (h < HIDDEN), 0).to(tl.float32)
        total = total + value * weight
    tl.store(Out + token * HIDDEN + h, total, h < HIDDEN)
