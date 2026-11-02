"""NPU metadata preparation for stably packed expert copies."""

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
        component = (position // 2) * 3 + position % 2 if SKIP_OFFSETS else position
        source = tl.load(
            Sources + component * SOURCE_STRIDE + START + e, e < EXPERTS, 0
        )
        tl.store(OutSources + position * EXPERTS + e, source, e < EXPERTS)
