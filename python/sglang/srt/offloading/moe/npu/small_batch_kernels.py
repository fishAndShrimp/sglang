"""NPU kernels for the fixed-slot offload path."""

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
    tl.store(OutCounts + e, tl.where(error == 0, raw, 0), e < EXPERTS)
    enabled = tl.load(Enabled) != 0
    accesses = tl.load(Access + START + e, e < EXPERTS, 0)
    tl.store(Access + START + e, accesses + tl.where(enabled, count, 0), e < EXPERTS)
    needed = (count > 0) & (error == 0)
    for position in tl.static_range(COMPONENT_COUNT):
        component = (position // 2) * 3 + position % 2 if SKIP_OFFSETS else position
        source = tl.load(
            Sources + component * SOURCE_STRIDE + START + e, e < EXPERTS, 0
        )
        length = tl.load(Lengths + component * EXPERTS + e, e < EXPERTS, 0)
        tl.store(OutSources + position * EXPERTS + e, source, e < EXPERTS)
        tl.store(
            OutLengths + position * EXPERTS + e,
            tl.where(needed, length, 0),
            e < EXPERTS,
        )
