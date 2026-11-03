"""Candidate packer for small offload batches with an authoritative live mask.

Only the input prefix is copied. Padding is intentionally not touched: callers
must supply a device live mask covering the entire MC2 padded allocation and
must never mark rows outside the current prefix active. No routing changes.
"""
import triton
import triton.language as tl


@triton.jit
def _pack_mc2_prefix(
    X, IDs, Weights, OutX, OutIDs, OutWeights,
    ROWS: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
    XS0: tl.constexpr, XS1: tl.constexpr,
    IS0: tl.constexpr, IS1: tl.constexpr,
    WS0: tl.constexpr, WS1: tl.constexpr,
    BLOCK: tl.constexpr, TOPK_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    per_row: tl.constexpr = triton.cdiv(H, BLOCK)
    row = pid // per_row
    part = pid % per_row
    x = part * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + row * XS0 + x * XS1, (row < ROWS) & (x < H), 0)
    tl.store(OutX + row * H + x, value, (row < ROWS) & (x < H))
    # No division by non-power-of-two top-k and no scalar control-flow branch.
    r = tl.arange(0, TOPK_BLOCK)
    active = (part == 0) & (row < ROWS) & (r < K)
    ids = tl.load(IDs + row * IS0 + r * IS1, active, 0)
    weights = tl.load(Weights + row * WS0 + r * WS1, active, 0)
    tl.store(OutIDs + row * K + r, ids, active)
    tl.store(OutWeights + row * K + r, weights, active)


def pack_prefix(x, ids, weights, out_x, out_ids, out_weights):
    rows, width = x.shape
    k = ids.shape[1]
    if not (0 < rows <= 64 and 0 < k <= 32):
        raise ValueError("Candidate supports 1..64 physical input rows and topk <=32")
    if any(not t.is_contiguous() for t in (out_x, out_ids, out_weights)):
        raise ValueError("Padded output buffers must be contiguous")
    if ids.shape != weights.shape or ids.shape[0] != rows:
        raise ValueError("Routing shape mismatch")
    if out_x.shape[0] < rows or out_x.shape[1] != width:
        raise ValueError("Hidden output shape mismatch")
    if out_ids.shape != out_weights.shape or out_ids.shape != (out_x.shape[0], k):
        raise ValueError("Routing output shape mismatch")
    _pack_mc2_prefix[(rows * triton.cdiv(width, 2048),)](
        x, ids, weights, out_x, out_ids, out_weights,
        ROWS=rows, H=width, K=k,
        XS0=x.stride(0), XS1=x.stride(1), IS0=ids.stride(0), IS1=ids.stride(1),
        WS0=weights.stride(0), WS1=weights.stride(1),
        BLOCK=2048, TOPK_BLOCK=triton.next_power_of_2(k),
    )
