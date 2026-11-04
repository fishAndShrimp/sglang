"""Publish a large catalog's immutable per-forward source table in bounded tiles."""

import torch
import triton
import triton.language as tl


@triton.jit
def publish_tiles(K2S, S2K, Valid, Sources, Output, Bad,
                  N: tl.constexpr, H: tl.constexpr, C: tl.constexpr,
                  STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    key = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inside = key < N
    slot = tl.load(K2S + key, inside, -1)
    safe = tl.minimum(tl.maximum(slot, 0), H - 1)
    hit = slot >= 0
    valid = tl.load(Valid + safe) != 0
    reverse = tl.load(S2K + safe)
    bad = (slot < -1) | (slot >= H) | (hit & (~valid | (reverse != key)))
    tl.store(Bad + tl.program_id(0), tl.sum((inside & bad).to(tl.int32), 0))
    # Both source banks contain a sentinel. Clamp before gathering even when
    # corruption has been detected; the whole-forward check retires the fault.
    index = tl.where(hit, N + 2 + safe, key + 1)
    for c in tl.static_range(C):
        value = tl.load(Sources + c * STRIDE + index, inside, 0)
        tl.store(Output + c * N + key, value, inside)


@triton.jit
def latch_publish_error(Bad, Error, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    bad = tl.sum(tl.load(Bad + i, i < PARTS, 0), 0) > 0
    tl.store(Error, tl.load(Error) | bad.to(tl.int32))


class TiledSourcePublisher:
    def __init__(self, owner):
        self.owner = owner
        self.n, self.h = owner.catalog.num_experts, owner.hot_slots
        self.parts = triton.cdiv(self.n, 256)
        self.reduction_block = triton.next_power_of_2(self.parts)
        self.bad = torch.empty(self.parts, dtype=torch.int32, device=owner.device)
        self.output = torch.empty(
            (owner.sources.shape[0], self.n), dtype=owner.sources.dtype,
            device=owner.device,
        )

    def run(self):
        o = self.owner
        publish_tiles[(self.parts,)](
            o.state.key_to_slot, o.state.slot_to_key, o.state.slot_valid,
            o.sources, self.output, self.bad,
            N=self.n, H=self.h, C=o.sources.shape[0],
            STRIDE=o.sources.stride(0), BLOCK=256,
        )
        latch_publish_error[(1,)](
            self.bad, o.error, PARTS=self.parts, BLOCK=self.reduction_block,
        )
