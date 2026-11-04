"""Bounded per-program decode metadata for large rank-local expert banks.

The first pass validates all tiles before any execution identity is changed.
The second pass keeps logical token order while packing physical GMM groups.
No device-to-host count read or dynamic allocation is required.
"""

import triton
import triton.language as tl
import torch


@triton.jit
def validate_prepare_tiles(Counts, HotSlots, Bad, ActiveCount,
                           E: tl.constexpr, H: tl.constexpr, KEY_START: tl.constexpr,
                           CUMULATIVE: tl.constexpr, DIRECT: tl.constexpr,
                           BLOCK: tl.constexpr):
    tile = tl.program_id(0)
    e = tile * BLOCK + tl.arange(0, BLOCK)
    inside = e < E
    count = tl.load(Counts + e, inside, 0)
    if CUMULATIVE:
        count -= tl.load(Counts + e - 1, inside & (e > 0), 0)
    invalid = (count < 0) & inside
    if DIRECT:
        slot = tl.load(HotSlots + KEY_START + e, inside, -1)
        invalid |= inside & ((slot < -1) | (slot >= H))
    tl.store(Bad + tile, tl.sum(invalid.to(tl.int32), 0))
    tl.store(ActiveCount + tile, tl.sum((inside & (count > 0)).to(tl.int32), 0))


@triton.jit
def prepare_copy_tiles(Counts, Error, Enabled, Access, Sources, Lengths,
                        OutCounts, OutSources, OutLengths, HotSlots, Groups,
                        ExecutionKeys, Bad, ActiveCount,
                        E: tl.constexpr, H: tl.constexpr, KEY_START: tl.constexpr,
                        SOURCE_STRIDE: tl.constexpr, COMPONENT_COUNT: tl.constexpr,
                        COMPONENT_MAP: tl.constexpr, CUMULATIVE: tl.constexpr,
                        DIRECT: tl.constexpr, DIRECT_CAPACITY: tl.constexpr,
                        HOT_OFFSET: tl.constexpr, TILES: tl.constexpr,
                        TILE_BLOCK: tl.constexpr, BLOCK: tl.constexpr):
    tile = tl.program_id(0)
    t = tl.arange(0, TILE_BLOCK)
    invalid = tl.sum(tl.load(Bad + t, t < TILES, 0), 0) > 0
    # Every tile derives the same error; only tile zero publishes the latch.
    error = tl.load(Error) | invalid.to(tl.int32)
    if tile == 0:
        tl.store(Error, error)
    e = tile * BLOCK + tl.arange(0, BLOCK)
    inside = e < E
    raw = tl.load(Counts + e, inside, 0)
    count = raw
    if CUMULATIVE:
        count -= tl.load(Counts + e - 1, inside & (e > 0), 0)
    needed = inside & (count > 0) & (error == 0)
    tl.store(OutCounts + e, tl.where(error == 0, raw, 0), inside)
    access = tl.load(Access + e, inside, 0)
    enabled = tl.load(Enabled) != 0
    tl.store(Access + e, access + tl.where(enabled & (error == 0), count, 0), inside)
    copy = needed
    if DIRECT:
        slot = tl.load(HotSlots + KEY_START + e, inside, -1)
        hit = (slot >= 0) & (slot < DIRECT_CAPACITY)
        physical = tl.where(hit, HOT_OFFSET + slot, e)
        preceding = tl.sum(tl.load(ActiveCount + t, (t < TILES) & (t < tile), 0), 0)
        position = preceding + tl.cumsum(needed.to(tl.int32), 0) - 1
        tl.store(Groups + 2 * position, physical.to(tl.int64), needed)
        tl.store(Groups + 2 * position + 1, count.to(tl.int64), needed)
        copy &= ~hit
    previous = tl.load(ExecutionKeys + e, inside, -1)
    copy &= previous != KEY_START + e
    tl.store(ExecutionKeys + e, KEY_START + e, copy)
    for position in tl.static_range(COMPONENT_COUNT):
        component = (COMPONENT_MAP >> (4 * position)) & 15
        source = tl.load(Sources + component * SOURCE_STRIDE + e, inside, 0)
        length = tl.load(Lengths + component * E + e, inside, 0)
        tl.store(OutSources + position * E + e, source, inside)
        tl.store(OutLengths + position * E + e, tl.where(copy, length, 0), inside)


class TiledPreparation:
    """Allocate a fixed plan once; all route-dependent values remain on device."""

    def __init__(self, owner):
        self.owner = owner
        self.tiles = triton.cdiv(owner.local_experts, 64)
        self.tile_block = triton.next_power_of_2(self.tiles)
        self.bad = torch.empty(self.tiles, dtype=torch.int32, device=owner.device)
        self.active = torch.empty_like(self.bad)

    def run(self, counts, access, sources, key_start, cumulative):
        o = self.owner
        if o.direct_hot:
            o.direct_groups.zero_()
        validate_prepare_tiles[(self.tiles,)](
            counts, o.state.key_to_slot, self.bad, self.active,
            E=o.local_experts, H=o.hot_slots, KEY_START=key_start,
            CUMULATIVE=cumulative, DIRECT=o.direct_hot, BLOCK=64,
        )
        prepare_copy_tiles[(self.tiles,)](
            counts, o.error, o.state.update_enabled, access, sources, o.lengths,
            o.fused_counts, o.fused_sources, o.fused_lengths, o.state.key_to_slot,
            getattr(o, "direct_groups", o.fused_counts), o.execution_keys,
            self.bad, self.active, E=o.local_experts, H=o.hot_slots,
            KEY_START=key_start, SOURCE_STRIDE=o.catalog.num_experts,
            COMPONENT_COUNT=len(o.fused_components), COMPONENT_MAP=o.fused_component_map,
            CUMULATIVE=cumulative, DIRECT=o.direct_hot,
            DIRECT_CAPACITY=o.direct_hot_slots, HOT_OFFSET=o.execution_slots,
            TILES=self.tiles, TILE_BLOCK=self.tile_block, BLOCK=64,
        )
