"""Bounded-UB implementation of the same canonical Hot-cache transaction.

Large banks use global device prefix scans between fixed-size kernels. No Host
reads, CPU routing, or changes to LFU selection/slot assignment are introduced.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def select_tiles(K2S, S2K, Valid, Target, Membership, Survivors, Free, NewFlags, Bad,
                 N: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inside = i < H
    old = tl.load(S2K + i, inside, -1)
    valid = tl.load(Valid + i, inside, 0) != 0
    forward = tl.load(K2S + tl.minimum(tl.maximum(old, 0), N - 1))
    bad_old = valid & ((old < 0) | (old >= N) | (forward != i))
    target = tl.load(Target + i, inside, 0)
    previous = tl.load(Target + i - 1, inside & (i > 0), -1)
    bad_target = (target < 0) | (target >= N) | ((i > 0) & (target <= previous))
    slot = tl.load(K2S + tl.minimum(tl.maximum(target, 0), N - 1))
    tl.store(NewFlags + i, (slot < 0).to(tl.int32), inside)
    # Scratch membership is prepared by a supported framework scatter. This
    # avoids H binary searches without unsafe cross-program indexed writes.
    found = valid & (tl.load(Membership + tl.minimum(tl.maximum(old, 0), N - 1)) != 0)
    tl.store(Survivors + i, found.to(tl.int32), inside)
    tl.store(Free + i, ~found, inside)
    tl.store(Bad + 2 * tl.program_id(0), tl.sum((inside & bad_old).to(tl.int32), 0))
    tl.store(Bad + 2 * tl.program_id(0) + 1, tl.sum((inside & bad_target).to(tl.int32), 0))


@triton.jit
def validate_tiles(ForwardBad, SlotBad, Enabled, Due, Error, Active,
                   FN: tl.constexpr, SN: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    enabled = (tl.load(Enabled) != 0) & (tl.load(Due) != 0)
    bad = tl.sum(tl.load(ForwardBad + i, i < FN, 0), 0) > 0
    old_bad = tl.sum(tl.load(SlotBad + 2 * i, i < SN, 0), 0) > 0
    target_bad = tl.sum(tl.load(SlotBad + 2 * i + 1, i < SN, 0), 0) > 0
    error = tl.load(Error) | (bad | old_bad | (enabled & target_bad)).to(tl.int32)
    tl.store(Error, error)
    tl.store(Active, enabled & (error == 0))


@triton.jit
def compact_new(Target, NewFlags, Prefix, NewKeys, NewCount, Active,
                H: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = tl.load(Active) != 0
    new = tl.load(NewFlags + i, i < H, 0) != 0
    position = tl.load(Prefix + i, i < H, 0) - 1
    key = tl.load(Target + i, i < H, 0)
    tl.store(NewKeys + position, key, (i < H) & active & new)
    if tl.program_id(0) == 0:
        tl.store(NewCount, tl.where(active, tl.load(Prefix + H - 1), 0))


@triton.jit
def prepare_tiles(S2K, SlotValid, Generation, Survivors, FreePrefix, NewKeys,
                  NewCount, Active, Proposed, Expected, Kind, Source, CopyValid,
                  H: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inside = i < H
    old = tl.load(S2K + i, inside, -1)
    old_valid = tl.load(SlotValid + i, inside, 0) != 0
    survivor = tl.load(Survivors + i, inside, 0) != 0
    position = tl.load(FreePrefix + i, inside, 0) - 1
    count = tl.load(NewCount)
    replacement = tl.load(NewKeys + tl.maximum(position, 0),
                          inside & ~survivor & (position < count), -1)
    active = tl.load(Active) != 0
    proposed = tl.where(active, tl.where(survivor, old, replacement), old)
    changed = inside & active & ((proposed != old) | (~old_valid & (proposed >= 0)))
    copy = changed & (proposed >= 0)
    generation = tl.load(Generation + i, inside, 0) + changed.to(tl.int64)
    tl.store(SlotValid + i, old_valid & ~changed, inside)
    tl.store(Generation + i, generation, inside)
    tl.store(Proposed + i, proposed, inside)
    tl.store(Expected + i, generation, inside)
    tl.store(Kind + i, tl.where(copy, 1, 0), inside)
    tl.store(Source + i, proposed, inside)
    tl.store(CopyValid + i, copy, inside)


@triton.jit
def check_generations(Generation, Expected, Bad, H: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    old = tl.load(Generation + i, i < H, 0)
    expected = tl.load(Expected + i, i < H, 0)
    tl.store(Bad + tl.program_id(0), tl.sum(((i < H) & (old != expected)).to(tl.int32), 0))


@triton.jit
def finish_validation(Bad, Error, Active, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    bad = tl.sum(tl.load(Bad + i, i < PARTS, 0), 0) > 0
    tl.store(Error, tl.load(Error) | ((tl.load(Active) != 0) & bad).to(tl.int32))


@triton.jit
def publish_tiles(K2S, S2K, SlotValid, Proposed, Active, Error,
                  H: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    publish = (i < H) & (tl.load(Active) != 0) & (tl.load(Error) == 0)
    key = tl.load(Proposed + i, i < H, -1)
    tl.store(S2K + i, key, publish)
    tl.store(SlotValid + i, key >= 0, publish)


class TiledCacheUpdate:
    def __init__(self, parent):
        self.p = parent
        h, device = parent.h, parent.owner.device
        self.parts = triton.cdiv(h, 256)
        self.bad = torch.zeros(self.parts * 2, dtype=torch.int32, device=device)
        self.flags = torch.empty(h, dtype=torch.int32, device=device)
        self.free = torch.empty(h, dtype=torch.bool, device=device)
        self.new_prefix = torch.empty(h, dtype=torch.int64, device=device)
        self.free_prefix = torch.empty_like(self.new_prefix)
        self.inverse = torch.empty(parent.n, dtype=torch.int64, device=device)
        self.membership = torch.empty(parent.n, dtype=torch.int32, device=device)
        self.slot_ids = torch.arange(h, dtype=torch.int64, device=device)

    def apply(self):
        from .cache_update import invalidate_mapping, validate_forward_map
        p = self.p
        o, s, h, n = p.owner, p.owner.state, p.h, p.n
        o.context.retain(self)
        nparts = triton.cdiv(n, 256)
        validate_forward_map[(nparts,)](s.key_to_slot, s.slot_to_key, s.slot_valid,
                                        p.partial_bad, N=n, H=h, BLOCK=256)
        self.membership.zero_()
        self.membership.scatter_(0, o.target_keys.clamp(0, n - 1), 1)
        select_tiles[(self.parts,)](s.key_to_slot, s.slot_to_key, s.slot_valid,
            o.target_keys, self.membership, p.survivors, self.free, self.flags,
            self.bad, N=n, H=h, BLOCK=256)
        validate_tiles[(1,)](p.partial_bad, self.bad, s.update_enabled, o.due,
            o.error, p.active, FN=nparts, SN=self.parts,
            BLOCK=triton.next_power_of_2(max(nparts, self.parts)))
        torch.cumsum(self.flags, 0, out=self.new_prefix)
        compact_new[(self.parts,)](o.target_keys, self.flags, self.new_prefix,
            p.new_keys, p.new_count, p.active, H=h, BLOCK=256)
        torch.cumsum(self.free, 0, out=self.free_prefix)
        prepare_tiles[(self.parts,)](s.slot_to_key, s.slot_valid, s.slot_generation,
            p.survivors, self.free_prefix, p.new_keys, p.new_count, p.active,
            p.proposed, p.expected, p.plan.source_kind, p.plan.source_index,
            p.plan.valid, H=h, BLOCK=256)
        p.plan.count.copy_(p.new_count)
        invalidate_mapping[(nparts,)](s.key_to_slot, s.slot_valid, p.active,
                                       N=n, H=h, BLOCK=256)
        ticket = o.access.device_ops.materialize_into(p.plan, p.bindings, o.context.compute_stream)
        ticket.completion.wait_on(o.context.compute_stream)
        check_generations[(self.parts,)](s.slot_generation, p.expected, self.bad, H=h, BLOCK=256)
        finish_validation[(1,)](self.bad, o.error, p.active, PARTS=self.parts,
                                BLOCK=triton.next_power_of_2(self.parts))
        # NPU Triton cannot lower masked int64 exchange here. Build a separate
        # inverse with the framework's scatter, then publish only on success.
        self.inverse.fill_(-1)
        self.inverse.scatter_(0, p.proposed.clamp(0, n - 1), self.slot_ids)
        torch.where(p.active & (o.error == 0), self.inverse, s.key_to_slot, out=s.key_to_slot)
        publish_tiles[(self.parts,)](s.key_to_slot, s.slot_to_key, s.slot_valid,
            p.proposed, p.active, o.error, H=h, BLOCK=256)
