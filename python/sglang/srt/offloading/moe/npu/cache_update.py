"""Optional full-Hot-bank update, preserving canonical target/slot placement.

LFU still selects the sorted target set. Surviving experts retain their slots;
ascending new keys fill ascending free slots, exactly as TensorResidencyOps.
Initialization and partial pools use the original residency implementation.
"""

import torch
import triton
import triton.language as tl

from ...transfer import DeviceCopyPlan


@triton.jit
def validate_forward_map(
    KeyToSlot,
    SlotToKey,
    SlotValid,
    Bad,
    N: tl.constexpr,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    slots = tl.load(KeyToSlot + i, i < N, -1)
    safe = tl.minimum(tl.maximum(slots, 0), H - 1)
    inverse = tl.load(SlotToKey + safe)
    valid = tl.load(SlotValid + safe) != 0
    bad = (slots < -1) | (slots >= H) | ((slots >= 0) & ((inverse != i) | ~valid))
    tl.store(Bad + tl.program_id(0), tl.sum(((i < N) & bad).to(tl.int32), 0))


@triton.jit
def validate_and_select(
    KeyToSlot,
    SlotToKey,
    SlotValid,
    Target,
    Enabled,
    Due,
    Error,
    Survivors,
    NewKeys,
    NewCount,
    Active,
    PartialBad,
    N: tl.constexpr,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    keys = tl.load(SlotToKey + i, i < H, -1)
    valid = tl.load(SlotValid + i, i < H, 0) != 0
    safe_keys = tl.minimum(tl.maximum(keys, 0), N - 1)
    forward = tl.load(KeyToSlot + safe_keys)
    bad_slot = valid & ((keys < 0) | (keys >= N) | (forward != i))
    target = tl.load(Target + i, i < H, 0)
    previous = tl.load(Target + i - 1, (i > 0) & (i < H), -1)
    bad_target = (target < 0) | (target >= N) | ((i > 0) & (target <= previous))
    enabled = (tl.load(Enabled) != 0) & (tl.load(Due) != 0)
    bad = tl.sum(tl.load(PartialBad + i, i < tl.cdiv(N, 256), 0), 0) > 0
    bad = bad | (tl.sum(((i < H) & bad_slot).to(tl.int32), 0) > 0)
    bad = bad | (enabled & (tl.sum(((i < H) & bad_target).to(tl.int32), 0) > 0))
    error = tl.load(Error) | bad.to(tl.int32)
    tl.store(Error, error)
    active = enabled & (error == 0)
    tl.store(Active, active)
    target_slot = tl.load(KeyToSlot + tl.minimum(tl.maximum(target, 0), N - 1))
    new = (i < H) & (target_slot < 0) & active
    position = tl.cumsum(new.to(tl.int32), 0) - 1
    tl.store(NewKeys + position, target, new)
    tl.store(NewCount, tl.sum(new.to(tl.int32), 0))
    # Sorted unique target keys and coherent inverse mapping prove that these
    # indexed writes have no duplicate destinations. The buffer is zeroed once
    # before this kernel; no atomics or full membership scatter-add is needed.
    tl.store(Survivors + target_slot, 1, (i < H) & (target_slot >= 0) & active)


@triton.jit
def prepare_update(
    KeyToSlot,
    SlotToKey,
    SlotValid,
    Generation,
    Survivors,
    NewKeys,
    NewCount,
    Active,
    Proposed,
    Expected,
    Kind,
    Source,
    Valid,
    Count,
    N: tl.constexpr,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    old = tl.load(SlotToKey + i, i < H, -1)
    old_valid = tl.load(SlotValid + i, i < H, 0) != 0
    survivor = tl.load(Survivors + i, i < H, 0) != 0
    free = (i < H) & ~survivor
    position = tl.cumsum(free.to(tl.int32), 0) - 1
    new_count = tl.load(NewCount)
    replacement = tl.load(
        NewKeys + tl.maximum(position, 0), free & (position < new_count), -1
    )
    active = tl.load(Active) != 0
    proposed = tl.where(active, tl.where(survivor, old, replacement), old)
    changed = (i < H) & active & ((proposed != old) | (~old_valid & (proposed >= 0)))
    copy = changed & (proposed >= 0)
    generation = tl.load(Generation + i, i < H, 0) + changed.to(tl.int64)
    retained = old_valid & ~changed
    tl.store(SlotValid + i, retained, i < H)
    tl.store(Generation + i, generation, i < H)
    tl.store(Proposed + i, proposed, i < H)
    tl.store(Expected + i, generation, i < H)
    tl.store(Kind + i, tl.where(copy, 1, 0), i < H)
    tl.store(Source + i, proposed, i < H)
    tl.store(Valid + i, copy, i < H)
    tl.store(Count, tl.sum(copy.to(tl.int32), 0))


@triton.jit
def invalidate_mapping(
    KeyToSlot, SlotValid, Active, N: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    old = tl.load(KeyToSlot + i, i < N, -1)
    valid = tl.load(SlotValid + tl.minimum(tl.maximum(old, 0), H - 1)) != 0
    active = tl.load(Active) != 0
    tl.store(KeyToSlot + i, tl.where(active & ~valid, -1, old), i < N)


@triton.jit
def publish_update(
    KeyToSlot,
    SlotToKey,
    SlotValid,
    Generation,
    Proposed,
    Expected,
    Active,
    Error,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    slot = tl.arange(0, BLOCK)
    active = tl.load(Active) != 0
    expected = tl.load(Expected + slot, slot < H, 0)
    generation = tl.load(Generation + slot, slot < H, 0)
    bad = tl.sum(((slot < H) & (generation != expected)).to(tl.int32), 0) > 0
    error = tl.load(Error) | (bad & active).to(tl.int32)
    tl.store(Error, error)
    publish = active & (error == 0) & (slot < H)
    proposed = tl.load(Proposed + slot, slot < H, -1)
    valid = proposed >= 0
    # Canonical sorted input and survivor/free-slot assignment guarantee unique
    # published keys. Faulted updates never reach these indexed stores.
    tl.store(KeyToSlot + proposed, slot.to(tl.int64), publish & valid)
    tl.store(SlotToKey + slot, proposed, publish)
    tl.store(SlotValid + slot, valid, publish)


class FusedCacheUpdate:
    def __init__(self, owner):
        self.owner = owner
        self.bindings = owner.residency.update_bindings
        self.n, self.h = owner.catalog.num_experts, owner.hot_slots
        if not (0 < self.h <= min(self.n, 4096) and self.n <= 16384):
            raise ValueError("Fused cache update supports up to 4096 Hot slots and 16384 keys")
        device = owner.device

        def tensor(shape, dtype=torch.int64):
            return torch.zeros(shape, dtype=dtype, device=device)

        self.partial_bad = tensor((triton.cdiv(self.n, 256),), torch.int32)
        self.survivors = tensor((self.h,), torch.bool)
        self.new_keys = tensor((self.h,))
        self.new_count = tensor((), torch.int32)
        self.active = tensor((), torch.bool)
        self.proposed = tensor((self.h,))
        self.expected = tensor((self.h,))
        self.plan = DeviceCopyPlan(
            tensor((self.h,)),
            tensor((self.h,)),
            torch.arange(self.h, dtype=torch.int64, device=device),
            tensor((self.h,), torch.bool),
            self.expected,
            tensor((), torch.int32),
            self.h,
        )

    def apply(self):
        o, s = self.owner, self.owner.state
        o.context.retain(self)
        validate_forward_map[(triton.cdiv(self.n, 256),)](
            s.key_to_slot,
            s.slot_to_key,
            s.slot_valid,
            self.partial_bad,
            N=self.n,
            H=self.h,
            BLOCK=256,
        )
        self.survivors.zero_()
        validate_and_select[(1,)](
            s.key_to_slot,
            s.slot_to_key,
            s.slot_valid,
            o.target_keys,
            s.update_enabled,
            o.due,
            o.error,
            self.survivors,
            self.new_keys,
            self.new_count,
            self.active,
            self.partial_bad,
            N=self.n,
            H=self.h,
            BLOCK=triton.next_power_of_2(max(self.h, triton.cdiv(self.n, 256))),
        )
        prepare_update[(1,)](
            s.key_to_slot,
            s.slot_to_key,
            s.slot_valid,
            s.slot_generation,
            self.survivors,
            self.new_keys,
            self.new_count,
            self.active,
            self.proposed,
            self.expected,
            self.plan.source_kind,
            self.plan.source_index,
            self.plan.valid,
            self.plan.count,
            N=self.n,
            H=self.h,
            BLOCK=triton.next_power_of_2(self.h),
        )
        invalidate_mapping[(triton.cdiv(self.n, 256),)](
            s.key_to_slot,
            s.slot_valid,
            self.active,
            N=self.n,
            H=self.h,
            BLOCK=256,
        )
        ticket = o.access.device_ops.materialize_into(
            self.plan, self.bindings, o.context.compute_stream
        )
        ticket.completion.wait_on(o.context.compute_stream)
        publish_update[(1,)](
            s.key_to_slot,
            s.slot_to_key,
            s.slot_valid,
            s.slot_generation,
            self.proposed,
            self.expected,
            self.active,
            o.error,
            H=self.h,
            BLOCK=triton.next_power_of_2(self.h),
        )
