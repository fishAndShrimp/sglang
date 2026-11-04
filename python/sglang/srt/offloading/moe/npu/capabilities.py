"""Resolve the decode implementation from tensor/API capabilities, not EP size.

This module has no device imports so launchers/tests can check the same policy.
Runtime counters and graph-phase changes belong to the owner, not this policy.
"""

from dataclasses import asdict, dataclass
import math


COPY_DESCRIPTOR_LIMIT = 256
PHYSICAL_GMM_GROUP_LIMIT = 1024
SINGLE_PROGRAM_HOT_LIMIT = 2048
TILED_SOURCE_CATALOG_THRESHOLD = 8192


@dataclass(frozen=True)
class CapacityPlan:
    local_experts: int
    catalog_size: int
    hot_slots: int
    execution_slots: int
    metadata_block: int
    decode_mask_capacity: int

    def to_dict(self):
        return asdict(self)


def plan_capacity(*, global_experts, ep_size, offloaded_layers, cache_ratio,
                  execution_slots=None, decode_mask_capacity=128):
    """Resolve static sizes before allocating pools or loading checkpoint bytes."""
    for name, value in (("global_experts", global_experts), ("ep_size", ep_size),
                        ("offloaded_layers", offloaded_layers), ("decode_mask_capacity", decode_mask_capacity)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if global_experts % ep_size:
        raise ValueError("Routed experts must divide evenly across EP ranks")
    if isinstance(cache_ratio, bool) or not isinstance(cache_ratio, (int, float)) or not math.isfinite(cache_ratio) or not 0 < cache_ratio <= 1:
        raise ValueError("cache_ratio must be finite and in (0, 1]")
    local = global_experts // ep_size
    execution = local if execution_slots is None else execution_slots
    if type(execution) is not int or execution < local:
        raise ValueError("Fixed execution bank must hold all local experts")
    catalog = offloaded_layers * local
    hot = math.floor(cache_ratio * catalog)
    if hot < 1:
        raise ValueError("Cache ratio produces an empty Hot bank")
    if hot < local and execution < 2 * local:
        raise ValueError("Whole-layer prefill needs Hot >= local experts or two execution banks")
    return CapacityPlan(local, catalog, hot, execution,
                        1 << (local - 1).bit_length(), decode_mask_capacity)


def descriptor_slices(experts, components):
    """Static native batches; the descriptor limit is not an expert limit."""
    extent = experts * components
    return tuple((start, min(start + COPY_DESCRIPTOR_LIMIT, extent))
                 for start in range(0, extent, COPY_DESCRIPTOR_LIMIT))


def plan_decode_mask(*, request_bound, graph_bound, tokens_per_request,
                     dispatch_padding, configured=None):
    """A conservative per-rank bound; request/graph bounds may be global.

    Do not divide by DP: uneven scheduling can place all admitted requests on
    one attention rank. The MC2 padding bound is an independent contract.
    """
    for name, value in (("request_bound", request_bound), ("graph_bound", graph_bound),
                        ("dispatch_padding", dispatch_padding)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if type(tokens_per_request) is not int or tokens_per_request <= 0:
        raise ValueError("tokens_per_request must be positive")
    required = max(1, dispatch_padding, max(request_bound, graph_bound) * tokens_per_request)
    if configured is not None:
        if type(configured) is not int or configured < required:
            raise ValueError(f"decode_mask_capacity must be >= planned token bound {required}")
        return configured
    return required


@dataclass(frozen=True)
class DecodeCapabilities:
    local_experts: int
    hot_slots: int
    execution_slots: int
    direct_hot_slots: int
    direct_hot_reasons: tuple
    execution_reuse: bool = True
    weight_layout: str = "fixed_local_slots"
    fused_metadata: bool = True
    published_sources: bool = True
    w2_overlap: bool = True

    def to_dict(self):
        result = asdict(self)
        result["direct_hot"] = (
            "none" if not self.direct_hot_slots
            else "all" if self.direct_hot_slots == self.hot_slots else "partial"
        )
        result["direct_group_pack"] = (
            "not_applicable" if not self.direct_hot_slots
            else "matrix_reduce" if self.local_experts <= 64 else "tiled_prefix_scatter"
        )
        result["metadata_prepare"] = "single_program" if self.local_experts <= 64 else "tiled_64"
        result["copy_descriptor_limit"] = COPY_DESCRIPTOR_LIMIT
        return result


def resolve_decode_capabilities(*, quant_kind, local_experts, hot_slots,
                                execution_slots, speculative=False,
                                disable_direct_hot=False):
    if not 0 < local_experts <= execution_slots:
        raise ValueError("Decode requires positive local experts and enough execution slots")
    if hot_slots <= 0:
        raise ValueError("Decode requires a nonempty Hot bank")
    reasons = []
    if quant_kind != "w8a8":
        reasons.append("physical-slot GMM requires W8A8 in this implementation")
    if speculative:
        reasons.append("physical-slot GMM speculative integration is not validated")
    if disable_direct_hot:
        reasons.append("explicit diagnostic override")
    available = max(0, PHYSICAL_GMM_GROUP_LIMIT - execution_slots)
    if available == 0:
        reasons.append("execution bank consumes the 1024 physical GMM groups")
    direct = 0 if reasons else min(hot_slots, available)
    if direct and direct < hot_slots:
        reasons.append("Hot slots beyond the physical GMM view use fixed execution slots")
    return DecodeCapabilities(local_experts, hot_slots, execution_slots, direct, tuple(reasons))
