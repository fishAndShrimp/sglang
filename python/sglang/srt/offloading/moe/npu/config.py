"""Strict JSON startup settings for the experimental native V4 Ascend binding."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from ..config import MoEOffloadConfig


def validate_native_environment():
    """Reject incompatible EP layouts before allocating or loading any weights."""
    if os.environ.get("SGLANG_DSV4_FP4_DEQUANT", "0").lower() in (
        "true",
        "1",
        "yes",
        "y",
    ):
        raise ValueError("Native offload requires SGLANG_DSV4_FP4_DEQUANT=0")
    if os.environ.get("DEEP_USE_MODE", "default").lower() != "default":
        raise ValueError("Native offload requires DEEP_USE_MODE=default")
    try:
        count_type = int(os.environ.get("MOE_EXPERT_TOKEN_NUMS_TYPE", "1"))
    except ValueError:
        count_type = None
    if count_type != 1:
        raise ValueError("Native offload requires MOE_EXPERT_TOKEN_NUMS_TYPE=1")
    if os.environ.get("MOE_ENABLE_TOPK_NEG_ONE", "0") != "1":
        raise ValueError("Native offload requires MOE_ENABLE_TOPK_NEG_ONE=1")


@dataclass(frozen=True)
class NpuOffloadSettings:
    cache_ratio: float
    execution_slots: int
    host_pool_gib: int
    workspace_limit_bytes: int
    metadata_limit_bytes: int
    max_dispatch_rows: int
    prefill_group_size: int = 8
    prefill_microbatch_size: int = 1
    prefill_max_inflight: int = 1
    native_extension_directory: str | None = None
    version: int = 1

    def __post_init__(self):
        if type(self.version) is not int or self.version != 1:
            raise ValueError("Unsupported MoE offload configuration version")
        for name in (
            "execution_slots",
            "host_pool_gib",
            "workspace_limit_bytes",
            "metadata_limit_bytes",
            "max_dispatch_rows",
            "prefill_group_size",
            "prefill_microbatch_size",
            "prefill_max_inflight",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be an explicitly positive integer")
        if type(self.cache_ratio) not in (float, int):
            raise ValueError("cache_ratio must be a finite number")
        if not 0 <= self.cache_ratio <= 1:
            raise ValueError("cache_ratio must be in [0, 1]")
        if self.native_extension_directory is not None and (
            not isinstance(self.native_extension_directory, str)
            or not self.native_extension_directory
        ):
            raise ValueError("native_extension_directory must be a nonempty path")
        if self.prefill_microbatch_size > self.prefill_group_size:
            raise ValueError("Prefill microbatch cannot exceed its FFN group")
        if self.prefill_group_size > self.execution_slots:
            raise ValueError("Execution bank cannot hold the prefill FFN group")

    @classmethod
    def from_file(cls, path):
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate configuration key: " + key)
                result[key] = value
            return result

        data = json.loads(Path(path).read_text(), object_pairs_hook=unique)
        if not isinstance(data, dict):
            raise ValueError("MoE offload configuration must be a JSON object")
        unknown = set(data) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(
                "Unknown MoE offload settings: " + ", ".join(sorted(unknown))
            )
        return cls(**data)

    def owner_config(self, *, ep_size, ep_rank, tp_size, moe_tp_size):
        return MoEOffloadConfig(
            cache_ratio=self.cache_ratio,
            execution_slots=self.execution_slots,
            ep_size=ep_size,
            ep_rank=ep_rank,
            tp_size=tp_size,
            moe_tp_size=moe_tp_size,
            workspace_limit_bytes=self.workspace_limit_bytes,
            metadata_limit_bytes=self.metadata_limit_bytes,
            host_allocation_limit_bytes=self.host_pool_gib * 1024**3,
        )

    def to_dict(self):
        return asdict(self)


def create_npu_offloader(path, *, hf_config, parallel, device):
    # Called after distributed/device setup, before constructing model weights.
    from ..offloader import MoEOffloaderV2
    from .backend import NpuMoEBackend

    settings = NpuOffloadSettings.from_file(path)
    config = settings.owner_config(
        ep_size=parallel.moe_ep_size,
        ep_rank=parallel.moe_ep_rank,
        tp_size=parallel.tp_size,
        moe_tp_size=parallel.moe_tp_size,
    )
    if parallel.pp_size != 1 or parallel.moe_dp_size != 1:
        raise ValueError("V4 offload currently requires PP=1 and MoE DP=1")
    return MoEOffloaderV2(config, NpuMoEBackend(settings, config, hf_config, device))
