"""Experimental expert weights for the native Ascend Ops runner.

Communication and token layout stay in the mainline dispatcher/runner. One
cross-layer bank has a fixed slot for every local expert. Device-side counts
select Host/Hot copies; cache publication occurs only after the whole forward.
"""

import json
import logging
import math
import os
from contextlib import contextmanager
from pathlib import Path

import torch

from sglang.srt.distributed.parallel_state import get_moe_ep_group
from sglang.srt.runtime_context import get_device, get_exec, get_parallel, get_schedule
from sglang.srt.utils.offloader import BaseOffloader

from ...transfer import EventCompletion, TransferTicket
from ..config import ExpertCatalog, ExpertLayerSpec, Phase
from ..execution import CacheUpdateCoordinator, ExpertWeightAccess
from ..periodic_lfu import PeriodicLFUConfig, PeriodicLFUPolicy
from ..residency import TensorResidencyOps
from ..resources import ResourceKind
from ..state import ResidencyState
from .format import ExpertLayerStorage
from .native_acl import get_native_acl
from .storage import MemFabricHostPool
from .transfer import (
    MemFabricAddressCatalog,
    MemFabricDeviceTransferOps,
    MemFabricTransferBindings,
    _group_pack_copy,
)
from .w8a8_format import NpuV4W8A8Format
from .w4a8_format import NpuV4W4A8Format

logger = logging.getLogger(__name__)


class _Context:
    """A cache transaction on the current stream, retired after its completion."""

    phase = Phase.DECODE
    capturing = False

    def __init__(self):
        self.compute_stream = self.copy_stream = torch.npu.current_stream()
        self.invocation_id = object()
        self.resources = []

    def retain(self, resource):
        self.resources.append(resource)

    retain_completion = retain

    def record_compute_completion(self):
        event = torch.npu.Event()
        event.record(self.compute_stream)
        ticket = TransferTicket(EventCompletion(event))
        self.retain(ticket)
        return ticket


class AscendOpsOffloader(BaseOffloader):
    def __init__(self, hf_config, path):
        p = get_parallel()
        if (
            p.tp_size != p.moe_ep_size
            or p.moe_tp_size != 1
            or p.pp_size != 1
            or p.nnodes != 1
        ):
            raise ValueError("Expert offload requires single-stage TP=EP and MoE TP=1")
        if get_exec().moe.moe_a2a_backend != "ascend_ops":
            raise ValueError("Expert offload currently requires Ascend Ops")
        if not get_schedule().disable_overlap_schedule:
            raise ValueError(
                "Single execution bank requires --disable-overlap-schedule"
            )
        if (
            get_exec().moe.enable_eplb
            or get_exec().moe.ep_num_redundant_experts
            or get_exec().moe.init_expert_location != "trivial"
            or get_exec().overlap.enable_two_batch_overlap
            or get_exec().overlap.enable_single_batch_overlap
        ):
            raise ValueError(
                "Initial fixed-slot offload requires trivial expert placement and does not support EPLB, redundant experts, or batch overlap"
            )
        if hf_config.model_type != "deepseek_v4":
            raise ValueError(
                "Initial format/model adapter supports DeepSeek V4 W8A8 only"
            )
        cfg = json.loads(Path(path).read_text())
        for option in (
            "skip_zero_offsets",
            "overlap_w2",
            "bundle_copies",
            "cached_source_tables",
            "fixed_copy_lengths",
            "graph_cache_tail",
            "fused_copy_metadata",
            "fused_allgather_combine",
            "fused_cache_transfer",
            "incremental_cache",
            "mask_decode_padding",
        ):
            if option in cfg and not isinstance(cfg[option], bool):
                raise ValueError(f"{option} must be boolean")
        self.cfg, self.hf_config = cfg, hf_config
        # Diagnostic ablation: keep both copy groups and their stream unchanged,
        # but make GMM1 wait for W2 as well. Default preserves W2 overlap.
        self.wait_w2_before_gmm1 = (
            os.environ.get("SGLANG_NPU_WAIT_W2_BEFORE_GMM1", "0") == "1"
        )
        self.static_replica_mode = cfg.get("static_replica_mode", "off")
        if self.static_replica_mode not in ("off", "unused", "balance"):
            raise ValueError("Unsupported static replica mode")
        self.static_replica = None
        self.static_replica_load = cfg.get("static_replica_load", "shared")
        if self.static_replica_load not in ("shared", "gloo_init"):
            raise ValueError("Unsupported static replica initialization transport")
        if self.static_replica_mode != "off" and not (
            cfg.get("fused_copy_metadata") and cfg.get("mask_decode_padding")
            and cfg.get("dispatch_prefetch_mode", "off") == "off"
            and cfg.get("router_ahead_mode", "off") == "off"
        ):
            raise ValueError("Static replicas require fused metadata/mask and no predictor")
        if cfg.get("incremental_cache", False) and not cfg.get(
            "fused_cache_transfer", False
        ):
            raise ValueError("Incremental cache requires fixed-slot cache transfer")
        if cfg.get("fused_copy_metadata", False) and not (
            cfg.get("cached_source_tables", False)
            and cfg.get("fixed_copy_lengths", False)
        ):
            raise ValueError(
                "Fused metadata requires published sources and fixed copy lengths"
            )
        ratio = cfg.get("cache_ratio", 0.28125)
        if (
            not isinstance(ratio, (int, float))
            or not math.isfinite(ratio)
            or not 0 < ratio <= 1
        ):
            raise ValueError("cache_ratio must be in (0, 1]")
        resident = max(3, getattr(hf_config, "num_hash_layers", 0))
        self.catalog = ExpertCatalog(
            tuple(
                ExpertLayerSpec(
                    i,
                    hf_config.n_routed_experts,
                    p.moe_ep_size,
                    p.moe_ep_rank,
                    resident=i < resident,
                )
                for i in range(hf_config.num_hidden_layers)
            )
        )
        self.local_experts = self.catalog.offloaded_layers[0].local_experts
        self.copy_capacity = self.local_experts + (4 if self.static_replica_mode != "off" else 0)
        # MemFabric batches use at most 256 entries on A3. Larger local expert
        # counts need explicit chunking; never overrun the native metadata UB.
        if self.copy_capacity > 256:
            raise ValueError(
                "Initial fixed-slot copy supports at most 256 local experts"
            )
        self.hot_slots = int(self.catalog.num_experts * ratio)
        if self.hot_slots < 1:
            raise ValueError("The initial implementation requires a nonempty Hot cache")
        self.device = torch.device("npu", get_device().gpu_id)
        self.storages = {}
        self.format = self.host_pool = None
        self.ready = False
        self.step = 0
        profile = cfg.get("offline_stats_path")
        if profile and not Path(profile).is_absolute():
            profile = str(Path(path).resolve().parent / profile)
        self.policy = PeriodicLFUPolicy(
            PeriodicLFUConfig(
                offline_stats_path=profile,
                **{
                    k: cfg[k]
                    for k in (
                        "update_interval",
                        "min_usage",
                        "replacement_min_gain",
                        "max_swaps",
                        "usage_decay",
                    )
                    if k in cfg
                },
            ),
            self.catalog,
        )
        self.initial_keys = self.policy.initial_keys(
            self.hot_slots, cfg.get("offline_model_id", "DeepSeek-V4-Flash")
        )

    def prepare_layer(self, layer, num_experts, hidden_size, intermediate_size, attrs):
        spec = self.catalog.layer(layer.layer_id)
        if spec.resident:
            return False
        schemes = (type(layer.w13_scheme).__name__, type(layer.w2_scheme).__name__)
        formats = {"ModelSlimW8A8Int8MoE": NpuV4W8A8Format,
                   "ModelSlimW4A8Int8MoE": NpuV4W4A8Format}
        if num_experts != spec.local_experts or schemes[0] != schemes[1] or schemes[0] not in formats:
            raise ValueError("Offload requires homogeneous ModelSlim W8A8 or per-channel W4A8 experts")
        format_type = formats[schemes[0]]
        if format_type is NpuV4W4A8Format:
            if any(not scheme.is_per_channel_weight or scheme.activation_use_clip
                   for scheme in (layer.w13_scheme, layer.w2_scheme)):
                raise ValueError("W4A8 offload currently requires per-channel quantization without activation clipping")
            if self.cfg.get("skip_zero_offsets", False):
                raise ValueError("W4A8 uses eight components; disable the six-component skip_zero_offsets optimization")
        if self.format is not None and type(self.format) is not format_type:
            raise ValueError("Offloaded layers must use the same weight format")
        if self.format is None:
            native = get_native_acl(self.cfg.get("native_extension_directory"))
            self.format = format_type(
                hidden_size, intermediate_size, self.device, native_acl=native
            )
            total = sum(
                self.format.host_nbytes(s.local_experts)
                for s in self.catalog.offloaded_layers
            )
            if self.static_replica_mode != "off" and self.static_replica_load == "gloo_init":
                total += self.format.expert_nbytes + 64  # one startup staging expert
            shared = self.static_replica_mode != "off" and self.static_replica_load == "shared"
            self.host_pool = MemFabricHostPool(
                total,
                self.device.index,
                physical_bytes=int(self.cfg.get("host_pool_gib", 64)) * 1024**3,
                native_acl=native,
                shared_world_size=get_parallel().moe_ep_size if shared else 1,
                shared_rank=get_parallel().moe_ep_rank if shared else 0,
                shared_store_url=self.cfg.get("static_replica_store", "tcp://127.0.0.1:29713"),
            )
        if (hidden_size, intermediate_size) != (
            self.format.hidden_size,
            self.format.intermediate_size,
        ):
            raise ValueError("This execution bank requires identical expert shapes")
        storage = ExpertLayerStorage(layer, spec, self.format, self.host_pool, attrs)
        self.storages[layer.layer_id] = storage
        layer._ascend_ops_offloader = self
        return True

    def post_init(self):
        if set(self.storages) != {s.layer_id for s in self.catalog.offloaded_layers}:
            raise RuntimeError("Not all planned expert layers were intercepted")
        for storage in self.storages.values():
            storage.require_sealed()
        self.sparse_hot = self.cfg.get("sparse_hot_gmm", False)
        self.direct_hot_slots = self.cfg.get("sparse_direct_hot_slots", self.hot_slots)
        if self.sparse_hot and (type(self.direct_hot_slots) is not int
                                or not 0 < self.direct_hot_slots <= self.hot_slots):
            raise ValueError("Direct Hot slots must be within the cache capacity")
        if self.sparse_hot and self.copy_capacity + self.direct_hot_slots > 1024:
            raise ValueError("Sparse GMM supports at most 1024 slots; limit sparse_direct_hot_slots")
        self.maintenance_busy_limit = int(
            os.environ.get("SGLANG_NPU_MAINTENANCE_BUSY_LIMIT", "0")
        )
        if self.maintenance_busy_limit not in (0, 8, 24):
            raise ValueError("Busy maintenance limit must be 0, 8, or 24")
        if self.maintenance_busy_limit and not self.cfg.get("incremental_cache", False):
            raise ValueError("Adaptive maintenance requires incremental cache")
        self.sparse_compact_groups = int(
            os.environ.get("SGLANG_NPU_SPARSE_COMPACT_GROUPS", "0")
        )
        if self.sparse_compact_groups not in (0, 8, 16):
            raise ValueError("Sparse compact groups must be 0, 8, or 16")
        if self.sparse_compact_groups and not self.sparse_hot:
            raise ValueError("Compact descriptors require sparse Hot GMM")
        if self.sparse_hot:
            if (type(self.format) is not NpuV4W8A8Format
                or not self.cfg.get("fused_copy_metadata", False)
                or not self.cfg.get("cached_source_tables", False)
                or self.static_replica_mode != "off"
                or self.cfg.get("dispatch_prefetch_mode", "off") != "off"
                or self.cfg.get("router_ahead_mode", "off") != "off"
                or os.environ.get("SGLANG_NPU_PROMOTE_EXECUTION", "0") != "0"):
                raise ValueError("Sparse Hot probe requires plain fixed-slot W8A8 fused path")
            self.union = self.format.allocate_bank(
                self.copy_capacity + self.hot_slots,
                ResourceKind.EXECUTION_WEIGHTS, "ascend.union")
            self.execution = self.union.view(0, self.copy_capacity)
            self.hot = self.union.view(self.copy_capacity, self.hot_slots)
            self.sparse_groups = torch.zeros(
                (self.copy_capacity + self.direct_hot_slots, 2),
                dtype=torch.int64, device=self.device)
        else:
            self.execution = self.format.allocate_bank(
                self.copy_capacity, ResourceKind.EXECUTION_WEIGHTS, "ascend.execution"
            )
            self.hot = self.format.allocate_bank(
                self.hot_slots, ResourceKind.HOT_WEIGHTS, "ascend.hot"
            )
        compact_cache = os.environ.get("SGLANG_NPU_COMPACT_CACHE_TRANSFER", "0") == "1"
        self.promote_execution = os.environ.get("SGLANG_NPU_PROMOTE_EXECUTION", "0") == "1"
        self.reuse_execution = os.environ.get("SGLANG_NPU_REUSE_EXECUTION", "0") == "1"
        track_execution = self.promote_execution or self.reuse_execution
        self.execution_keys = None
        if track_execution:
            if (
                type(self.format) is not NpuV4W8A8Format
                or not self.cfg.get("fused_copy_metadata", False)
                or self.copy_capacity != self.local_experts
                or self.static_replica_mode != "off"
                or self.cfg.get("dispatch_prefetch_mode", "off") != "off"
                or self.cfg.get("router_ahead_mode", "off") != "off"
                or compact_cache
                or (self.promote_execution and not self.cfg.get("fused_cache_transfer", False))
            ):
                raise ValueError("Execution reuse requires fixed-slot W8A8 fused metadata without predictors, replicas or compact maintenance")
            self.execution_keys = torch.full(
                (self.local_experts,), -1, dtype=torch.int64, device=self.device
            )
        if compact_cache:
            if (
                type(self.format) is not NpuV4W8A8Format
                or not self.cfg.get("skip_zero_offsets", False)
                or self.static_replica_mode != "off"
                or not 0 < self.policy.config.max_swaps <= 256
            ):
                raise ValueError("Compact maintenance requires zero-offset W8A8, no replicas, and 1..256 swaps")
            # Loaded Host offsets were certified zero. Initialize the whole Hot
            # offset bank before any transaction or graph may refer to it.
            for name, tensor in self.hot.tensors.items():
                if name.endswith("_offset"):
                    tensor.zero_()
        compute_bank = (self.union.view(0, self.copy_capacity + self.direct_hot_slots)
                        if self.sparse_hot else self.execution)
        self.quant_info = compute_bank.quant_info()
        names = tuple(self.format.layouts)
        hosts = {
            name: tuple(
                cap
                for s in self.catalog.offloaded_layers
                for cap in self.storages[s.layer_id].component_capabilities[name]
            )
            for name in names
        }
        self.addresses = MemFabricAddressCatalog(
            hosts,
            self.hot.component_regions,
            {n: self.format.layouts[n].nbytes for n in names},
            device=self.device,
        )
        self.sources = torch.stack(
            [torch.cat(pair) for pair in self.addresses.source_tables]
        )
        self.destinations = torch.tensor(
            [[r.address for r in self.execution.component_regions[n]] for n in names],
            dtype=torch.int64,
            device=self.device,
        )
        self.lengths = torch.tensor(
            [[self.format.layouts[n].nbytes] * self.copy_capacity for n in names],
            dtype=torch.uint32,
            device=self.device,
        )
        self.scan = torch.tensor(
            self.copy_capacity, dtype=torch.uint32, device=self.device
        )
        self.ids = torch.arange(self.local_experts, device=self.device)
        self._init_copy_plan(names)
        self.packed = torch.zeros_like(self.destinations)
        self.error = torch.zeros((), dtype=torch.int32, device=self.device)
        self.state = ResidencyState.allocate(
            self.catalog.num_experts, self.hot_slots, self.local_experts, self.device
        )
        self.keys = {
            s.layer_id: self.ids + self.catalog.key(s.layer_id, 0)
            for s in self.catalog.offloaded_layers
        }
        self.policy_state = self.policy.allocate_state(
            self.catalog.num_experts, self.device
        )
        self.target_keys = torch.tensor(
            self.initial_keys, dtype=torch.int64, device=self.device
        )
        bindings = MemFabricTransferBindings(
            self.addresses,
            self.hot.component_regions,
            capacity=self.hot_slots,
            hot_generations=self.state.slot_generation,
            error_status=self.error,
        )
        self.residency = TensorResidencyOps(
            self.state,
            self.catalog,
            execution_waves=[],
            update_bindings=bindings,
            error_status=self.error,
        )
        self.context = _Context()
        device_ops = MemFabricDeviceTransferOps(retain=self._retain)
        if self.cfg.get("fused_cache_transfer", False):
            from .cache_transfer import FixedSlotCacheTransfer

            device_ops = FixedSlotCacheTransfer(bindings, retain=self._retain)
        self.access = ExpertWeightAccess(self.residency, device_ops=device_ops)
        self.due = torch.zeros((), dtype=torch.bool, device=self.device)
        self.coordinator = CacheUpdateCoordinator(
            self.policy,
            self.policy_state,
            self.state,
            self.target_keys,
            self.access,
            due_out=self.due,
        )
        self.state.update_enabled.fill_(True)
        self.access.apply_update(
            self.target_keys, self.context
        ).completion.synchronize()
        if self.promote_execution:
            from .promotion_transfer import PromotionCacheTransfer

            # Initial fill uses Host. Later maintenance follows GMM2, which
            # already waited for the complete W13/W2 execution-bank writes.
            self.access.device_ops = PromotionCacheTransfer(
                bindings, retain=self._retain,
                execution_sources=self.execution.component_regions,
                execution_keys=self.execution_keys,
            )
        if compact_cache:
            # Initial fill can exceed max_swaps, so install only after the
            # original full-bank transaction has completed.
            from .compact_cache_transfer import CompactCacheTransfer

            self.access.device_ops = CompactCacheTransfer(
                bindings,
                retain=self._retain,
                max_updates=min(self.policy.config.max_swaps, self.hot_slots),
                skip_zero_components=tuple(n for n in names if n.endswith("_offset")),
            )
            logger.info("Experimental compact maintenance: %d copy submissions", len(self.access.device_ops.descriptors))
        if self.cfg.get("cached_source_tables", False):
            self._publish_sources()
        if self.cfg.get("incremental_cache", False):
            from .cache_update import FusedCacheUpdate

            self.cache_update = FusedCacheUpdate(self)
        self._check_error()
        self.context.resources.clear()
        self.state.update_enabled.fill_(False)  # warmup/capture never trains LFU
        self.decode_active_rows = None
        if self.cfg.get("mask_decode_padding", False):
            parallel = get_parallel()
            if parallel.attn_dp_size != parallel.moe_ep_size or parallel.attn_cp_size != 1:
                raise ValueError("Decode row mask requires attention DP=EP and CP=1")
            capacity = self.cfg.get("decode_mask_capacity", 128)
            if type(capacity) is not int or capacity <= 0:
                raise ValueError("decode_mask_capacity must be positive")
            # Allocated before capture; every graph keeps this same address.
            self.decode_active_rows = torch.zeros(capacity, dtype=torch.bool, device=self.device)
        self.diagnostics = None
        if self.cfg.get("diagnostics_directory"):
            from .diagnostics import ForwardDiagnostics

            self.diagnostics_rank = get_parallel().moe_ep_rank
            self.diagnostics = ForwardDiagnostics(
                self, self.cfg["diagnostics_directory"],
                self.cfg.get("diagnostics_capacity", 2048),
            )
        if self.cfg.get("graph_cache_tail", False):
            self._capture_cache_tails()
        self.ready = True
        logger.info(
            "Ascend expert offload: local=%d hot=%d/%d execution=%.3f GiB host=%.3f GiB; fixed-slot copies, W2 overlap=%s, fused metadata=%s",
            self.local_experts,
            self.hot_slots,
            self.catalog.num_experts,
            self.format.expert_nbytes * self.local_experts / 1024**3,
            self.host_pool.used_bytes / 1024**3,
            self.cfg.get("overlap_w2", False),
            self.cfg.get("fused_copy_metadata", False),
        )

    def initialize_router_ahead(self, model):
        if self.static_replica_mode != "off":
            from .static_replica import StaticReplica
            self.static_replica = StaticReplica(self)
        # Decode masking is independent of who owns the expert weights.
        self.resident_diagnostics = None
        resident = sorted((m for m in model.modules()
                           if type(m).__name__ == "DeepseekV2MoE" and not m.is_nextn
                           and getattr(m.experts, "_ascend_ops_offloader", None) is None),
                          key=lambda m: m.layer_id)
        if self.cfg.get("mask_resident_decode_padding", False):
            if self.decode_active_rows is None:
                raise ValueError("Resident masking requires the existing decode effective-row mask")
            for module in resident:
                module.experts._ascend_ops_decode_mask_owner = self
        if self.cfg.get("resident_diagnostics_directory"):
            from .resident_diagnostics import ResidentDispatchDiagnostics
            self.resident_diagnostics = ResidentDispatchDiagnostics(
                self, resident, self.cfg["resident_diagnostics_directory"])
            for module in resident:
                module.experts._resident_dispatch_observer = self.resident_diagnostics
        self.router_ahead = None
        self.dispatch_prefetch = None
        if self.cfg.get("dispatch_prefetch_mode", "off") != "off":
            if self.cfg.get("router_ahead_mode", "off") != "off":
                raise ValueError("Dispatch prefetch and router shadow are separate experiments")
            from .dispatch_prefetch import DispatchPrefetch
            self.dispatch_prefetch = DispatchPrefetch(self, model)
        if self.cfg.get("router_ahead_mode", "off") != "off":
            from .router_ahead import RouterAhead
            self.router_ahead = RouterAhead(self, model)
            logger.info("Router-ahead observation installed for %d layers", len(self.router_ahead.layers))

    def _retain(self, value):
        self.context.retain(value)

    def decode_row_mask(self, padded_bs):
        if self.decode_active_rows is None:
            return None
        if padded_bs > self.decode_active_rows.numel():
            raise ValueError("MC2 padded batch exceeds decode_mask_capacity")
        return self.decode_active_rows[:padded_bs]

    def prepare_weights(self, layer_id, runner_input):
        if not self.ready:
            raise RuntimeError("Expert bank is not initialized")
        if self.cfg.get("fused_copy_metadata", False):
            return self._prepare_fused(layer_id, runner_input)
        counts = runner_input.expert_tokens
        if counts.shape != (self.local_experts,):
            raise ValueError(f"Unexpected local expert counts shape: {counts.shape}")
        if runner_input.group_list_type == 0:
            counts = torch.diff(counts, prepend=torch.zeros_like(counts[:1]))
        elif runner_input.group_list_type != 1:
            raise ValueError("Only cumulative or per-expert group lists are supported")
        keys = self.keys[layer_id]
        if self.cfg.get("cached_source_tables", False):
            bad = counts < 0
        else:
            slots = self.state.key_to_slot.gather(0, keys)
            safe_slots = slots.clamp(0, self.hot_slots - 1)
            hit = slots >= 0
            coherent = self.state.slot_valid.to(torch.int32).gather(
                0, safe_slots
            ).bool() & (self.state.slot_to_key.gather(0, safe_slots) == keys)
            bad = (
                (counts < 0)
                | (slots < -1)
                | (slots >= self.hot_slots)
                | (hit & ~coherent)
            )
        self.error.bitwise_or_(bad.any().to(torch.int32))
        needed = (counts > 0) & (self.error == 0)
        # Do not send invalid group counts into GMM while the batch-level fault
        # is waiting to be reported consistently across the EP ranks.
        runner_input.expert_tokens = torch.where(
            self.error == 0, runner_input.expert_tokens, 0
        )
        start = self.catalog.key(layer_id, 0)
        self.state.step_access_count[start : start + self.local_experts].add_(
            torch.where(self.state.update_enabled, counts, 0)
        )
        if self.cfg.get("cached_source_tables", False):
            src = self.published_sources[:, start : start + self.local_experts]
        else:
            indices = torch.where(
                hit, safe_slots + self.catalog.num_experts + 2, keys + 1
            )
            src = self.sources.gather(
                1, indices.unsqueeze(0).expand(self.sources.shape[0], -1)
            )
        if not self.cfg.get("fixed_copy_lengths", False):
            src = torch.where(needed.unsqueeze(0), src, 0)
        # Build descriptors while cache placement is frozen for this forward.
        self.context.retain((src, counts, needed))
        copies = self._plan_copies(src, needed)
        self._launch_copies(layer_id, copies)
        return self.quant_info

    def _launch_copies(self, layer_id, copies):
        if self.cfg.get("overlap_w2", False):
            ready, first, second = self.copy_events[layer_id]
            compute = torch.npu.current_stream()
            ready.record(compute)
            with torch.npu.stream(self.weight_copy_stream):
                self.weight_copy_stream.wait_event(ready)
                self._submit_copies(copies[0], phase=0)
                first.record(self.weight_copy_stream)
                self._submit_copies(copies[1], phase=1)
                second.record(self.weight_copy_stream)
            compute.wait_event(second if self.wait_w2_before_gmm1 else first)
        else:
            self._submit_copies(copies[0])

    def _init_copy_plan(self, names):
        self.copy_budget_kernel = None
        self.copy_core_budgets = (32, 32)
        if self.cfg.get("copy_kernel_library"):
            from .copy_budget import CopyBudgetKernel

            self.copy_core_budgets = tuple(
                self.cfg.get(key, 32) for key in ("w13_copy_cores", "w2_copy_cores")
            )
            if not self.cfg.get("overlap_w2", False) or any(
                type(n) is not int or not 1 <= n <= 32 for n in self.copy_core_budgets
            ):
                raise ValueError("Copy budgets require W2 overlap and 1..32 AIV cores")
            self.copy_budget_kernel = CopyBudgetKernel(self.cfg["copy_kernel_library"])
        # Zero offsets are a checked invariant of this format, not an assumption
        # about arbitrary quantized checkpoints. They remain valid in every slot.
        selected = []
        for i, name in enumerate(names):
            if self.cfg.get("skip_zero_offsets", False) and name.endswith("_offset"):
                self.execution.tensors[name].zero_()
            else:
                selected.append(i)
        phases = [selected]
        if self.cfg.get("overlap_w2", False):
            phases = [
                [i for i in selected if names[i].startswith(prefix)]
                for prefix in ("w13", "w2")
            ]
            self.weight_copy_stream = torch.npu.Stream(device=self.device)
            self.copy_events = {
                s.layer_id: tuple(torch.npu.Event() for _ in range(3))
                for s in self.catalog.offloaded_layers
            }
        self.fixed_groups = torch.ones(256, dtype=torch.int64, device=self.device)
        self.copy_plans = []
        chunk = (
            max(1, 256 // self.copy_capacity)
            if self.cfg.get("bundle_copies", False)
            else 1
        )
        for phase in phases:
            plans = []
            for begin in range(0, len(phase), chunk):
                components = torch.tensor(
                    phase[begin : begin + chunk], device=self.device
                )
                n = components.numel() * self.copy_capacity
                plans.append(
                    (
                        tuple(phase[begin : begin + chunk]),
                        components,
                        self.destinations.index_select(0, components).flatten(),
                        self.lengths.view(torch.int32)
                        .index_select(0, components)
                        .flatten()
                        .view(torch.uint32),
                        torch.tensor(n, dtype=torch.uint32, device=self.device),
                        torch.arange(n, device=self.device),
                        torch.zeros(n, dtype=torch.int64, device=self.device),
                    )
                )
            self.copy_plans.append(plans)

        if self.cfg.get("fused_copy_metadata", False):
            self.fused_components = tuple(selected)
            self.fused_sources = torch.empty(
                len(selected) * self.copy_capacity,
                dtype=torch.int64,
                device=self.device,
            )
            self.fused_lengths = torch.empty_like(
                self.fused_sources, dtype=torch.uint32
            )
            self.fused_counts = torch.empty(
                self.copy_capacity, dtype=torch.int64, device=self.device
            )
            self.fused_descriptors = []
            offset = 0
            for phase in self.copy_plans:
                descriptors = []
                for indices, _, destinations, _, scan, ids, packed in phase:
                    n = len(indices) * self.copy_capacity
                    descriptors.append(
                        (
                            self.fused_sources[offset : offset + n],
                            destinations,
                            self.fused_lengths[offset : offset + n],
                            scan,
                            self.fixed_groups[:n],
                            packed,
                        )
                    )
                    offset += n
                self.fused_descriptors.append(descriptors)

    def _prepare_fused(self, layer_id, runner_input):
        if self.static_replica is not None:
            return self.static_replica.prepare_weights(layer_id, runner_input)
        if getattr(self, "dispatch_prefetch", None) is not None and self.dispatch_prefetch.consume:
            return self.dispatch_prefetch.prepare(layer_id, runner_input)
        import triton

        from .small_batch_kernels import prepare_fixed_copies

        counts = runner_input.expert_tokens
        if (
            counts.shape != (self.local_experts,)
            or counts.dtype != torch.int64
            or not counts.is_contiguous()
        ):
            raise ValueError(
                "Fused copy metadata expects one int64 count per local expert"
            )
        if runner_input.group_list_type not in (0, 1):
            raise ValueError("Unsupported group list type")
        start = self.catalog.key(layer_id, 0)
        end = start + self.local_experts
        prepare_fixed_copies[(1,)](
            counts,
            self.error,
            self.state.update_enabled,
            self.state.step_access_count[start:end],
            self.published_sources[:, start:end],
            self.lengths,
            self.fused_counts,
            self.fused_sources,
            self.fused_lengths,
            EXPERTS=self.local_experts,
            SOURCE_STRIDE=self.catalog.num_experts,
            START=0,
            COMPONENT_COUNT=len(self.fused_components),
            SKIP_OFFSETS=self.cfg.get("skip_zero_offsets", False),
            CUMULATIVE=runner_input.group_list_type == 0,
            BLOCK=triton.next_power_of_2(self.local_experts),
            ExecutionKeys=self.execution_keys,
            KEY_START=start if self.execution_keys is not None else 0,
            TRACK_EXECUTION=self.execution_keys is not None,
            SKIP_REUSED=self.reuse_execution,
            HOT_DIRECT=self.sparse_hot,
            HotSlots=self.state.key_to_slot,
            Groups=self.sparse_groups if self.sparse_hot else self.fused_counts,
            HOT_KEY_START=start,
            HOT_CAPACITY=self.hot_slots,
            DIRECT_HOT_CAPACITY=self.direct_hot_slots,
            COMPACT_GROUPS=self.sparse_compact_groups,
        )
        if self.sparse_hot:
            runner_input.expert_tokens = self.sparse_groups
            runner_input.group_list_type = 2
        else:
            runner_input.expert_tokens = self.fused_counts
        self._launch_copies(layer_id, self.fused_descriptors)
        return self.quant_info

    def _plan_copies(self, src, needed):
        if self.cfg.get("fixed_copy_lengths", False):
            # All groups participate, including zero-byte entries: native packed
            # index j then equals source index i. This preserves fixed slots
            # without sorting / gathering destinations. Zero lengths never
            # dereference their source, even when every expert is inactive.
            lengths = self.lengths.view(torch.int32) * needed.to(torch.int32).unsqueeze(
                0
            )
            self.context.retain(lengths)
            result = []
            for phase in self.copy_plans:
                descriptors = []
                for indices, components, dst, _, scan, ids, packed in phase:
                    if len(indices) == 1:
                        source, sizes = (
                            src[indices[0]],
                            lengths[indices[0]].view(torch.uint32),
                        )
                    else:
                        source = src.index_select(0, components).flatten()
                        sizes = (
                            lengths.index_select(0, components)
                            .flatten()
                            .view(torch.uint32)
                        )
                        self.context.retain((source, sizes))
                    descriptors.append(
                        (
                            source,
                            dst,
                            sizes,
                            scan,
                            self.fixed_groups[: ids.numel()],
                            packed,
                        )
                    )
                result.append(descriptors)
            return result
        result = []
        order = torch.argsort(
            torch.where(needed, self.ids, self.ids + self.local_experts).float()
        )
        ordered_destinations = self.destinations.gather(
            1, order.unsqueeze(0).expand_as(self.destinations)
        )
        groups = needed.to(torch.int64)
        self.context.retain((order, ordered_destinations, groups))
        for phase in self.copy_plans:
            descriptors = []
            for indices, components, destinations, lengths, scan, ids, packed in phase:
                if len(indices) == 1:
                    source, dst, active_groups = (
                        src[indices[0]],
                        ordered_destinations[indices[0]],
                        groups,
                    )
                else:
                    active = needed.repeat(len(indices))
                    order = torch.argsort(
                        torch.where(active, ids, ids + ids.numel()).float()
                    )
                    dst = destinations.gather(0, order)
                    source = src.index_select(0, components).flatten()
                    active_groups = active.to(torch.int64)
                    self.context.retain((source, dst, active_groups, order, active))
                descriptors.append((source, dst, lengths, scan, active_groups, packed))
            result.append(descriptors)
        return result

    def _submit_copies(self, descriptors, *, phase=0):
        for source, dst, lengths, scan, groups, packed in descriptors:
            # Only the valid packed prefix is written, and no caller consumes
            # its unused tail. No per-layer clearing of this output is needed.
            if self.copy_budget_kernel is None:
                status = _group_pack_copy(
                    source, dst, lengths, scan, groups, packed, self.device
                )
            else:
                status = self.copy_budget_kernel(
                    source,
                    dst,
                    lengths,
                    scan,
                    groups,
                    packed,
                    self.device,
                    self.copy_core_budgets[phase],
                )
            if status is not None and status != 0:
                raise RuntimeError(f"Expert copy submission failed: {status}")

    def wait_for_w2(self, layer_id):
        if self.cfg.get("overlap_w2", False):
            torch.npu.current_stream().wait_event(self.copy_events[layer_id][2])

    def combine_allgather(self, hidden_states, state):
        from sglang.srt.layers.dp_attention import get_is_extend_in_batch

        if (
            not self.cfg.get("fused_allgather_combine", False)
            or get_is_extend_in_batch()
        ):
            return None
        tokens, topk = state.topk_weights.shape
        if tokens > 64 or not all(
            t.is_contiguous()
            for t in (
                hidden_states,
                state.topk_weights,
                state.topk_ids,
                state.expanded_row_idx,
            )
        ):
            return None
        import triton

        from .small_batch_kernels import weighted_local_sum

        output = torch.empty(
            (tokens, hidden_states.shape[1]), dtype=torch.float32, device=self.device
        )
        weighted_local_sum[(tokens, triton.cdiv(hidden_states.shape[1], 256))](
            hidden_states,
            state.expanded_row_idx,
            state.topk_weights,
            state.topk_ids,
            output,
            HIDDEN=hidden_states.shape[1],
            TOPK=topk,
            EXPERTS=self.local_experts,
            Y_ROWS=hidden_states.shape[0],
            BLOCK=256,
            enable_fp_fusion=False,
        )
        return output

    def _publish_sources(self):
        # Cache placement is immutable throughout a forward. Publish one stable
        # device table after initial fill / cache maintenance; graph replays read
        # these addresses without rebuilding forty identical placement queries.
        keys = torch.arange(self.catalog.num_experts, device=self.device)
        slots = self.state.key_to_slot
        safe = slots.clamp(0, self.hot_slots - 1)
        hit = slots >= 0
        coherent = self.state.slot_valid.to(torch.int32).gather(0, safe).bool() & (
            self.state.slot_to_key.gather(0, safe) == keys
        )
        bad = (slots < -1) | (slots >= self.hot_slots) | (hit & ~coherent)
        self.error.bitwise_or_(bad.any().to(torch.int32))
        indices = torch.where(hit, safe + self.catalog.num_experts + 2, keys + 1)
        values = self.sources.gather(
            1, indices.unsqueeze(0).expand(self.sources.shape[0], -1)
        )
        if not hasattr(self, "published_sources"):
            self.published_sources = torch.empty_like(values)
        self.published_sources.copy_(values)

    def _cache_tail(self, maintenance):
        if maintenance and self.cfg.get("incremental_cache", False):
            swap_budget = None
            if self.maintenance_busy_limit:
                # Only current-rank observed work; no Host read or collective.
                light = (self.state.step_access_count > 0).sum() <= (
                    2 * len(self.catalog.offloaded_layers)
                )
                swap_budget = torch.where(
                    light, self.policy.config.max_swaps, self.maintenance_busy_limit
                )
            self.policy.observe(
                self.policy_state,
                self.state.step_access_count,
                self.state.update_enabled,
                self.due,
            )
            self.policy.select_hot_set(
                self.policy_state, self.state.key_to_slot, self.target_keys,
                swap_budget=swap_budget,
            )
            self.policy.finish_step(self.policy_state)
            self.cache_update.apply()
        else:
            self.coordinator.emit_step_end(self.context, maintenance=maintenance)
        if self.cfg.get("cached_source_tables", False) and maintenance:
            self._publish_sources()

    def _capture_cache_tails(self):
        # Capture with updates disabled: neither usage nor placement may change
        # during warmup. Each graph owns a separate pool and retains all raw
        # pointer descriptors for its entire lifetime.
        self.cache_tail_graphs = {}
        self.cache_tail_contexts = {}
        previous = self.context
        for maintenance in (False, True):
            self.context = _Context()
            self._cache_tail(maintenance)
            self.context.record_compute_completion().completion.synchronize()
            self.context.resources.clear()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                self.context = _Context()
                self.context.capturing = True
                self._cache_tail(maintenance)
            torch.npu.synchronize()
            self.cache_tail_graphs[maintenance] = graph
            self.cache_tail_contexts[maintenance] = self.context
        self.context = previous
        self._check_error()

    def _check_error(self):
        flag = self.error.clone()
        if get_parallel().moe_ep_size > 1:
            torch.distributed.all_reduce(
                flag,
                op=torch.distributed.ReduceOp.MAX,
                group=get_moe_ep_group().device_group,
            )
        code = int(flag.item())
        if code:
            raise RuntimeError(
                f"Expert offload device fault (all EP ranks stopped): {code}"
            )

    @contextmanager
    def forward_scope(self, forward_batch=None):
        if not self.ready:
            yield
            return
        self.context = _Context()
        if self.decode_active_rows is not None:
            if forward_batch is None:
                raise ValueError("Decode padding mask requires the original forward batch")
            live = 0 if forward_batch.forward_mode.is_idle() else forward_batch.input_ids.numel()
            self.decode_active_rows.fill_(False)
            self.decode_active_rows[:min(live, self.decode_active_rows.numel())].fill_(True)
        if getattr(self, "router_ahead", None) is not None:
            self.router_ahead.begin()
        if getattr(self, "dispatch_prefetch", None) is not None:
            self.dispatch_prefetch.begin(forward_batch)
        if getattr(self, "resident_diagnostics", None) is not None:
            self.resident_diagnostics.begin()
        self.state.step_access_count.zero_()
        if self.static_replica is not None:
            self.static_replica.begin(forward_batch)
        self.state.update_enabled.fill_(True)
        try:
            yield
            if self.static_replica is not None:
                self.static_replica.finish()
            if getattr(self, "dispatch_prefetch", None) is not None:
                self.dispatch_prefetch.drain()
            self.step += 1
            if getattr(self, "resident_diagnostics", None) is not None:
                self.resident_diagnostics.snapshot(forward_batch, self.step)
            maintenance = self.step % self.policy.config.update_interval == 0
            diagnostic_index = None
            if self.diagnostics is not None:
                diagnostic_index = self.diagnostics.before_maintenance(
                    forward_batch, self.step, maintenance
                )
            if getattr(self, "router_ahead", None) is not None:
                self.router_ahead.snapshot(diagnostic_index)
            if self.cfg.get("graph_cache_tail", False):
                self.cache_tail_graphs[maintenance].replay()
            else:
                self._cache_tail(maintenance)
            if self.diagnostics is not None:
                self.diagnostics.after_maintenance(diagnostic_index, maintenance)
            # One error check per whole forward, outside the captured graph.
            # It also retires cache transactions before shared-bank reuse.
            self._check_error()
            if self.static_replica is not None:
                self.static_replica.maybe_dump()
            if getattr(self, "resident_diagnostics", None) is not None:
                self.resident_diagnostics.maybe_dump()
            if self.diagnostics is not None:
                self.diagnostics.maybe_dump()
            if getattr(self, "router_ahead", None) is not None:
                self.router_ahead.maybe_dump()
            if getattr(self, "dispatch_prefetch", None) is not None:
                self.dispatch_prefetch.maybe_dump()
        finally:
            if getattr(self, "dispatch_prefetch", None) is not None:
                self.dispatch_prefetch.drain()
            if self.cfg.get("overlap_w2", False):
                self.context.compute_stream.wait_stream(self.weight_copy_stream)
            self.context.record_compute_completion().completion.synchronize()
            self.context.resources.clear()
            self.state.update_enabled.fill_(False)
