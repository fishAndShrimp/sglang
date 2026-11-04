"""Expert weights for the native Ascend Ops runner.

Communication and token layout stay in the mainline dispatcher/runner. Decode
uses fixed local slots with independent reuse and partial Hot views; prefill alternates
complete layers across two bank views. Cache publication follows the whole forward.
"""

import json
import logging
import math
import os
from contextlib import contextmanager
from pathlib import Path

import torch

from sglang.srt.distributed.parallel_state import get_moe_ep_group
from sglang.srt.runtime_context import (
    get_device,
    get_exec,
    get_parallel,
    get_schedule,
    get_spec,
)
from sglang.srt.utils.offloader import BaseOffloader

from ...transfer import EventCompletion, TransferTicket
from ..config import Phase
from ..execution import CacheUpdateCoordinator, ExpertWeightAccess
from ..periodic_lfu import PeriodicLFUConfig, PeriodicLFUPolicy
from ..residency import TensorResidencyOps
from ..resources import ResourceKind
from ..state import ResidencyState
from .format import ExpertLayerStorage
from .model_adapter import build_expert_catalog, get_model_id, get_quant_spec
from .model_profile import inspect_model
from .capabilities import (
    COPY_DESCRIPTOR_LIMIT,
    TILED_SOURCE_CATALOG_THRESHOLD,
    descriptor_slices,
    plan_capacity,
    plan_decode_mask,
)
from .native_acl import get_native_acl
from .storage import MemFabricHostPool
from .transfer import (
    MemFabricAddressCatalog,
    MemFabricTransferBindings,
    _group_pack_copy,
)
from .w8a8_format import NpuV4W8A8Format

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
        self.model_profile = inspect_model(hf_config)
        self.model_profile.require_runtime()
        self.model_id = get_model_id(hf_config)
        cfg = json.loads(Path(path).read_text())
        allowed = {
            "cache_ratio",
            "host_pool_gib",
            "offline_stats_path",
            "offline_model_id",
            "native_extension_directory",
            "decode_mask_capacity",
            "execution_slots",
            "update_interval",
            "min_usage",
            "replacement_min_gain",
            "max_swaps",
            "usage_decay",
        }
        if not isinstance(cfg, dict) or set(cfg) - allowed:
            raise ValueError(
                "Unknown offload configuration keys; experimental switches are not supported"
            )
        self.cfg, self.hf_config = cfg, hf_config
        ratio = cfg.get("cache_ratio", 0.25 if self.model_id == "GLM-5.2" else 0.28125)
        if (
            not isinstance(ratio, (int, float))
            or not math.isfinite(ratio)
            or not 0 < ratio <= 1
        ):
            raise ValueError("cache_ratio must be in (0, 1]")
        self.catalog = build_expert_catalog(hf_config, p.moe_ep_size, p.moe_ep_rank)
        if tuple(s.layer_id for s in self.catalog.layers) != self.model_profile.moe_layer_ids:
            raise ValueError("Architecture profile and expert catalog disagree on MoE layers")
        from sglang.srt.environ import envs
        graph_config = get_exec().graph.cuda_graph_config
        graph_bound = (graph_config.decode.max_bs or 0) if graph_config is not None else 0
        tokens_per_request = (get_spec().speculative_num_draft_tokens or 1) if get_spec().speculative_algorithm else 1
        mask_capacity = plan_decode_mask(
            request_bound=get_schedule().max_running_requests or 0,
            graph_bound=graph_bound,
            tokens_per_request=tokens_per_request,
            dispatch_padding=envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get() or 0,
            configured=cfg.get("decode_mask_capacity"),
        )
        self.capacity_plan = plan_capacity(
            global_experts=self.model_profile.global_experts,
            ep_size=p.moe_ep_size,
            offloaded_layers=len(self.catalog.offloaded_layers),
            cache_ratio=ratio,
            execution_slots=cfg.get("execution_slots"),
            decode_mask_capacity=mask_capacity,
        )
        self.local_experts = self.copy_capacity = self.capacity_plan.local_experts
        self.execution_slots = self.capacity_plan.execution_slots
        self.hot_slots = self.capacity_plan.hot_slots
        if self.catalog.num_experts != self.capacity_plan.catalog_size:
            raise ValueError("Execution bank requires a homogeneous expert catalog")
        from .prefill import whole_layer_bank_layout

        whole_layer_bank_layout(
            self.local_experts, self.hot_slots, self.execution_slots
        )
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
            self.hot_slots, cfg.get("offline_model_id", self.model_id)
        )

    def prepare_layer(self, layer, num_experts, hidden_size, intermediate_size, attrs):
        spec = self.catalog.layer(layer.layer_id)
        if spec.resident:
            return False
        if num_experts != spec.local_experts:
            raise ValueError(
                "Offload expects only rank-local routed experts; "
                "use --disable-shared-experts-fusion"
            )
        if (hidden_size, intermediate_size) != (
            self.model_profile.expert_hidden_size, self.model_profile.intermediate_size
        ):
            raise ValueError("Actual expert tensor dimensions disagree with the startup model profile")
        runner_config = getattr(layer, "moe_runner_config", None)
        if runner_config is not None and (
            runner_config.top_k != self.model_profile.top_k
            or runner_config.activation != self.model_profile.activation
        ):
            raise ValueError("Actual expert top-k/activation disagrees with the startup model profile")
        quant_spec = get_quant_spec(layer.w13_scheme, layer.w2_scheme)
        if self.format is None:
            native = get_native_acl(self.cfg.get("native_extension_directory"))
            self.quant_spec = quant_spec
            if quant_spec.kind == "w4a8":
                from .w4a8_format import NpuW4A8Format

                self.format = NpuW4A8Format(
                    hidden_size,
                    intermediate_size,
                    self.device,
                    native_acl=native,
                    group_size=quant_spec.group_size,
                    tp_size=quant_spec.tp_size,
                    activation_use_clip=quant_spec.activation_use_clip,
                )
            else:
                self.format = NpuV4W8A8Format(
                    hidden_size, intermediate_size, self.device, native_acl=native
                )
            total = sum(
                self.format.host_nbytes(s.local_experts)
                for s in self.catalog.offloaded_layers
            )
            host_gib = self.cfg.get("host_pool_gib", 64)
            if type(host_gib) is not int or host_gib <= 0:
                raise ValueError("host_pool_gib must be a positive integer per rank")
            if total > host_gib * 1024**3:
                raise ValueError(
                    f"{self.model_id} {quant_spec.kind} needs at least "
                    f"{math.ceil(total / 1024**3)} GiB host_pool_gib per rank "
                    f"for expert storage; configured {host_gib} GiB"
                )
            self.host_pool = MemFabricHostPool(
                total,
                self.device.index,
                physical_bytes=host_gib * 1024**3,
                native_acl=native,
            )
        if quant_spec != self.quant_spec:
            raise ValueError(
                "This execution bank requires identical expert quantization"
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
        from .capabilities import resolve_decode_capabilities

        self.capabilities = resolve_decode_capabilities(
            quant_kind=self.quant_spec.kind,
            local_experts=self.copy_capacity,
            hot_slots=self.hot_slots,
            execution_slots=self.execution_slots,
            speculative=bool(get_spec().speculative_algorithm),
            disable_direct_hot=os.environ.get("MOE_DIAGNOSTIC_DISABLE_DIRECT_HOT") == "1",
        )
        self.direct_hot_slots = self.capabilities.direct_hot_slots
        self.direct_hot = self.direct_hot_slots > 0
        # Execution identity is independent of the Hot addressing format.
        self.execution_keys = torch.full(
            (self.copy_capacity,), -1, dtype=torch.int64, device=self.device
        )
        if self.direct_hot:
            self.union = self.format.allocate_bank(
                self.execution_slots + self.hot_slots,
                ResourceKind.EXECUTION_WEIGHTS, "ascend.union",
            )
            self.execution = self.union.view(0, self.execution_slots)
            self.hot = self.union.view(self.execution_slots, self.hot_slots)
            groups = self.execution_slots + self.direct_hot_slots
            self.direct_groups = torch.zeros((groups, 2), dtype=torch.int64, device=self.device)
            self.quant_info = self.union.view(0, groups).quant_info()
        else:
            self.execution = self.format.allocate_bank(
                self.execution_slots, ResourceKind.EXECUTION_WEIGHTS, "ascend.execution"
            )
            self.hot = self.format.allocate_bank(
                self.hot_slots, ResourceKind.HOT_WEIGHTS, "ascend.hot"
            )
            self.quant_info = self.execution.view(0, self.copy_capacity).quant_info()
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
            [
                [
                    r.address
                    for r in self.execution.component_regions[n][: self.copy_capacity]
                ]
                for n in names
            ],
            dtype=torch.int64,
            device=self.device,
        )
        self.lengths = torch.tensor(
            [[self.format.layouts[n].nbytes] * self.copy_capacity for n in names],
            dtype=torch.int32,
            device=self.device,
        )
        self._init_copy_plan(names)
        self.error = torch.zeros((), dtype=torch.int32, device=self.device)
        self.state = ResidencyState.allocate(
            self.catalog.num_experts, self.hot_slots, self.local_experts, self.device
        )
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
        self._publish_sources()
        from .cache_update import FusedCacheUpdate

        self.cache_update = FusedCacheUpdate(self)
        self._check_error()
        self.context.resources.clear()
        self.state.update_enabled.fill_(False)  # warmup/capture never trains LFU
        self.decode_active_rows = None
        parallel = get_parallel()
        if parallel.attn_dp_size != parallel.moe_ep_size or parallel.attn_cp_size != 1:
            raise ValueError("Decode row mask requires attention DP=EP and CP=1")
        capacity = self.capacity_plan.decode_mask_capacity
        # Allocated before capture; every graph keeps this same address.
        self.decode_active_rows = torch.zeros(
            capacity, dtype=torch.bool, device=self.device
        )
        from .prefill import AscendPrefillWeights

        self.prefill = AscendPrefillWeights(self)
        self._capture_cache_tails()
        self.diagnostics = None
        if os.getenv("MOE_DIAGNOSTIC_DIR"):
            from .diagnostics import ForwardSnapshots
            self.diagnostics = ForwardSnapshots(self)
        self.ready = True
        self.optimization_manifest = self.capabilities.to_dict()
        self.optimization_manifest.update(
            tp=parallel.tp_size, ep=parallel.moe_ep_size, rank=parallel.moe_ep_rank,
            attention_dp=parallel.attn_dp_size, quant=self.quant_spec.kind,
            catalog=self.catalog.num_experts, decode_mask_capacity=capacity,
            communication="local" if parallel.moe_ep_size == 1 else "mc2",
            cache_update="tiled" if self.cache_update.tiled else "single_program",
            source_publish="tiled" if hasattr(self, "source_publisher") else "framework",
            model_multi_stream=os.environ.get("SGLANG_NPU_USE_MULTI_STREAM"),
            model=self.model_profile.model_id,
            top_k=self.model_profile.top_k,
            expert_hidden_size=self.model_profile.expert_hidden_size,
            metadata_block=self.capacity_plan.metadata_block,
            copy_batches_per_projection=[len(p) for p in self.fused_descriptors],
            weight_bytes_per_expert=self.format.expert_nbytes,
            execution_weight_bytes=self.execution_slots * self.format.expert_nbytes,
            hot_weight_bytes=self.hot_slots * self.format.expert_nbytes,
            host_used_bytes=self.host_pool.used_bytes,
        )
        logger.info("OFFLOAD_CAPABILITIES %s", json.dumps(self.optimization_manifest, sort_keys=True))
        logger.info(
            "Ascend expert offload: local=%d hot=%d/%d execution=%.3f GiB host=%.3f GiB; "
            "W2 overlap=%s, fused metadata=%s, Hot direct=%s, execution reuse=True",
            self.local_experts,
            self.hot_slots,
            self.catalog.num_experts,
            self.format.expert_nbytes * self.execution_slots / 1024**3,
            self.host_pool.used_bytes / 1024**3,
            True,
            True,
            self.direct_hot,
        )

    def _retain(self, value):
        self.context.retain(value)

    def decode_row_mask(self, padded_bs):
        if self.decode_active_rows is None:
            return None
        if padded_bs > self.decode_active_rows.numel():
            raise ValueError("MC2 padded batch exceeds decode_mask_capacity")
        return self.decode_active_rows[:padded_bs]

    def start_prefill_layer(self, layer_id):
        self.prefill.pipeline.start(layer_id, self.context)

    def prepare_weights(self, layer_id, runner_input):
        if not self.ready:
            raise RuntimeError("Expert bank is not initialized")
        from sglang.srt.layers.dp_attention import get_is_extend_in_batch

        if get_is_extend_in_batch():
            return self.prefill.weights(layer_id, runner_input)
        return self._prepare_fused(layer_id, runner_input)

    def _launch_copies(self, layer_id, copies):
        ready, first, second = self.copy_events[layer_id]
        compute = torch.npu.current_stream()
        # This frontier also follows the previous layer's GMM2, so neither the
        # execution weights nor their packed counts are overwritten while read.
        ready.record(compute)
        with torch.npu.stream(self.weight_copy_stream):
            self.weight_copy_stream.wait_event(ready)
            self._submit_copies(copies[0])
            first.record(self.weight_copy_stream)
            self._submit_copies(copies[1])
            second.record(self.weight_copy_stream)
        compute.wait_event(first)

    def _init_copy_plan(self, names):
        # Checkpoint-only W4 metadata is not read by GMM. W8 offsets are
        # separately validated as zero and remain valid without decode copies.
        selected = []
        for i, name in enumerate(names):
            if self.format.layouts[name].runtime_name is None:
                continue
            if name.endswith("_offset"):
                self.execution.tensors[name].zero_()
            else:
                selected.append(i)
        phases = [
            [i for i in selected if names[i].startswith(prefix)]
            for prefix in ("w13", "w2")
        ]
        self.weight_copy_stream = torch.npu.Stream(device=self.device)
        self.copy_events = {
            s.layer_id: tuple(torch.npu.Event() for _ in range(3))
            for s in self.catalog.offloaded_layers
        }
        self.fused_components = tuple(selected)
        if any(component >= 16 for component in selected):
            raise ValueError("Component map supports at most 16 format components")
        self.fused_component_map = sum(c << (4 * i) for i, c in enumerate(selected))
        self._init_fixed_copy_plan(phases)
        self.metadata_block = 1 << (self.copy_capacity - 1).bit_length()
        self.tiled_prepare = None
        if self.copy_capacity > 64:
            from .small_batch_tiled import TiledPreparation
            self.tiled_prepare = TiledPreparation(self)

    def _init_fixed_copy_plan(self, phases):
        # Direct Hot GMM indices address fixed physical slots. Keep one copy
        # group per slot, including zero-byte copies, so misses are never packed.
        self.fixed_groups = torch.ones(COPY_DESCRIPTOR_LIMIT, dtype=torch.int64, device=self.device)
        self.fused_sources = torch.empty(
            len(self.fused_components) * self.copy_capacity,
            dtype=torch.int64,
            device=self.device,
        )
        self.fused_lengths = torch.empty_like(self.fused_sources, dtype=torch.uint32)
        self.fused_counts = torch.empty(
            self.copy_capacity, dtype=torch.int64, device=self.device
        )
        self.fused_descriptors = []
        offset = 0
        for phase in phases:
            descriptors = []
            components = torch.tensor(phase, device=self.device)
            destinations = self.destinations.index_select(0, components).flatten()
            extent = len(phase) * self.copy_capacity
            for begin, end in descriptor_slices(self.copy_capacity, len(phase)):
                count = end - begin
                descriptors.append(
                    (
                        self.fused_sources[offset + begin : offset + end],
                        destinations[begin:end],
                        self.fused_lengths[offset + begin : offset + end],
                        torch.tensor(count, dtype=torch.uint32, device=self.device),
                        self.fixed_groups[:count],
                        torch.zeros(count, dtype=torch.int64, device=self.device),
                    )
                )
            offset += extent
            self.fused_descriptors.append(descriptors)

    def _prepare_fused(self, layer_id, runner_input):
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
        if self.tiled_prepare is not None:
            self.tiled_prepare.run(
                counts, self.state.step_access_count[start:end],
                self.published_sources[:, start:end], start,
                runner_input.group_list_type == 0,
            )
        else:
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
                SKIP_OFFSETS=True,
                COMPONENT_MAP=self.fused_component_map,
                CUMULATIVE=runner_input.group_list_type == 0,
                BLOCK=self.metadata_block,
                DIRECT_HOT=self.direct_hot,
                DIRECT_HOT_CAPACITY=self.direct_hot_slots,
                HOT_OFFSET=self.execution_slots,
                HotSlots=self.state.key_to_slot,
                HOT_CAPACITY=self.hot_slots,
                Groups=getattr(self, "direct_groups", self.fused_counts),
                ExecutionKeys=self.execution_keys,
                KEY_START=start,
            )
        if self.direct_hot:
            runner_input.expert_tokens = self.direct_groups
            runner_input.group_list_type = 2
        else:
            runner_input.expert_tokens = self.fused_counts
        self._launch_copies(layer_id, self.fused_descriptors)
        return self.quant_info

    def _submit_copies(self, descriptors):
        for source, dst, lengths, scan, groups, packed in descriptors:
            # GMM uses logical counts or physical-slot groups, independently
            # of these native copy outputs; no per-copy output clearing.
            status = _group_pack_copy(
                source, dst, lengths, scan, groups, packed, self.device
            )
            if status is not None and status != 0:
                raise RuntimeError(f"Expert copy submission failed: {status}")

    def wait_for_w2(self, layer_id):
        from sglang.srt.layers.dp_attention import get_is_extend_in_batch

        if get_is_extend_in_batch():
            self.prefill.pipeline.wait_for_w2(layer_id, self.context)
            return
        torch.npu.current_stream().wait_event(self.copy_events[layer_id][2])

    def _publish_sources(self):
        if self.catalog.num_experts > TILED_SOURCE_CATALOG_THRESHOLD:
            if not hasattr(self, "source_publisher"):
                from .source_publish import TiledSourcePublisher

                self.source_publisher = TiledSourcePublisher(self)
                self.published_sources = self.source_publisher.output
            self.source_publisher.run()
            return
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
        if maintenance:
            self.policy.observe(
                self.policy_state,
                self.state.step_access_count,
                self.state.update_enabled,
                self.due,
            )
            self.policy.select_hot_set(
                self.policy_state, self.state.key_to_slot, self.target_keys
            )
            self.policy.finish_step(self.policy_state)
            self.cache_update.apply()
        else:
            self.coordinator.emit_step_end(self.context, maintenance=maintenance)
        if maintenance:
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
                raise ValueError(
                    "Decode padding mask requires the original forward batch"
                )
            live = (
                0
                if forward_batch.forward_mode.is_idle()
                else forward_batch.input_ids.numel()
            )
            if not forward_batch.is_extend_in_batch and live > self.decode_active_rows.numel():
                raise ValueError(
                    f"Decode has {live} token rows but startup mask capacity is "
                    f"{self.decode_active_rows.numel()}; increase the planned request/graph bounds"
                )
            self.decode_active_rows.fill_(False)
            self.decode_active_rows[: min(live, self.decode_active_rows.numel())].fill_(
                True
            )
        if self.diagnostics is not None:
            self.diagnostics.begin(forward_batch)
        self.state.step_access_count.zero_()
        self.state.update_enabled.fill_(True)
        try:
            # RuntimeContext's flag is installed later by _forward_raw; use this
            # batch's global phase, including locally idle/decode attention ranks.
            if forward_batch is not None and forward_batch.is_extend_in_batch:
                # Prefill overwrites execution banks independently of decode's
                # reuse keys. Invalidate before any prefetch, including on idle
                # attention ranks, so the next decode cannot reuse stale bytes.
                if self.execution_keys is not None:
                    self.execution_keys.fill_(-1)
                self.prefill.pipeline.begin_forward(self.context)
            yield
            self.prefill.pipeline.finish_forward(self.context)
            if self.diagnostics is not None:
                self.diagnostics.before_maintenance()
            self.step += 1
            maintenance = self.step % self.policy.config.update_interval == 0
            self.cache_tail_graphs[maintenance].replay()
            if self.diagnostics is not None:
                self.diagnostics.after_maintenance(maintenance)
            # One error check per whole forward, outside the captured graph.
            # It also retires cache transactions before shared-bank reuse.
            self._check_error()
            if self.diagnostics is not None:
                self.diagnostics.flush_if_complete()
        finally:
            try:
                self.prefill.restore_hot()
            finally:
                self.context.compute_stream.wait_stream(self.weight_copy_stream)
                self.context.record_compute_completion().completion.synchronize()
                self.context.resources.clear()
                self.prefill.pipeline.reset()
                self.state.update_enabled.fill_(False)
