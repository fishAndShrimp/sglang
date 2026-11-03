"""Expert weights for the native Ascend Ops runner.

Communication and token layout stay in the mainline dispatcher/runner. Decode
stably packs active experts into one execution bank; eager prefill alternates
complete layers across two bank views. Cache publication follows the whole forward.
"""

import json
import logging
import math
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
        if hf_config.model_type != "deepseek_v4":
            raise ValueError(
                "Initial format/model adapter supports DeepSeek V4 W8A8 only"
            )
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
        self.copy_capacity = self.local_experts
        self.execution_slots = cfg.get("execution_slots", self.local_experts)
        if type(self.execution_slots) is not int:
            raise ValueError("execution_slots must be an integer")
        # MemFabric batches use at most 256 entries on A3. Larger local expert
        # counts need explicit chunking; never overrun the native metadata UB.
        if self.copy_capacity > 256:
            raise ValueError("Packed expert copies support at most 256 local experts")
        self.hot_slots = int(self.catalog.num_experts * ratio)
        if self.hot_slots < 1:
            raise ValueError("The initial implementation requires a nonempty Hot cache")
        if self.hot_slots > 1280 or self.catalog.num_experts > 8192:
            raise ValueError(
                "Cache maintenance supports at most 1280 Hot slots and 8192 expert keys per rank"
            )
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
            self.hot_slots, cfg.get("offline_model_id", "DeepSeek-V4-Flash")
        )

    def prepare_layer(self, layer, num_experts, hidden_size, intermediate_size, attrs):
        spec = self.catalog.layer(layer.layer_id)
        if spec.resident:
            return False
        if (
            num_experts != spec.local_experts
            or type(layer.w13_scheme).__name__ != "ModelSlimW8A8Int8MoE"
            or type(layer.w2_scheme).__name__ != "ModelSlimW8A8Int8MoE"
        ):
            raise ValueError("Offload expects local routed experts with ModelSlim W8A8")
        if self.format is None:
            native = get_native_acl(self.cfg.get("native_extension_directory"))
            self.format = NpuV4W8A8Format(
                hidden_size, intermediate_size, self.device, native_acl=native
            )
            total = sum(
                self.format.host_nbytes(s.local_experts)
                for s in self.catalog.offloaded_layers
            )
            self.host_pool = MemFabricHostPool(
                total,
                self.device.index,
                physical_bytes=int(self.cfg.get("host_pool_gib", 64)) * 1024**3,
                native_acl=native,
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
        capacity = self.cfg.get("decode_mask_capacity", 128)
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("decode_mask_capacity must be positive")
        # Allocated before capture; every graph keeps this same address.
        self.decode_active_rows = torch.zeros(
            capacity, dtype=torch.bool, device=self.device
        )
        from .prefill import AscendPrefillWeights

        self.prefill = AscendPrefillWeights(self)
        self._capture_cache_tails()
        self.ready = True
        logger.info(
            "Ascend expert offload: local=%d hot=%d/%d execution=%.3f GiB host=%.3f GiB; packed expert copies, W2 overlap=%s, fused metadata=%s",
            self.local_experts,
            self.hot_slots,
            self.catalog.num_experts,
            self.format.expert_nbytes * self.execution_slots / 1024**3,
            self.host_pool.used_bytes / 1024**3,
            True,
            True,
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
        # Zero offsets are a checked invariant of this format, not an assumption
        # about arbitrary quantized checkpoints. They remain valid in every slot.
        selected = []
        for i, name in enumerate(names):
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
        self.fused_sources = torch.empty(
            (len(selected), self.copy_capacity),
            dtype=torch.int64,
            device=self.device,
        )
        self.copy_groups = torch.empty(
            self.copy_capacity, dtype=torch.int64, device=self.device
        )
        self.copy_scan = torch.tensor(
            self.copy_capacity, dtype=torch.int32, device=self.device
        )
        self.fused_descriptors = []
        for phase in phases:
            descriptors = []
            for component in phase:
                # Each component packs the same expert sequence into its own
                # destination prefix. Flattening components together would put
                # the scale's first output at slot M of the weight address table.
                packed = torch.zeros_like(self.copy_groups)
                descriptors.append(
                    (
                        self.fused_sources[selected.index(component)],
                        self.destinations[component],
                        self.lengths[component],
                        self.copy_scan,
                        self.copy_groups,
                        packed,
                    )
                )
            self.fused_descriptors.append(descriptors)
        # GMM reads the first W13 copy's output after the W13 event. All other
        # calls own separate outputs, especially W2 copies overlapping GMM1.
        self.packed_counts = self.fused_descriptors[0][0][-1]

    def _prepare_fused(self, layer_id, runner_input):
        import triton

        from .small_batch_kernels import prepare_packed_copies

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
        prepare_packed_copies[(1,)](
            counts,
            self.error,
            self.state.update_enabled,
            self.state.step_access_count[start:end],
            self.published_sources[:, start:end],
            self.copy_groups,
            self.fused_sources,
            EXPERTS=self.local_experts,
            SOURCE_STRIDE=self.catalog.num_experts,
            START=0,
            COMPONENT_COUNT=len(self.fused_components),
            SKIP_OFFSETS=True,
            CUMULATIVE=runner_input.group_list_type == 0,
            BLOCK=triton.next_power_of_2(self.local_experts),
        )
        self._launch_copies(layer_id, self.fused_descriptors)
        # Dispatch already sorts rows by local expert. Removing zero-token
        # groups preserves row order, including the original combine metadata.
        runner_input.expert_tokens = self.packed_counts
        runner_input.group_list_type = 1
        return self.quant_info

    def _submit_copies(self, descriptors):
        for source, dst, lengths, scan, groups, packed in descriptors:
            # Native packing leaves the unused tail untouched. GMM consumes a
            # fixed-capacity count vector, so stale groups must be cleared.
            packed.zero_()
            status = _group_pack_copy(
                source, dst, lengths, scan, groups, packed, self.device
            )
            if status is not None and status != 0:
                raise RuntimeError(f"Expert copy submission failed: {status}")

    def wait_for_w2(self, layer_id):
        from sglang.srt.layers.dp_attention import get_is_extend_in_batch

        if get_is_extend_in_batch():
            return
        torch.npu.current_stream().wait_event(self.copy_events[layer_id][2])

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
            self.decode_active_rows.fill_(False)
            self.decode_active_rows[: min(live, self.decode_active_rows.numel())].fill_(
                True
            )
        self.state.step_access_count.zero_()
        self.state.update_enabled.fill_(True)
        try:
            yield
            self.prefill.pipeline.finish_forward(self.context)
            self.step += 1
            maintenance = self.step % self.policy.config.update_interval == 0
            self.cache_tail_graphs[maintenance].replay()
            # One error check per whole forward, outside the captured graph.
            # It also retires cache transactions before shared-bank reuse.
            self._check_error()
        finally:
            try:
                self.prefill.restore_hot()
            finally:
                self.context.compute_stream.wait_stream(self.weight_copy_stream)
                self.context.record_compute_completion().completion.synchronize()
                self.context.resources.clear()
                self.prefill.pipeline.reset()
                self.state.update_enabled.fill_(False)
