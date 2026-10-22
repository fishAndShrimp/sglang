"""Ascend invocation, model-hook and full-graph lifecycle binding.

Native dependencies are initialized explicitly by the backend.
The native extension, registered master and EP/FFN adapters are assembled by
``NpuMoEBackend``; no capture-time ticket is used to certify a later replay.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import wraps

from ...transfer import EventCompletion, TransferTicket
from ..config import Phase
from ..interfaces import BoundMoERuntime, RuntimeCapabilities
from ..resources import Allocation, AllocationLedger, ResourceKind
from ..runtime import ManagedExecutionContext

logger = logging.getLogger(__name__)


class NpuDeviceRuntime:
    def __init__(self, device):
        import torch
        import torch_npu  # noqa: F401

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type != "npu" or self.device.index is None:
            raise ValueError("Offload requires an explicit npu:<rank device>")
        with torch.npu.device(self.device):
            self.copy_stream = torch.npu.Stream(device=self.device)

    def current_stream(self):
        return self.torch.npu.current_stream(self.device)

    @contextmanager
    def stream_scope(self, stream):
        with self.torch.npu.device(self.device), self.torch.npu.stream(stream):
            yield

    def record(self, stream):
        with self.stream_scope(stream):
            event = self.torch.npu.Event()
            event.record(stream)
        return TransferTicket(EventCompletion(event), (event, stream))

    def synchronize(self):
        # Failure/phase transition/teardown only. Includes internal EP streams.
        self.torch.npu.synchronize(self.device)

    def allocated(self):
        return self.torch.npu.memory_allocated(self.device)

    def reset_peak(self):
        self.torch.npu.reset_peak_memory_stats(self.device)

    def peak(self):
        return self.torch.npu.max_memory_allocated(self.device)


class _LayerProgramMap:
    def __init__(self, runtime, phase):
        self.runtime, self.phase = runtime, phase

    def layer_start(self, layer_id, context):
        programs = self.runtime.layer_programs[layer_id]
        program = programs.prefill if self.phase is Phase.PREFILL else programs.decode
        program.layer_start(layer_id, context)

    def run(self, layer, hidden_states, routing, context):
        programs = self.runtime.layer_programs[layer]
        program = programs.prefill if self.phase is Phase.PREFILL else programs.decode
        return program.run(layer, hidden_states, routing, context)


class NpuBoundMoERuntime(BoundMoERuntime):
    """Concrete lifecycle around statically assembled Ascend layer programs.

    Capabilities describe the emitted full-graph path. Startup capture still has
    to succeed on the installed stack; this is not a hardware certification.
    Eager decode is rejected instead of silently falling back after graph miss.
    """

    capabilities = RuntimeCapabilities(True, True, True)

    def __init__(
        self,
        *,
        device_runtime,
        residency,
        residency_ops,
        cache_updates,
        snapshot,
        reload_weights,
        layer_programs,
        resources,
        expert_nbytes,
        storage,
        cache_stats=None,
        diagnostics=None,
        pipelines=(),
        release=None,
    ):
        self.device_runtime = device_runtime
        self.residency = residency
        self.ops = residency_ops
        self.cache_updates = cache_updates
        self.cache_stats = cache_stats
        self.diagnostics = diagnostics
        self.snapshot = snapshot
        self.reload_weights = reload_weights
        self.layer_programs = dict(layer_programs)
        if not self.layer_programs:
            raise ValueError("At least one offloaded layer must be bound")
        self.resources = tuple(resources)
        self.expert_nbytes = expert_nbytes
        self.storage = tuple(storage)
        self.pipelines = tuple(pipelines)
        self._release = release
        self.prefill_program = _LayerProgramMap(self, Phase.PREFILL)
        self.decode_program = _LayerProgramMap(self, Phase.DECODE)
        self._context = None
        self._step_started = False
        self._capture = None
        self._hooks = []
        self._method_hooks = []
        self._experts = []
        self._failed_contexts = []
        self._transition_contexts = []
        self._closed = False
        self._installed = False
        self._owner = None
        self._static_allocated = None
        self._captured_workspace = 0
        self._cache_initialized = False
        self.residency.update_enabled.fill_(True)

    def initialize_hot_cache(self, keys):
        """Materialize offline Top-K through the production transfer/publication.

        No synthetic decode step runs here: online usage and the periodic clock
        remain zero. The same fault latch and complete-bundle fence used during
        serving protect initial publication before any graph can capture banks.
        """
        import torch

        self._require_open()
        if self._installed or self._cache_initialized:
            raise RuntimeError("Initialize the hot cache exactly once before binding")
        keys = tuple(keys)
        if len(keys) != self.residency.slot_to_key.numel():
            raise ValueError("Offline Top-K must fill the configured hot capacity")
        if len(set(keys)) != len(keys) or any(
            type(key) is not int or not 0 <= key < self.residency.key_to_slot.numel()
            for key in keys
        ):
            raise ValueError("Offline Top-K must contain distinct catalog keys")

        def initialize(context):
            targets = self.cache_updates.target_keys
            source = torch.tensor(keys, dtype=torch.int64, device=targets.device)
            context.retain(source)
            targets.copy_(source)
            return self.cache_updates.access.apply_update(targets, context)

        ticket = self._transition(initialize)
        ticket.completion.synchronize()
        if self.ops.error_status.item() != 0:
            raise RuntimeError(
                "Device residency/transfer fault during cache initialization"
            )
        self._transition_contexts.clear()
        self._cache_initialized = True
        logger.info(
            "MoE offload initial cache ready: slots=%s layers=%s",
            len(keys),
            len(self.layer_programs),
        )

    def _new_context(self, phase, *, stream=None, capturing=False):
        context = ManagedExecutionContext(
            phase,
            compute_stream=stream or self.device_runtime.current_stream(),
            copy_stream=self.device_runtime.copy_stream,
            record_completion=self.device_runtime.record,
            failure_drain=self.synchronize,
            capturing=capturing,
        )
        context.started_layers = set()
        return context

    def _require_open(self):
        if self._closed:
            raise RuntimeError("MoE runtime is closed")

    @property
    def context(self):
        self._require_open()
        if self._context is None:
            raise RuntimeError("MoE execution requires a forward or capture scope")
        return self._context

    def install_model_hooks(self, offloader, model_runner):
        self._require_open()
        if self._installed:
            raise RuntimeError("Model hooks are already installed")
        self._owner = offloader.owner
        found = {}
        # register_forward_pre_hook runs before Router, allowing eager H2D/D
        # overlap. Routed experts enter through FusedMoE.forward_impl below.
        try:
            for module in model_runner.model.modules():
                mlp = getattr(module, "mlp", None)
                layer_id = getattr(mlp, "layer_id", None)
                experts = getattr(mlp, "experts", None)
                if layer_id not in self.layer_programs or experts is None:
                    continue
                if not getattr(mlp, "is_deepseek_v4", False):
                    continue
                if layer_id in found:
                    raise ValueError("Duplicate V4 MoE layer hooks")
                if getattr(experts, "_moe_offload_runtime", None) is not None:
                    raise ValueError("Experts already belong to another offloader")
                found[layer_id] = module
                direct = getattr(module, "forward_hc_pre_from_prev", None)
                if direct is not None:
                    self._wrap_direct_layer_entry(module, layer_id, direct)
                experts._moe_offload_runtime = self
                self._experts.append(experts)
                self._hooks.append(
                    module.register_forward_pre_hook(self._layer_start_hook(layer_id))
                )
            if set(found) != set(self.layer_programs):
                raise ValueError("Model does not contain every bound V4 MoE layer")
            model_runner.capture_head_hooks.append(self._capture_head)
            model_runner.capture_tail_hooks.append(self._capture_tail)
            self._runner = model_runner
            self._installed = True
        except BaseException:
            self._remove_hooks()
            raise

    def _enter_layer(self, layer_id):
        self._ensure_step()
        context = self.context
        if layer_id not in context.started_layers:
            self.emit_layer_start(layer_id, context)
            context.started_layers.add(layer_id)

    def _layer_start_hook(self, layer_id):
        def hook(module, args):
            self._enter_layer(layer_id)

        return hook

    def _wrap_direct_layer_entry(self, module, layer_id, original):
        # NPU V4 calls this method directly, bypassing nn.Module.__call__ hooks.
        # Keep the signature and restore the original instance/class lookup at
        # teardown. The per-context set prevents duplicate starts if __call__
        # delegates to the same method on another model path.
        name = "forward_hc_pre_from_prev"
        had_instance = name in module.__dict__
        previous = module.__dict__.get(name)

        @wraps(original)
        def wrapped(*args, **kwargs):
            self._enter_layer(layer_id)
            return original(*args, **kwargs)

        self._method_hooks.append((module, name, had_instance, previous))
        setattr(module, name, wrapped)

    def _ensure_step(self):
        context = self.context
        if context.phase is Phase.DECODE and self._capture is None:
            raise RuntimeError("MoE offload decode must use a captured full graph")
        if not self._step_started:
            self.emit_step_begin(context)
            self._step_started = True

    def emit_step_begin(self, context):
        if context.phase is Phase.DECODE:
            self.ops.begin_step(context)

    def emit_layer_start(self, layer_id, context):
        if context.phase is Phase.PREFILL:
            # All layers share A/B storage. Include the preceding layer/forward
            # reader frontier before its next worker may overwrite either bank.
            frontier = context.record_compute_completion()
            frontier.completion.wait_on(context.copy_stream)
        super().emit_layer_start(layer_id, context)

    def emit_after_router(self, layer_id, routing, context):
        # Counters are updated by residency from complete post-dispatch demand.
        # Router-local top-k is retained, never substituted for received counts.
        if layer_id not in self.layer_programs:
            raise ValueError("Unbound expert layer")
        context.retain(routing)

    def run_layer(self, layer_id, hidden_states, routing, *, pre_quant_input=None):
        self._ensure_step()
        if pre_quant_input is not None:
            raise ValueError("Prequantized routed side streams are not supported")
        context = self.context
        # Eager preprocessing may replace the local-padding Tensor after
        # forward_scope entry. Read the finalized batch binding at actual D.
        context.num_token_non_padded = context.forward_batch.moe_num_token_non_padded()
        program = (
            self.prefill_program
            if context.phase is Phase.PREFILL
            else self.decode_program
        )
        # Dispatch/combine join native communication onto compute_stream. This
        # matches resident inference: release receive tensors/handles after C so
        # the graph allocator can reuse them for the next layer. Events and copy
        # descriptors remain pinned by the invocation; failed scopes keep tensors.
        with context.temporary_scope():
            self.emit_after_router(layer_id, routing, context)
            return program.run(layer_id, hidden_states, routing, context)

    @staticmethod
    def _phase(forward_batch):
        mode = forward_batch.forward_mode
        if mode.is_decode_or_idle():
            return Phase.DECODE
        if mode.is_extend_without_speculative():
            return Phase.PREFILL
        raise ValueError("MoE offload accepts decode/idle or ordinary extend only")

    @staticmethod
    def _has_global_work(forward_batch):
        for name in ("original_global_num_tokens_cpu", "global_num_tokens_cpu"):
            counts = getattr(forward_batch, name, None)
            if counts is not None:
                return any(count > 0 for count in counts)
        count = getattr(forward_batch, "global_num_token_non_padded_cpu", None)
        if count is not None:
            return count > 0
        return forward_batch.input_ids.numel() > 0

    @contextmanager
    def forward_scope(self, owner, forward_batch):
        self._require_open()
        if self._context is not None or self._capture is not None:
            raise RuntimeError("Concurrent forwards/capture are not supported")
        phase = self._phase(forward_batch)
        stream = self.device_runtime.current_stream()
        token = owner.begin_forward(phase, stream)
        context = self._new_context(phase, stream=stream)
        self._context, self._step_started = context, False
        logger.debug(
            "MoE offload forward.begin: seq=%s phase=%s bs=%s tokens=%s",
            token.sequence,
            phase.value,
            getattr(forward_batch, "batch_size", 0),
            getattr(getattr(forward_batch, "input_ids", None), "numel", lambda: 0)(),
        )
        try:
            if phase is Phase.DECODE:
                # Shared host metadata decides whether the whole batch is idle;
                # a rank receiving no routes still advances with active peers.
                # The graph consumes the live fixed-address flag on replay.
                with self.device_runtime.stream_scope(stream):
                    self.residency.update_enabled.fill_(
                        self._has_global_work(forward_batch)
                    )
            context.forward_batch = forward_batch
            context.num_token_non_padded = None
            context.retain(forward_batch)
            yield context
            # Replay does not run Python hooks. Its tail was emitted in capture;
            # record fresh events only, without updating the policy twice.
            tail = self.emit_step_end(context) if self._step_started else None
            record = None
            if self.diagnostics is not None:
                try:
                    runner = getattr(
                        getattr(self, "_runner", None), "decode_cuda_graph_runner", None
                    )
                    record = self.diagnostics.snapshot(
                        phase,
                        token.sequence,
                        forward_batch,
                        bucket=getattr(runner, "bs", None)
                        if phase is Phase.DECODE
                        else None,
                    )
                    context.retain(record)
                except Exception:
                    logger.warning(
                        "MoE offload batch snapshot unavailable", exc_info=True
                    )
            completion = context.finish(residency_ready=tail)
            owner.finish_forward(token, completion)
            if self.diagnostics is not None:
                try:
                    self.diagnostics.publish(record, completion.output_ready.completion)
                except Exception:
                    logger.warning(
                        "MoE offload batch log publication failed", exc_info=True
                    )
            logger.debug(
                "MoE offload forward.submitted: seq=%s phase=%s",
                token.sequence,
                phase.value,
            )
        except BaseException:
            logger.exception(
                "MoE offload forward.failed: seq=%s phase=%s",
                token.sequence,
                phase.value,
            )
            self._failed_contexts.append(context)
            owner.fail_forward(token)
            # Stop workers before removing their active retention context. Any
            # failed drain leaves all contexts pinned for an explicit close retry.
            self.synchronize()
            raise
        finally:
            self._context, self._step_started = None, False

    def _capture_head(self, graph_runner, forward_batch, num_tokens):
        if self._capture is None or self._context is not None:
            raise RuntimeError("Invalid offload capture invocation")
        stream = self.device_runtime.current_stream()
        context = self._new_context(Phase.DECODE, stream=stream, capturing=True)
        context.forward_batch = forward_batch
        context.num_token_non_padded = None
        context.retain(forward_batch)
        self._capture.append(context)
        self._context = context
        self.emit_step_begin(context)
        self._step_started = True

    def _capture_tail(self, graph_runner, output, forward_batch, num_tokens):
        if self._capture is None or not self._step_started:
            raise RuntimeError("Capture tail requires its matching head")
        tail = self.emit_step_end(self.context)
        self.context.finish(residency_ready=tail)
        self._context, self._step_started = None, False

    @contextmanager
    def capture_scope(self, owner, graph_runner, stream):
        self._require_open()
        if self._capture is not None or self._context is not None:
            raise RuntimeError("Nested MoE graph capture")
        self.synchronize()
        self.residency.update_enabled.fill_(False)
        self.synchronize()  # flag must be visible before any warmup stream
        if self._static_allocated is None:
            # KV pools and graph input buffers are allocated after model hooks;
            # take the baseline only when the first capture session starts.
            self._static_allocated = self.device_runtime.allocated()
        self.device_runtime.reset_peak()
        retained = []
        self._capture = retained
        try:
            yield retained
            if self._context is not None:
                raise RuntimeError("Capture model invocation did not finish")
            self.synchronize()
            if self.ops.error_status.item() != 0:
                raise RuntimeError("Device residency fault during graph warmup")
            self._account_capture_workspace(owner)
        except BaseException:
            self._failed_contexts.extend(retained)
            raise
        finally:
            self._capture = None
            self._context, self._step_started = None, False
            # Outside the captured graph: replay observes the live device flag.
            self.residency.update_enabled.fill_(True)
            self.synchronize()

    def after_graph_warmup(self):
        self._require_open()
        if self._capture is None or self._context is not None or not self._capture:
            raise RuntimeError("Warmup retirement requires a finished invocation")
        # capture_one invokes its reset hook after one eager forward, before the
        # next warmup/capture. Older entries may belong to captured graph buckets.
        context = self._capture[-1]
        self.synchronize()  # Includes native EP streams and partially queued work.
        context.release_completed()
        self._capture.pop()

    def _account_capture_workspace(self, owner):
        if self._static_allocated is None:
            self._static_allocated = self.device_runtime.allocated()
        used = max(0, self.device_runtime.peak() - self._static_allocated)
        self._captured_workspace = max(used, self._captured_workspace)
        dynamic = Allocation(
            "npu.capture_and_temporary_workspace",
            ResourceKind.WORKSPACE,
            self._captured_workspace,
            ("capture_workspace", id(self)),
            self,
        )
        # Includes whole-model capture growth conservatively; no expert payload
        # can hide in an uncharged graph-pool allocation.
        allocations = self.resources + (dynamic,)
        ledger = AllocationLedger(owner.allocation_ledger.limits, allocations)
        owner.allocation_ledger = ledger

    def _transition(self, operation):
        if self._context is not None:
            raise RuntimeError("Phase transition cannot overlap a forward")
        # The owner drained prior work before entering this transition.
        self._transition_contexts.clear()
        context = self._new_context(Phase.DECODE)
        self._transition_contexts.append(context)
        self._context = context
        try:
            with self.device_runtime.stream_scope(context.compute_stream):
                ticket = operation(context)
                completion = context.finish(residency_ready=ticket)
            return completion.resources_idle
        except BaseException:
            self._failed_contexts.append(context)
            raise
        finally:
            self._context = None

    def snapshot_decode_state(self):
        return self._transition(self.snapshot.save)

    def borrow_hot_arena(self):
        # Physical aliases are constructed before capture; no allocation/rebind.
        self.residency.execution_valid.zero_()

    def restore_decode_state(self):
        ticket = self._transition(
            lambda context: self.snapshot.restore(context, self.reload_weights)
        )
        # Transition tickets are synchronized by the owner. Retain their context
        # until that fence, then release at the next transition or teardown.
        return ticket

    def synchronize(self):
        contexts = list(self._failed_contexts)
        if self._context is not None:
            contexts.append(self._context)
        abort_error = None
        for context in {id(item): item for item in contexts}.values():
            if context.phase is not Phase.PREFILL or context._closed:
                continue
            for pipeline in self.pipelines:
                try:
                    ticket = pipeline.abort(context)
                    context.retain(ticket)
                except BaseException as error:
                    # A failed event record can follow a successful worker join.
                    # Still stop the other submitters and attempt a device drain;
                    # retain all resources until a later abort succeeds as well.
                    if abort_error is None:
                        abort_error = error
        self.device_runtime.synchronize()
        if abort_error is not None:
            raise abort_error

    def _remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        for module, name, had_instance, previous in self._method_hooks:
            if had_instance:
                setattr(module, name, previous)
            else:
                delattr(module, name)
        self._method_hooks.clear()
        for experts in self._experts:
            if getattr(experts, "_moe_offload_runtime", None) is self:
                del experts._moe_offload_runtime
        self._experts.clear()
        runner = getattr(self, "_runner", None)
        if runner is not None:
            for hooks, callback in (
                (runner.capture_head_hooks, self._capture_head),
                (runner.capture_tail_hooks, self._capture_tail),
            ):
                if callback in hooks:
                    hooks.remove(callback)

    def close(self):
        if self._closed:
            return
        self.synchronize()
        if self.diagnostics is not None:
            try:
                self.diagnostics.close()
            except Exception:
                logger.warning(
                    "MoE offload batch logger teardown failed", exc_info=True
                )
        self._remove_hooks()
        if self._release is not None:
            self._release()
        self._failed_contexts.clear()
        self._transition_contexts.clear()
        self.layer_programs.clear()
        self.resources = self.storage = self.pipelines = ()
        self.snapshot = self.cache_updates = self.ops = self.residency = None
        self.cache_stats = self.diagnostics = None
        self.reload_weights = self._release = None
        self._closed = True
