"""Reusable invocation and exact-restore machinery for accelerator bindings.

No device is initialized by importing this module. Native stream/event factories
and weight reloads are supplied by the binding; none of these helpers certifies
an accelerator graph or creates another expert-weight allocation.
"""

import logging

from contextlib import contextmanager
from dataclasses import dataclass, fields
from threading import RLock, get_ident
from weakref import ref

from ..transfer import Completion, JoinedCompletion, TransferTicket
from .config import Phase
from .interfaces import BoundMoERuntime, ExecutionContext, ForwardCompletion
from .resources import Allocation, ResourceKind

logger = logging.getLogger(__name__)


class ManagedExecutionContext(ExecutionContext):
    """Thread-safe keepalives and fresh completion events for one submission.

    Stop/join host workers before finish(). failure_drain must stop workers and
    drain *every* device stream, including partially submitted native calls.
    On a failed drain the context deliberately keeps all resources alive.
    Capture contexts remain pinned by their graph bucket, not by replay tickets.
    """

    def __init__(
        self,
        phase,
        *,
        compute_stream,
        copy_stream,
        record_completion,
        failure_drain,
        capturing=False,
    ):
        if not isinstance(phase, Phase):
            raise ValueError("Unsupported execution phase")
        if phase is Phase.PREFILL and capturing:
            raise ValueError("Prefill cannot be captured")
        self._invocation_id = object()
        self.phase = phase
        self.capturing = capturing
        self.compute_stream = compute_stream
        self.copy_stream = copy_stream
        self._record = record_completion
        self._failure_drain = failure_drain
        self._lock = RLock()
        self._resources = []
        self._resource_ids = set()
        self._temporary_resources = []
        self._submission_thread = get_ident()
        self._dependencies = {}
        self._frontiers = {}
        self._recorded_completions = set()
        self._completion_streams = {}
        self._disposed = False
        self._workers = 0
        self._closed = False
        self._completion = None

    def retain(self, resource):
        with self._lock:
            if self._closed:
                raise RuntimeError(
                    "Invocation already closed; join workers before finish"
                )
            if (
                not isinstance(resource, (TransferTicket, Completion))
                and self._temporary_resources
                and get_ident() == self._submission_thread
            ):
                self._temporary_resources[-1].append(resource)
            elif id(resource) not in self._resource_ids:
                self._resource_ids.add(id(resource))
                self._resources.append(resource)

    def register_completion(self, dependency, *, stream=None):
        """Track dependencies without implicitly retaining arbitrary resources.

        Only the recorder may supply stream: it establishes actual submission
        order. Unknown transport/communication completions remain independent.
        Nested joins are flattened and deduplicated by completion identity.
        """
        completion = (
            dependency.completion
            if isinstance(dependency, TransferTicket)
            else dependency
        )
        if not isinstance(completion, Completion):
            raise TypeError("Expected a recorded completion")
        with self._lock:
            if self._closed:
                raise RuntimeError("Invocation already closed")
            if stream is not None:
                self._frontiers[id(stream)] = completion
                self._recorded_completions.add(id(completion))
                self._completion_streams[id(completion)] = id(stream)
            pending = [completion]
            while pending:
                item = pending.pop()
                if type(item) is JoinedCompletion:
                    pending.extend(item.completions)
                else:
                    self._dependencies[id(item)] = item
        return dependency

    def wait_for(self, dependency, stream):
        """Skip only leaves already enqueued by our recorder on this stream.

        Unknown transport and auxiliary-reader events retain their real waits.
        No ordering is inferred from an event name, tensor device or graph key.
        """
        completion = (
            dependency.completion
            if isinstance(dependency, TransferTicket)
            else dependency
        )
        leaves = (
            completion.completions
            if type(completion) is JoinedCompletion
            else (completion,)
        )
        for item in leaves:
            if self._completion_streams.get(id(item)) != id(stream):
                item.wait_on(stream)

    def worker_started(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot start workers after invocation seal")
            self._workers += 1

    def worker_finished(self):
        with self._lock:
            if self._workers <= 0:
                raise RuntimeError("Unbalanced worker completion")
            self._workers -= 1

    @contextmanager
    def temporary_scope(self):
        """Release submission-thread wave/layer temporaries in prefill and decode.

        Only use after joining off-stream readers onto compute_stream. Ordinary
        tensor allocation/free then follows the same allocator rules as resident
        inference; graph pools keep addresses reserved without pinning every
        intermediate Tensor. Owners/tickets keep registrations and events alive.
        Background submitters bypass these scopes, retaining their resources for
        the invocation even while the submission thread has an open scope.
        """
        if get_ident() != self._submission_thread:
            raise RuntimeError("Only the submission thread may open temporary scopes")
        resources = []
        with self._lock:
            if self._closed:
                raise RuntimeError("Invocation already closed")
            self._temporary_resources.append(resources)
        try:
            yield
        except BaseException:
            # Partial submission may have no usable event. Do not release these
            # resources until the runtime drains every device/communication stream.
            with self._lock:
                self._resources.extend(resources)
            raise
        finally:
            with self._lock:
                self._temporary_resources.pop()

    def record_compute_completion(self):
        return self._record_on(self.compute_stream)

    def _record_on(self, stream):
        ticket = self._record(stream)
        if not isinstance(ticket, TransferTicket):
            raise TypeError("Completion factory must return TransferTicket")
        self.retain(ticket)
        self.register_completion(ticket, stream=stream)
        return ticket

    def finish(self, *, output_ready=None, residency_ready=None):
        with self._lock:
            if self._closed:
                raise RuntimeError("Invocation already finished")
            if self._temporary_resources:
                raise RuntimeError("Close temporary scopes before finishing invocation")
            if self._workers:
                raise RuntimeError("Stop and join workers before finishing invocation")
            # Replay requires fresh events. When neither milestone was supplied,
            # both describe the same compute frontier and share one record.
            output_ready = output_ready or self.record_compute_completion()
            residency_ready = residency_ready or output_ready
            for ticket in (output_ready, residency_ready):
                if not isinstance(ticket, TransferTicket):
                    raise TypeError("Forward completion requires TransferTicket")
                self.retain(ticket)
                self.register_completion(ticket)
            copy_done = self._record_on(self.copy_stream)
            if self.capturing:
                # CaptureEnd must see the auxiliary stream's LAST task, including
                # its final event, joined back onto the capture origin.
                copy_done.completion.wait_on(self.compute_stream)
            dependencies = {
                key: value
                for key, value in self._dependencies.items()
                if key not in self._recorded_completions
            }
            dependencies.update(
                (id(value), value) for value in self._frontiers.values()
            )
            idle = TransferTicket(
                JoinedCompletion(dependencies.values()), (self._resources,)
            )
            self._completion = ForwardCompletion(
                output_ready, residency_ready, idle, self._resources, ref(self)
            )
            self._closed = True
            return self._completion

    def dispose_completed(self, *, graph_destroyed=False):
        """Retire after existing completion proof; no query or device wait.

        The owner calls this only after done()/wait succeeds. Captured contexts
        additionally require successful graph destruction. Failed submissions use
        dispose_after_drain because they may not have a usable completion ticket.
        """
        with self._lock:
            if self._disposed:
                return
            if not self._closed or self._completion is None:
                raise RuntimeError("Finish the invocation before releasing resources")
            if self.capturing and not graph_destroyed:
                raise RuntimeError(
                    "Destroy the captured graph before releasing resources"
                )
            self._dispose()

    def dispose_after_drain(self, *, graph_destroyed=False):
        """Internal teardown path after workers stop and every stream drains."""
        with self._lock:
            if self._disposed:
                return
            if self._workers:
                raise RuntimeError("Join workers before releasing resources")
            if self.capturing and not graph_destroyed:
                raise RuntimeError(
                    "Destroy the captured graph before releasing resources"
                )
            self._closed = True
            self._dispose()

    def _dispose(self):
        self._resources.clear()
        self._resource_ids.clear()
        self._dependencies.clear()
        self._frontiers.clear()
        self._recorded_completions.clear()
        self._completion_streams.clear()
        self._completion = None
        self._record = self._failure_drain = None
        # These optional bindings otherwise retain the entire serving batch or
        # captured inputs through external references to an already retired ctx.
        self.__dict__.pop("forward_batch", None)
        self.__dict__.pop("num_token_non_padded", None)
        self._disposed = True

    def release_completed(self):
        """Compatibility eager wait-and-release; serving retirement never calls it."""
        with self._lock:
            if not self._closed or self._completion is None:
                raise RuntimeError("Finish the invocation before releasing resources")
            completion = self._completion
        completion.resources_idle.completion.synchronize()
        # Graph warmups use capturing=True to emit exactly the capture program,
        # but this method is called only outside actual graph capture.
        self.dispose_completed(graph_destroyed=True)

    def abort(self):
        """Drain failed eager submissions before releasing retained objects."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Invocation already finished")
        self._failure_drain()
        for completion in tuple(self._dependencies.values()):
            completion.synchronize()
        self.dispose_after_drain()


@dataclass
class HotSnapshot:
    """Only persistent residency belongs to the cross-phase snapshot."""

    key_to_slot: object
    slot_to_key: object
    slot_valid: object
    slot_generation: object


class DecodeStateSnapshot:
    """Preallocated metadata snapshot with weights reloaded into original slots.

    reload_weights(saved_residency, context) must load every saved valid slot
    from the immutable host master and return its complete bundle ticket. It
    must not publish live maps. This helper restores maps/policy only after the
    reload dependency; a failure leaves the owner poisoned for explicit teardown.
    """

    def __init__(self, residency, policy, policy_state, *, error_status):
        import torch

        self.residency = residency
        self.policy = policy
        self.policy_state = policy_state
        if (
            error_status.shape != ()
            or error_status.dtype != torch.int32
            or error_status.device != residency.key_to_slot.device
        ):
            raise ValueError("Snapshot needs the shared scalar device error_status")
        self.error_status = error_status
        self.saved = HotSnapshot(
            **{
                field.name: torch.empty_like(getattr(residency, field.name))
                for field in fields(HotSnapshot)
            }
        )
        self.saved_policy = policy.allocate_state(
            residency.key_to_slot.numel(), residency.key_to_slot.device
        )
        self._snapshot = None

    def save(self, context):
        if context.capturing:
            raise RuntimeError("Mode snapshots are outside graph capture")
        if self._snapshot is not None:
            raise RuntimeError("Decode snapshot already exists")
        context.retain(self)
        # Mode changes already drain the owner. Observe device faults at this
        # eager boundary, never from dynamic decode planning or graph replay.
        context.record_compute_completion().completion.synchronize()
        self._require_healthy()
        self._copy_residency(self.residency, self.saved)
        self.policy.copy_state(self.policy_state, self.saved_policy)
        ticket = context.record_compute_completion()
        self._snapshot = TransferTicket(ticket.completion, (ticket, self))
        context.retain(self._snapshot)
        return self._snapshot

    def restore(self, context, reload_weights):
        if context.capturing or self._snapshot is None:
            raise RuntimeError("Restore requires an eager saved decode snapshot")
        context.retain(self)
        # Establish on both streams: restore lowering may inspect saved maps
        # on compute before scheduling weight loads on copy.
        self._snapshot.completion.wait_on(context.compute_stream)
        self._snapshot.completion.wait_on(context.copy_stream)
        copied = reload_weights(self.saved, context)
        if not isinstance(copied, TransferTicket):
            raise TypeError("Snapshot reload must return TransferTicket")
        context.retain(copied)
        # Native submission success does not imply a valid copy: device lowering
        # can latch a fault and mask every write. A transition is eager and can
        # wait here; never publish saved valid maps for an unsuccessful reload.
        copied.completion.synchronize()
        self._require_healthy()
        copied.completion.wait_on(context.compute_stream)
        self._copy_residency(self.saved, self.residency)
        self.policy.copy_state(self.saved_policy, self.policy_state)
        self.residency.execution_key.fill_(-1)
        self.residency.execution_valid.zero_()
        self.residency.step_access_count.zero_()
        # update_enabled belongs to the current invocation/capture, not history.
        published = context.record_compute_completion()
        ticket = TransferTicket(
            JoinedCompletion((copied.completion, published.completion)),
            (self, self._snapshot, copied, published),
        )
        context.retain(ticket)
        self._snapshot = None
        return ticket

    def _require_healthy(self):
        if self.error_status.item() != 0:
            raise RuntimeError("Device residency/transfer fault during mode transition")

    @staticmethod
    def _copy_residency(source, destination):
        for field in fields(HotSnapshot):
            getattr(destination, field.name).copy_(getattr(source, field.name))

    def allocations(self):
        """Include both saved residency and policy in the owner's metadata budget."""
        tensors = [
            ("residency." + field.name, getattr(self.saved, field.name))
            for field in fields(self.saved)
        ]
        # Cache policies can expose custom allocation reporting; the bundled
        # TensorLFUPolicy is a dataclass of tensors.
        if hasattr(self.saved_policy, "allocations"):
            policy_allocations = tuple(self.saved_policy.allocations())
        else:
            policy_allocations = ()
            tensors.extend(
                ("policy." + field.name, getattr(self.saved_policy, field.name))
                for field in fields(self.saved_policy)
            )
        result, seen = (
            list(policy_allocations),
            {x.storage_key for x in policy_allocations},
        )
        for name, tensor in tensors:
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage.data_ptr())
            if storage.nbytes() and key not in seen:
                seen.add(key)
                result.append(
                    Allocation(
                        "snapshot." + name,
                        ResourceKind.METADATA,
                        storage.nbytes(),
                        key,
                        self,
                    )
                )
        return tuple(result)


class InvocationRuntime(BoundMoERuntime):
    """Shared invocation, transition and worker teardown for device bindings.

    A binding supplies streams/events, model hooks, storage and programs. Phase
    and pending-forward ownership remain with the single MoEOffloadOwner; this
    class does not duplicate that state machine.
    """

    def _initialize_lifecycle(self):
        self._context = None
        self._step_started = False
        self._capture = None
        self._failed_contexts = []
        self._transition_contexts = []
        self._closed = False
        self._closing = False
        self._invocation_sequence = 0
        self._runtime_identity = object()

    def retire_transitions(self):
        """Called only after the owner/initializer completed its existing wait."""
        for context in self._transition_contexts:
            context.dispose_completed()
        self._transition_contexts.clear()

    def retire_graph_resources(self, contexts):
        """Forget only contexts disposed by their own successful graph cleanup.

        A runtime may serve several graph backends. Resetting one backend is no
        proof that another backend's failed capture can release its resources.
        Unknown failed contexts remain pinned until their owner resets or closes.
        """
        retired = {
            id(context)
            for context in contexts
            if context.capturing and context._disposed
        }
        self._failed_contexts = [
            context for context in self._failed_contexts if id(context) not in retired
        ]

    def _new_context(self, phase, *, stream=None, capturing=False):
        context = ManagedExecutionContext(
            phase,
            compute_stream=stream or self.device_runtime.current_stream(),
            copy_stream=self.device_runtime.copy_stream,
            record_completion=self.device_runtime.record,
            failure_drain=self.synchronize,
            capturing=capturing,
        )
        self._invocation_sequence += 1
        context._invocation_id = (
            self._runtime_identity,
            self._invocation_sequence,
            "capture" if capturing else "eager",
        )
        context.started_layers = set()
        return context

    def _require_open(self):
        if self._closed or self._closing:
            raise RuntimeError("MoE runtime is closed or closing")

    @property
    def context(self):
        # Closing rejects new submissions, but existing workers must still pin
        # partial native submissions until stop/join and device drain finish.
        if self._closed:
            raise RuntimeError("MoE runtime is closed")
        if self._context is None:
            raise RuntimeError("MoE execution requires a forward or capture scope")
        return self._context

    def _transition(self, operation):
        if self._context is not None:
            raise RuntimeError("Phase transition cannot overlap a forward")
        # The owner drained prior work before entering this transition.
        self.retire_transitions()
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
        # The owner retires the transition context immediately after its wait.
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

    def close(self):
        if self._closed:
            return
        if self._context is not None or self._capture is not None:
            raise RuntimeError("Finish active host submissions before closing runtime")
        self._closing = True
        self.synchronize()
        self.close_after_drain()

    def close_after_drain(self):
        """Release after the owner/close path stopped submissions and drained."""
        if self._closed:
            return
        if self._context is not None or self._capture is not None:
            raise RuntimeError("Finish active host submissions before closing runtime")
        owner = getattr(self, "_owner", None)
        if owner is not None and getattr(owner, "_graphs", ()):
            raise RuntimeError("Destroy graph buckets before closing runtime")
        self._closing = True
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
        for context in self._failed_contexts + self._transition_contexts:
            context.dispose_after_drain(graph_destroyed=True)
        self._failed_contexts.clear()
        self._transition_contexts.clear()
        if self.snapshot is not None:
            self.snapshot._snapshot = None
        self.layer_programs.clear()
        self.prefill_program = self.decode_program = None
        self.resources = self.storage = self.pipelines = ()
        self.snapshot = self.cache_updates = self.ops = self.residency = None
        self.cache_stats = self.diagnostics = None
        self.reload_weights = self._release = None
        self._closed = True
