"""Eager prefill contracts and bounded host submission over caller-owned pools.

The worker and bank schedule are device-neutral; accelerator operations and
physical bindings remain injected and require separate hardware validation.
"""

import logging
import math
from dataclasses import dataclass, field
from threading import Condition, RLock, Thread

from ..transfer import Completion, ImmediateCompletion, JoinedCompletion, TransferTicket
from .config import ExpertWave, Phase
from .execution import ExecutionRequest, require_prefill

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PrefillPipelineConfig:
    """Physical capacities of caller-owned A/B views, in complete experts.

    No storage is allocated here. The binding callbacks must validate that A and
    B do not overlap and that the combined first group is physically contiguous.
    Small layers truncate requested microbatch/group sizes to their expert count.
    """

    local_experts: int
    initial_capacity: int
    group_size: int
    secondary_capacity: int
    microbatch_size: int = 1
    max_inflight: int = 1
    initial_bank: int = 0
    secondary_bank: int = 1
    poll_interval: float = 0.001

    def __post_init__(self):
        integers = (
            self.local_experts,
            self.initial_capacity,
            self.group_size,
            self.microbatch_size,
            self.max_inflight,
        )
        if any(type(value) is not int or value <= 0 for value in integers):
            raise ValueError("Prefill capacities and P/Q must be positive integers")
        if type(self.secondary_capacity) is not int or self.secondary_capacity < 0:
            raise ValueError("Secondary capacity must be a nonnegative integer")
        for name in ("initial_capacity", "group_size", "microbatch_size"):
            object.__setattr__(self, name, min(getattr(self, name), self.local_experts))
        if not self.microbatch_size <= self.group_size <= self.initial_capacity:
            raise ValueError("Prefill requires P <= S <= K_cap <= N")
        # Early dispatch feedback can seal at P even when A could hold all N.
        if (
            self.local_experts > self.microbatch_size
            and self.secondary_capacity < self.group_size
        ):
            raise ValueError(
                "Prefill requires a separately budgeted B bank of S experts"
            )
        if (
            type(self.initial_bank) is not int
            or type(self.secondary_bank) is not int
            or min(self.initial_bank, self.secondary_bank) < 0
            or self.initial_bank == self.secondary_bank
        ):
            raise ValueError("Prefill requires distinct nonnegative bank identifiers")
        if not math.isfinite(self.poll_interval) or self.poll_interval <= 0:
            raise ValueError("Poll interval must be positive and finite")


@dataclass
class _PrefillSession:
    layer_id: int
    invocation_id: object
    condition: Condition = field(default_factory=Condition)
    previous_idle: object = None
    worker: object = None
    worker_started: bool = False
    worker_done: bool = False
    stop: bool = False
    error: object = None
    dispatch_ready: object = None
    loaded: int = 0
    waves: tuple = ()
    leases: list = field(default_factory=list)
    next_wave: int = 0
    closed: bool = False


class BoundedPrefillWeightPipeline:
    """Bounded eager submitter and A/B/a scheduling over existing weight pools.

    ``make_request(layer_id, wave, bank_offset, context)`` returns a PREFILL
    ExecutionRequest. Initial microbatches write at offsets 0, P, 2P inside A;
    subsequent waves write at offset zero in B or the small alias inside A.
    ``combine_bindings(layer_id, wave, leases, context)`` validates and returns a
    contiguous view over the already loaded initial microbatches, without copying.
    ``worker_scope(context)`` is a mandatory context-manager factory which enters
    the correct device/context and copy stream on the worker thread.
    ``record_copy_completion(context)`` records a fresh TransferTicket after all
    copy submissions, including partially failed driver calls. It must preserve
    the backend's host-queue ordering and use the invocation's copy stream.

    All allocation, residency, actual copies and event creation remain injected.
    This implementation establishes a scheduling contract, not accelerator/graph
    certification. The context's retain method must be thread-safe.
    """

    def __init__(
        self,
        access,
        config,
        *,
        make_request,
        combine_bindings,
        worker_scope,
        record_copy_completion,
    ):
        self.access = access
        if not isinstance(config, PrefillPipelineConfig):
            raise TypeError("Expected PrefillPipelineConfig")
        for callback in (
            make_request,
            combine_bindings,
            worker_scope,
            record_copy_completion,
        ):
            if not callable(callback):
                raise TypeError("Prefill binding callbacks must be callable")
        self.config = config
        self.make_request = make_request
        self.combine_bindings = combine_bindings
        self.worker_scope = worker_scope
        self.record_copy_completion = record_copy_completion
        self._lock = RLock()
        self._active_session = None
        self._aborted = {}
        self._previous_idle = None

    def _session(self, layer, context):
        require_prefill(context)
        with self._lock:
            session = self._active_session
        if (
            session is None
            or session.invocation_id is not context.invocation_id
            or session.layer_id != layer
            or session.closed
        ):
            raise RuntimeError(
                "No active prefill layer-start session for this invocation"
            )
        return session

    @staticmethod
    def _raise_failed(session):
        if session.error is not None:
            raise RuntimeError("Prefill submitter failed") from session.error
        if session.stop:
            raise RuntimeError("Prefill session was aborted")

    def start(self, layer_id, context):
        """Pin one layer's session before starting its bounded eager submitter."""
        require_prefill(context)
        if type(layer_id) is not int or layer_id < 0:
            raise ValueError("A prefill layer needs a nonnegative integer identifier")
        session = _PrefillSession(layer_id, context.invocation_id)
        with self._lock:
            if context.invocation_id in self._aborted:
                raise RuntimeError("Cannot restart an aborted prefill invocation")
            # A/B views are shared: a sealed but unread layer still owns them.
            if self._active_session is not None:
                raise RuntimeError(
                    "Finish the previous prefill session before starting another"
                )
            session.previous_idle = self._previous_idle
            self._active_session = session
        context.retain(self)
        context.retain(session)
        session.worker = Thread(
            target=self._load_initial,
            args=(session, context),
            name=f"moe-prefill-{layer_id}",
            daemon=False,
        )
        context.worker_started()
        try:
            session.worker.start()
            session.worker_started = True
        except BaseException as error:
            context.worker_finished()
            with session.condition:
                session.error = error
                session.worker_done = True
                session.condition.notify_all()
            raise

    def validate_session(self, layer, context):
        session = self._session(layer, context)
        with session.condition:
            self._raise_failed(session)
            if session.waves:
                raise RuntimeError("Prefill session was already sealed")

    def _request(self, session, wave, bank_offset, context):
        request = self.make_request(session.layer_id, wave, bank_offset, context)
        if (
            not isinstance(request, ExecutionRequest)
            or request.layer_id != session.layer_id
            or request.wave != wave
            or request.phase is not Phase.PREFILL
        ):
            raise ValueError("Prefill binding must request exactly this prefill wave")
        return request

    def _load_initial(self, session, context):
        config = self.config
        inflight = []
        try:
            with self.worker_scope(context):
                if session.previous_idle is not None:
                    context.retain(session.previous_idle)
                    session.previous_idle.completion.wait_on(context.copy_stream)
                while session.loaded < config.initial_capacity:
                    with session.condition:
                        if session.stop:
                            break
                        # Never query an event before the dispatcher publishes it.
                        if (
                            session.loaded
                            and session.dispatch_ready is not None
                            and session.dispatch_ready.done()
                        ):
                            break
                    inflight = [
                        lease
                        for lease in inflight
                        if not lease.ticket.completion.done()
                    ]
                    if len(inflight) >= config.max_inflight:
                        with session.condition:
                            if not session.stop:
                                session.condition.wait(config.poll_interval)
                        continue
                    start = session.loaded
                    end = min(start + config.microbatch_size, config.initial_capacity)
                    wave = ExpertWave(
                        len(session.leases), start, end, config.initial_bank
                    )
                    lease = self.access.stage(
                        self._request(session, wave, start, context), context
                    )
                    # A complete expert bundle was submitted; readiness is separate.
                    with session.condition:
                        session.leases.append(lease)
                        session.loaded = end
                        inflight.append(lease)
                        session.condition.notify_all()
        except BaseException as error:
            with session.condition:
                session.error = error
                session.stop = True
        finally:
            context.worker_finished()
            with session.condition:
                session.worker_done = True
                session.condition.notify_all()

    @staticmethod
    def _join_worker(session):
        if session.worker_started:
            session.worker.join()

    def seal(self, layer, dispatch_ready, context):
        """Stop new batches at dispatch feedback; join and combine submitted leases."""
        session = self._session(layer, context)
        if not isinstance(dispatch_ready, Completion):
            raise TypeError("Dispatch feedback must be a recorded Completion")
        context.retain(dispatch_ready)
        with session.condition:
            self._raise_failed(session)
            if session.waves or session.dispatch_ready is not None:
                raise RuntimeError("Prefill session may only be sealed once")
            session.dispatch_ready = dispatch_ready
            session.condition.notify_all()
            while not session.worker_done:
                session.condition.wait()
        self._join_worker(session)
        self._raise_failed(session)
        if not session.leases or not session.loaded:
            raise RuntimeError("Prefill must submit at least one complete microbatch")
        config = self.config
        logger.debug(
            "MoE offload prefill.sealed: layer=%s initial_submitted=%s "
            "microbatch=%s max_inflight=%s group=%s",
            session.layer_id,
            session.loaded,
            config.microbatch_size,
            config.max_inflight,
            config.group_size,
        )
        waves = [ExpertWave(0, 0, session.loaded, config.initial_bank)]
        for index, start in enumerate(
            range(session.loaded, config.local_experts, config.group_size), 1
        ):
            bank = config.secondary_bank if index % 2 else config.initial_bank
            waves.append(
                ExpertWave(
                    index,
                    start,
                    min(start + config.group_size, config.local_experts),
                    bank,
                )
            )
        bindings = self.combine_bindings(
            session.layer_id, waves[0], tuple(session.leases), context
        )
        lease = self.access.combine_prefill(session.leases, bindings, context)
        session.leases = [lease]
        session.waves = tuple(waves)
        return session.waves

    def acquire(self, layer, wave, context):
        session = self._session(layer, context)
        self._raise_failed(session)
        index = session.next_wave
        if index >= len(session.waves) or session.waves[index] != wave:
            raise ValueError("Acquire prefill waves exactly once in sealed order")
        current = session.leases[index]
        if index and session.leases[index - 1].read_done is None:
            raise RuntimeError("Previous prefill wave has no final reader completion")
        if current.read_done is not None:
            raise RuntimeError("Current prefill wave was already retired")
        next_index = index + 1
        if next_index < len(session.waves):
            next_wave = session.waves[next_index]
            # The first reuse of A must wait for ALL first-group reservations.
            old_index = next_index - 2
            if old_index >= 0:
                previous = session.leases[old_index]
                if previous.read_done is None:
                    raise RuntimeError("Cannot overwrite a bank without its old reader")
                previous.read_done.completion.wait_on(context.copy_stream)
            lease = self.access.stage(
                self._request(session, next_wave, 0, context), context
            )
            session.leases.append(lease)
        session.next_wave += 1
        return current

    @staticmethod
    def _tickets(session):
        tickets = []
        for lease in session.leases:
            tickets.append(lease.ticket)
            if lease.read_done is not None:
                tickets.append(lease.read_done)
        return tickets

    def _copy_fence(self, context):
        ticket = self.record_copy_completion(context)
        if not isinstance(ticket, TransferTicket):
            raise TypeError("Copy-stream completion must be a TransferTicket")
        context.retain_completion(ticket)
        return ticket

    def finish(self, layer, context):
        """Retire this session with every copy/reader dependency, without host wait."""
        session = self._session(layer, context)
        self._raise_failed(session)
        if not session.waves or session.next_wave != len(session.waves):
            raise RuntimeError("All sealed prefill waves must execute before finish")
        if any(lease.read_done is None for lease in session.leases):
            raise RuntimeError("Every prefill wave must retire its final reader")
        self._join_worker(session)
        tickets = self._tickets(session) + [self._copy_fence(context)]
        ticket = TransferTicket(
            JoinedCompletion(item.completion for item in tickets),
            (session, *tickets),
        )
        context.retain_completion(ticket)
        session.closed = True
        with self._lock:
            # Keep the dependency but avoid chaining prior sessions/contexts.
            # Their owning forward tickets retain storage until device idle.
            self._previous_idle = TransferTicket(ticket.completion)
            self._active_session = None
        return ticket

    def abort(self, context):
        """Idempotently stop host submission and fence partial work before release."""
        require_prefill(context)
        with self._lock:
            previous = self._aborted.get(context.invocation_id)
            if previous is not None:
                return previous
            session = self._active_session
            if (
                session is not None
                and session.invocation_id is not context.invocation_id
            ):
                session = None
        if session is not None:
            with session.condition:
                session.stop = True
                session.condition.notify_all()
            self._join_worker(session)
        # Record AFTER the submitter exits: failed driver calls may have queued
        # writes without returning a lease. If fencing fails, retain sessions and
        # propagate; the runtime must drain all streams before freeing resources.
        tickets = (
            [self._copy_fence(context), *self._tickets(session)]
            if session is not None
            else []
        )
        ticket = TransferTicket(
            (
                JoinedCompletion(item.completion for item in tickets)
                if tickets
                else ImmediateCompletion()
            ),
            (session, *tickets),
        )
        context.retain_completion(ticket)
        with self._lock:
            if session is not None:
                session.closed = True
                self._active_session = None
            self._aborted[context.invocation_id] = ticket
        return ticket
