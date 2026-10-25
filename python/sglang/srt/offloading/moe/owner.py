"""Shared phase, budget and submission ownership for all device runtimes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from threading import RLock

from .config import Phase
from .resources import AllocationLedger, ResourceKind

logger = logging.getLogger(__name__)


def _serialized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return call


class OwnerState(Enum):
    READY = "ready"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass(frozen=True)
class ForwardToken:
    owner_token: object
    sequence: int
    phase: Phase


class GraphLease:
    """Held by a graph bucket until its graph has been destroyed."""

    def __init__(self, owner, key):
        self.owner = owner
        self.key = key

    def close(self, *, drained=False):
        if self.owner is None:
            return
        self.owner._release_graph(self.key, drained=drained)
        self.owner = None


class MoEOffloadOwner:
    def __init__(self, config, catalog, runtime):
        self._lock = RLock()
        if any(
            (layer.ep_size, layer.ep_rank) != (config.ep_size, config.ep_rank)
            for layer in catalog.layers
        ):
            raise ValueError("Owner configuration and expert catalog disagree on EP")
        runtime.capabilities.require_decode()
        if runtime.expert_nbytes <= 0:
            raise ValueError("Backend must report physical bytes of a complete expert")
        self.config = config
        self.catalog = catalog
        self.runtime = runtime
        hot_slots = config.hot_slots(catalog)
        host_payload_bytes = catalog.num_experts * runtime.expert_nbytes
        host_limit = config.host_allocation_limit_bytes
        if host_limit is None:
            host_limit = host_payload_bytes
        if host_limit < host_payload_bytes:
            raise ValueError("Host budget cannot hold the immutable expert master")
        self.allocation_ledger = AllocationLedger(
            {
                ResourceKind.HOST_WEIGHTS: host_limit,
                ResourceKind.HOT_WEIGHTS: hot_slots * runtime.expert_nbytes,
                ResourceKind.EXECUTION_WEIGHTS: config.execution_slots
                * runtime.expert_nbytes,
                ResourceKind.WORKSPACE: config.workspace_limit_bytes,
                ResourceKind.METADATA: config.metadata_limit_bytes,
            },
            runtime.resources,
        )
        if self.allocation_ledger.usage[ResourceKind.HOST_WEIGHTS] < host_payload_bytes:
            raise ValueError(
                "host_weights allocation does not hold the complete master"
            )
        for kind in (
            ResourceKind.HOT_WEIGHTS,
            ResourceKind.EXECUTION_WEIGHTS,
        ):
            if (
                self.allocation_ledger.usage[kind]
                != self.allocation_ledger.limits[kind]
            ):
                raise ValueError(f"{kind.value} allocation does not match slot budget")
        self.state = OwnerState.READY
        self.phase = Phase.DECODE
        self._token = object()
        self._sequence = 0
        self._active = None
        self._capture_session = None
        self._pending = []
        self._transition_ticket = None
        self._snapshot_saved = False
        self._graphs = set()
        self._closing = False
        self._graph_cleanups = set()

    def _require_ready(self):
        if self.state is not OwnerState.READY or self._closing or self._graph_cleanups:
            raise RuntimeError(f"Offload owner is {self.state.value}")

    def _require_no_forward(self, *, capture_identity=None):
        if self._active is not None or (
            self._capture_session is not None
            and self._capture_session is not capture_identity
        ):
            raise RuntimeError(
                "Concurrent forwards, captures or transitions are not allowed"
            )

    @_serialized
    def begin_capture(self, identity):
        """Atomically prepare decode and gate submissions for the capture scope."""
        if identity is None:
            raise ValueError("Capture sessions require an identity")
        self._require_ready()
        self._require_no_forward()
        self.switch_phase(Phase.DECODE)
        self.wait_idle()
        self._capture_session = identity

    @_serialized
    def end_capture(self, identity, *, failed=False):
        if identity is not self._capture_session or identity is None:
            raise ValueError("Foreign or already finished capture session")
        # Poison before reopening the gate, so another thread cannot submit in
        # the interval between a failed scope unwinding and graph cleanup.
        if failed and self.state is not OwnerState.CLOSED:
            self.state = OwnerState.FAILED
        self._capture_session = None

    @_serialized
    def wait_idle(self):
        self._require_no_forward()
        try:
            for completion in self._pending:
                completion.resources_idle.completion.synchronize()
                completion.dispose_completed()
            self._pending.clear()
        except BaseException:
            self.state = OwnerState.FAILED
            raise

    @_serialized
    def switch_phase(self, phase):
        self._require_ready()
        self._require_no_forward()
        if not isinstance(phase, Phase):
            raise ValueError("Unsupported offload phase")
        if phase is self.phase:
            return
        logger.info(
            "MoE offload phase.begin: rank=%s from=%s to=%s sequence=%s",
            self.config.ep_rank,
            self.phase.value,
            phase.value,
            self._sequence,
        )
        self.wait_idle()
        try:
            if phase is Phase.PREFILL:
                self._transition_ticket = self.runtime.snapshot_decode_state()
                self._transition_ticket.completion.synchronize()
                self._snapshot_saved = True
                self.runtime.borrow_hot_arena()
            else:
                if not self._snapshot_saved:
                    raise RuntimeError("Cannot restore decode without a snapshot")
                self._transition_ticket = self.runtime.restore_decode_state()
                self._transition_ticket.completion.synchronize()
                self._snapshot_saved = False
            retire = getattr(self.runtime, "retire_transitions", None)
            if retire is not None:
                retire()
            self.phase = phase
            self._transition_ticket = None
            logger.info(
                "MoE offload phase.ready: rank=%s phase=%s",
                self.config.ep_rank,
                phase.value,
            )
        except BaseException:
            # A backend may already have enqueued work. Retain all resources until
            # explicit close() drains every stream, including unreported work.
            self.state = OwnerState.FAILED
            raise

    @_serialized
    def begin_forward(self, phase, compute_stream):
        self._require_ready()
        self._require_no_forward()
        self.switch_phase(phase)
        # Host submissions may be asynchronous, but device forwards are serialized
        # by this join. Do not poll LFU/miss state or host-synchronize each step.
        try:
            pending = []
            for item in self._pending:
                if item.resources_idle.completion.done():
                    item.dispose_completed()
                else:
                    pending.append(item)
            self._pending = pending
            if self._pending:
                self._pending[-1].resources_idle.completion.wait_on(compute_stream)
        except BaseException:
            self.state = OwnerState.FAILED
            raise
        self._sequence += 1
        token = ForwardToken(self._token, self._sequence, phase)
        self._active = token
        return token

    def _check_token(self, token):
        if (
            token is None
            or token is not self._active
            or token.owner_token is not self._token
        ):
            raise ValueError("Foreign, stale or already finished forward token")

    @_serialized
    def finish_forward(self, token, completion):
        self._check_token(token)
        self._pending.append(completion)
        self._active = None

    @_serialized
    def fail_forward(self, token):
        self._check_token(token)
        self.state = OwnerState.FAILED
        self._active = None
        # runtime.synchronize() in close covers work that failed before returning
        # a completion ticket. No new forward is allowed after partial execution.

    @_serialized
    def fail_graph_capture(self):
        """Poison a partial warmup/capture; keep graph/storage leases for cleanup."""
        self._require_no_forward()
        if self.state is not OwnerState.CLOSED:
            self.state = OwnerState.FAILED

    @_serialized
    def acquire_graph(self, key, *, capture_identity=None):
        self._require_ready()
        self._require_no_forward(capture_identity=capture_identity)
        if key in self._graphs:
            raise ValueError("Graph bucket already bound")
        self._graphs.add(key)
        return GraphLease(self, key)

    @_serialized
    def _release_graph(self, key, *, drained=False):
        if not drained:
            self.wait_idle()
        # Also cover capture/warmup work, which is not a serving forward ticket.
        try:
            if not drained:
                self.runtime.synchronize()
            self._graphs.remove(key)
        except BaseException:
            self.state = OwnerState.FAILED
            raise

    @_serialized
    def begin_cleanup(self):
        """Close the submission gate before joining workers or draining devices."""
        self._require_no_forward()
        self._closing = True

    @_serialized
    def begin_graph_cleanup(self, identity):
        """Temporarily gate forwards while one graph backend is being retired."""
        self._require_no_forward()
        self._graph_cleanups.add(identity)

    @_serialized
    def finish_graph_cleanup(self, identity):
        # Distinct wrappers can be closed concurrently. Keep the gate until all
        # registered cleanup transactions finish; failure also poisons state.
        self._graph_cleanups.discard(identity)

    @_serialized
    def dispose_pending_after_drain(self):
        for completion in self._pending:
            completion.dispose_completed()
        self._pending.clear()

    @_serialized
    def close(self):
        if self.state is OwnerState.CLOSED:
            return
        self._require_no_forward()
        if self._graphs:
            raise RuntimeError("Destroy graph buckets and release graph leases first")
        self.begin_cleanup()
        try:
            self.runtime.synchronize()
            self.dispose_pending_after_drain()
            close_drained = getattr(self.runtime, "close_after_drain", None)
            if close_drained is not None:
                close_drained()
            else:
                self.runtime.close()
        except BaseException:
            self.state = OwnerState.FAILED
            raise
        self._transition_ticket = None
        # Release allocation keepalives only after the backend is quiescent.
        self.allocation_ledger = None
        self.state = OwnerState.CLOSED
