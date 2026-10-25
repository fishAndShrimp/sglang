"""Opt-in graph backend ownership; serving completion stays in forward_scope."""

import logging
from contextlib import contextmanager
from threading import RLock

from .owner import GraphLease, _serialized

logger = logging.getLogger(__name__)


class GraphLeaseBackend:
    """Wrap the existing CUDA/NPU backend without replacing its graph mechanics.

    Capture scope encompasses ALL warmups, including runner-side autotuning.
    The runtime supplies capture hook state and disables updates outside the graph
    before yielding a keepalive. Captured state stays alive until graph destruction.
    Normal replay runs inside the runtime's existing serving forward_scope, which
    must record fresh completion events after each submission. Capture-time tickets
    are never reused here as evidence that a later replay has completed.
    """

    def __init__(self, backend, owner, graph_runner):
        self._lock = RLock()
        self.backend = backend
        self.owner = owner
        self._rank = getattr(getattr(owner, "config", None), "ep_rank", None)
        self.graph_runner = graph_runner
        self._identity = object()
        self._leases = {}
        self._capture_resources = []
        self._capturing = False
        self._failed = False
        self._closed = False
        self._closing = False

    def __getattr__(self, name):
        # NPU-specific attributes and backend diagnostics remain available.
        return getattr(self.backend, name)

    def _require_open(self):
        if self._closed or self._failed or self._closing:
            raise RuntimeError("MoE graph backend is closed or failed")

    def _capture_failed(self):
        self._failed = True
        self.owner.fail_graph_capture()
        # Drain BEFORE destroying graphs; release leases only after destruction.
        # If either fails, cleanup preserves resources/leases for a later retry.
        self.cleanup()

    @contextmanager
    def capture_session(self, stream):
        with self._lock:
            with self._capture_session(stream):
                yield

    @contextmanager
    def _capture_session(self, stream):
        self._require_open()
        if self._capturing:
            raise RuntimeError("Nested MoE graph capture sessions are not supported")
        self.owner.begin_capture(self._identity)
        self._capturing = True
        failed = True
        try:
            try:
                with self.owner.runtime.capture_scope(
                    self.owner, self.graph_runner, stream
                ) as resources:
                    if resources is None:
                        raise TypeError(
                            "MoE capture_scope must yield capture keepalive state"
                        )
                    self._capture_resources.append(resources)
                    with self.backend.capture_session(stream):
                        yield
                    if self._failed:
                        raise RuntimeError("A failed MoE capture cannot be resumed")
                failed = False
            finally:
                # Keep the owner gate through runtime/backend scope teardown.
                # Failed capture cleanup may begin only after both unwind.
                self._capturing = False
                self.owner.end_capture(self._identity, failed=failed)
        except BaseException:
            self._capture_failed()
            raise

    @_serialized
    def capture_one(
        self,
        shape_key,
        forward_fn,
        capture_inputs=None,
        post_warmup_hook=None,
    ):
        self._require_open()
        if not self._capturing:
            raise RuntimeError("Capture graphs inside a MoE capture_session")
        if shape_key in self._leases:
            raise ValueError(
                "Destroy an existing MoE graph before replacing its bucket"
            )
        logger.info(
            "MoE offload graph.capture.begin: rank=%s bucket=%s", self._rank, shape_key
        )
        lease = self.owner.acquire_graph(
            (self._identity, shape_key), capture_identity=self._identity
        )
        self._leases[shape_key] = lease
        try:
            warmup_hook = post_warmup_hook
            retire_warmup = getattr(self.owner.runtime, "after_graph_warmup", None)
            if retire_warmup is not None:

                def warmup_hook():
                    if post_warmup_hook is not None:
                        post_warmup_hook()
                    retire_warmup()

            result = self.backend.capture_one(
                shape_key,
                forward_fn,
                capture_inputs=capture_inputs,
                post_warmup_hook=warmup_hook,
            )
            logger.info(
                "MoE offload graph.capture.ready: rank=%s bucket=%s",
                self._rank,
                shape_key,
            )
            return result
        except BaseException:
            logger.exception(
                "MoE offload graph.capture.failed: rank=%s bucket=%s",
                self._rank,
                shape_key,
            )
            self._failed = True
            # capture_session unwinds backend/runtime scope before cleanup; do
            # not destroy a graph while still inside the capture context.
            raise

    @_serialized
    def can_run(self, forward_batch, shape_key):
        self._require_open()
        return shape_key in self._leases and self.backend.can_run(
            forward_batch, shape_key
        )

    @contextmanager
    def replay_session(self):
        with self._lock:
            self._require_open()
            with self.backend.replay_session():
                yield

    @_serialized
    def replay(self, shape_key, static_forward_batch, **kwargs):
        self._require_open()
        if shape_key not in self._leases:
            raise ValueError("MoE graph bucket has no owner lease")
        return self.backend.replay(shape_key, static_forward_batch, **kwargs)

    @_serialized
    def replay_with_input_update(self, shape_key, *args, **kwargs):
        self._require_open()
        if shape_key not in self._leases:
            raise ValueError("MoE graph bucket has no owner lease")
        return self.backend.replay_with_input_update(shape_key, *args, **kwargs)

    @_serialized
    def cleanup(self):
        if self._closed:
            return
        # Block new submissions before stopping host workers. A failed cleanup
        # stays closed to replay and can be retried without freeing graph storage.
        begin_cleanup = getattr(self.owner, "begin_graph_cleanup", None)
        if begin_cleanup is not None:
            begin_cleanup(self._identity)
        else:
            self.owner.wait_idle()
        self._closing = True
        try:
            stop = getattr(self.backend, "stop_submissions", None)
            if stop is not None:
                stop()
            self.owner.runtime.synchronize()
            failed_cleanup = getattr(self.backend, "cleanup_failed_capture", None)
            if self._failed and callable(failed_cleanup):
                failed_cleanup()
            else:
                self.backend.cleanup()
            # Only successful reset permits captured resource retirement. A
            # partial reset failure leaves all leases and resources pinned.
            retired_contexts = []
            for resources in self._capture_resources:
                if isinstance(resources, (list, tuple)):
                    for context in resources:
                        retire = getattr(context, "dispose_after_drain", None)
                        if retire is not None:
                            retire(graph_destroyed=True)
                            retired_contexts.append(context)
            retired = getattr(self.owner.runtime, "retire_graph_resources", None)
            if retired is not None:
                retired(retired_contexts)
            for key, lease in tuple(self._leases.items()):
                if isinstance(lease, GraphLease):
                    lease.close(drained=True)
                else:
                    lease.close()
                del self._leases[key]
            dispose = getattr(self.owner, "dispose_pending_after_drain", None)
            if dispose is not None:
                dispose()
        except BaseException:
            self._failed = True
            self.owner.fail_graph_capture()
            raise
        self._capture_resources.clear()
        self._closed = True
        finish_cleanup = getattr(self.owner, "finish_graph_cleanup", None)
        if finish_cleanup is not None:
            finish_cleanup(self._identity)
