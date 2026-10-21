"""Opt-in graph backend ownership; serving completion stays in forward_scope."""

from contextlib import contextmanager

from .config import Phase


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
        self.backend = backend
        self.owner = owner
        self.graph_runner = graph_runner
        self._identity = object()
        self._leases = {}
        self._capture_resources = []
        self._capturing = False
        self._failed = False
        self._closed = False

    def __getattr__(self, name):
        # NPU-specific attributes and backend diagnostics remain available.
        return getattr(self.backend, name)

    def _require_open(self):
        if self._closed or self._failed:
            raise RuntimeError("MoE graph backend is closed or failed")

    def _capture_failed(self):
        self._failed = True
        self.owner.fail_graph_capture()
        # Drain BEFORE destroying graphs; release leases only after destruction.
        # If either fails, cleanup preserves resources/leases for a later retry.
        self.cleanup()

    @contextmanager
    def capture_session(self, stream):
        self._require_open()
        if self._capturing:
            raise RuntimeError("Nested MoE graph capture sessions are not supported")
        self.owner.switch_phase(Phase.DECODE)
        self.owner.wait_idle()
        self._capturing = True
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
        except BaseException:
            self._capture_failed()
            raise
        finally:
            self._capturing = False

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
        lease = self.owner.acquire_graph((self._identity, shape_key))
        self._leases[shape_key] = lease
        try:
            return self.backend.capture_one(
                shape_key,
                forward_fn,
                capture_inputs=capture_inputs,
                post_warmup_hook=post_warmup_hook,
            )
        except BaseException:
            self._failed = True
            # capture_session unwinds backend/runtime scope before cleanup; do
            # not destroy a graph while still inside the capture context.
            raise

    def can_run(self, forward_batch, shape_key):
        self._require_open()
        return shape_key in self._leases and self.backend.can_run(
            forward_batch, shape_key
        )

    def replay_session(self):
        self._require_open()
        return self.backend.replay_session()

    def replay(self, shape_key, static_forward_batch, **kwargs):
        self._require_open()
        if shape_key not in self._leases:
            raise ValueError("MoE graph bucket has no owner lease")
        return self.backend.replay(shape_key, static_forward_batch, **kwargs)

    def replay_with_input_update(self, shape_key, *args, **kwargs):
        self._require_open()
        if shape_key not in self._leases:
            raise ValueError("MoE graph bucket has no owner lease")
        return self.backend.replay_with_input_update(shape_key, *args, **kwargs)

    def cleanup(self):
        if self._closed:
            return
        # Failed warmup/capture can have work absent from serving completion
        # tickets. A successful drain covers those streams before graph reset.
        self.owner.wait_idle()
        try:
            self.owner.runtime.synchronize()
            failed_cleanup = getattr(self.backend, "cleanup_failed_capture", None)
            if self._failed and callable(failed_cleanup):
                failed_cleanup()
            else:
                self.backend.cleanup()
            for key, lease in tuple(self._leases.items()):
                lease.close()
                del self._leases[key]
        except BaseException:
            # A partial reset must never leave the remaining graph replayable.
            self._failed = True
            self.owner.fail_graph_capture()
            raise
        self._capture_resources.clear()
        self._closed = True
