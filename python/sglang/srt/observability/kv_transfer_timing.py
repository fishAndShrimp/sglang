"""Opt-in P-side timing. No device synchronization or transfer policy changes."""

import contextvars
import json
import os
import threading
import time
from contextlib import contextmanager

ENABLED = os.environ.get("SGLANG_KV_TIMING", "0") == "1"
_current = contextvars.ContextVar("kv_transfer_timing", default=None)
_phase = contextvars.ContextVar("kv_transfer_timing_phase", default="other")


class ChunkTiming:
    def __init__(self, chunk, dequeued):
        self.event = {
            "event": "chunk",
            "version": 2,
            "pid": os.getpid(),
            "room": chunk.room,
            "chunk": chunk._kv_timing_id,
            "attempt": chunk._kv_timing_attempts,
            "last": chunk.is_last_chunk,
            "enqueue": chunk._kv_timing_enqueued,
            "dequeue": dequeued,
            "outcome": "failed",
            "phases": [],
            "calls": [],
        }
        self.lock = threading.Lock()

    def record(self, key, value):
        # Calls may execute in several layer-transfer executor threads.
        with self.lock:
            self.event[key].append(value)

    def activate(self):
        return _current.set(self)

    def finish(self, token, logger):
        _current.reset(token)
        now = time.perf_counter()
        self.event.setdefault("work_end", now)
        self.event["cleanup_end"] = now
        # Emit after worker cleanup; the scheduler may log finish first.
        logger.info("KV_TIMING %s", json.dumps(self.event, separators=(",", ":")))


@contextmanager
def phase(name):
    timing = _current.get()
    if timing is None:
        yield
        return
    token = _phase.set(name)
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        _phase.reset(token)
        timing.record("phases", [name, start, end])


def submit(executor, function, *args, **kwargs):
    """Propagate the request/phase context to layer-transfer executor threads."""
    if _current.get() is None:
        return executor.submit(function, *args, **kwargs)
    context = contextvars.copy_context()
    return executor.submit(context.run, function, *args, **kwargs)


def batch_transfer_sync(engine, session, sources, destinations, lengths):
    timing = _current.get()
    if timing is None:
        return engine.batch_transfer_sync(session, sources, destinations, lengths)
    size = int(sum(lengths))
    phase_name = _phase.get()
    result = None
    start = time.perf_counter()
    try:
        result = engine.batch_transfer_sync(session, sources, destinations, lengths)
        return result
    finally:
        end = time.perf_counter()
        # No memory addresses or payloads are logged. None denotes an exception.
        timing.record("calls", [phase_name, start, end, size, result])
