"""Bounded, asynchronous batch logging outside the inference graph.

Only existing route counters are read. Copies use the current compute stream
and the caller's existing forward completion; no event, stream, wait or device
synchronization is introduced here. A slow logger drops records, never batches.
"""

from __future__ import annotations

import logging
import queue
import threading
from contextlib import nullcontext
from dataclasses import dataclass

from .config import Phase
from .resources import Allocation, ResourceKind

logger = logging.getLogger(__name__)


@dataclass
class _BatchRecord:
    slot: int
    sequence: int
    phase: Phase
    mode: str
    batch_size: int
    tokens: int
    bucket: object
    completion: object = None
    copy_error: str | None = None


class BatchDiagnostics:
    """Snapshot four scalars into fixed pinned slots, then log after completion.

    The runtime must call close() only AFTER its existing device drain. Slots
    stay owned until their completion, including partial copy/query failures.
    host_buffer/device_scope are injectable for CPU contract tests only.
    """

    def __init__(
        self,
        stats,
        error_status,
        *,
        rank,
        device,
        interval=1,
        capacity=64,
        host_buffer=None,
        device_scope=nullcontext,
    ):
        import torch

        if type(interval) is not int or interval < 0:
            raise ValueError("batch_log_interval must be a nonnegative integer")
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("batch_log_capacity must be a positive integer")
        self.stats, self.error_status = stats, error_status
        self.rank, self.device = rank, str(device)
        self.interval, self.capacity = interval, capacity
        self._device_scope = device_scope
        self._host = None
        self._errors = None
        if interval:
            self._host = (
                torch.empty(
                    (capacity, 4), dtype=torch.int64, device="cpu", pin_memory=True
                )
                if host_buffer is None
                else host_buffer
            )
            if (
                self._host.device.type != "cpu"
                or self._host.shape != (capacity, 4)
                or self._host.dtype != torch.int64
            ):
                raise ValueError("Batch diagnostics require a capacity x 4 CPU buffer")
            # Preserve the native int32 fault scalar without a device-side dtype
            # conversion. The fourth int64 lane reserves space for this view.
            self._errors = self._host[:, 3:4].view(torch.int32)[:, 0]
        self._free = queue.SimpleQueue()
        for slot in range(capacity):
            self._free.put(slot)
        self._pending = queue.SimpleQueue()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._disabled = self._closed = False
        self._dropped = 0
        self._reported_drops = 0

    def allocations(self):
        if self._host is None:
            return ()
        storage = self._host.untyped_storage()
        return (
            Allocation(
                "moe.batch_log.pinned_snapshots",
                ResourceKind.METADATA,
                storage.nbytes(),
                ("cpu", storage.data_ptr()),
                self,
            ),
        )

    def snapshot(self, phase, sequence, forward_batch, *, bucket=None):
        """Called outside capture, before the existing forward completion record."""
        if (
            not self.interval
            or self._disabled
            or self._closed
            or sequence % self.interval
            or not logger.isEnabledFor(logging.INFO)
        ):
            return None
        try:
            slot = self._free.get_nowait()
        except queue.Empty:
            self._dropped += 1
            self._wake.set()
            return None
        record = _BatchRecord(
            slot,
            sequence,
            phase,
            getattr(forward_batch.forward_mode, "name", phase.value),
            getattr(forward_batch, "batch_size", 0),
            forward_batch.input_ids.numel(),
            bucket,
        )
        # Retain a reserved slot even if a later copy fails after earlier writes
        # were queued. Publication attaches the same forward's completion.
        try:
            row = self._host[slot]
            if phase is Phase.DECODE:
                for index, source in enumerate(
                    (self.stats.accesses, self.stats.hits, self.stats.misses)
                ):
                    row[index].copy_(source, non_blocking=True)
            self._errors[slot].copy_(self.error_status, non_blocking=True)
        except Exception as error:
            record.copy_error = str(error)
            self._disabled = True
        return record

    def publish(self, record, completion):
        if record is None:
            return
        record.completion = completion
        self._pending.put(record)
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run, name="moe-batch-logger", daemon=True
            )
            try:
                self._thread.start()
            except Exception:
                # Keep the pending slot until the runtime's existing close drain.
                self._disabled = True
                self._thread = None
                logger.warning(
                    "MoE offload batch logger could not start", exc_info=True
                )
        self._wake.set()

    def _emit(self, record):
        if record.copy_error is not None:
            logger.warning(
                "MoE offload batch logging disabled: rank=%s seq=%s copy_error=%s",
                self.rank,
                record.sequence,
                record.copy_error,
            )
            return
        # This is CPU memory, read only after the existing completion is ready.
        row = self._host[record.slot]
        error = int(self._errors[record.slot])
        accesses = hits = misses = rate = "N/A"
        if record.phase is Phase.DECODE:
            accesses, hits, misses = row[:3].tolist()
            if (
                not error
                and accesses > 0
                and 0 <= hits <= accesses
                and hits + misses == accesses
            ):
                rate = f"{100 * hits / accesses:.2f}%"
        logger.log(
            logging.ERROR if error else logging.INFO,
            "MoE offload batch: rank=%s device=%s seq=%s phase=%s mode=%s "
            "bs=%s tokens=%s bucket=%s assignments=%s hits=%s misses=%s "
            "hit_rate=%s error=%s",
            self.rank,
            self.device,
            record.sequence,
            record.phase.value,
            record.mode,
            record.batch_size,
            record.tokens,
            record.bucket if record.bucket is not None else "N/A",
            accesses,
            hits,
            misses,
            rate,
            error,
        )

    def _run(self):
        waiting = []
        try:
            with self._device_scope():
                while True:
                    self._wake.clear()
                    while True:
                        try:
                            waiting.append(self._pending.get_nowait())
                        except queue.Empty:
                            break
                    while waiting:
                        record = waiting[0]
                        # close follows the runtime's successful device drain.
                        if not self._stop.is_set() and not record.completion.done():
                            break
                        self._emit(record)
                        waiting.pop(0)
                        self._free.put(record.slot)
                    if self._dropped != self._reported_drops:
                        logger.warning(
                            "MoE offload batch log buffer full: rank=%s dropped=%s; "
                            "inference continues without waiting for logging",
                            self.rank,
                            self._dropped,
                        )
                        self._reported_drops = self._dropped
                    if self._stop.is_set():
                        break
                    self._wake.wait(0.01)
        except Exception:
            # Query/logging failures cannot poison the inference owner. The
            # fixed host buffer remains alive through runtime teardown.
            self._disabled = True
            logger.warning("MoE offload batch logging disabled", exc_info=True)

    def close(self):
        """Call after the runtime's existing drain; performs no device waits."""
        self._closed = True
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            # A blocked log handler must not stall teardown indefinitely either.
            self._thread.join(timeout=1)
