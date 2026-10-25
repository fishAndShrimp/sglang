"""Ascend invocation, model-hook and full-graph lifecycle binding.

Native dependencies are initialized explicitly by the backend.
The native extension, registered master and EP/FFN adapters are assembled by
``NpuMoEBackend``; no capture-time ticket is used to certify a later replay.
"""

from __future__ import annotations

from contextlib import contextmanager

from ...transfer import EventCompletion, TransferTicket


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
