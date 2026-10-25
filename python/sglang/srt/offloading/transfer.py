"""Backend-neutral transfers into caller-owned storage.

Host-planned copies are useful for V2 and prefill. They do not implement dynamic
decode replay: a DeviceCopyPlan requires an explicitly supplied device lowering.
No transport may allocate an unaccounted expert-weight destination or staging copy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Tuple


class Completion(ABC):
    """A dependency covering every stream used by the associated operation."""

    @abstractmethod
    def wait_on(self, stream: Any) -> None:
        """Enqueue a device wait; do not synchronize the host."""

    @abstractmethod
    def synchronize(self) -> None:
        """Host wait, reserved for transitions, teardown and diagnostics."""

    @abstractmethod
    def done(self) -> bool:
        """Nonblocking host query; never use this to select graph work."""


class ImmediateCompletion(Completion):
    def wait_on(self, stream):
        pass

    def synchronize(self):
        pass

    def done(self):
        return True


class EventCompletion(Completion):
    """Works with CUDA/NPU events exposing query/synchronize/wait_event."""

    def __init__(self, event):
        self.event = event

    def wait_on(self, stream):
        stream.wait_event(self.event)

    def synchronize(self):
        self.event.synchronize()

    def done(self):
        return self.event.query()


@dataclass(frozen=True)
class TransferTicket:
    completion: Completion
    # Retain storage, registration handles, plans and destinations until complete.
    keepalive: Tuple[Any, ...] = ()


class JoinedCompletion(Completion):
    """Join unique completion facts without adding host synchronization.

    Flatten only our own aggregates. Distinct events are preserved even when
    they name the same stream; their submission order is not implied here.
    """

    def __init__(self, completions):
        pending, unique, seen = list(reversed(tuple(completions))), [], set()
        while pending:
            completion = pending.pop()
            if type(completion) is JoinedCompletion:
                pending.extend(reversed(completion.completions))
                continue
            key = (
                (EventCompletion, id(completion.event))
                if type(completion) is EventCompletion
                else (type(completion), id(completion))
            )
            if key not in seen:
                seen.add(key)
                unique.append(completion)
        self.completions = tuple(unique)

    def wait_on(self, stream):
        for completion in self.completions:
            completion.wait_on(stream)

    def synchronize(self):
        for completion in self.completions:
            completion.synchronize()

    def done(self):
        return all(completion.done() for completion in self.completions)


@dataclass(frozen=True)
class HostCopyRegion:
    """Byte ranges relative to backend-recognized allocation bases, not tensor shapes.

    The backend validates physical layout, registration, direction and aliasing.
    No dtype conversion or weight staging is implied by this descriptor.
    """

    source: Any
    destination: Any
    source_offset: int
    destination_offset: int
    nbytes: int

    def __post_init__(self):
        values = (self.source_offset, self.destination_offset, self.nbytes)
        if any(type(value) is not int for value in values):
            raise TypeError("Copy byte ranges must be integers")
        if min(values) < 0:
            raise ValueError("Copy byte ranges must be nonnegative")
        if self.source_offset + self.nbytes > self.source.nbytes:
            raise ValueError("Copy exceeds source allocation")
        if self.destination_offset + self.nbytes > self.destination.nbytes:
            raise ValueError("Copy exceeds destination allocation")


@dataclass(frozen=True)
class HostCopyPlan:
    """Eager-only descriptors covering complete expert bundles, including scales."""

    regions: Tuple[HostCopyRegion, ...]
    keepalive: Tuple[Any, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "regions", tuple(self.regions))
        if any(type(region) is not HostCopyRegion for region in self.regions):
            raise TypeError("Host plans require immutable HostCopyRegion entries")
        object.__setattr__(self, "keepalive", tuple(self.keepalive))


class CopySourceKind(IntEnum):
    """Device-plan ABI shared by residency planners and transfer lowerings."""

    NONE = 0
    HOST = 1
    HOT_SLOT = 2


@dataclass(frozen=True)
class DeviceCopyPlan:
    """Fixed-capacity tensors; active entries/count are device values.

    source_kind distinguishes HOST, HOT_SLOT and NONE. expected_generation guards
    reused slots. count is the number of valid entries (not a scan-prefix length).
    An expert entry includes all weights and quantization metadata.
    The backend validates shapes/dtypes once when binding, outside capture.
    """

    source_kind: Any
    source_index: Any
    destination_slot: Any
    valid: Any
    expected_generation: Any
    count: Any
    capacity: int


class HostTransferOps(ABC):
    """Host-planned copies into owned storage; never a dynamic replay fallback.

    An ACL batch implementation must preserve torch_npu host queue ordering,
    reject dependent/overlapping batch entries and record bundle readiness only
    after every submitted copy. A failed submission may still have live writes.
    """

    @abstractmethod
    def materialize_into(self, plan: HostCopyPlan, bindings, stream) -> TransferTicket:
        """Submit on stream; retain registrations/descriptors through completion."""


class DeviceTransferOps(ABC):
    """Hardware lowering for device-selected sources and fixed destinations."""

    @abstractmethod
    def materialize_into(self, plan, bindings, stream) -> TransferTicket:
        """Emit graph operations. No .item(), .tolist() or CPU route decisions."""
