"""Physical allocation accounting for caller-owned tensor storage."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ResourceKind(Enum):
    HOST_WEIGHTS = "host_weights"
    HOT_WEIGHTS = "hot_weights"
    EXECUTION_WEIGHTS = "execution_weights"
    WORKSPACE = "workspace"
    METADATA = "metadata"


@dataclass(frozen=True)
class Allocation:
    name: str
    kind: ResourceKind
    nbytes: int
    storage_key: Any
    keepalive: Any


class AllocationLedger:
    """Account unique physical allocations and enforce byte budgets."""

    def __init__(self, limits, allocations):
        self.limits = dict(limits)
        self.allocations = tuple(allocations)
        self.usage = {kind: 0 for kind in ResourceKind}
        names, storages = set(), set()
        for item in self.allocations:
            if item.name in names or item.storage_key in storages:
                raise ValueError("Report physical allocations once; aliases are views")
            if item.nbytes < 0:
                raise ValueError("Allocation size cannot be negative")
            names.add(item.name)
            storages.add(item.storage_key)
            self.usage[item.kind] += item.nbytes
        for kind, used in self.usage.items():
            if used > self.limits.get(kind, 0):
                raise ValueError(f"{kind.value} budget exceeded: {used} bytes")


def tensor_allocations(prefix, tensors, kind=ResourceKind.METADATA):
    """Report physical storage, including views, without hiding duplicate reports.

    The ledger rejects aliases reported more than once. Empty storage has no
    allocation identity and is omitted; each record retains its tensor owner.
    """
    result = []
    for name, tensor in tensors:
        storage = tensor.untyped_storage()
        if storage.nbytes():
            result.append(
                Allocation(
                    prefix + str(name),
                    kind,
                    storage.nbytes(),
                    (str(tensor.device), storage.data_ptr()),
                    tensor,
                )
            )
    return tuple(result)
