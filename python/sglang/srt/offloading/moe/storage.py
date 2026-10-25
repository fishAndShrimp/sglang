"""Pinned native expert storage, independent of CUDA UVA or NPU registration."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class HostMemoryCapability:
    """Backend-certified accessibility of one physical host allocation.

    Registration must own the allocation/mapping for its entire lifetime. A
    MemFabric allocation is not necessarily recognized by Tensor.is_pinned();
    this descriptor is explicit certification by the allocator, not a way to
    infer registration from a CPU pointer. Device and host addresses may differ.
    """

    host_address: int
    nbytes: int
    acl_async_readable: bool
    device_address: Optional[int]
    registration: object
    allocation_nbytes: Optional[int] = None

    def __post_init__(self):
        if self.host_address <= 0 or self.nbytes <= 0:
            raise ValueError("Host registration needs a nonempty address range")
        if self.registration is None:
            raise ValueError("Host registration requires an owning keepalive")
        if self.device_address is not None and self.device_address <= 0:
            raise ValueError("Device-visible address must be positive")
        if self.physical_nbytes < self.nbytes:
            raise ValueError("Physical allocation cannot be smaller than its range")

    @property
    def physical_nbytes(self):
        return self.nbytes if self.allocation_nbytes is None else self.allocation_nbytes

    def validate_range(self, address, nbytes):
        offset = address - self.host_address
        if offset < 0 or nbytes < 0 or offset + nbytes > self.nbytes:
            raise ValueError("Tensor/copy range is outside registered host storage")
        return offset

    def device_pointer(self, host_pointer, nbytes):
        offset = self.validate_range(host_pointer, nbytes)
        if self.device_address is None:
            raise ValueError("Host allocation has no certified device mapping")
        return self.device_address + offset
