"""Pinned native expert storage, independent of CUDA UVA or NPU registration."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Optional

from .resources import Allocation, ResourceKind


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


class HostExpertStore:
    """Own caller-created host bundles without copying or changing their format.

    Tensors are writable during checkpoint loading/repacking. seal() freezes the
    bindings; contents are logically immutable afterward. A registration object
    owns any backend host-memory registration and stays alive with this store.
    Physical allocations are counted once, even when bundle views share storage.
    """

    def __init__(self):
        self._components = {}
        self._capabilities = {}
        self._registrations = []
        self._sealed = False

    def add(self, layer_id, name, tensor, *, capability=None):
        if self._sealed:
            raise RuntimeError("Host expert storage is sealed")
        key = (layer_id, name)
        if key in self._components:
            raise ValueError("Duplicate expert component")
        if tensor.device.type != "cpu":
            raise ValueError("Host expert components must use CPU storage")
        storage = tensor.untyped_storage()
        if capability is None:
            if not tensor.is_pinned():
                raise ValueError("Host expert components must use pinned CPU storage")
            capability = HostMemoryCapability(
                storage.data_ptr(), storage.nbytes(), True, None, tensor
            )
        if not isinstance(capability, HostMemoryCapability):
            raise TypeError("Expected a certified HostMemoryCapability")
        if not capability.acl_async_readable:
            raise ValueError("Host weights require certified asynchronous host reads")
        capability.validate_range(storage.data_ptr(), storage.nbytes())
        for previous in self._capabilities.values():
            start, end = (
                capability.host_address,
                capability.host_address + capability.physical_nbytes,
            )
            old_start = previous.host_address
            old_end = old_start + previous.physical_nbytes
            if start < old_end and old_start < end:
                if (start, end) != (
                    old_start,
                    old_end,
                ) or capability.nbytes != previous.nbytes:
                    raise ValueError(
                        "Overlapping registrations must name the same allocation"
                    )
                if capability.device_address != previous.device_address:
                    raise ValueError(
                        "Aliased host storage has inconsistent device mappings"
                    )
        self._components[key] = tensor
        self._capabilities[key] = capability

    def retain_registration(self, registration):
        if self._sealed:
            raise RuntimeError("Register host memory before sealing graph bindings")
        self._registrations.append(registration)

    def seal(self):
        if not self._components:
            raise ValueError("Cannot seal empty expert storage")
        self._sealed = True

    @property
    def components(self):
        return MappingProxyType(self._components)

    @property
    def capabilities(self):
        return MappingProxyType(self._capabilities)

    def require_device_access(self):
        """Validate the shared ACL/MemFabric master before creating graph plans."""
        if not self._sealed:
            raise RuntimeError("Seal host storage before validating device access")
        for capability in self._capabilities.values():
            capability.device_pointer(capability.host_address, capability.nbytes)

    def allocations(self):
        if not self._sealed:
            raise RuntimeError(
                "Seal host storage before finalizing its allocation ledger"
            )
        seen, allocations = set(), []
        for (layer_id, name), tensor in self._components.items():
            capability = self._capabilities[(layer_id, name)]
            identity = ("cpu", capability.host_address)
            if identity in seen:
                continue
            seen.add(identity)
            allocations.append(
                Allocation(
                    f"host.{layer_id}.{name}",
                    ResourceKind.HOST_WEIGHTS,
                    capability.physical_nbytes,
                    identity,
                    self,
                )
            )
        return tuple(allocations)
