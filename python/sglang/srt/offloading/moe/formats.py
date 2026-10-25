"""Immutable expert bundle descriptions and the format behavior they parameterize."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class BundleComponent:
    name: str
    raw_shape: tuple
    runtime_shape: tuple
    runtime_stride: tuple
    dtype: object
    nbytes: int
    checkpoint_dtype: object
    host_nbytes: int
    shards: tuple
    runtime_name: str
    attributes: tuple = ()
    alignment: int = 64

    def __post_init__(self):
        if (
            not self.name
            or not self.shards
            or len(set(self.shards)) != len(self.shards)
        ):
            raise ValueError("Bundle components require a name and distinct shards")
        if (
            self.nbytes <= 0
            or self.host_nbytes < self.nbytes
            or self.alignment <= 0
            or self.alignment & (self.alignment - 1)
            or self.host_nbytes % self.alignment
        ):
            raise ValueError("Invalid physical component extent or alignment")


@dataclass(frozen=True)
class BundleSchema:
    """One storage-compatible arena; fingerprints include native format semantics."""

    format_id: str
    components: tuple
    conversion_groups: tuple
    new_outputs: tuple
    dispatch_config: tuple

    def __post_init__(self):
        names = tuple(c.name for c in self.components)
        converted = tuple(name for group in self.conversion_groups for name in group)
        if (
            not names
            or len(set(names)) != len(names)
            or sorted(converted) != sorted(names)
        ):
            raise ValueError("Conversion groups must cover each component exactly once")
        if not set(self.new_outputs) <= set(names):
            raise ValueError("Conversion workspace names must be bundle components")

    @property
    def fingerprint(self):
        return self.format_id, self.components

    @property
    def required_shards(self):
        return sum(len(c.shards) for c in self.components)

    @property
    def expert_nbytes(self):
        return sum(c.nbytes for c in self.components)


class ExpertFormat(ABC):
    """A complete native bundle, including checkpoint validation and bounded loading."""

    schema: BundleSchema

    @abstractmethod
    def write_checkpoint(self, name, target, value, shard, metadata):
        """Validate and write one owned shard; metadata is loader-local state."""

    @abstractmethod
    def repack_into(self, raw_components, expert, destinations):
        """Validate native layout before overwriting the authoritative host bytes."""

    @abstractmethod
    def allocate_bank(self, capacity, kind, name):
        """Allocate fixed native expert extents with an explicit byte budget."""
