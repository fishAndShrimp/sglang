"""Opt-in MoE offloading framework; accelerator execution is backend supplied."""

from .cache_backend import CacheBackend, CacheBackendBinding
from .cache_metadata import CacheMetadataAdapter, CacheRouteStats
from .cache_residency import CopySource
from .cache_residency import TensorResidencyOps as CacheResidencyOps
from .cache_runtime import CacheExecutionContext, CacheRuntime
from .config import ExpertCatalog, ExpertLayerSpec, MoEOffloadConfig, Phase
from .execution import (
    CacheUpdateCoordinator,
    ExecutionRequest,
    ExecutionReservation,
    ExpertWeightAccess,
    LocalExecution,
    SourcePolicy,
    UpdatePlan,
    WeightLease,
)
from .interfaces import ExecutionContext, LocalRoutes
from .offloader import MoEOffloaderV2
from .owner import MoEOffloadOwner
from .periodic_lfu import PeriodicLFUConfig, PeriodicLFUPolicy, PeriodicLFUState
from .policy import TensorLFUPolicy
from .prefill import (
    BoundedPrefillWeightPipeline,
    PrefillPipelineConfig,
    PrefillWeightPipeline,
)
from .programs import (
    ComposedDFCProgram,
    DecodeLocalExpertProgram,
    DispatchOnceProgram,
    PrefillDispatchOnceProgram,
    PrefillLocalExpertProgram,
)
from .residency import BoundExecutionWave, TensorResidencyOps
from .runtime import DecodeStateSnapshot, ManagedExecutionContext
from .storage import HostMemoryCapability

__all__ = [
    "BoundExecutionWave",
    "BoundedPrefillWeightPipeline",
    "CacheBackend",
    "CacheBackendBinding",
    "CacheExecutionContext",
    "CacheMetadataAdapter",
    "CacheResidencyOps",
    "CacheRouteStats",
    "CacheRuntime",
    "CacheUpdateCoordinator",
    "ComposedDFCProgram",
    "CopySource",
    "DecodeLocalExpertProgram",
    "DecodeStateSnapshot",
    "DispatchOnceProgram",
    "ExecutionContext",
    "ExecutionRequest",
    "ExecutionReservation",
    "ExpertCatalog",
    "ExpertLayerSpec",
    "ExpertWeightAccess",
    "LocalExecution",
    "ManagedExecutionContext",
    "LocalRoutes",
    "MoEOffloadConfig",
    "MoEOffloadOwner",
    "MoEOffloaderV2",
    "PeriodicLFUConfig",
    "PeriodicLFUPolicy",
    "PeriodicLFUState",
    "Phase",
    "HostMemoryCapability",
    "PrefillPipelineConfig",
    "PrefillDispatchOnceProgram",
    "PrefillLocalExpertProgram",
    "PrefillWeightPipeline",
    "SourcePolicy",
    "TensorLFUPolicy",
    "TensorResidencyOps",
    "UpdatePlan",
    "WeightLease",
]
