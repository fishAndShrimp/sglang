"""Opt-in MoE offloading framework; accelerator execution is backend supplied."""

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
from .interfaces import ExecutionContext
from .offloader import MoEOffloaderV2
from .owner import MoEOffloadOwner
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
    "CacheUpdateCoordinator",
    "ComposedDFCProgram",
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
    "MoEOffloadConfig",
    "MoEOffloaderV2",
    "MoEOffloadOwner",
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
