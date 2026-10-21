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
from .prefill import PrefillWeightPipeline
from .programs import (
    ComposedDFCProgram,
    DecodeLocalExpertProgram,
    DispatchOnceProgram,
    PrefillDispatchOnceProgram,
    PrefillLocalExpertProgram,
)

__all__ = [
    "CacheUpdateCoordinator",
    "ComposedDFCProgram",
    "DecodeLocalExpertProgram",
    "DispatchOnceProgram",
    "ExecutionContext",
    "ExecutionRequest",
    "ExecutionReservation",
    "ExpertCatalog",
    "ExpertLayerSpec",
    "ExpertWeightAccess",
    "LocalExecution",
    "MoEOffloadConfig",
    "MoEOffloaderV2",
    "MoEOffloadOwner",
    "Phase",
    "PrefillDispatchOnceProgram",
    "PrefillLocalExpertProgram",
    "PrefillWeightPipeline",
    "SourcePolicy",
    "TensorLFUPolicy",
    "UpdatePlan",
    "WeightLease",
]
