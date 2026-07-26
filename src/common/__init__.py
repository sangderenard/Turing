# -*- coding: utf-8 -*-
"""Public API: dt graph construction via registry and standardized runner.

This module intentionally exposes only the high-level entry points. Low-level
helpers (node dataclasses, scaling utilities) remain internal under
``src.common.dt_system`` and should be imported from there only when needed
inside the codebase.
"""

from .dt_system.dt import SuperstepPlan, SuperstepResult
from .dt_system.dt_graph import MetaLoopRunner, GraphBuilder
from .dt_system.engine_api import (
    DtCompatibleEngine,
    EngineRegistration,
)
from .dt_system.registry import (
    register_engine,
    list_engines,
    get_engine,
)
from .dt_system.time_runtime import (
    ManagedAdvance,
    ManagedCommitGate,
    TimeWindowRequest,
    TimeAdvanceReport,
    ManagedTimeRuntime,
)

__all__ = [
    # Planning/result types
    "SuperstepPlan",
    "SuperstepResult",
    # Runner and high-level builder
    "MetaLoopRunner",
    "GraphBuilder",
    # Engine registration contract
    "DtCompatibleEngine",
    "EngineRegistration",
    # Registry entry points
    "register_engine",
    "list_engines",
    "get_engine",
    # Stable external managed-time process boundary
    "ManagedAdvance",
    "ManagedCommitGate",
    "TimeWindowRequest",
    "TimeAdvanceReport",
    "ManagedTimeRuntime",
]
