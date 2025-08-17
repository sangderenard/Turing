# -*- coding: utf-8 -*-
"""Fluid mechanics adapters for the dt system.

This package complements ``classic_mechanics`` with wrappers that adapt
fluid simulators (discrete/SPH, voxel/MAC, hybrid particle-grid) and a
stub for a softbody simulator to the DtCompatibleEngine interface used by
``dt_graph``.
"""

from .discrete_fluid_engine import BathDiscreteFluidEngine
from .voxel_fluid_engine import VoxelFluidEngine
from .hybrid_fluid_engine import HybridFluidEngine
from .softbody_engine import SoftbodyEngineWrapper

__all__ = [
    "BathDiscreteFluidEngine",
    "VoxelFluidEngine",
    "HybridFluidEngine",
    "SoftbodyEngineWrapper",
]
