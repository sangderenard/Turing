# -*- coding: utf-8 -*-
"""DtCompatibleEngine adapter for Bath's DiscreteFluid.

This engine wraps the discrete/SPH fluid simulator found in
``src/cells/bath/discrete_fluid.py`` and exposes the minimal
DtCompatibleEngine interface used by the dt-graph runner. Time control
remains centralized in the dt graph: ``step(dt)`` advances exactly that
window (the underlying simulator may substep internally for stability).

Metrics are derived from particle velocities and mass conservation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import warnings

# Re-export implementation from the new location for backward compatibility
from .fluid_mechanics.discrete_fluid_engine import BathDiscreteFluidEngine  # type: ignore


# Emit a deprecation warning when importing from the old path at runtime
warnings.filterwarnings(
    "default",
    category=DeprecationWarning,
)

warnings.warn(
    "Importing BathDiscreteFluidEngine from dt_system.bath_fluid_engine is deprecated. "
    "Use dt_system.fluid_mechanics.discrete_fluid_engine instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["BathDiscreteFluidEngine"]
