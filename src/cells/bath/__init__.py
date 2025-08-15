"""Fluid mechanics namespace for cellular simulations.

The :class:`~transmogrifier.cells.bath.fluid.Bath` class lives in
:mod:`~transmogrifier.cells.bath.fluid` and is re-exported here for
convenience.
"""

from .fluid import Bath
from .adapter import (
    BathAdapter,
    SPHAdapter,
    MACAdapter,
    HybridAdapter,
    run_headless,
    run_opengl,
)
from .hybrid_fluid import HybridFluid

__all__ = [
    "Bath",
    "BathAdapter",
    "SPHAdapter",
    "MACAdapter",
    "HybridAdapter",
    "HybridFluid",
    "run_headless",
    "run_opengl",
]
