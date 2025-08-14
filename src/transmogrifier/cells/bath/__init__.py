"""Fluid mechanics namespace for cellular simulations.

The :class:`~transmogrifier.cells.bath.fluid.Bath` class lives in
:mod:`~transmogrifier.cells.bath.fluid` and is re-exported here for
convenience.
"""

from .fluid import Bath
from .adapter import BathAdapter, SPHAdapter, MACAdapter

__all__ = ["Bath", "BathAdapter", "SPHAdapter", "MACAdapter"]
