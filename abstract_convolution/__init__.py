"""Compatibility package for tests.

This thin wrapper re-exports the abstract convolution
implementation located under ``src.common.tensors`` so that it can
be imported as ``abstract_convolution`` in tests and examples.
"""

from importlib import import_module as _import_module

# Re-export the laplace_nd module and its public API.
laplace_nd = _import_module("src.common.tensors.abstract_convolution.laplace_nd")

# Lift common classes/functions into the package namespace for convenience.
from src.common.tensors.abstract_convolution import (
    BuildLaplace3D,
    GridDomain,
    Transform,
    RectangularTransform,
)

__all__ = [
    "laplace_nd",
    "BuildLaplace3D",
    "GridDomain",
    "Transform",
    "RectangularTransform",
]
