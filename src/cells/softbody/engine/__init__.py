"""XPBD engine and geometry utilities for softbody simulations."""

from .params import EngineParams
from .xpbd_core import XPBDSolver
from .constraints import (
    StretchConstraint,
    VolumeConstraint,
    DihedralBendingConstraint,
    PlaneContact,
)
from .hierarchy import Cell, Organelle
from .collisions import resolve_membrane_collisions, build_self_contacts_spatial_hash

__all__ = [
    "EngineParams",
    "XPBDSolver",
    "StretchConstraint",
    "VolumeConstraint",
    "DihedralBendingConstraint",
    "PlaneContact",
    "Cell",
    "Organelle",
    "resolve_membrane_collisions",
    "build_self_contacts_spatial_hash",
]
