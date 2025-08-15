"""Minimal OpenGL renderer package with mesh, line and point layers."""

from .renderer import GLRenderer, MeshLayer, LineLayer, PointLayer
from .api import (
    rainbow_colors,
    rainbow_history_points,
    pack_mesh,
    pack_lines,
    pack_points,
    cellsim_layers,
    fluid_layers,
)

__all__ = [
    "GLRenderer",
    "MeshLayer",
    "LineLayer",
    "PointLayer",
    # API helpers
    "rainbow_colors",
    "rainbow_history_points",
    "pack_mesh",
    "pack_lines",
    "pack_points",
    "cellsim_layers",
    "fluid_layers",
]
