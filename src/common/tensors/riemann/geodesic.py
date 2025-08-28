from __future__ import annotations

"""
GeodesicConv3D (scaffold)
------------------------

Local kernel aggregation over approximate geodesic neighborhoods.
Roadmap:
- Build local neighborhoods via exp‑map or discrete stencils
- Apply learned kernel weights over neighbors
- Optional heat‑kernel weighting for distance decay
"""

from typing import Any


class GeodesicConv3D:
    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int = 5) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = int(kernel_size)

    def forward(self, x: Any, *, manifold) -> Any:
        raise NotImplementedError("GeodesicConv3D.forward: to be implemented")

