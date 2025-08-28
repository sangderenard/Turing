from __future__ import annotations

"""
Parallel transport utilities (scaffold)
--------------------------------------

Provide discrete approximations to Levi‑Civita transport for aligning
feature frames along curves/edges on the grid.
"""

from typing import Any


class ParallelTransport:
    def transport(self, features: Any, *, manifold, path=None) -> Any:
        raise NotImplementedError("ParallelTransport.transport: to be implemented")

