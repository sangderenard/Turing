"""Compatibility entry for the living compiler graph visualization.

The display implementation is intentionally owned by
``rendering.precompiled_graph``: MultiNetworkFluxSpring supplies the physics
and LiveVizGLPoints supplies the visible pygame/OpenGL surface.  This module
keeps the short import used by the LLVM/OOP handoff without maintaining a
second renderer or a hidden GL context.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from ..rendering.precompiled_graph import run_precompiled_graph


@dataclass
class GraphGrowthDisplay:
    """Small lifecycle wrapper around the canonical visible surface."""

    context: Any = None
    duration: float = math.inf
    size: tuple[int, int] = (1100, 760)
    fps: int = 60
    release_hz: float = 6.0
    graph: Any = None
    active: bool = False

    def open(self) -> "GraphGrowthDisplay":
        # Context creation belongs to LiveVizGLPoints.  Keeping it there is
        # what makes this a visible, event-pumped window rather than the old
        # hidden-provider approximation.
        self.active = True
        return self

    def attach(self, graph: Any) -> "GraphGrowthDisplay":
        accessor = getattr(graph, "graph_accessor", None)
        self.graph = accessor() if callable(accessor) else graph
        return self

    def run(self) -> "GraphGrowthDisplay":
        if self.graph is None:
            raise RuntimeError("GraphGrowthDisplay.run requires attach(graph)")
        self.active = True
        try:
            run_precompiled_graph(
                self.graph,
                duration=self.duration,
                size=self.size,
                fps=self.fps,
                release_hz=self.release_hz,
            )
        finally:
            self.active = False
        return self

    def close(self) -> None:
        self.active = False

    def status(self) -> dict[str, Any]:
        return {"active": self.active, "attached": self.graph is not None}

    def __enter__(self) -> "GraphGrowthDisplay":
        return self.open()

    def __exit__(self, *_exception: Any) -> None:
        self.close()


def display_for(
    graph: Any,
    *,
    context: Any = None,
    duration: float = math.inf,
) -> GraphGrowthDisplay:
    """Launch the canonical visible surface and return after it closes."""

    return GraphGrowthDisplay(
        context=context,
        duration=duration,
    ).open().attach(graph).run()


__all__ = ["GraphGrowthDisplay", "display_for"]
