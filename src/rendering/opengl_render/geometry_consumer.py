"""Display shell geometry packets without simulation or camera policy."""

from dataclasses import dataclass

import numpy as np

from src.compiler.geometry_display import decode_geometry_frame
from .renderer import MeshLayer, LineLayer, PointLayer


@dataclass(frozen=True)
class GeometryPresentation:
    generation: int
    attempt: int
    revision: int
    status: str = "presented"


class GeometryConsumer:
    """Synchronous consumer for a dedicated generic GLRenderer.

    The caller owns context/events and transports the returned completion to
    the program. Errors propagate: they must become failed/closed completions,
    never successful frame acknowledgments. This object is used on one host
    render thread, not shared concurrently.
    """

    def __init__(self, renderer):
        self.renderer = renderer
        self.last_generation = -1

    def present(self, payload: bytes) -> GeometryPresentation:
        frame = decode_geometry_frame(payload)
        if frame.generation <= self.last_generation:
            raise ValueError("geometry generation must increase after presentation")
        # Replace every layer, including empty layers, so prior frames cannot
        # leave stale geometry or text. No auto-fit or simulation-specific logic.
        self.renderer.set_mesh(MeshLayer(
            frame.vertices[:, :3], frame.indices, colors=frame.vertices[:, 3:7],
        ))
        self.renderer.set_lines(LineLayer(frame.lines[:, :3], colors=frame.lines[:, 3:7]))
        self.renderer.set_points(PointLayer(
            frame.points[:, :3], colors=frame.points[:, 3:7], sizes_px=frame.points[:, 7],
        ))
        # Wire matrix is mathematical row-major; renderer uploads GL_FALSE.
        self.renderer.set_mvp(np.ascontiguousarray(frame.mvp.T))
        self.renderer.set_overlay_text(frame.overlay.splitlines())
        self.renderer.draw(frame.viewport)
        self.last_generation = frame.generation
        return GeometryPresentation(frame.generation, frame.attempt, frame.revision)
