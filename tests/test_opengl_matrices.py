import numpy as np
from dataclasses import dataclass

from src.opengl_render import api as gl_api
from src.opengl_render.api import (
    SecondOrderAutoFitState,
    _perspective,
    _look_at,
    advance_second_order_autofit,
    draw_layers,
)
from src.cells.softbody.geometry.geodesic import icosahedron

# Provide minimal stand-ins for GL dataclasses when OpenGL is unavailable
if gl_api.MeshLayer is object:  # pragma: no cover - depends on environment
    @dataclass
    class MeshLayer:
        positions: np.ndarray
        indices: np.ndarray
        colors: np.ndarray | None = None

    class LineLayer:
        pass

    class PointLayer:
        positions: np.ndarray

    gl_api.MeshLayer = MeshLayer  # type: ignore
    gl_api.LineLayer = LineLayer  # type: ignore
    gl_api.PointLayer = PointLayer  # type: ignore
else:  # pragma: no cover - real OpenGL path
    MeshLayer = gl_api.MeshLayer

class _StubRenderer:
    def __init__(self):
        self.mesh = None
        self.mvp = None
        self.viewport = None
    def set_mesh(self, mesh):
        self.mesh = mesh
    def set_lines(self, lines):
        self.lines = lines
    def set_points(self, points):
        self.points = points
    def set_mvp(self, mvp):
        self.mvp = mvp
    def draw(self, viewport):
        self.viewport = viewport


def test_perspective_and_look_at_row_major():
    p = _perspective(60.0, 1.0, 0.1, 100.0)
    # Last row contains perspective canonical values in row-major layout
    assert np.allclose(p[3], [0.0, 0.0, -1.0, 0.0])

    eye = np.array([0.0, 0.0, 5.0], np.float32)
    center = np.zeros(3, np.float32)
    up = np.array([0.0, 1.0, 0.0], np.float32)
    v = _look_at(eye, center, up)
    # Translation appears in final column, not final row
    assert np.allclose(v[3, :], [0.0, 0.0, 0.0, 1.0])
    assert np.allclose(v[:3, 3], -v[:3, :3] @ eye)


def test_draw_layers_transposes_mvp_and_handles_icosahedron():
    V, F = icosahedron()
    mesh = MeshLayer(V.astype(np.float32), F.astype(np.uint32))
    renderer = _StubRenderer()
    viewport = (640, 480)
    draw_layers(renderer, {"membrane": mesh}, viewport)
    # Geometry was uploaded
    assert renderer.mesh is not None and renderer.mesh.positions.shape == (12, 3)
    assert renderer.viewport == viewport

    # Compute expected MVP as used in draw_layers
    pts = mesh.positions
    center = pts.mean(axis=0)
    radius = float(np.linalg.norm(pts - center, axis=1).max())
    eye = center + np.array([0.0, 0.0, radius * 3.0], np.float32)
    up = np.array([0.0, 1.0, 0.0], np.float32)
    aspect = viewport[0] / viewport[1]
    expected = (
        _perspective(45.0, aspect, 0.1, radius * 10.0)
        @ _look_at(eye, center, up)
    ).T
    assert np.allclose(renderer.mvp, expected, atol=2e-7)


def test_draw_layers_tolerates_an_initially_empty_point_graph():
    points = gl_api.PointLayer(
        positions=np.empty((0, 3), dtype=np.float32),
        colors=np.empty((0, 4), dtype=np.float32),
    )
    renderer = _StubRenderer()

    draw_layers(renderer, {"points": points}, (640, 480))

    assert renderer.points is points
    assert renderer.mvp is None
    assert renderer.viewport == (640, 480)


def test_draw_layers_can_use_an_oblique_perspective_view():
    V, F = icosahedron()
    mesh = MeshLayer(V.astype(np.float32), F.astype(np.uint32))
    renderer = _StubRenderer()
    direction = np.array([0.55, 0.32, 1.0], dtype=np.float32)

    draw_layers(
        renderer,
        {"membrane": mesh},
        (640, 480),
        view_direction=tuple(direction),
    )

    center = mesh.positions.mean(axis=0)
    radius = float(np.linalg.norm(mesh.positions - center, axis=1).max())
    direction /= np.linalg.norm(direction)
    eye = center + direction * radius * 3.0
    expected = (
        _perspective(45.0, 640 / 480, 0.1, radius * 10.0)
        @ _look_at(eye, center, np.array([0.0, 1.0, 0.0], np.float32))
    ).T
    assert np.allclose(renderer.mvp, expected, atol=2e-7)


def test_second_order_autofit_double_smooths_bounds_changes():
    state = SecondOrderAutoFitState(
        center_1=np.zeros(3, dtype=np.float32),
        center_2=np.zeros(3, dtype=np.float32),
        radius_1=1.0,
        radius_2=1.0,
        updated_at=0.0,
    )
    state = advance_second_order_autofit(
        state,
        np.array([10.0, 0.0, 0.0], dtype=np.float32),
        10.0,
        now=0.1,
        time_constant=0.22,
    )

    assert 0.0 < state.center_2[0] < state.center_1[0] < 10.0
    assert 1.0 < state.radius_2 < state.radius_1 < 10.0
