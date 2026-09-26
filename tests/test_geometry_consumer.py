from dataclasses import replace

import numpy as np
import pytest

from src.compiler.geometry_display import GeometryFrame
from src.rendering.opengl_render.geometry_consumer import GeometryConsumer
from src.rendering.opengl_render import renderer as gl


def test_geometry_consumer_completion_camera_empty_layers_and_failure():
    class Renderer:
        fail = False

        def set_mesh(self, layer): self.mesh = layer
        def set_lines(self, layer): self.lines = layer
        def set_points(self, layer): self.points = layer
        def set_mvp(self, matrix): self.mvp = matrix
        def set_overlay_text(self, text): self.text = text
        def draw(self, viewport):
            self.viewport = viewport
            assert consumer.last_generation < generation[0]
            if self.fail:
                raise RuntimeError("presentation failed")

    renderer = Renderer()
    consumer = GeometryConsumer(renderer)
    generation = [4]
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 3] = 5
    vertices = np.array([[0, 0, 0, 1, 1, 1, 1]] * 3, dtype=np.float32)
    frame = GeometryFrame(4, 2, 1, (640, 480), matrix, vertices,
                          np.array([0, 1, 2], dtype=np.uint32), vertices[:2],
                          np.array([[0, 0, 0, 1, 1, 1, 1, 6]]), "frame 4")
    done = consumer.present(frame.encode())
    assert (done.generation, done.attempt, done.revision, done.status) == (4, 2, 1, "presented")
    np.testing.assert_array_equal(renderer.mvp, matrix.T)
    with pytest.raises(ValueError, match="generation"):
        consumer.present(frame.encode())
    generation[0] = 5
    empty = replace(frame, generation=5, vertices=np.empty((0, 7)),
                    indices=np.empty(0, dtype=np.uint32), lines=np.empty((0, 7)),
                    points=np.empty((0, 8)), overlay="")
    renderer.fail = True
    with pytest.raises(RuntimeError, match="presentation failed"):
        consumer.present(empty.encode())
    assert consumer.last_generation == 4
    renderer.fail = False
    assert consumer.present(empty.encode()).generation == 5
    assert renderer.mesh.indices.size == renderer.lines.positions.size == renderer.points.positions.size == 0
    assert renderer.text == []


def test_mesh_upload_reuses_buffers_and_clears_removed_attributes(monkeypatch):
    calls = []
    monkeypatch.setattr(gl, "_link_program", lambda *a: 1)
    for name in tuple(vars(gl)):
        if name.startswith("gl") and callable(getattr(gl, name)):
            monkeypatch.setattr(gl, name, lambda *args, _name=name: calls.append((_name, args)))
    allocations = []
    monkeypatch.setattr(gl, "glGenVertexArrays", lambda count: 10)

    def allocate(count):
        allocations.append(count)
        return [11, 12, 13, 14]

    monkeypatch.setattr(gl, "glGenBuffers", allocate)
    renderer = gl.GLRenderer(host=gl.RendererHost(lambda: None, lambda: 0))
    positions = np.ones((3, 3), dtype=np.float32)
    renderer.set_mesh(gl.MeshLayer(positions, np.array([0, 1, 2]), colors=np.ones((3, 4))))
    assert renderer._mesh["has_colors"] is True
    growth = renderer.buffer_growth_allocations
    renderer.set_mesh(gl.MeshLayer(positions * 2, np.array([2, 1, 0])))
    assert renderer._mesh["has_colors"] is False
    assert allocations == [4]
    assert renderer.buffer_growth_allocations == growth
    uploads = [args[3] for name, args in calls if name == "glBufferSubData"]
    np.testing.assert_array_equal(uploads[-2], np.zeros((3, 4)))
    renderer.set_mesh(gl.MeshLayer(np.empty((0, 3)), np.empty(0, dtype=np.uint32)))
    assert renderer._mesh["count"] == 0
    assert allocations == [4]
