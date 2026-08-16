# opengl_render/renderer.py
# Minimal, context-agnostic OpenGL renderer for Mesh + Line + Point layers.
# Requires an active OpenGL 3.3+ context (created by your host app or cli.py).

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Iterable, Mapping
import ctypes
import numpy as np

from OpenGL.GL import (
    glCreateShader, glShaderSource, glCompileShader, glGetShaderiv, glGetShaderInfoLog,
    glCreateProgram, glAttachShader, glLinkProgram, glGetProgramiv, glGetProgramInfoLog,
    glDeleteShader, glDeleteProgram, glUseProgram,
    glGenVertexArrays, glBindVertexArray, glDeleteVertexArrays,
    glGenBuffers, glBindBuffer, glBufferData, glBufferSubData, glDeleteBuffers,
    glEnableVertexAttribArray, glVertexAttribPointer, glVertexAttribIPointer,
    glGetUniformLocation, glGetAttribLocation, glUniformMatrix4fv, glUniform1f, glUniform1i, glUniform4fv,
    glDrawArrays, glDrawElements, glPolygonMode, glLineWidth,
    glGenTextures, glBindTexture, glTexBuffer, glDeleteTextures, glActiveTexture,
    glEnable, glDisable, glBlendFunc, glDepthMask, glCullFace, glViewport, glClearColor, glClear,
    glWindowPos2f, glDrawPixels, glPixelStorei, glReadPixels, glGetError,
    GL_COMPILE_STATUS, GL_LINK_STATUS,
    GL_VERTEX_SHADER, GL_FRAGMENT_SHADER,
    GL_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER, GL_DYNAMIC_DRAW, GL_STATIC_DRAW,
    GL_FLOAT, GL_FALSE, GL_TRIANGLES, GL_LINES, GL_POINTS, GL_INT,
    GL_TEXTURE_BUFFER, GL_TEXTURE0, GL_RGB32F,
    GL_DEPTH_TEST, GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_PROGRAM_POINT_SIZE,
    GL_POINT_SPRITE, GL_POINT_SMOOTH,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_CULL_FACE, GL_BACK,
    GL_UNPACK_ALIGNMENT, GL_RGBA, GL_UNSIGNED_BYTE
)

# ---------------------------
# shader helpers (canonical)
# ---------------------------

def _compile_shader(src: str, stype) -> int:
    sid = glCreateShader(stype)
    glShaderSource(sid, src)
    glCompileShader(sid)
    ok = glGetShaderiv(sid, GL_COMPILE_STATUS)
    if not ok:
        log = glGetShaderInfoLog(sid).decode()
        raise RuntimeError(f"Shader compile failed:\n{log}\n----\n{src}")
    return sid

def _link_program(vs_src: str, fs_src: str) -> int:
    vs = _compile_shader(vs_src, GL_VERTEX_SHADER)
    fs = _compile_shader(fs_src, GL_FRAGMENT_SHADER)
    pid = glCreateProgram()
    glAttachShader(pid, vs)
    glAttachShader(pid, fs)
    glLinkProgram(pid)
    ok = glGetProgramiv(pid, GL_LINK_STATUS)
    if not ok:
        log = glGetProgramInfoLog(pid).decode()
        raise RuntimeError(f"Program link failed:\n{log}")
    glDeleteShader(vs); glDeleteShader(fs)
    return pid

# ---------------------------
# default shaders (cartoon)
# ---------------------------

MESH_VS = """
#version 330 core
layout(location=0) in vec3 aPos;         // vertex position
layout(location=1) in vec3 aNrm;         // vertex normal (optional)
layout(location=2) in vec4 aColor;       // per-vertex color (optional)
uniform mat4 uMVP;
out vec4 vColor;
void main(){
    vColor = aColor;
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""

# Semi-transparent fill; edge lines are drawn in a separate pass for “toon” look.
MESH_FS = """
#version 330 core
in vec4 vColor;
out vec4 FragColor;
uniform vec4 uMeshColor;     // used if no per-vertex color bound
uniform float uAlpha;        // overall alpha multiplier
void main(){
    vec4 base = (vColor.a > 0.0) ? vColor : uMeshColor;
    FragColor = vec4(base.rgb, clamp(base.a, 0.0, 1.0) * uAlpha);
}
"""

LINE_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec4 aColor;
layout(location=2) in float aActive;
uniform mat4 uMVP;
uniform float uPulse;
out vec4 vColor;
void main(){
    float activation = clamp(aActive * uPulse, 0.0, 1.0);
    vColor = vec4(
        mix(aColor.rgb, vec3(1.0), activation),
        mix(aColor.a, 1.0, activation)
    );
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""

LINE_FS = """
#version 330 core
in vec4 vColor;
out vec4 FragColor;
uniform float uAlpha;
void main(){
    FragColor = vec4(vColor.rgb, vColor.a * uAlpha);
}
"""

# Point sprites with circular mask and soft edge (cartoon dots)
POINT_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec4 aColor;   // rgba; use .a as per-point alpha
layout(location=2) in float aSize;   // pixel size
layout(location=3) in float aActive;
uniform mat4 uMVP;
uniform float uPulse;
out vec4 vColor;
void main(){
    float activation = clamp(aActive * uPulse, 0.0, 1.0);
    vColor = vec4(mix(aColor.rgb, vec3(1.0), activation), aColor.a);
    gl_Position = uMVP * vec4(aPos, 1.0);
    gl_PointSize = aSize + 5.0 * activation;
}
"""

INDEXED_LINE_VS = """
#version 330 core
layout(location=0) in int aNodeIndex;
layout(location=1) in vec4 aColor;
layout(location=2) in float aActive;
uniform samplerBuffer uPositions;
uniform mat4 uMVP;
uniform float uPulse;
out vec4 vColor;
void main(){
    float activation = clamp(aActive * uPulse, 0.0, 1.0);
    vColor = vec4(
        mix(aColor.rgb, vec3(1.0), activation),
        mix(aColor.a, 1.0, activation)
    );
    vec3 position = texelFetch(uPositions, aNodeIndex).xyz;
    gl_Position = uMVP * vec4(position, 1.0);
}
"""

POINT_FS = """
#version 330 core
in vec4 vColor;
out vec4 FragColor;
void main(){
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(uv, uv);
    if (r2 > 1.0) discard;
    float alpha = vColor.a * (1.0 - smoothstep(0.70, 1.0, r2));
    FragColor = vec4(vColor.rgb, alpha);
}
"""

# ---------------------------
# Layer dataclasses
# ---------------------------

@dataclass
class MeshLayer:
    positions: np.ndarray        # (Nv, 3) float32
    indices:   np.ndarray        # (Ni,) uint32  (triangles)
    normals:   Optional[np.ndarray] = None   # (Nv, 3)
    colors:    Optional[np.ndarray] = None   # (Nv, 4) rgba
    rgba: Tuple[float,float,float,float] = (0.3, 0.6, 0.9, 0.35)
    alpha: float = 1.0

@dataclass
class LineLayer:
    positions: np.ndarray        # (Nl, 3) float32 (pairs form GL_LINES)
    colors:    Optional[np.ndarray] = None   # (Nl, 4) rgba
    width:     float = 2.0
    alpha:     float = 1.0
    active:    Optional[np.ndarray] = None
    pulse:     float = 0.0
    topology_revision: int = -1
    activation_revision: object = None

@dataclass
class PointLayer:
    positions: np.ndarray        # (Np, 3) float32
    colors:    Optional[np.ndarray] = None   # (Np, 4) rgba
    sizes_px:  Optional[np.ndarray] = None   # (Np,) float32
    size_px_default: float = 6.0
    alpha: float = 1.0
    active: Optional[np.ndarray] = None
    pulse: float = 0.0
    topology_revision: int = -1
    activation_revision: object = None


@dataclass
class CudaGraphLayer:
    """One leased CUDA position page plus immutable topology presentation."""

    positions: object
    node_count: int
    edge_indices: np.ndarray
    node_colors: np.ndarray
    node_sizes: np.ndarray
    edge_colors: np.ndarray
    node_active: np.ndarray
    edge_active: np.ndarray
    pulse: float = 0.0
    width: float = 1.5
    topology_revision: int = -1
    activation_revision: object = None
    camera_center: Optional[np.ndarray] = None
    camera_radius: float = 8.0
    release: Optional[Callable[[], None]] = None
    

# ---------------------------
# Debug renderer
# ---------------------------

class DebugRenderer:
    """Headless renderer that pretty-prints layer data.

    This bypasses all OpenGL calls while exercising the same layer gathering
    logic used by :class:`GLRenderer`.  Each call simply prints a small table of
    the received arrays, making it suitable for test environments or machines
    without a graphics context.
    """

    def __init__(self, *, file=None):
        import sys
        self.file = file or sys.stdout

    # The OpenGL renderer expects a hook with ``print_layers`` when running in
    # debug mode.  ``layers`` is a mapping from string name to either raw
    # ``numpy`` arrays or the dataclasses defined above.
    def print_layers(self, layers: Mapping[str, object]) -> None:
        import numpy as _np

        def _preview(arr: _np.ndarray, max_rows: int = 5) -> str:
            arr = _np.asarray(arr)
            with _np.printoptions(precision=3, suppress=True, threshold=10):
                if arr.ndim >= 2:
                    arr = arr[:max_rows]
                else:
                    arr = arr[:max_rows]
                return _np.array2string(arr)

        print("=== DebugRenderer Frame ===", file=self.file)
        for name, layer in layers.items():
            print(f"[{name}]", file=self.file)
            if isinstance(layer, MeshLayer):
                print(
                    f"  positions: {_preview(layer.positions)}", file=self.file
                )
                print(f"  indices:   {_preview(layer.indices)}", file=self.file)
            elif isinstance(layer, LineLayer):
                print(
                    f"  positions: {_preview(layer.positions)}", file=self.file
                )
            elif isinstance(layer, PointLayer):
                print(
                    f"  positions: {_preview(layer.positions)}", file=self.file
                )
            elif isinstance(layer, Mapping):
                for key, arr in layer.items():
                    print(f"  {key}: {_preview(arr)}", file=self.file)
            elif isinstance(layer, (list, tuple)):
                for line in layer:
                    print(f"  {line}", file=self.file)
            else:
                try:
                    print(f"  {_preview(layer)}", file=self.file)
                except Exception:
                    print("  (unprintable layer)", file=self.file)
        print("", file=self.file)

# ---------------------------
# Core renderer
# ---------------------------

class GLRenderer:
    """A minimal scene graph: Mesh → Lines → Points (draw order)."""

    def __init__(
        self,
        mvp: Optional[np.ndarray] = None,
        *,
        size: Tuple[int, int] = (640, 480),
        point_shader_sources: tuple[str, str] | None = None,
    ):
        """Create a renderer and its backing window.

        Parameters
        ----------
        mvp:
            Optional model-view-projection matrix.  When ``None`` an identity
            matrix is used.
        size:
            ``(width, height)`` of the window in pixels.  A new ``pygame``
            window is created on construction which also establishes the OpenGL
            context used for subsequent draw calls.
        """

        # Create an OpenGL context for this renderer (pygame based).
        import pygame
        from pygame.locals import DOUBLEBUF, OPENGL

        pygame.init()
        flags = DOUBLEBUF | OPENGL
        try:  # pragma: no cover - best effort for headless environments
            pygame.display.set_mode(size, flags)
        except Exception:  # noqa: BLE001
            # Fall back to dummy driver so tests can run headless.
            import os
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            pygame.display.set_mode(size, flags)
        self._window_size = size

        # programs
        self.prog_mesh = _link_program(MESH_VS,  MESH_FS)
        self.prog_line = _link_program(LINE_VS,  LINE_FS)
        self.prog_indexed_line = _link_program(INDEXED_LINE_VS, LINE_FS)
        self._point_shader_sources = point_shader_sources
        point_vertex, point_fragment = point_shader_sources or (POINT_VS, POINT_FS)
        self.prog_point = _link_program(point_vertex, point_fragment)

        # MVP (4x4 float32, column-major)
        self.mvp = np.identity(4, dtype=np.float32) if mvp is None else np.asarray(mvp, np.float32)

        # VAOs/VBOs per layer instance
        self._mesh = None
        self._line = None
        self._indexed_line = None
        self._point = None
        self._cuda_position_resource = None

        # GL global state
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_PROGRAM_POINT_SIZE)
        # The Windows compatibility contexts created by pygame require these
        # two enables before shader point sprites become rasterized at all.
        # Without them draw calls succeed with GL_NO_ERROR but emit no pixels.
        glEnable(GL_POINT_SPRITE)
        glEnable(GL_POINT_SMOOTH)
        glCullFace(GL_BACK)

        self._overlay_lines: list[str] = []
        self._font = None
        import os
        self._capture_path = os.environ.get("TURING_GL_CAPTURE_PATH")
        self._capture_after_ms = int(os.environ.get(
            "TURING_GL_CAPTURE_AFTER_MS", "4000"
        ))
        self.buffer_growth_allocations = 0
        self.buffer_updates = 0

    @staticmethod
    def _next_buffer_capacity(required: int, current: int = 0) -> int:
        """Return a geometric byte capacity suitable for a persistent VBO."""

        capacity = max(256, int(current))
        while capacity < int(required):
            capacity *= 2
        return capacity

    def _upload_persistent(
        self,
        state: dict,
        *,
        buffer_key: str,
        capacity_key: str,
        target: int,
        data: np.ndarray,
        usage: int = GL_DYNAMIC_DRAW,
    ) -> None:
        """Update one resident GL buffer, growing it only when necessary."""

        contiguous = np.ascontiguousarray(data)
        required = int(contiguous.nbytes)
        glBindBuffer(target, state[buffer_key])
        capacity = int(state.get(capacity_key, 0))
        if required > capacity:
            capacity = self._next_buffer_capacity(required, capacity)
            glBufferData(target, capacity, None, usage)
            state[capacity_key] = capacity
            self.buffer_growth_allocations += 1
        elif capacity == 0:
            capacity = self._next_buffer_capacity(required)
            glBufferData(target, capacity, None, usage)
            state[capacity_key] = capacity
            self.buffer_growth_allocations += 1
        if required:
            glBufferSubData(target, 0, required, contiguous)
            self.buffer_updates += 1

    # ---- Mesh API ----
    def set_mesh(self, layer: MeshLayer):
        # build VAO / VBO / EBO
        vao = glGenVertexArrays(1); glBindVertexArray(vao)

        vbo = glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER, vbo)
        pos = layer.positions.astype(np.float32, copy=False)
        nrm = (layer.normals.astype(np.float32, copy=False) if layer.normals is not None else None)
        clr = (layer.colors.astype(np.float32, copy=False)  if layer.colors  is not None else None)

        # pack attributes as tightly-separated buffers (simpler updates)
        glBufferData(GL_ARRAY_BUFFER, pos.nbytes, pos, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))

        nbo = None
        if nrm is not None:
            nbo = glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER, nbo)
            glBufferData(GL_ARRAY_BUFFER, nrm.nbytes, nrm, GL_DYNAMIC_DRAW)
            glEnableVertexAttribArray(1); glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))

        cbo = None
        if clr is not None:
            cbo = glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER, cbo)
            glBufferData(GL_ARRAY_BUFFER, clr.nbytes, clr, GL_DYNAMIC_DRAW)
            glEnableVertexAttribArray(2); glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))

        ebo = glGenBuffers(1); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        idx = layer.indices.astype(np.uint32, copy=False).ravel()
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL_STATIC_DRAW)

        glBindVertexArray(0)
        self._mesh = dict(vao=vao, vbo=vbo, nbo=nbo, cbo=cbo, ebo=ebo,
                          count=idx.size,
                          rgba=np.array(layer.rgba, np.float32),
                          alpha=float(layer.alpha))

    def update_mesh_positions(self, positions: np.ndarray):
        if not self._mesh: return
        glBindBuffer(GL_ARRAY_BUFFER, self._mesh["vbo"])
        data = positions.astype(np.float32, copy=False)
        glBufferSubData(GL_ARRAY_BUFFER, 0, data.nbytes, data)

    # ---- Line API ----
    def set_lines(self, layer: LineLayer):
        pos = np.ascontiguousarray(layer.positions, dtype=np.float32)
        col = (
            np.zeros((pos.shape[0], 4), dtype=np.float32)
            if layer.colors is None
            else np.ascontiguousarray(layer.colors, dtype=np.float32)
        )
        active = (
            np.zeros(pos.shape[0], dtype=np.float32)
            if layer.active is None
            else np.ascontiguousarray(layer.active, dtype=np.float32).reshape(-1)
        )
        if self._line is None:
            vao = glGenVertexArrays(1)
            vbo = glGenBuffers(1)
            cbo = glGenBuffers(1)
            abo = glGenBuffers(1)
            self._line = dict(
                vao=vao, vbo=vbo, cbo=cbo, abo=abo,
                vbo_capacity=0, cbo_capacity=0, abo_capacity=0,
                count=0, width=float(layer.width), alpha=float(layer.alpha),
                topology_revision=None, activation_revision=None, pulse=0.0,
            )
            glBindVertexArray(vao)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(
                0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0)
            )
            glBindBuffer(GL_ARRAY_BUFFER, cbo)
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(
                1, 4, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0)
            )
            glBindBuffer(GL_ARRAY_BUFFER, abo)
            glEnableVertexAttribArray(2)
            glVertexAttribPointer(
                2, 1, GL_FLOAT, GL_FALSE, 4, ctypes.c_void_p(0)
            )
            glBindVertexArray(0)
        self._upload_persistent(
            self._line,
            buffer_key="vbo",
            capacity_key="vbo_capacity",
            target=GL_ARRAY_BUFFER,
            data=pos,
        )
        topology_revision = int(getattr(layer, "topology_revision", -1))
        if (
            topology_revision < 0
            or self._line.get("topology_revision") != topology_revision
        ):
            self._upload_persistent(
                self._line,
                buffer_key="cbo",
                capacity_key="cbo_capacity",
                target=GL_ARRAY_BUFFER,
                data=col,
            )
            self._line["topology_revision"] = topology_revision
        activation_revision = getattr(layer, "activation_revision", None)
        if (
            activation_revision is None
            or self._line.get("activation_revision") != activation_revision
        ):
            self._upload_persistent(
                self._line,
                buffer_key="abo",
                capacity_key="abo_capacity",
                target=GL_ARRAY_BUFFER,
                data=active,
            )
            self._line["activation_revision"] = activation_revision
        self._line["count"] = pos.shape[0]
        self._line["width"] = float(layer.width)
        self._line["alpha"] = float(layer.alpha)
        self._line["pulse"] = float(getattr(layer, "pulse", 0.0))

    def update_lines(self, positions: np.ndarray):
        if not self._line: return
        self._upload_persistent(
            self._line,
            buffer_key="vbo",
            capacity_key="vbo_capacity",
            target=GL_ARRAY_BUFFER,
            data=np.ascontiguousarray(positions, dtype=np.float32),
        )

    def set_cuda_graph(self, layer: CudaGraphLayer) -> None:
        """Present resident CUDA positions without constructing edge geometry."""

        from ..cuda_gl_interop import CudaGLBuffer

        tensor = layer.positions
        if not getattr(tensor, "is_cuda", False):
            raise TypeError("CudaGraphLayer.positions must be CUDA resident")
        if getattr(tensor, "dtype", None) != __import__("torch").float32:
            raise TypeError("CudaGraphLayer.positions must be float32")
        if not tensor.is_contiguous():
            raise ValueError("CudaGraphLayer.positions must be contiguous")
        node_count = int(layer.node_count)
        required = node_count * 3 * 4

        if self._point is None:
            vao = glGenVertexArrays(1)
            vbo, cbo, sbo, abo = glGenBuffers(4)
            self._point = dict(
                mode="split", vao=vao, vbo=vbo, cbo=cbo, sbo=sbo, abo=abo,
                vbo_capacity=0, cbo_capacity=0, sbo_capacity=0, abo_capacity=0,
                count=0, alpha=1.0, fluxspring=False,
                topology_revision=None, activation_revision=None, pulse=0.0,
            )
            glBindVertexArray(vao)
            for location, buffer, count, stride in (
                (0, vbo, 3, 12), (1, cbo, 4, 16),
                (2, sbo, 1, 4), (3, abo, 1, 4),
            ):
                glBindBuffer(GL_ARRAY_BUFFER, buffer)
                glEnableVertexAttribArray(location)
                glVertexAttribPointer(
                    location, count, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0)
                )
            glBindVertexArray(0)
        elif self._point.get("mode") != "split":
            raise RuntimeError("CUDA graph cannot share the particles point ABI")

        capacity = int(self._point.get("vbo_capacity", 0))
        if required > capacity or capacity == 0:
            if self._cuda_position_resource is not None:
                self._cuda_position_resource.close()
                self._cuda_position_resource = None
            capacity = self._next_buffer_capacity(required, capacity)
            glBindBuffer(GL_ARRAY_BUFFER, self._point["vbo"])
            glBufferData(GL_ARRAY_BUFFER, capacity, None, GL_DYNAMIC_DRAW)
            self._point["vbo_capacity"] = capacity
            self.buffer_growth_allocations += 1
        if self._cuda_position_resource is None:
            device_index = int(getattr(tensor.device, "index", 0) or 0)
            self._cuda_position_resource = CudaGLBuffer(
                self._point["vbo"], device_index=device_index
            )
        if required:
            self._cuda_position_resource.copy_from_tensor(tensor, required)

        topology_revision = int(layer.topology_revision)
        if self._point.get("topology_revision") != topology_revision:
            self._upload_persistent(
                self._point, buffer_key="cbo", capacity_key="cbo_capacity",
                target=GL_ARRAY_BUFFER,
                data=np.ascontiguousarray(layer.node_colors, dtype=np.float32),
            )
            self._upload_persistent(
                self._point, buffer_key="sbo", capacity_key="sbo_capacity",
                target=GL_ARRAY_BUFFER,
                data=np.ascontiguousarray(layer.node_sizes, dtype=np.float32),
            )
            self._point["topology_revision"] = topology_revision
        if self._point.get("activation_revision") != layer.activation_revision:
            self._upload_persistent(
                self._point, buffer_key="abo", capacity_key="abo_capacity",
                target=GL_ARRAY_BUFFER,
                data=np.ascontiguousarray(layer.node_active, dtype=np.float32),
            )
            self._point["activation_revision"] = layer.activation_revision
        self._point["count"] = node_count
        self._point["pulse"] = float(layer.pulse)

        if self._indexed_line is None:
            vao = glGenVertexArrays(1)
            ibo, cbo, abo = glGenBuffers(3)
            texture = glGenTextures(1)
            self._indexed_line = dict(
                vao=vao, ibo=ibo, cbo=cbo, abo=abo, texture=texture,
                ibo_capacity=0, cbo_capacity=0, abo_capacity=0,
                count=0, width=1.5, alpha=1.0, pulse=0.0,
                topology_revision=None, activation_revision=None,
            )
            glBindVertexArray(vao)
            glBindBuffer(GL_ARRAY_BUFFER, ibo)
            glEnableVertexAttribArray(0)
            glVertexAttribIPointer(0, 1, GL_INT, 4, ctypes.c_void_p(0))
            for location, buffer, count, stride in (
                (1, cbo, 4, 16), (2, abo, 1, 4),
            ):
                glBindBuffer(GL_ARRAY_BUFFER, buffer)
                glEnableVertexAttribArray(location)
                glVertexAttribPointer(
                    location, count, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0)
                )
            glBindVertexArray(0)
            glBindTexture(GL_TEXTURE_BUFFER, texture)
            glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB32F, self._point["vbo"])
            glBindTexture(GL_TEXTURE_BUFFER, 0)

        if self._indexed_line.get("topology_revision") != topology_revision:
            endpoints = np.ascontiguousarray(
                layer.edge_indices, dtype=np.int32
            ).reshape(-1)
            self._upload_persistent(
                self._indexed_line, buffer_key="ibo", capacity_key="ibo_capacity",
                target=GL_ARRAY_BUFFER, data=endpoints,
            )
            self._upload_persistent(
                self._indexed_line, buffer_key="cbo", capacity_key="cbo_capacity",
                target=GL_ARRAY_BUFFER,
                data=np.ascontiguousarray(layer.edge_colors, dtype=np.float32),
            )
            self._indexed_line["topology_revision"] = topology_revision
            self._indexed_line["count"] = int(endpoints.size)
        if self._indexed_line.get("activation_revision") != layer.activation_revision:
            self._upload_persistent(
                self._indexed_line, buffer_key="abo", capacity_key="abo_capacity",
                target=GL_ARRAY_BUFFER,
                data=np.ascontiguousarray(layer.edge_active, dtype=np.float32),
            )
            self._indexed_line["activation_revision"] = layer.activation_revision
        self._indexed_line["width"] = float(layer.width)
        self._indexed_line["pulse"] = float(layer.pulse)

    # ---- Point API ----
    def set_points(self, layer: PointLayer):
        # The older particles surface uses a nine-float interleaved ABI. Keep
        # supporting it for callers that explicitly supply that shader, while
        # FluxSpring's real LiveViz shader follows the ordinary position/color/
        # size attribute layout below.
        particles_abi = (
            self._point_shader_sources is not None
            and "in_position" in self._point_shader_sources[0]
            and "in_radius" in self._point_shader_sources[0]
        )
        fluxspring_abi = (
            self._point_shader_sources is not None
            and "in vec3 in_pos" in self._point_shader_sources[0]
            and "in vec3 in_col" in self._point_shader_sources[0]
            and "in float in_size" in self._point_shader_sources[0]
        )
        if particles_abi:
            pos = np.ascontiguousarray(layer.positions, dtype=np.float32)
            col = (
                np.ones((pos.shape[0], 4), np.float32)
                if layer.colors is None
                else layer.colors.astype(np.float32, copy=False)
            )
            size = (
                np.full((pos.shape[0],), layer.size_px_default, np.float32)
                if layer.sizes_px is None
                else layer.sizes_px.astype(np.float32, copy=False)
            )
            # This is the original particles.py ABI:
            # position3, color3, alpha, radius, kinetic-energy.
            interleaved = np.column_stack((
                pos,
                col[:, :3],
                col[:, 3],
                size / 20.0,
                np.zeros((pos.shape[0],), dtype=np.float32),
            )).astype(np.float32, copy=False)
            if self._point is None:
                vao = glGenVertexArrays(1)
                vbo = glGenBuffers(1)
                self._point = dict(
                    mode="particles", vao=vao, vbo=vbo, cbo=None, sbo=None,
                    vbo_capacity=0, count=0, alpha=float(layer.alpha),
                    fluxspring=False,
                )
                glBindVertexArray(vao)
                glBindBuffer(GL_ARRAY_BUFFER, vbo)
                stride = 9 * 4
                for name, count, offset in (
                    ("in_position", 3, 0),
                    ("in_color", 3, 3 * 4),
                    ("in_alpha", 1, 6 * 4),
                    ("in_radius", 1, 7 * 4),
                    ("in_ke", 1, 8 * 4),
                ):
                    location = glGetAttribLocation(self.prog_point, name)
                    glEnableVertexAttribArray(location)
                    glVertexAttribPointer(
                        location, count, GL_FLOAT, GL_FALSE, stride,
                        ctypes.c_void_p(offset),
                    )
                glBindVertexArray(0)
            self._upload_persistent(
                self._point,
                buffer_key="vbo",
                capacity_key="vbo_capacity",
                target=GL_ARRAY_BUFFER,
                data=interleaved,
            )
            self._point["count"] = pos.shape[0]
            self._point["alpha"] = float(layer.alpha)
            return

        pos = np.ascontiguousarray(layer.positions, dtype=np.float32)

        # colors
        # Falling back to all-zero RGBA keeps points fully transparent when a
        # layer omits colors.  This invisibility is deliberate and should not be
        # treated as an error; callers needing visible points must supply their
        # own colors.
        col = np.zeros((pos.shape[0], 4), np.float32) if layer.colors is None else layer.colors.astype(np.float32, copy=False)
        if fluxspring_abi:
            # Match LiveVizGLPoints exactly: tightly packed RGB, not the
            # generic renderer's RGBA attribute. This distinction is visible
            # on drivers that enforce the shader's vec3 declaration strictly.
            col = np.ascontiguousarray(col[:, :3], dtype=np.float32)
        color_components = 3 if fluxspring_abi else 4
        color_stride = color_components * 4

        # sizes
        size = (np.full((pos.shape[0],), layer.size_px_default, np.float32) if layer.sizes_px is None
                else np.ascontiguousarray(layer.sizes_px, dtype=np.float32))
        supports_pulse = self._point_shader_sources is None
        active = (
            np.zeros(pos.shape[0], dtype=np.float32)
            if layer.active is None
            else np.ascontiguousarray(layer.active, dtype=np.float32).reshape(-1)
        )
        if self._point is None:
            vao = glGenVertexArrays(1)
            if supports_pulse:
                vbo, cbo, sbo, abo = glGenBuffers(4)
            else:
                vbo, cbo, sbo = glGenBuffers(3)
                abo = None
            self._point = dict(
                mode="split", vao=vao, vbo=vbo, cbo=cbo, sbo=sbo, abo=abo,
                vbo_capacity=0, cbo_capacity=0, sbo_capacity=0,
                abo_capacity=0,
                count=0, alpha=float(layer.alpha),
                fluxspring=self._point_shader_sources is not None,
                topology_revision=None, activation_revision=None, pulse=0.0,
            )
            glBindVertexArray(vao)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(
                0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0)
            )
            glBindBuffer(GL_ARRAY_BUFFER, cbo)
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(
                1, color_components, GL_FLOAT, GL_FALSE, color_stride,
                ctypes.c_void_p(0),
            )
            glBindBuffer(GL_ARRAY_BUFFER, sbo)
            glEnableVertexAttribArray(2)
            glVertexAttribPointer(
                2, 1, GL_FLOAT, GL_FALSE, 4, ctypes.c_void_p(0)
            )
            if supports_pulse:
                glBindBuffer(GL_ARRAY_BUFFER, abo)
                glEnableVertexAttribArray(3)
                glVertexAttribPointer(
                    3, 1, GL_FLOAT, GL_FALSE, 4, ctypes.c_void_p(0)
                )
            glBindVertexArray(0)
        self._upload_persistent(
            self._point,
            buffer_key="vbo",
            capacity_key="vbo_capacity",
            target=GL_ARRAY_BUFFER,
            data=pos,
        )
        topology_revision = int(getattr(layer, "topology_revision", -1))
        if (
            topology_revision < 0
            or self._point.get("topology_revision") != topology_revision
        ):
            for buffer_key, capacity_key, data in (
                ("cbo", "cbo_capacity", col),
                ("sbo", "sbo_capacity", size),
            ):
                self._upload_persistent(
                    self._point,
                    buffer_key=buffer_key,
                    capacity_key=capacity_key,
                    target=GL_ARRAY_BUFFER,
                    data=data,
                )
            self._point["topology_revision"] = topology_revision
        activation_revision = getattr(layer, "activation_revision", None)
        if supports_pulse and (
            activation_revision is None
            or self._point.get("activation_revision") != activation_revision
        ):
            self._upload_persistent(
                self._point,
                buffer_key="abo",
                capacity_key="abo_capacity",
                target=GL_ARRAY_BUFFER,
                data=active,
            )
            self._point["activation_revision"] = activation_revision
        self._point["count"] = pos.shape[0]
        self._point["alpha"] = float(layer.alpha)
        self._point["pulse"] = float(getattr(layer, "pulse", 0.0))
        self._point["size_min"] = float(size.min()) if size.size else 0.0
        self._point["size_max"] = float(size.max()) if size.size else 0.0

    def update_points(self, positions: Optional[np.ndarray] = None,
                      colors: Optional[np.ndarray] = None,
                      sizes_px: Optional[np.ndarray] = None):
        #if not self._point: return
        if positions is not None:
            self._upload_persistent(
                self._point,
                buffer_key="vbo",
                capacity_key="vbo_capacity",
                target=GL_ARRAY_BUFFER,
                data=np.ascontiguousarray(positions, dtype=np.float32),
            )
        if colors is not None:
            col = np.ascontiguousarray(colors, dtype=np.float32)
            if self._point.get("fluxspring") and col.ndim == 2:
                col = np.ascontiguousarray(col[:, :3], dtype=np.float32)
            self._upload_persistent(
                self._point,
                buffer_key="cbo",
                capacity_key="cbo_capacity",
                target=GL_ARRAY_BUFFER,
                data=col,
            )
        if sizes_px is not None:
            self._upload_persistent(
                self._point,
                buffer_key="sbo",
                capacity_key="sbo_capacity",
                target=GL_ARRAY_BUFFER,
                data=np.ascontiguousarray(sizes_px, dtype=np.float32),
            )

    # ---- MVP / draw ----
    def set_mvp(self, mvp: np.ndarray):
        self.mvp = np.asarray(mvp, dtype=np.float32)

    def draw(self, viewport_px: Tuple[int,int]):
        import pygame

        w, h = viewport_px
        glViewport(0, 0, int(w), int(h))
        glClearColor(0.08, 0.08, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # 1) Meshes (base)
        if self._mesh:
            glUseProgram(self.prog_mesh)
            uMVP  = glGetUniformLocation(self.prog_mesh, "uMVP")
            uCol  = glGetUniformLocation(self.prog_mesh, "uMeshColor")
            uAlph = glGetUniformLocation(self.prog_mesh, "uAlpha")
            glUniformMatrix4fv(uMVP, 1, GL_FALSE, self.mvp)
            glUniform4fv(uCol, 1, self._mesh["rgba"])
            glUniform1f(uAlph, self._mesh["alpha"])
            glBindVertexArray(self._mesh["vao"])
            glDrawElements(GL_TRIANGLES, self._mesh["count"], 0x1405, ctypes.c_void_p(0))  # GL_UNSIGNED_INT = 0x1405
            glBindVertexArray(0)

        # 2) Lines (edges). CUDA graphs fetch endpoints from the shared
        # resident point buffer, so neither edge expansion nor position upload
        # occurs on the host.
        if self._indexed_line:
            glUseProgram(self.prog_indexed_line)
            glUniformMatrix4fv(
                glGetUniformLocation(self.prog_indexed_line, "uMVP"),
                1, GL_FALSE, self.mvp,
            )
            glUniform1f(
                glGetUniformLocation(self.prog_indexed_line, "uAlpha"),
                self._indexed_line["alpha"],
            )
            glUniform1f(
                glGetUniformLocation(self.prog_indexed_line, "uPulse"),
                self._indexed_line.get("pulse", 0.0),
            )
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_BUFFER, self._indexed_line["texture"])
            glUniform1i(
                glGetUniformLocation(self.prog_indexed_line, "uPositions"), 0
            )
            glLineWidth(max(1.0, self._indexed_line["width"]))
            glBindVertexArray(self._indexed_line["vao"])
            glDrawArrays(GL_LINES, 0, self._indexed_line["count"])
            glBindVertexArray(0)
            glBindTexture(GL_TEXTURE_BUFFER, 0)
        elif self._line:
            glUseProgram(self.prog_line)
            uMVP  = glGetUniformLocation(self.prog_line, "uMVP")
            uAlph = glGetUniformLocation(self.prog_line, "uAlpha")
            uPulse = glGetUniformLocation(self.prog_line, "uPulse")
            glUniformMatrix4fv(uMVP, 1, GL_FALSE, self.mvp)
            glUniform1f(uAlph, self._line["alpha"])
            glUniform1f(uPulse, self._line.get("pulse", 0.0))
            glLineWidth(max(1.0, self._line["width"]))
            glBindVertexArray(self._line["vao"])
            glDrawArrays(GL_LINES, 0, self._line["count"])
            glBindVertexArray(0)

        # 3) Points (peaks)
        if self._point:
            glDisable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)

            glUseProgram(self.prog_point)
            fluxspring = bool(self._point.get("fluxspring"))
            uMVP = glGetUniformLocation(
                self.prog_point,
                "u_mvp" if fluxspring else "uMVP",
            )
            glUniformMatrix4fv(uMVP, 1, GL_FALSE, self.mvp)
            if fluxspring:
                uTime = glGetUniformLocation(self.prog_point, "u_time")
                glUniform1f(uTime, float(pygame.time.get_ticks()))
            elif self._point_shader_sources is None:
                uPulse = glGetUniformLocation(self.prog_point, "uPulse")
                glUniform1f(uPulse, self._point.get("pulse", 0.0))
            glBindVertexArray(self._point["vao"])
            glDrawArrays(GL_POINTS, 0, self._point["count"])
            glBindVertexArray(0)

        if (
            self._capture_path
            and pygame.time.get_ticks() >= self._capture_after_ms
            and self._point
            and self._point["count"]
        ):
            from PIL import Image

            pixels = glReadPixels(
                0, 0, int(w), int(h), GL_RGBA, GL_UNSIGNED_BYTE
            )
            rgba = np.frombuffer(pixels, dtype=np.uint8).reshape(h, w, 4)
            Image.fromarray(np.flipud(rgba), "RGBA").save(self._capture_path)
            print(
                "[fluxspring-capture]",
                {
                    "path": self._capture_path,
                    "points": self._point["count"],
                    "size_min": self._point.get("size_min"),
                    "size_max": self._point.get("size_max"),
                    "gl_error": int(glGetError()),
                },
                flush=True,
            )
            self._capture_path = None

        glUseProgram(0)
        self._draw_overlay()
        pygame.display.flip()
    # ---- disposal ----
    def dispose(self):
        if self._cuda_position_resource is not None:
            try:
                self._cuda_position_resource.close()
            except Exception:
                pass
            self._cuda_position_resource = None
        if self._indexed_line:
            try:
                glDeleteTextures(1, [int(self._indexed_line["texture"])])
            except Exception:
                pass
        for state in (self._mesh, self._line, self._indexed_line, self._point):
            if not state:
                continue
            buffers = [
                int(state[key])
                for key in ("vbo", "nbo", "cbo", "sbo", "abo", "ebo", "ibo")
                if state.get(key) is not None
            ]
            if buffers:
                try:
                    glDeleteBuffers(len(buffers), buffers)
                except Exception:
                    pass
            try:
                glDeleteVertexArrays(1, [int(state["vao"])])
            except Exception:
                pass
        for pid in (
            self.prog_mesh, self.prog_line, self.prog_indexed_line, self.prog_point
        ):
            try:
                glDeleteProgram(pid)
            except Exception:
                pass

    def set_overlay_text(self, lines: Iterable[str]) -> None:
        self._overlay_lines = [str(l) for l in lines]

    def _draw_overlay(self) -> None:
        if not self._overlay_lines:
            return
        try:
            import pygame
            if self._font is None:
                pygame.font.init()
                self._font = pygame.font.SysFont("Courier", 14)
            # Ensure alpha blending is enabled for RGBA text surfaces
            try:
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            except Exception:
                pass

            y = self._window_size[1] - 20
            for line in self._overlay_lines:
                # Draw a subtle outline/drop shadow so text is visible on any background
                text_surf = self._font.render(line, True, (255, 255, 255))
                shadow_surf = self._font.render(line, True, (0, 0, 0))

                w, h = text_surf.get_size()
                glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

                # Offsets for a simple 4-direction outline
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    sdata = pygame.image.tostring(shadow_surf, "RGBA", True)
                    glWindowPos2f(5 + dx, y + dy)
                    glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, sdata)

                # Main text on top
                data = pygame.image.tostring(text_surf, "RGBA", True)
                glWindowPos2f(5, y)
                glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, data)
                y -= h
        except Exception:
            pass
