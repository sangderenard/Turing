# opengl_render/renderer.py
# Minimal, context-agnostic OpenGL renderer for Mesh + Line + Point layers.
# Requires an active OpenGL 3.3+ context (created by your host app or cli.py).

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Iterable, Mapping
import ctypes
import numpy as np

from OpenGL.GL import (
    glCreateShader, glShaderSource, glCompileShader, glGetShaderiv, glGetShaderInfoLog,
    glCreateProgram, glAttachShader, glLinkProgram, glGetProgramiv, glGetProgramInfoLog,
    glDeleteShader, glDeleteProgram, glUseProgram,
    glGenVertexArrays, glBindVertexArray,
    glGenBuffers, glBindBuffer, glBufferData, glBufferSubData,
    glEnableVertexAttribArray, glVertexAttribPointer,
    glGetUniformLocation, glUniformMatrix4fv, glUniform1f, glUniform4fv,
    glDrawArrays, glDrawElements, glPolygonMode, glLineWidth,
    glEnable, glDisable, glBlendFunc, glDepthMask, glCullFace, glViewport, glClearColor, glClear,
    GL_COMPILE_STATUS, GL_LINK_STATUS,
    GL_VERTEX_SHADER, GL_FRAGMENT_SHADER,
    GL_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER, GL_DYNAMIC_DRAW, GL_STATIC_DRAW,
    GL_FLOAT, GL_FALSE, GL_TRIANGLES, GL_LINES, GL_POINTS,
    GL_DEPTH_TEST, GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_PROGRAM_POINT_SIZE,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_CULL_FACE, GL_BACK
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
uniform mat4 uMVP;
out vec4 vColor;
void main(){
    vColor = aColor;
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
uniform mat4 uMVP;
out vec4 vColor;
void main(){
    vColor = aColor;
    gl_Position = uMVP * vec4(aPos, 1.0);
    gl_PointSize = aSize;
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
    float edge = smoothstep(1.0, 0.7, r2);
    FragColor = vec4(vColor.rgb, vColor.a * (1.0 - edge));
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

@dataclass
class PointLayer:
    positions: np.ndarray        # (Np, 3) float32
    colors:    Optional[np.ndarray] = None   # (Np, 4) rgba
    sizes_px:  Optional[np.ndarray] = None   # (Np,) float32
    size_px_default: float = 6.0
    alpha: float = 1.0

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

    def __init__(self, mvp: Optional[np.ndarray] = None):
        # programs
        self.prog_mesh = _link_program(MESH_VS,  MESH_FS)
        self.prog_line = _link_program(LINE_VS,  LINE_FS)
        self.prog_point= _link_program(POINT_VS, POINT_FS)

        # MVP (4x4 float32, column-major)
        self.mvp = np.identity(4, dtype=np.float32) if mvp is None else np.asarray(mvp, np.float32)

        # VAOs/VBOs per layer instance
        self._mesh = None
        self._line = None
        self._point = None

        # GL global state
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_PROGRAM_POINT_SIZE)
        glCullFace(GL_BACK)

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
        vao = glGenVertexArrays(1); glBindVertexArray(vao)

        # positions
        vbo = glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER, vbo)
        pos = layer.positions.astype(np.float32, copy=False)
        glBufferData(GL_ARRAY_BUFFER, pos.nbytes, pos, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))

        # colors
        cbo = None
        if layer.colors is not None:
            cbo = glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER, cbo)
            col = layer.colors.astype(np.float32, copy=False)
            glBufferData(GL_ARRAY_BUFFER, col.nbytes, col, GL_DYNAMIC_DRAW)
            glEnableVertexAttribArray(1); glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))

        glBindVertexArray(0)
        self._line = dict(vao=vao, vbo=vbo, cbo=cbo, count=pos.shape[0], width=float(layer.width), alpha=float(layer.alpha))

    def update_lines(self, positions: np.ndarray):
        if not self._line: return
        glBindBuffer(GL_ARRAY_BUFFER, self._line["vbo"])
        data = positions.astype(np.float32, copy=False)
        glBufferSubData(GL_ARRAY_BUFFER, 0, data.nbytes, data)

    # ---- Point API ----
    def set_points(self, layer: PointLayer):
        vao = glGenVertexArrays(1); glBindVertexArray(vao)

        # positions
        vbo = glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER, vbo)
        pos = layer.positions.astype(np.float32, copy=False)
        glBufferData(GL_ARRAY_BUFFER, pos.nbytes, pos, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))

        # colors
        col = np.zeros((pos.shape[0], 4), np.float32) if layer.colors is None else layer.colors.astype(np.float32, copy=False)
        cbo = glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER, cbo)
        glBufferData(GL_ARRAY_BUFFER, col.nbytes, col, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(1); glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))

        # sizes
        size = (np.full((pos.shape[0],), layer.size_px_default, np.float32) if layer.sizes_px is None
                else layer.sizes_px.astype(np.float32, copy=False))
        sbo = glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER, sbo)
        glBufferData(GL_ARRAY_BUFFER, size.nbytes, size, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(2); glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 4, ctypes.c_void_p(0))

        glBindVertexArray(0)
        self._point = dict(vao=vao, vbo=vbo, cbo=cbo, sbo=sbo, count=pos.shape[0], alpha=float(layer.alpha))

    def update_points(self, positions: Optional[np.ndarray] = None,
                      colors: Optional[np.ndarray] = None,
                      sizes_px: Optional[np.ndarray] = None):
        if not self._point: return
        if positions is not None:
            glBindBuffer(GL_ARRAY_BUFFER, self._point["vbo"])
            data = positions.astype(np.float32, copy=False)
            glBufferSubData(GL_ARRAY_BUFFER, 0, data.nbytes, data)
        if colors is not None:
            glBindBuffer(GL_ARRAY_BUFFER, self._point["cbo"])
            col = colors.astype(np.float32, copy=False)
            glBufferSubData(GL_ARRAY_BUFFER, 0, col.nbytes, col)
        if sizes_px is not None:
            glBindBuffer(GL_ARRAY_BUFFER, self._point["sbo"])
            s = sizes_px.astype(np.float32, copy=False)
            glBufferSubData(GL_ARRAY_BUFFER, 0, s.nbytes, s)

    # ---- MVP / draw ----
    def set_mvp(self, mvp: np.ndarray):
        self.mvp = np.asarray(mvp, dtype=np.float32)

    def draw(self, viewport_px: Tuple[int,int]):
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

        # 2) Lines (edges)
        if self._line:
            glUseProgram(self.prog_line)
            uMVP  = glGetUniformLocation(self.prog_line, "uMVP")
            uAlph = glGetUniformLocation(self.prog_line, "uAlpha")
            glUniformMatrix4fv(uMVP, 1, GL_FALSE, self.mvp)
            glUniform1f(uAlph, self._line["alpha"])
            glLineWidth(max(1.0, self._line["width"]))
            glBindVertexArray(self._line["vao"])
            glDrawArrays(GL_LINES, 0, self._line["count"])
            glBindVertexArray(0)

        # 3) Points (peaks)
        if self._point:
            glUseProgram(self.prog_point)
            uMVP  = glGetUniformLocation(self.prog_point, "uMVP")
            glUniformMatrix4fv(uMVP, 1, GL_FALSE, self.mvp)
            glBindVertexArray(self._point["vao"])
            glDrawArrays(GL_POINTS, 0, self._point["count"])
            glBindVertexArray(0)

        glUseProgram(0)

    # ---- disposal ----
    def dispose(self):
        for pid in (self.prog_mesh, self.prog_line, self.prog_point):
            try: glDeleteProgram(pid)
            except Exception: pass
