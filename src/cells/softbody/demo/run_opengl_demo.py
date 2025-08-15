import os, sys, time, math, random
import argparse
import ctypes
from typing import Dict
import numpy as np

# ----- Cellsim backend (uses your code) -------------------------------------
from .run_numpy_demo import (
    make_cellsim_backend as base_make_cellsim_backend,
    step_cellsim,
    build_numpy_parser,
    get_numpy_tag_names,
    extract_numpy_kwargs,
    run_fluid_demo,
    _perspective,
    _look_at,
    _translate,
    _rotate_y,
    _compute_center_radius_pts,
)


def make_cellsim_backend_from_args(args):
    """Construct backend using shared numpy parameters from CLI args."""
    api, provider = base_make_cellsim_backend(
        cell_vols=args.cell_vols,
        cell_imps=args.cell_imps,
        cell_elastic_k=args.cell_elastic_k,
        bath_na=args.bath_na,
        bath_cl=args.bath_cl,
        bath_pressure=args.bath_pressure,
        bath_volume_factor=args.bath_volume_factor,
        substeps=args.substeps,
        dt_provider=args.dt_provider,
        dim=args.sim_dim,
    )

    # color identity (rendering only)
    levels = [64, 144, 208]
    for i, c in enumerate(api.cells):
        setattr(c, "_identity_green", levels[i % len(levels)])

    return api, provider
def _normalize_vec(x):
    x = np.asarray(x, dtype=np.float64)
    m = float(np.max(x)) if x.size else 0.0
    return (x / m) if m > 1e-12 else np.zeros_like(x, dtype=np.float64)


def compute_curl(vec_field: np.ndarray,
                 spacing: float | tuple[float, float, float] = 1.0) -> np.ndarray:
    """Compute curl of a vector field on a regular grid using finite differences.

    Parameters
    ----------
    vec_field:
        Array of shape ``(..., d)`` where ``d`` is ``2`` or ``3`` representing the
        components of the vector field arranged on a regular grid. The spatial
        dimensions precede the last axis.
    spacing:
        Grid spacing along each axis. Either a scalar applied to all axes or a
        sequence of length ``d``.

    Returns
    -------
    np.ndarray
        Curl with the same spatial shape. For ``d=2`` the result is a scalar
        ``(...,)`` representing the out-of-plane component. For ``d=3`` the
        result has shape ``(..., 3)``.
    """

    vec_field = np.asarray(vec_field, dtype=np.float64)
    dim = vec_field.shape[-1]
    spacing = np.broadcast_to(np.array(spacing, dtype=np.float64), (dim,))

    if dim == 2:
        Fx = vec_field[..., 0]
        Fy = vec_field[..., 1]
        dFydx, dFydy = np.gradient(Fy, *spacing, edge_order=2)
        dFxdx, dFxdy = np.gradient(Fx, *spacing, edge_order=2)
        return dFydx - dFxdy
    elif dim == 3:
        Fx = vec_field[..., 0]
        Fy = vec_field[..., 1]
        Fz = vec_field[..., 2]
        dFxdx, dFxdy, dFxdz = np.gradient(Fx, *spacing, edge_order=2)
        dFydx, dFydy, dFydz = np.gradient(Fy, *spacing, edge_order=2)
        dFzdx, dFzdy, dFzdz = np.gradient(Fz, *spacing, edge_order=2)
        curl_x = dFzdy - dFydz
        curl_y = dFxdz - dFzdx
        curl_z = dFydx - dFxdy
        return np.stack((curl_x, curl_y, curl_z), axis=-1)
    else:
        raise ValueError("vec_field must have 2 or 3 components in its last axis")

def _measure_pressure_mass(h, api):
    """Return (pressures, masses, greens255) per cell index."""
    pressures, masses, greens = [], [], []
    for i, c in enumerate(h.cells):
        # --- pressure ---
        p = 0.0
        if hasattr(c, "contact_pressure_estimate"):
            try:
                p = float(c.contact_pressure_estimate())
            except Exception:
                p = 0.0
        elif hasattr(api.cells[i], "internal_pressure"):
            p = float(getattr(api.cells[i], "internal_pressure", 0.0))
        else:
            # Laplace fallback from tension + radius (if available)
            try:
                V = abs(c.enclosed_volume()); R = ((3.0*V)/(4.0*math.pi))**(1.0/3.0)
                gamma = float(getattr(api.cells[i], "membrane_tension", getattr(c, "membrane_tension", 0.0)))
                p = 2.0 * gamma / max(R, 1e-6)
            except Exception:
                p = 0.0

        # --- total dissolved mass ---
        n = getattr(api.cells[i], "n", None)
        masses.append(sum(n.values()) if isinstance(n, dict) else 0.0)

        # --- identity green (0..255) ---
        g = getattr(api.cells[i], "_identity_green", getattr(c, "_identity_green", 128))
        greens.append(int(g))

        pressures.append(p)
    return np.array(pressures), np.array(masses), np.array(greens, dtype=np.int32)

# ----- OpenGL minimal renderer (pygame + PyOpenGL) --------------------------
try:
    import pygame
    from pygame.locals import DOUBLEBUF, OPENGL
    from OpenGL.GL import (
        # shader / program
        glCreateShader, glShaderSource, glCompileShader, glGetShaderiv, glGetShaderInfoLog,
        glCreateProgram, glAttachShader, glLinkProgram, glGetProgramiv, glGetProgramInfoLog, glDeleteShader,
        # buffers / vao
        glGenVertexArrays, glGenBuffers, glBindVertexArray, glBindBuffer, glBufferData, glBufferSubData,
        glEnableVertexAttribArray, glVertexAttribPointer, glVertexAttribDivisor,
        # uniforms / draw
        glGetUniformLocation, glUniform4fv, glUniform1f, glUniformMatrix4fv,
        glDrawElements, glDrawArrays, glDrawArraysInstanced, glUseProgram,
        # state
        glEnable, glBlendFunc, glViewport, glClearColor, glClear, glDepthMask, glCullFace,
        # enums
        GL_COMPILE_STATUS, GL_LINK_STATUS,
        GL_VERTEX_SHADER, GL_FRAGMENT_SHADER,
    GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW, GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW,
        GL_FLOAT, GL_FALSE, GL_TRUE,
        GL_TRIANGLES, GL_UNSIGNED_INT, GL_POINTS,
        GL_DEPTH_TEST, GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_PROGRAM_POINT_SIZE,
        GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
        GL_CULL_FACE, GL_BACK,
    )
except Exception as e:
    print("This demo needs pygame + PyOpenGL. Try: pip install pygame PyOpenGL")
    raise

# --- tiny math helpers (column-major) ---
def perspective(fovy_deg, aspect, znear, zfar):
    f = 1.0 / math.tan(math.radians(fovy_deg) * 0.5)
    m = np.zeros((4,4), dtype=np.float32)
    m[0,0] = f / aspect
    m[1,1] = f
    m[2,2] = (zfar + znear) / (znear - zfar)
    m[2,3] = (2 * zfar * znear) / (znear - zfar)
    m[3,2] = -1.0
    return m


def ortho(left, right, bottom, top, znear, zfar):
    m = np.identity(4, dtype=np.float32)
    m[0,0] = 2.0 / (right - left)
    m[1,1] = 2.0 / (top - bottom)
    m[2,2] = -2.0 / (zfar - znear)
    m[3,3] = 1.0
    m[0,3] = -(right + left) / (right - left)
    m[1,3] = -(top + bottom) / (top - bottom)
    m[2,3] = -(zfar + znear) / (zfar - znear)
    return m

def look_at(eye, center, up):
    eye = np.array(eye, dtype=np.float32)
    center = np.array(center, dtype=np.float32)
    up = np.array(up, dtype=np.float32)
    f = center - eye; f = f / (np.linalg.norm(f) + 1e-12)
    s = np.cross(f, up); s = s / (np.linalg.norm(s) + 1e-12)
    u = np.cross(s, f)
    m = np.identity(4, dtype=np.float32)
    m[0,0:3] = s
    m[1,0:3] = u
    m[2,0:3] = -f
    m[:3,3] = -np.array([np.dot(s, eye), np.dot(u, eye), np.dot(-f, eye)], dtype=np.float32)
    return m

def compile_shader(src, stype):
    sid = glCreateShader(stype)
    glShaderSource(sid, src)
    glCompileShader(sid)
    ok = glGetShaderiv(sid, GL_COMPILE_STATUS)
    if not ok:
        log = glGetShaderInfoLog(sid).decode()
        raise RuntimeError(f"Shader compile failed: {log}\n{src}")
    return sid

def link_program(vs_src, fs_src):
    vs = compile_shader(vs_src, GL_VERTEX_SHADER)
    fs = compile_shader(fs_src, GL_FRAGMENT_SHADER)
    pid = glCreateProgram()
    glAttachShader(pid, vs)
    glAttachShader(pid, fs)
    glLinkProgram(pid)
    ok = glGetProgramiv(pid, GL_LINK_STATUS)
    if not ok:
        log = glGetProgramInfoLog(pid).decode()
        raise RuntimeError(f"Program link failed: {log}")
    glDeleteShader(vs); glDeleteShader(fs)
    return pid

# --- model/world transforms -------------------------------------------------
def translate(tvec):
    """Return a 4x4 translation matrix for tvec (x,y,z)."""
    m = np.identity(4, dtype=np.float32)
    m[0, 3] = float(tvec[0])
    m[1, 3] = float(tvec[1])
    m[2, 3] = float(tvec[2])
    return m

def rotate_y(angle_rad):
    """Return a 4x4 rotation matrix around +Y axis."""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    m = np.identity(4, dtype=np.float32)
    m[0, 0] =  c; m[0, 2] = s
    m[2, 0] = -s; m[2, 2] = c
    return m
# ---- color mapping knobs (globals) -----------------------------------------
PRESSURE_GAIN = 0.75  # how strongly pressure boosts blue (0..1+)
MASS_GAIN     = 0.75  # how strongly total dissolved mass boosts red (0..1+)

BASE_R = 0.15         # base red component for every cell (0..1)
BASE_B = 0.25         # base blue component for every cell (0..1)
BASE_A = 0.35         # mesh alpha (0..1)

MESH_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uMVP;
void main(){
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""
MESH_FS = """
#version 330 core
out vec4 FragColor;
uniform vec4 uColor; // rgba
void main(){
    FragColor = uColor; // premult not needed; straight alpha ok with src-alpha blending
}
"""

# Instanced arrow shader ------------------------------------------------------
POINT_VS = """
#version 330 core
layout(location=0) in vec3 aBase;   // arrow mesh vertex (unit scale, +X axis)
layout(location=1) in vec3 aOffset; // per-instance position
layout(location=2) in vec3 aVec;    // per-instance velocity vector
layout(location=3) in float aScalar; // per-instance scalar (magnitude, curl, ...)
out float vScalar;
uniform mat4 uMVP;
uniform float uArrowScale;
void main(){
    float len = length(aVec);
    vec3 dir = (len > 1e-6) ? normalize(aVec) : vec3(1.0, 0.0, 0.0);
    float ang = atan(dir.y, dir.x);
    float c = cos(ang), s = sin(ang);
    mat2 rot = mat2(c, -s, s, c);
    vec3 base = aBase;
    base.x *= len * uArrowScale;            // scale by vector length (length along +X)
    vec2 xy = rot * base.xy;  // rotate into direction
    vec3 world = vec3(xy, base.z) + aOffset;
    vScalar = aScalar;
    gl_Position = uMVP * vec4(world, 1.0);
}
"""
POINT_FS = """
#version 330 core
in float vScalar;
out vec4 FragColor;
uniform vec4 uColor; // rgba (alpha used)
uniform float uTime;

vec3 hsv2rgb(vec3 c){
    vec3 rgb = clamp(abs(mod(c.x*6.0 + vec3(0.0,4.0,2.0),6.0)-3.0)-1.0,0.0,1.0);
    return c.z * mix(vec3(1.0), rgb, c.y);
}

void main(){
    float t = clamp(vScalar, 0.0, 1.0);
    vec3 col = hsv2rgb(vec3(0.7*(1.0 - t), 1.0, 1.0));
    float pulse = 0.5 + 0.5 * sin(uTime + t * 6.2831);
    FragColor = vec4(col * pulse, uColor.a);
}
"""

# Point sprite shader ---------------------------------------------------------
SPRITE_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uMVP;
uniform float uPointScale; // pixels
void main(){
    gl_Position = uMVP * vec4(aPos, 1.0);
    gl_PointSize = uPointScale;
}
"""
SPRITE_FS = """
#version 330 core
out vec4 FragColor;
uniform vec4 uColor; // rgba
uniform float uTime;
void main(){
    // circular sprite mask
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(uv, uv);
    if (r2 > 1.0) discard;
    float edge = smoothstep(1.0, 0.7, r2);
    float pulse = 0.5 + 0.5 * sin(uTime);
    FragColor = vec4(uColor.rgb * pulse, uColor.a * (1.0 - edge));
}
"""

# Sprite fragment shader with a simple specular highlight used for droplets.
DROPLET_FS = """
#version 330 core
out vec4 FragColor;
uniform vec4 uColor; // rgba
void main(){
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(uv, uv);
    if (r2 > 1.0) discard;
    float edge = smoothstep(1.0, 0.7, r2);
    float spec = pow(max(0.0, 1.0 - r2), 4.0);
    FragColor = vec4(uColor.rgb * (0.5 + 0.5 * spec), uColor.a * (1.0 - edge));
}
"""

class CellGL:
    def __init__(self, cell):
        self.cell = cell
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.ebo = glGenBuffers(1)
        self.count = int(cell.faces.shape[0] * 3)
        self._last_vbytes = 0
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        X32 = cell.X.astype(np.float32, copy=False)
        glBufferData(GL_ARRAY_BUFFER, X32.nbytes, X32, GL_DYNAMIC_DRAW)
        self._last_vbytes = int(X32.nbytes)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        I = cell.faces.astype(np.uint32, copy=False).ravel()
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, I.nbytes, I, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glBindVertexArray(0)
        # color: use your R (ionic), G (identity), B (pressure) idea — here pick per-cell green with alpha
        g = getattr(cell, "_identity_green", 128) / 255.0
        base = np.array([0.3, g, 0.45, 0.35], dtype=np.float32)  # semi-transparent
        self.color = base

    def upload(self):
        # Bind VAO so EBO binding applies to the right VAO
        glBindVertexArray(self.vao)
        # Update vertex buffer (positions)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        X32 = self.cell.X.astype(np.float32, copy=False)
        nbytes = int(X32.nbytes)
        if nbytes != self._last_vbytes:
            glBufferData(GL_ARRAY_BUFFER, nbytes, X32, GL_DYNAMIC_DRAW)
            self._last_vbytes = nbytes
        else:
            glBufferSubData(GL_ARRAY_BUFFER, 0, nbytes, X32)
        # Update element buffer (indices) unconditionally; topology may change
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        I = self.cell.faces.astype(np.uint32, copy=False).ravel()
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, I.nbytes, I, GL_STATIC_DRAW)
        glBindVertexArray(0)

    def draw(self, prog, u_mvp, u_color):
        glBindVertexArray(self.vao)
        glUniform4fv(u_color, 1, self.color)
        glDrawElements(GL_TRIANGLES, self.count, GL_UNSIGNED_INT, ctypes.c_void_p(0))
        glBindVertexArray(0)

def _rebuild_gl_cells(h):
    """Build GL wrappers for the current hierarchy cells."""
    return [CellGL(c) for c in getattr(h, 'cells', [])]

def _sync_gl_cells(gl_cells, h):
    """Ensure GL cells match the current hierarchy.

    If topology or count changed, rebuild GL objects. Otherwise, rebind cell
    references so we always upload the latest positions.
    Returns (gl_cells, rebuilt: bool)
    """
    cells = getattr(h, 'cells', [])
    if len(gl_cells) != len(cells):
        return _rebuild_gl_cells(h), True
    # Check topology consistency (faces shape)
    for i, (cg, c) in enumerate(zip(gl_cells, cells)):
        try:
            if cg.cell.faces.shape != c.faces.shape:
                return _rebuild_gl_cells(h), True
        except Exception:
            return _rebuild_gl_cells(h), True
    # Rebind to current cell objects
    for i, cg in enumerate(gl_cells):
        cg.cell = cells[i]
    return gl_cells, False

def gather_organelles(h):
    pts = []
    cols = []
    for c in h.cells:
        g = getattr(c, "_identity_green", 128) / 255.0
        color = np.array([0.9, g, 0.2, 0.5], dtype=np.float32)
        for o in c.organelles:
            pts.append([o.pos[0], o.pos[1], o.pos[2]])
            cols.append(color)
    if not pts:
        return None, None
    return np.array(pts, dtype=np.float32), np.array(cols, dtype=np.float32)

def gather_vertices(h):
    """Collect all vertex positions from all cells as a single Nx3 array (float32)."""
    cells = getattr(h, 'cells', [])
    if not cells:
        return None
    try:
        allX = np.concatenate([c.X for c in cells], axis=0).astype(np.float32)
        return allX
    except Exception:
        return None

class PointsGL:
    def __init__(self, pts, vecs=None, cols=None):
        self.n = len(pts)
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # Position buffer
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, pts.nbytes, pts, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))

        # Optional vector buffer
        self.vbo_vec = None
        if vecs is not None:
            self.vbo_vec = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vec)
            glBufferData(GL_ARRAY_BUFFER, vecs.nbytes, vecs, GL_DYNAMIC_DRAW)
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))

        glBindVertexArray(0)

        self.color = np.array([1,1,1,0.5], dtype=np.float32)  # fallback if no per-pt color
        self.cols = cols  # not used per-point to keep shader minimal

    def upload(self, pts, vecs=None):
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        # If size changed, reallocate buffer
        if len(pts) != self.n:
            glBufferData(GL_ARRAY_BUFFER, pts.nbytes, pts, GL_DYNAMIC_DRAW)
            self.n = len(pts)
        else:
            glBufferSubData(GL_ARRAY_BUFFER, 0, pts.nbytes, pts)

        if self.vbo_vec is not None and vecs is not None:
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vec)
            if len(vecs) != self.n:
                glBufferData(GL_ARRAY_BUFFER, vecs.nbytes, vecs, GL_DYNAMIC_DRAW)
            else:
                glBufferSubData(GL_ARRAY_BUFFER, 0, vecs.nbytes, vecs)

    def draw(self, prog, u_mvp, u_color, u_psize):
        glUniform4fv(u_color, 1, self.color)
        glUniform1f(u_psize, 6.0)  # pixels per point; bump if you want larger organelles
        glBindVertexArray(self.vao)
        glDrawArrays(GL_POINTS, 0, self.n)
        glBindVertexArray(0)

class ArrowsGL:
    """Instanced arrow renderer using per-instance velocity vectors."""

    def __init__(self, pts, vecs, scalars=None):
        self.n = len(pts)
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # Base arrow mesh oriented along +X consisting of two triangles
        arrow = np.array([
            [0.0, -0.02, 0.0],
            [0.0,  0.02, 0.0],
            [0.6,  0.0, 0.0],
            [0.6, -0.05, 0.0],
            [0.6,  0.05, 0.0],
            [1.0,  0.0, 0.0],
        ], dtype=np.float32)

        self.base_count = len(arrow)
        self.vbo_base = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_base)
        glBufferData(GL_ARRAY_BUFFER, arrow.nbytes, arrow, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))

        # Instance positions
        self.vbo_pos = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_pos)
        glBufferData(GL_ARRAY_BUFFER, pts.nbytes, pts, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glVertexAttribDivisor(1, 1)

        # Instance velocity vectors
        self.vbo_vec = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vec)
        glBufferData(GL_ARRAY_BUFFER, vecs.nbytes, vecs, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glVertexAttribDivisor(2, 1)

        # Optional scalar for color mapping (defaults to vector magnitude)
        if scalars is None:
            scalars = np.linalg.norm(vecs, axis=1)
        scalars = _normalize_vec(np.asarray(scalars, dtype=np.float32))
        self.vbo_scalar = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_scalar)
        glBufferData(GL_ARRAY_BUFFER, scalars.nbytes, scalars, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 4, ctypes.c_void_p(0))
        glVertexAttribDivisor(3, 1)

        glBindVertexArray(0)

        self.color = np.array([1, 1, 1, 1], dtype=np.float32)

    def upload(self, pts, vecs, scalars=None):
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_pos)
        if len(pts) != self.n:
            glBufferData(GL_ARRAY_BUFFER, pts.nbytes, pts, GL_DYNAMIC_DRAW)
            self.n = len(pts)
        else:
            glBufferSubData(GL_ARRAY_BUFFER, 0, pts.nbytes, pts)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vec)
        if len(vecs) != self.n:
            glBufferData(GL_ARRAY_BUFFER, vecs.nbytes, vecs, GL_DYNAMIC_DRAW)
        else:
            glBufferSubData(GL_ARRAY_BUFFER, 0, vecs.nbytes, vecs)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_scalar)
        if scalars is None:
            scalars = np.linalg.norm(vecs, axis=1)
        scalars = _normalize_vec(np.asarray(scalars, dtype=np.float32))
        if len(scalars) != self.n:
            glBufferData(GL_ARRAY_BUFFER, scalars.nbytes, scalars, GL_DYNAMIC_DRAW)
        else:
            glBufferSubData(GL_ARRAY_BUFFER, 0, scalars.nbytes, scalars)

    def draw(self, prog, u_mvp, u_color, _u_unused=None):
        glUniform4fv(u_color, 1, self.color)
        glBindVertexArray(self.vao)
        glDrawArraysInstanced(GL_TRIANGLES, 0, self.base_count, self.n)
        glBindVertexArray(0)

def compute_cells_center_of_mass(h):
    """Approximate a global center-of-mass from all cell meshes.
    We weight each cell's centroid by its vertex count to better reflect size.
    Fallback to simple average if anything looks odd.
    """
    try:
        total_w = 0.0
        acc = np.zeros(3, dtype=np.float32)
        for c in getattr(h, 'cells', []):
            X = getattr(c, 'X', None)
            if X is None or len(X) == 0:
                continue
            ctr = np.mean(X, axis=0).astype(np.float32)
            w = float(len(X))
            acc += ctr * w
            total_w += w
        if total_w > 0:
            return acc / total_w
    except Exception:
        pass
    # fallback origin
    return np.array([0.0, 0.0, 0.0], dtype=np.float32)

def main():
    # CLI: share numpy demo params and add any GL-specific knobs later
    parser = argparse.ArgumentParser(parents=[build_numpy_parser(add_help=False)], conflict_handler='resolve')
    parser.add_argument("--frames", type=int, default=0,
                        help="Frame cap (0 runs until window close)")
    parser.add_argument("--render", choices=["points", "mesh"], default="points",
                        help="Rendering mode: 'points' shows white vertex point cloud (default); 'mesh' shows shaded meshes.")
    parser.add_argument("--stream-npz", type=str, default="",
                        help="If provided, play a pre-rendered NPZ stream (points or mesh)")
    parser.add_argument("--flat2d", action="store_true",
                        help="Render in a fixed top-down 2D orthographic view")
    args = parser.parse_args()

    # When a fluid demo is requested, delegate to shared numpy handler
    if getattr(args, "fluid", ""):
        run_fluid_demo(args)
        return

    # If streaming mode, we still init pygame/GL, but won't step the sim.
    streaming = bool(getattr(args, "stream_npz", ""))
    if not streaming:
        # ---------- init sim ----------
        api, provider = make_cellsim_backend_from_args(args)
        # Prime mechanics so hierarchy exists before rendering (provider builds _h on first step)
        api.step(1e-3)
        h = getattr(provider, "_h", None)
        if h is None:
            print("Softbody mechanics provider did not initialize its hierarchy (_h) after a step.\n"
                  "This likely means the provider wasn't attached or the engine didn't call sync().")
            return

    # ---------- init pygame + GL ----------
    pygame.init()
    w, hwin = 1100, 800
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.set_mode((w, hwin), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("XPBD Softbody — pygame+OpenGL")

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_PROGRAM_POINT_SIZE)
#    glEnable(GL_CULL_FACE)
#    glCullFace(GL_BACK)

    mesh_prog = link_program(MESH_VS, MESH_FS)
    mesh_u_mvp = glGetUniformLocation(mesh_prog, "uMVP")
    mesh_u_color = glGetUniformLocation(mesh_prog, "uColor")

    pt_prog = link_program(SPRITE_VS, SPRITE_FS)
    pt_u_mvp = glGetUniformLocation(pt_prog, "uMVP")
    pt_u_color = glGetUniformLocation(pt_prog, "uColor")
    pt_u_psize = glGetUniformLocation(pt_prog, "uPointScale")
    pt_u_time = glGetUniformLocation(pt_prog, "uTime")

    # build GL objects depending on render mode or stream type
    gl_cells = []
    gl_pts = None           # organelles (mesh mode only)
    gl_vtx_pts = None       # vertex point cloud (points mode)
    stream = None
    if streaming:
        stream = np.load(args.stream_npz)
        stype = stream['stream_type'].item() if ('stream_type' in getattr(stream, 'files', [])) else None
        if stype not in ('opengl_points_v1', 'opengl_mesh_v1'):
            print(f"Unsupported stream type: {stype}")
            return
        # No GL buffers to allocate up-front for points; for mesh we still need VAOs/EBOs per cell topology
        if stype == 'opengl_mesh_v1':
            # Create dummy cells just to hold EBO topology and VBO size; we'll upload positions per-frame
            n_cells = int(stream['n_cells'])
            # We emulate CellGL using a minimal shim with faces array; we'll create one CellGL per cell with initial dummy X
            class _CellShim:
                pass
            vtx_counts = stream['vtx_counts'].astype(np.int32)
            faces_concat = stream['faces_concat'].astype(np.uint32)
            face_counts = stream['face_counts'].astype(np.int32)
            faces_per_cell = []
            off = 0
            for i in range(n_cells):
                nf = int(face_counts[i])
                F = faces_concat[off*3:(off+nf)*3].reshape(nf, 3)
                off += nf
                # build cell shim
                cs = _CellShim()
                cs.X = np.zeros((int(vtx_counts[i]), 3), dtype=np.float32)
                cs.faces = F
                gl_cells.append(CellGL(cs))
        else:
            gl_vtx_pts = PointsGL(np.zeros((1,3), dtype=np.float32))
            gl_vtx_pts.color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    else:
        if args.render == "mesh":
            gl_cells = _rebuild_gl_cells(h)
            pts, cols = gather_organelles(h)
            gl_pts = PointsGL(pts, cols=cols) if pts is not None else None
        else:
            vtx = gather_vertices(h)
            if vtx is not None:
                gl_vtx_pts = PointsGL(vtx)
                gl_vtx_pts.color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    # camera (initialized to look near the initial cell cluster)
    eye = np.array([0.5, 0.5, 1.7], dtype=np.float32)
    center = compute_cells_center_of_mass(h) if not streaming else np.array([0.0, 0.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    fovy = 45.0
    cam_dir = (center - eye)
    cam_dir = cam_dir / (np.linalg.norm(cam_dir) + 1e-12)
    cam_dist = float(np.linalg.norm(center - eye))
    center_s = center.copy()
    dist_s = cam_dist

    clock = pygame.time.Clock()
    running = True
    dt = float(getattr(args, "dt", 1e-3))
    t = 0.0  # simulation time (kept for sim; not used for rotation)
    t0 = time.perf_counter()  # wall-clock start for steady rotation

    frame = 0
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

        # step or stream
        if not streaming:
            dt = step_cellsim(api, dt)
            t += dt
            # Reacquire hierarchy each tick (provider may replace _h)
            h = getattr(provider, "_h", h)

        if not streaming:
            # Update GL resources per render mode
            if args.render == "mesh":
                # Keep GL cells bound to the latest hierarchy cells; rebuild if needed
                gl_cells, _ = _sync_gl_cells(gl_cells, h)
                # update meshes from XPBD positions
                for cg in gl_cells:
                    cg.upload()
                # organelles as points (optional)
                if gl_pts is not None:
                    pts, _ = gather_organelles(h)
                    if pts is None:
                        gl_pts = None
                    else:
                        if gl_pts.n != len(pts):
                            gl_pts = PointsGL(pts)
                        else:
                            gl_pts.upload(pts)
            else:
                # points mode: update vertex point cloud
                vtx = gather_vertices(h)
                if vtx is None:
                    gl_vtx_pts = None
                else:
                    if gl_vtx_pts is None:
                        gl_vtx_pts = PointsGL(vtx)
                        gl_vtx_pts.color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
                    elif gl_vtx_pts.n != len(vtx):
                        gl_vtx_pts = PointsGL(vtx)
                        gl_vtx_pts.color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
                    else:
                        gl_vtx_pts.upload(vtx)

        # view/proj and model (either from stream or computed)
        viewport = pygame.display.get_surface().get_size()
        aspect = viewport[0] / max(1, viewport[1])
        time_s = pygame.time.get_ticks() * 0.001
        if streaming:
            # Pull MVP from stream for current frame index
            fidx = frame
            if 'mvps' not in getattr(stream, 'files', []):
                print('Stream missing mvps')
                return
            MVP = stream['mvps'][fidx].astype(np.float32)
        else:
            if args.flat2d:
                allX = np.concatenate([c.X for c in h.cells], axis=0).astype(np.float32)
                bmin, bmax = allX.min(0), allX.max(0)
                margin = 0.1 * np.linalg.norm(bmax - bmin)
                left, right = bmin[0] - margin, bmax[0] + margin
                bottom, top = bmin[1] - margin, bmax[1] + margin
                P = ortho(left, right, bottom, top, -1.0, 1.0)
                MVP = P.astype(np.float32)
            else:
                # Auto-frame camera: fit to current geometry (and smooth it)
                new_center = compute_cells_center_of_mass(h)
                allX = np.concatenate([c.X for c in h.cells], axis=0).astype(np.float32)
                bmin, bmax = allX.min(0), allX.max(0)
                radius = 0.5 * np.linalg.norm(bmax - bmin)
                desired_dist = max(0.2, radius / math.tan(math.radians(fovy * 0.5)) * 2.0)
                alpha = 0.15  # smoothing factor
                center_s = (1.0 - alpha) * center_s + alpha * new_center
                dist_s   = (1.0 - alpha) * dist_s   + alpha * desired_dist
                center   = center_s
                cam_dist = float(dist_s)
                eye      = center - cam_dir * cam_dist

                # view/proj
                near = 0.05
                far  = max(10.0, cam_dist + 3.0*radius)
                P = perspective(fovy, aspect, near, far)
                V = look_at(eye, center, up)
                # scene/world rotation using wall-clock
                # Turn off rotation for lower-dimensional sims (1D/2D) to match NumPy demo behavior
                sim_dim = int(getattr(args, "sim_dim", 3))
                rot_speed = 0.25 if (not args.flat2d and sim_dim == 3) else 0.0
                t_wall = time.perf_counter() - t0
                theta = rot_speed * t_wall
                T_neg = translate(-center)
                R_y  = rotate_y(theta)
                T_pos = translate(center)
                M = (T_pos @ R_y @ T_neg).astype(np.float32)
                MVP = (P @ V @ M).astype(np.float32)

        # clear
        glViewport(0, 0, viewport[0], viewport[1])
        glClearColor(0.06, 0.07, 0.10, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if (streaming and stype == 'opengl_mesh_v1') or (not streaming and args.render == "mesh"):
            glUseProgram(mesh_prog)
            glUniformMatrix4fv(mesh_u_mvp, 1, GL_FALSE, MVP.T.flatten())

            if streaming:
                # Upload per-frame vertices and colors; draw in precomputed order
                fidx = frame
                vtx_concat = stream['vtx_concat'][fidx]
                draw_order = stream['draw_order'][fidx]
                colors = stream['colors'][fidx]
                vtx_counts = stream['vtx_counts']
                vtx_offsets = stream['vtx_offsets']
                for draw_i in draw_order:
                    i = int(draw_i)
                    cg = gl_cells[i]
                    off = int(vtx_offsets[i]); n = int(vtx_counts[i])
                    cg.cell.X = vtx_concat[off:off+n, :]
                    cg.color = colors[i]
                    cg.upload()
                    cg.draw(mesh_prog, mesh_u_mvp, mesh_u_color)
            else:
                # back-to-front sort by view depth (centroid)
                view_dir = (center - eye); view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-12)
                depths = []
                for cg in gl_cells:
                    c = cg.cell
                    ctr = np.mean(c.X, axis=0).astype(np.float32)
                    ctr_h = np.array([ctr[0], ctr[1], ctr[2], 1.0], dtype=np.float32)
                    ctr_rot = (M @ ctr_h)[:3]
                    d = np.dot(ctr_rot - eye, view_dir)
                    depths.append((d, cg))
                depths.sort(key=lambda x: x[0])

                pressures, masses, greens255 = _measure_pressure_mass(h, api)
                pN = _normalize_vec(pressures)
                mN = _normalize_vec(masses)

                glDepthMask(GL_FALSE)
                for _, cg in depths:
                    i = h.cells.index(cg.cell)
                    G = float(greens255[i]) / 255.0
                    R = min(1.0, BASE_R + MASS_GAIN * float(mN[i]))
                    B = min(1.0, BASE_B + PRESSURE_GAIN * float(pN[i]))
                    cg.color = np.array([R, G, B, BASE_A], dtype=np.float32)
                    cg.draw(mesh_prog, mesh_u_mvp, mesh_u_color)
                glDepthMask(GL_TRUE)

                if gl_pts is not None and gl_pts.n > 0:
                    glUseProgram(pt_prog)
                    glUniformMatrix4fv(pt_u_mvp, 1, GL_FALSE, MVP.T.flatten())
                    glUniform1f(pt_u_time, time_s)
                    gl_pts.draw(pt_prog, pt_u_mvp, pt_u_color, pt_u_psize)
        else:
            # points mode
            glUseProgram(pt_prog)
            glUniformMatrix4fv(pt_u_mvp, 1, GL_FALSE, MVP.T.flatten())
            glUniform1f(pt_u_time, time_s)
            if streaming:
                fidx = frame
                pts_offsets = stream['pts_offsets']
                pts_concat = stream['pts_concat']
                start = int(pts_offsets[fidx]); end = int(pts_offsets[fidx+1])
                pts = pts_concat[start:end]
                if gl_vtx_pts is None or gl_vtx_pts.n != len(pts):
                    gl_vtx_pts = PointsGL(pts)
                    gl_vtx_pts.color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
                else:
                    gl_vtx_pts.upload(pts)
                gl_vtx_pts.draw(pt_prog, pt_u_mvp, pt_u_color, pt_u_psize)
            else:
                if gl_vtx_pts is not None and gl_vtx_pts.n > 0:
                    gl_vtx_pts.draw(pt_prog, pt_u_mvp, pt_u_color, pt_u_psize)

        pygame.display.flip()
        clock.tick(60)
        frame += 1
        if getattr(args, "frames", 0) and frame >= args.frames:
            break

    pygame.quit()


def play_points_stream_from_dir(
    dir_path: str,
    *,
    viewport_w: int = 1100,
    viewport_h: int = 800,
    loop_mode: str = "none",
    fps: float = 60.0,
) -> None:
    """Render a disk-backed OpenGL point stream.

    Frames are read as ``pts_XXXXXX.npy`` and ``mvp_XXXXXX.npy`` in ``dir_path``.
    A file named ``done`` signals no further frames will arrive.
    """
    import os
    import time

    pygame = _ensure_gl_context(viewport_w, viewport_h)
    from OpenGL.GL import (
        glUseProgram,
        glUniformMatrix4fv,
        glUniform4fv,
        glUniform1f,
        glViewport,
        glClearColor,
        glClear,
        GL_COLOR_BUFFER_BIT,
        GL_DEPTH_BUFFER_BIT,
        GL_FALSE,
    )
    from OpenGL.GL import glGetUniformLocation

    prog = link_program(SPRITE_VS, SPRITE_FS)
    u_mvp = glGetUniformLocation(prog, "uMVP")
    u_color = glGetUniformLocation(prog, "uColor")
    u_psize = glGetUniformLocation(prog, "uPointScale")
    u_time = glGetUniformLocation(prog, "uTime")

    gl_pts = PointsGL(np.zeros((0, 3), dtype=np.float32))
    clock = pygame.time.Clock()
    frame = 0
    direction = 1
    while True:
        pts_path = os.path.join(dir_path, f"pts_{frame:06d}.npy")
        mvp_path = os.path.join(dir_path, f"mvp_{frame:06d}.npy")
        while not (os.path.exists(pts_path) and os.path.exists(mvp_path)):
            if os.path.exists(os.path.join(dir_path, "done")):
                break
            pygame.event.pump()
            time.sleep(0.05)
        if not (os.path.exists(pts_path) and os.path.exists(mvp_path)):
            if loop_mode == "loop":
                frame = 0
                direction = 1
                continue
            if loop_mode == "bounce":
                direction = -direction
                frame += direction
                if frame < 0:
                    frame = 0
                    direction = 1
                continue
            break

        for e in pygame.event.get():
            if e.type == pygame.QUIT or (
                e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_q)
            ):
                pygame.quit()
                return

        pts = np.load(pts_path).astype(np.float32, copy=False)
        mvp = np.load(mvp_path).astype(np.float32, copy=False)

        glUseProgram(prog)
        glUniformMatrix4fv(u_mvp, 1, GL_FALSE, mvp.T.flatten())
        glUniform1f(u_time, pygame.time.get_ticks() * 0.001)
        gl_pts.upload(pts)

        viewport = pygame.display.get_surface().get_size()
        glViewport(0, 0, viewport[0], viewport[1])
        glClearColor(0.06, 0.07, 0.10, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        gl_pts.draw(prog, u_mvp, u_color, u_psize)
        pygame.display.flip()
        clock.tick(fps)
        frame += direction

    pygame.quit()


def play_points_stream(
    pts_offsets: np.ndarray | None = None,
    pts_concat: np.ndarray | None = None,
    mvps: np.ndarray | None = None,
    vec_concat: np.ndarray | None = None,
    scalar_fields: Dict[str, np.ndarray] | None = None,
    droplet_offsets: np.ndarray | None = None,
    droplet_concat: np.ndarray | None = None,
    *,
    gather_func=None,
    step_func=None,
    frames: int = 0,
    dt: float = 0.0,
    fovy: float = 60.0,
    rot_speed: float = 0.0,
    show_vectors: bool = False,
    show_droplets: bool = False,
    initial_metric: str | None = "speed",
    arrow_scale: float = 1.0,
    flow_anim_speed: float = 1.0,
    viewport_w: int = 1100,
    viewport_h: int = 800,
    loop_mode: str = "none",
    fps: float = 60.0,
) -> None:
    """Render OpenGL points either from arrays or directly from a live simulator.

    When ``gather_func`` and ``step_func`` are provided, this function streams
    frames on-the-fly by calling ``gather_func`` to obtain the current point
    positions and ``step_func`` to advance the simulation.  In this mode the
    ``frames``, ``dt``, ``fovy`` and ``rot_speed`` parameters control playback.
    Otherwise, precomputed arrays ``pts_offsets``, ``pts_concat`` and ``mvps``
    are used along with optional per-vertex vectors and scalar fields.
    """

    pygame = _ensure_gl_context(viewport_w, viewport_h)
    from OpenGL.GL import (
        glUseProgram,
        glUniformMatrix4fv,
        glUniform4fv,
        glUniform1f,
        glViewport,
        glClearColor,
        glClear,
        GL_COLOR_BUFFER_BIT,
        GL_DEPTH_BUFFER_BIT,
        GL_FALSE,
    )
    from OpenGL.GL import glGetUniformLocation

    pt_prog = link_program(SPRITE_VS, SPRITE_FS)
    pt_u_mvp = glGetUniformLocation(pt_prog, "uMVP")
    pt_u_color = glGetUniformLocation(pt_prog, "uColor")
    pt_u_psize = glGetUniformLocation(pt_prog, "uPointScale")
    pt_u_time = glGetUniformLocation(pt_prog, "uTime")

    drop_prog = link_program(SPRITE_VS, DROPLET_FS)
    drop_u_mvp = glGetUniformLocation(drop_prog, "uMVP")
    drop_u_color = glGetUniformLocation(drop_prog, "uColor")
    drop_u_psize = glGetUniformLocation(drop_prog, "uPointScale")

    arrow_prog = link_program(POINT_VS, POINT_FS)
    arrow_u_mvp = glGetUniformLocation(arrow_prog, "uMVP")
    arrow_u_color = glGetUniformLocation(arrow_prog, "uColor")
    arrow_u_time = glGetUniformLocation(arrow_prog, "uTime")
    arrow_u_scale = glGetUniformLocation(arrow_prog, "uArrowScale")

    live = gather_func is not None and step_func is not None

    clock = pygame.time.Clock()

    if live:
        # --- initialize first frame and camera ---
        try:
            pts, vecs, drops = gather_func()
        except ValueError:
            pts, vecs = gather_func()
            drops = None

        center, radius = _compute_center_radius_pts(pts)
        eye = np.array([0.5, 0.5, 1.7], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        cam_dir = center - eye
        cam_dir = cam_dir / (np.linalg.norm(cam_dir) + 1e-12)
        cam_dist = float(np.linalg.norm(center - eye))
        center_s = center.copy()
        dist_s = cam_dist
        aspect = viewport_w / max(1, viewport_h)

        gl_pts = None
        gl_drops = None
        use_arrows = False
        t_sim = 0.0
        frame = 0
        running = True
        while running and (frames <= 0 or frame < frames):
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

            new_center, radius = _compute_center_radius_pts(pts)
            desired_dist = max(0.2, radius / math.tan(math.radians(fovy * 0.5)) * 2.0)
            alpha = 0.15
            center_s = (1.0 - alpha) * center_s + alpha * new_center
            dist_s = (1.0 - alpha) * dist_s + alpha * desired_dist
            center = center_s
            cam_dist = float(dist_s)
            eye = center - cam_dir * cam_dist

            P = _perspective(fovy, aspect, 0.05, max(10.0, cam_dist + 3.0 * radius))
            V = _look_at(eye, center, up)
            theta = rot_speed * t_sim
            T_neg = _translate(-center)
            R_y = _rotate_y(theta)
            T_pos = _translate(center)
            M = (T_pos @ R_y @ T_neg).astype(np.float32)
            MVP = (P @ V @ M).astype(np.float32)

            if show_vectors and vecs is not None:
                scalars = np.linalg.norm(vecs, axis=1).astype(np.float32, copy=False)
                if gl_pts is None or gl_pts.n != len(pts):
                    gl_pts = ArrowsGL(
                        pts.astype(np.float32, copy=False),
                        vecs.astype(np.float32, copy=False),
                        scalars,
                    )
                    gl_pts.color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
                else:
                    gl_pts.upload(
                        pts.astype(np.float32, copy=False),
                        vecs.astype(np.float32, copy=False),
                        scalars,
                    )
                use_arrows = True
            else:
                if gl_pts is None or gl_pts.n != len(pts):
                    gl_pts = PointsGL(pts.astype(np.float32, copy=False))
                    gl_pts.color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
                else:
                    gl_pts.upload(pts.astype(np.float32, copy=False))
                use_arrows = False

            if show_droplets and drops is not None:
                if gl_drops is None or gl_drops.n != len(drops):
                    gl_drops = PointsGL(drops.astype(np.float32, copy=False))
                    gl_drops.color = np.array([0.8, 0.9, 1.0, 0.9], dtype=np.float32)
                else:
                    gl_drops.upload(drops.astype(np.float32, copy=False))
            elif gl_drops is not None:
                gl_drops = None

            viewport = pygame.display.get_surface().get_size()
            glViewport(0, 0, viewport[0], viewport[1])
            glClearColor(0.06, 0.07, 0.10, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            if use_arrows:
                glUseProgram(arrow_prog)
                glUniformMatrix4fv(arrow_u_mvp, 1, GL_FALSE, MVP.T.flatten())
                glUniform1f(arrow_u_time, t_sim * flow_anim_speed)
                glUniform1f(arrow_u_scale, arrow_scale)
                gl_pts.draw(arrow_prog, arrow_u_mvp, arrow_u_color, None)
            else:
                glUseProgram(pt_prog)
                glUniformMatrix4fv(pt_u_mvp, 1, GL_FALSE, MVP.T.flatten())
                glUniform1f(pt_u_time, t_sim * flow_anim_speed)
                gl_pts.draw(pt_prog, pt_u_mvp, pt_u_color, pt_u_psize)

            if gl_drops is not None:
                glUseProgram(drop_prog)
                glUniformMatrix4fv(drop_u_mvp, 1, GL_FALSE, MVP.T.flatten())
                gl_drops.draw(drop_prog, drop_u_mvp, drop_u_color, drop_u_psize)

            pygame.display.flip()
            clock.tick(fps)

            dt = step_func(dt)
            t_sim += dt
            frame += 1
            try:
                pts, vecs, drops = gather_func()
            except ValueError:
                pts, vecs = gather_func()
                drops = None

        pygame.quit()
        return

    # --- Precomputed playback path ---
    if pts_offsets is None or pts_concat is None or mvps is None:
        raise ValueError("precomputed playback requires pts_offsets, pts_concat and mvps")

    pts = pts_concat[: max(1, int(pts_offsets[1] - pts_offsets[0]))]
    vecs = (
        vec_concat[: max(1, int(pts_offsets[1] - pts_offsets[0]))]
        if vec_concat is not None
        else None
    )

    metric_names = list(scalar_fields.keys()) if scalar_fields else []
    if initial_metric == "none":
        current_metric = None
    else:
        current_metric = (
            initial_metric
            if initial_metric in metric_names
            else (metric_names[0] if metric_names else None)
        )
    if metric_names:
        menu = "Scalar fields:" + ", ".join(f" {i+1}:{n}" for i, n in enumerate(metric_names)) + " (0:none)"
        print(menu)
    scalar_concat = scalar_fields[current_metric] if current_metric and scalar_fields else None
    scalars = (
        scalar_concat[: max(1, int(pts_offsets[1] - pts_offsets[0]))]
        if scalar_concat is not None
        else None
    )
    if show_vectors and vecs is not None:
        if scalars is None:
            scalars = np.linalg.norm(vecs, axis=1)
        gl_pts = ArrowsGL(
            pts.astype(np.float32, copy=False),
            vecs.astype(np.float32, copy=False),
            scalars.astype(np.float32, copy=False),
        )
        use_arrows = True
    else:
        gl_pts = PointsGL(pts.astype(np.float32, copy=False))
        use_arrows = False
    gl_pts.color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    if show_droplets and droplet_offsets is not None and droplet_concat is not None:
        dstart = int(droplet_offsets[0])
        dend = int(droplet_offsets[1]) if len(droplet_offsets) > 1 else dstart
        dpts = droplet_concat[dstart:dend]
        gl_drops = PointsGL(dpts.astype(np.float32, copy=False))
        gl_drops.color = np.array([0.8, 0.9, 1.0, 0.9], dtype=np.float32)
    else:
        gl_drops = None

    running = True
    frame = 0
    F = int(mvps.shape[0])
    direction = 1
    while running:
        time_s = pygame.time.get_ticks() * 0.001 * flow_anim_speed
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
            elif e.type == pygame.KEYDOWN and metric_names:
                if pygame.K_0 <= e.key <= pygame.K_9:
                    idx = e.key - pygame.K_0
                    if idx == 0:
                        current_metric = None
                    elif 1 <= idx <= len(metric_names):
                        current_metric = metric_names[idx - 1]
                    scalar_concat = scalar_fields[current_metric] if current_metric else None

        fidx = int(frame)
        if fidx >= F:
            if loop_mode == "loop":
                fidx = 0
                frame = 0
            elif loop_mode == "bounce":
                direction = -1
                frame = F - 1
                fidx = frame
            else:
                break
        elif fidx < 0 and loop_mode == "bounce":
            direction = 1
            frame = 0
            fidx = 0

        start = int(pts_offsets[fidx])
        end = int(pts_offsets[fidx + 1])
        cur = pts_concat[start:end]
        if use_arrows:
            cur_vecs = vec_concat[start:end]
            cur_scalars = (
                scalar_concat[start:end]
                if scalar_concat is not None
                else np.linalg.norm(cur_vecs, axis=1)
            )
            if gl_pts.n != len(cur):
                gl_pts = ArrowsGL(
                    cur.astype(np.float32, copy=False),
                    cur_vecs.astype(np.float32, copy=False),
                    cur_scalars.astype(np.float32, copy=False),
                )
                gl_pts.color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
            else:
                gl_pts.upload(
                    cur.astype(np.float32, copy=False),
                    cur_vecs.astype(np.float32, copy=False),
                    cur_scalars.astype(np.float32, copy=False),
                )
        else:
            if gl_pts.n != len(cur):
                gl_pts = PointsGL(cur.astype(np.float32, copy=False))
                gl_pts.color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
            else:
                gl_pts.upload(cur.astype(np.float32, copy=False))

        if gl_drops is not None and droplet_offsets is not None and droplet_concat is not None:
            dstart = int(droplet_offsets[fidx])
            dend = int(droplet_offsets[fidx + 1])
            dcur = droplet_concat[dstart:dend]
            if gl_drops.n != len(dcur):
                gl_drops = PointsGL(dcur.astype(np.float32, copy=False))
                gl_drops.color = np.array([0.8, 0.9, 1.0, 0.9], dtype=np.float32)
            else:
                gl_drops.upload(dcur.astype(np.float32, copy=False))

        MVP = mvps[fidx].astype(np.float32, copy=False)

        viewport = pygame.display.get_surface().get_size()
        glViewport(0, 0, viewport[0], viewport[1])
        glClearColor(0.06, 0.07, 0.10, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if use_arrows:
            glUseProgram(arrow_prog)
            glUniformMatrix4fv(arrow_u_mvp, 1, GL_FALSE, MVP.T.flatten())
            glUniform1f(arrow_u_time, time_s)
            glUniform1f(arrow_u_scale, arrow_scale)
            gl_pts.draw(arrow_prog, arrow_u_mvp, arrow_u_color, None)
        else:
            glUseProgram(pt_prog)
            glUniformMatrix4fv(pt_u_mvp, 1, GL_FALSE, MVP.T.flatten())
            glUniform1f(pt_u_time, time_s)
            gl_pts.draw(pt_prog, pt_u_mvp, pt_u_color, pt_u_psize)

        if gl_drops is not None:
            glUseProgram(drop_prog)
            glUniformMatrix4fv(drop_u_mvp, 1, GL_FALSE, MVP.T.flatten())
            gl_drops.draw(drop_prog, drop_u_mvp, drop_u_color, drop_u_psize)

        pygame.display.flip()
        clock.tick(fps)
        frame += direction

    pygame.quit()

def play_mesh_stream_from_dir(
    dir_path: str,
    *,
    viewport_w: int = 1100,
    viewport_h: int = 800,
    loop_mode: str = "none",
    fps: float = 60.0,
) -> None:
    """Render a disk-backed OpenGL mesh stream.

    Directory must contain ``vtx_counts.npy``, ``faces_concat.npy``,
    ``face_counts.npy`` and ``vtx_offsets.npy`` along with per-frame
    ``vtx_XXXXXX.npy``, ``colors_XXXXXX.npy``, ``draw_XXXXXX.npy`` and
    ``mvp_XXXXXX.npy`` files.  A ``done`` file signals completion.
    """
    import os
    import time

    pygame = _ensure_gl_context(viewport_w, viewport_h)
    from OpenGL.GL import (
        glUseProgram,
        glUniformMatrix4fv,
        glUniform4fv,
        glViewport,
        glClearColor,
        glClear,
        GL_COLOR_BUFFER_BIT,
        GL_DEPTH_BUFFER_BIT,
        GL_FALSE,
        glDepthMask,
    )
    from OpenGL.GL import glGetUniformLocation

    vtx_counts = np.load(os.path.join(dir_path, "vtx_counts.npy")).astype(np.int32)
    faces_concat = np.load(os.path.join(dir_path, "faces_concat.npy")).astype(np.uint32)
    face_counts = np.load(os.path.join(dir_path, "face_counts.npy")).astype(np.int32)
    vtx_offsets = np.load(os.path.join(dir_path, "vtx_offsets.npy")).astype(np.int32)
    n_cells = int(len(vtx_counts))

    mesh_prog = link_program(MESH_VS, MESH_FS)
    mesh_u_mvp = glGetUniformLocation(mesh_prog, "uMVP")
    mesh_u_color = glGetUniformLocation(mesh_prog, "uColor")

    class _CellShim:
        pass

    gl_cells: list[CellGL] = []
    f_off = 0
    for i in range(n_cells):
        nf = int(face_counts[i])
        F = faces_concat[f_off * 3 : (f_off + nf) * 3].reshape(nf, 3)
        f_off += nf
        cs = _CellShim()
        cs.X = np.zeros((int(vtx_counts[i]), 3), dtype=np.float32)
        cs.faces = F.astype(np.uint32, copy=False)
        gl_cells.append(CellGL(cs))

    clock = pygame.time.Clock()
    frame = 0
    direction = 1
    while True:
        vtx_path = os.path.join(dir_path, f"vtx_{frame:06d}.npy")
        col_path = os.path.join(dir_path, f"colors_{frame:06d}.npy")
        draw_path = os.path.join(dir_path, f"draw_{frame:06d}.npy")
        mvp_path = os.path.join(dir_path, f"mvp_{frame:06d}.npy")
        while not all(os.path.exists(p) for p in (vtx_path, col_path, draw_path, mvp_path)):
            if os.path.exists(os.path.join(dir_path, "done")):
                break
            pygame.event.pump()
            time.sleep(0.05)
        if not all(os.path.exists(p) for p in (vtx_path, col_path, draw_path, mvp_path)):
            if loop_mode == "loop":
                frame = 0
                direction = 1
                continue
            if loop_mode == "bounce":
                direction = -direction
                frame += direction
                if frame < 0:
                    frame = 0
                    direction = 1
                continue
            break

        for e in pygame.event.get():
            if e.type == pygame.QUIT or (
                e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_q)
            ):
                pygame.quit()
                return

        vtx_concat = np.load(vtx_path).astype(np.float32, copy=False)
        cols = np.load(col_path).astype(np.float32, copy=False)
        order = np.load(draw_path).astype(np.int32, copy=False)
        MVP = np.load(mvp_path).astype(np.float32, copy=False)

        viewport = pygame.display.get_surface().get_size()
        glViewport(0, 0, viewport[0], viewport[1])
        glClearColor(0.06, 0.07, 0.10, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(mesh_prog)
        glUniformMatrix4fv(mesh_u_mvp, 1, GL_FALSE, MVP.T.flatten())
        glDepthMask(GL_TRUE)
        for draw_i in order:
            i = int(draw_i)
            off = int(vtx_offsets[i])
            n = int(vtx_counts[i])
            gl_cells[i].cell.X = vtx_concat[off : off + n, :]
            gl_cells[i].color = cols[i, :]
            gl_cells[i].upload()
            gl_cells[i].draw(mesh_prog, mesh_u_mvp, mesh_u_color)

        pygame.display.flip()
        clock.tick(fps)
        frame += direction

    pygame.quit()

if __name__ == "__main__":
    main()

# ---- In-process streaming API ----------------------------------------------
# These helpers allow another module (like the NumPy demo) to feed frames
# directly as NumPy arrays without writing/reading NPZ files or using sockets.

def _ensure_gl_context(width: int = 1100, height: int = 800):
    try:
        import pygame
        from pygame.locals import DOUBLEBUF, OPENGL
    except Exception:
        raise
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.set_mode((int(width), int(height)), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("XPBD Softbody — OpenGL Stream")
    return pygame


# ---- NumPy tag bridge -------------------------------------------------------
def numpy_tag_names() -> list[str]:
    """Expose the shared NumPy CLI tag names for external callers."""
    return get_numpy_tag_names()


def numpy_kwargs_from_args(args: dict | object) -> dict:
    """Extract only NumPy-relevant kwargs from a larger args dict/namespace."""
    if isinstance(args, dict):
        return extract_numpy_kwargs(args)
    # generic object/namespace
    d = {k: getattr(args, k) for k in numpy_tag_names() if hasattr(args, k)}
    return d


def play_mesh_stream(*,
                     n_cells: int,
                     vtx_counts: np.ndarray,
                     vtx_offsets: np.ndarray,
                     faces_concat: np.ndarray,
                     face_counts: np.ndarray,
                     vtx_concat: np.ndarray,
                     colors: np.ndarray,
                     draw_order: np.ndarray,
                     mvps: np.ndarray,
                     viewport_w: int = 1100,
                     viewport_h: int = 800,
                     loop_mode: str = "none",
                     fps: float = 60.0):
    """Render a precomputed OpenGL mesh stream in-process.

    Arguments mirror the NPZ keys used in streaming mode, but are NumPy arrays.
    loop_mode: 'none' | 'loop' | 'bounce'.
    """
    pygame = _ensure_gl_context(viewport_w, viewport_h)
    from OpenGL.GL import (
        glUseProgram, glUniformMatrix4fv, glUniform4fv, glViewport,
        glClearColor, glClear, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_FALSE,
        glDepthMask
    )
    from OpenGL.GL import glGetUniformLocation

    mesh_prog = link_program(MESH_VS, MESH_FS)
    mesh_u_mvp = glGetUniformLocation(mesh_prog, "uMVP")
    mesh_u_color = glGetUniformLocation(mesh_prog, "uColor")

    # Build GL wrappers (CellGL) per cell topology once
    class _CellShim:
        pass
    gl_cells = []
    off = 0
    # Build faces per cell
    f_off = 0
    faces_per_cell = []
    for i in range(int(n_cells)):
        nf = int(face_counts[i])
        F = faces_concat[f_off*3:(f_off+nf)*3].reshape(nf, 3)
        f_off += nf
        cs = _CellShim()
        cs.X = np.zeros((int(vtx_counts[i]), 3), dtype=np.float32)
        cs.faces = F.astype(np.uint32, copy=False)
        gl_cells.append(CellGL(cs))

    clock = pygame.time.Clock()
    running = True
    frame = 0
    F = int(mvps.shape[0])
    direction = 1
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False

        fidx = int(frame)
        if fidx >= F:
            if loop_mode == "loop":
                fidx = 0; frame = 0
            elif loop_mode == "bounce":
                direction = -1; frame = F-1; fidx = frame
            else:
                running = False; break
        elif fidx < 0 and loop_mode == "bounce":
            direction = 1; frame = 0; fidx = 0

        MVP = mvps[fidx].astype(np.float32, copy=False)

        viewport = pygame.display.get_surface().get_size()
        glViewport(0, 0, viewport[0], viewport[1])
        glClearColor(0.06, 0.07, 0.10, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(mesh_prog)
        glUniformMatrix4fv(mesh_u_mvp, 1, GL_FALSE, MVP.T.flatten())

        # Draw in precomputed order
        order = draw_order[fidx].astype(np.int32, copy=False)
        for draw_i in order:
            i = int(draw_i)
            off = int(vtx_offsets[i]); n = int(vtx_counts[i])
            gl_cells[i].cell.X = vtx_concat[fidx, off:off+n, :].astype(np.float32, copy=False)
            gl_cells[i].color = colors[fidx, i, :].astype(np.float32, copy=False)
            gl_cells[i].upload()
            gl_cells[i].draw(mesh_prog, mesh_u_mvp, mesh_u_color)

        pygame.display.flip()
        clock.tick(fps)
        frame += direction

    pygame.quit()

