import os, sys, time, math, random
import argparse
import ctypes
import numpy as np

# ----- Cellsim backend (uses your code) -------------------------------------
from .run_numpy_demo import (
    make_cellsim_backend as base_make_cellsim_backend,
    step_cellsim,
    build_numpy_parser,
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
        glEnableVertexAttribArray, glVertexAttribPointer,
        # uniforms / draw
        glGetUniformLocation, glUniform4fv, glUniform1f, glUniformMatrix4fv,
        glDrawElements, glDrawArrays, glUseProgram,
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
POINT_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uMVP;
uniform float uPointScale; // pixels
void main(){
    gl_Position = uMVP * vec4(aPos, 1.0);
    gl_PointSize = uPointScale;
}
"""
POINT_FS = """
#version 330 core
out vec4 FragColor;
uniform vec4 uColor; // rgba
void main(){
    // circular sprite mask
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(uv, uv);
    if (r2 > 1.0) discard;
    float edge = smoothstep(1.0, 0.7, r2);
    FragColor = vec4(uColor.rgb, uColor.a * (1.0 - edge));
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
    def __init__(self, pts, cols):
        self.n = len(pts)
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, pts.nbytes, pts, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glBindVertexArray(0)
        self.color = np.array([1,1,1,0.5], dtype=np.float32)  # fallback if no per-pt color
        self.cols = cols  # not used per-point to keep shader minimal

    def upload(self, pts):
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        # If size changed, reallocate buffer
        if len(pts) != self.n:
            glBufferData(GL_ARRAY_BUFFER, pts.nbytes, pts, GL_DYNAMIC_DRAW)
            self.n = len(pts)
        else:
            glBufferSubData(GL_ARRAY_BUFFER, 0, pts.nbytes, pts)

    def draw(self, prog, u_mvp, u_color, u_psize):
        glUniform4fv(u_color, 1, self.color)
        glUniform1f(u_psize, 6.0)  # pixels per point; bump if you want larger organelles
        glBindVertexArray(self.vao)
        glDrawArrays(GL_POINTS, 0, self.n)
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

    pt_prog = link_program(POINT_VS, POINT_FS)
    pt_u_mvp = glGetUniformLocation(pt_prog, "uMVP")
    pt_u_color = glGetUniformLocation(pt_prog, "uColor")
    pt_u_psize = glGetUniformLocation(pt_prog, "uPointScale")

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
            gl_vtx_pts = PointsGL(np.zeros((1,3), dtype=np.float32), None)
            gl_vtx_pts.color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    else:
        if args.render == "mesh":
            gl_cells = _rebuild_gl_cells(h)
            pts, cols = gather_organelles(h)
            gl_pts = PointsGL(pts, cols) if pts is not None else None
        else:
            vtx = gather_vertices(h)
            if vtx is not None:
                gl_vtx_pts = PointsGL(vtx, None)
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
                            gl_pts = PointsGL(pts, None)
                        else:
                            gl_pts.upload(pts)
            else:
                # points mode: update vertex point cloud
                vtx = gather_vertices(h)
                if vtx is None:
                    gl_vtx_pts = None
                else:
                    if gl_vtx_pts is None:
                        gl_vtx_pts = PointsGL(vtx, None)
                        gl_vtx_pts.color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
                    elif gl_vtx_pts.n != len(vtx):
                        gl_vtx_pts = PointsGL(vtx, None)
                        gl_vtx_pts.color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
                    else:
                        gl_vtx_pts.upload(vtx)

        # view/proj and model (either from stream or computed)
        viewport = pygame.display.get_surface().get_size()
        aspect = viewport[0] / max(1, viewport[1])
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
                    gl_pts.draw(pt_prog, pt_u_mvp, pt_u_color, pt_u_psize)
        else:
            # points mode
            glUseProgram(pt_prog)
            glUniformMatrix4fv(pt_u_mvp, 1, GL_FALSE, MVP.T.flatten())
            if streaming:
                fidx = frame
                pts_offsets = stream['pts_offsets']
                pts_concat = stream['pts_concat']
                start = int(pts_offsets[fidx]); end = int(pts_offsets[fidx+1])
                pts = pts_concat[start:end]
                if gl_vtx_pts is None or gl_vtx_pts.n != len(pts):
                    gl_vtx_pts = PointsGL(pts, None)
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

def play_points_stream(pts_offsets: np.ndarray,
                       pts_concat: np.ndarray,
                       mvps: np.ndarray,
                       vec_concat: np.ndarray | None = None,
                       *,
                       viewport_w: int = 1100,
                       viewport_h: int = 800,
                       loop_mode: str = "none",
                       fps: float = 60.0):
    """Render a precomputed OpenGL points stream in-process.

    pts_offsets: (F+1,) int64
    pts_concat:  (N,3) float32
    mvps:        (F,4,4) float32
    vec_concat:  optional (N,3) float32 per-vertex vectors
    loop_mode:   'none' | 'loop' | 'bounce'
    """
    pygame = _ensure_gl_context(viewport_w, viewport_h)
    from OpenGL.GL import (
        glUseProgram, glUniformMatrix4fv, glUniform4fv, glUniform1f, glViewport,
        glClearColor, glClear, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_FALSE
    )
    from OpenGL.GL import glGetUniformLocation

    # Reuse existing shader and point VAO logic
    mesh_prog = link_program(MESH_VS, MESH_FS)
    pt_prog = link_program(POINT_VS, POINT_FS)
    pt_u_mvp = glGetUniformLocation(pt_prog, "uMVP")
    pt_u_color = glGetUniformLocation(pt_prog, "uColor")
    pt_u_psize = glGetUniformLocation(pt_prog, "uPointScale")

    # Set up a single reusable point cloud object
    pts = pts_concat[: max(1, int(pts_offsets[1]-pts_offsets[0]))]
    gl_pts = PointsGL(pts.astype(np.float32, copy=False), None)
    gl_pts.color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

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

        # Frame indices with loop/bounce
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

        # Upload per-frame points
        start = int(pts_offsets[fidx]); end = int(pts_offsets[fidx+1])
        cur = pts_concat[start:end]
        if gl_pts.n != len(cur):
            gl_pts = PointsGL(cur.astype(np.float32, copy=False), None)
            gl_pts.color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        else:
            gl_pts.upload(cur.astype(np.float32, copy=False))

        MVP = mvps[fidx].astype(np.float32, copy=False)

        viewport = pygame.display.get_surface().get_size()
        glViewport(0, 0, viewport[0], viewport[1])
        glClearColor(0.06, 0.07, 0.10, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(pt_prog)
        glUniformMatrix4fv(pt_u_mvp, 1, GL_FALSE, MVP.T.flatten())
        gl_pts.draw(pt_prog, pt_u_mvp, pt_u_color, pt_u_psize)

        pygame.display.flip()
        clock.tick(fps)
        frame += direction

    pygame.quit()

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

