import os, sys, time, math, random
import ctypes
import numpy as np

# ----- Cellsim backend (uses your code) -------------------------------------
from .run_numpy_demo import make_cellsim_backend as base_make_cellsim_backend, step_cellsim


def make_cellsim_backend():
    api, provider = base_make_cellsim_backend(
        cell_vols=[0.8, 0.8, 0.8],
        cell_imps=[600.0 + 80 * i for i in range(3)],
        cell_elastic_k=[0.2, 0.2, 0.2],
        bath_na=5.0,
        bath_cl=5.0,
        bath_pressure=0.0,
        bath_volume_factor=3.0,
        substeps=5,
        dt_provider=0.005,
    )

    # color identity (rendering only)
    levels = [64, 144, 208]
    for i, c in enumerate(api.cells):
        setattr(c, "_identity_green", levels[i % len(levels)])

    return api, provider

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
        self.count = cell.faces.shape[0] * 3
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        X32 = cell.X.astype(np.float32, copy=False)
        glBufferData(GL_ARRAY_BUFFER, X32.nbytes, X32, GL_DYNAMIC_DRAW)
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

    def upload_vertices(self):
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        X32 = self.cell.X.astype(np.float32, copy=False)
        glBufferSubData(GL_ARRAY_BUFFER, 0, X32.nbytes, X32)

    def draw(self, prog, u_mvp, u_color):
        glBindVertexArray(self.vao)
        glUniform4fv(u_color, 1, self.color)
        glDrawElements(GL_TRIANGLES, self.count, GL_UNSIGNED_INT, ctypes.c_void_p(0))
        glBindVertexArray(0)

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
    # ---------- init sim ----------
    api, provider = make_cellsim_backend()
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
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)

    mesh_prog = link_program(MESH_VS, MESH_FS)
    mesh_u_mvp = glGetUniformLocation(mesh_prog, "uMVP")
    mesh_u_color = glGetUniformLocation(mesh_prog, "uColor")

    pt_prog = link_program(POINT_VS, POINT_FS)
    pt_u_mvp = glGetUniformLocation(pt_prog, "uMVP")
    pt_u_color = glGetUniformLocation(pt_prog, "uColor")
    pt_u_psize = glGetUniformLocation(pt_prog, "uPointScale")

    # build GL objects per cell
    gl_cells = [CellGL(c) for c in h.cells]
    pts, cols = gather_organelles(h)
    gl_pts = PointsGL(pts, cols) if pts is not None else None

    # camera (initialized to look near the initial cell cluster)
    eye = np.array([0.5, 0.5, 1.7], dtype=np.float32)
    center = compute_cells_center_of_mass(h)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    fovy = 45.0
    # lock initial view direction; we’ll track COM and adapt distance
    cam_dir = (center - eye)
    cam_dir = cam_dir / (np.linalg.norm(cam_dir) + 1e-12)
    cam_dist = float(np.linalg.norm(center - eye))
    center_s = center.copy()  # smoothed center
    dist_s = cam_dist         # smoothed distance

    clock = pygame.time.Clock()
    running = True
    dt = 1e-3
    t = 0.0

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

        # step cellsim (keep dt sane)
        dt = min(max(step_cellsim(api, dt), 1e-4), 5e-2)
        t += dt

        # update buffers from XPBD positions
        for cg in gl_cells:
            cg.upload_vertices()
        if gl_pts is not None:
            pts, _ = gather_organelles(h)
            if pts is not None:
                gl_pts.upload(pts)

        # Reacquire hierarchy each tick (provider may replace _h)
        h = getattr(provider, "_h", h)

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
        viewport = pygame.display.get_surface().get_size()
        aspect = viewport[0] / max(1, viewport[1])
        near = 0.05
        far  = max(10.0, cam_dist + 3.0*radius)
        P = perspective(fovy, aspect, near, far)
        V = look_at(eye, center, up)
        MVP = (P @ V).astype(np.float32)

        # clear
        glViewport(0, 0, viewport[0], viewport[1])
        glClearColor(0.06, 0.07, 0.10, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # rough back-to-front sort by view depth (centroid)
        view_dir = (center - eye); view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-12)
        depths = []
        for cg in gl_cells:
            c = cg.cell
            ctr = np.mean(c.X, axis=0).astype(np.float32)
            d = np.dot(ctr - eye, view_dir)
            depths.append((d, cg))
        # back (more negative d) to front for correct blending
        depths.sort(key=lambda x: x[0])  # ascending

        # draw semi-transparent meshes
        glUseProgram(mesh_prog)
        # IMPORTANT: NumPy is row-major; GLSL expects column-major.
        # Send the transpose so the shader sees the correct transform.
        glUniformMatrix4fv(mesh_u_mvp, 1, GL_FALSE, MVP.T.flatten())
        glDepthMask(GL_FALSE)  # do not write depth when blending
        for _, cg in depths:
            cg.draw(mesh_prog, mesh_u_mvp, mesh_u_color)
        glDepthMask(GL_TRUE)

        # draw organelles (points)
        if gl_pts is not None and gl_pts.n > 0:
            glUseProgram(pt_prog)
            glUniformMatrix4fv(pt_u_mvp, 1, GL_FALSE, MVP.T.flatten())
            gl_pts.draw(pt_prog, pt_u_mvp, pt_u_color, pt_u_psize)

        pygame.display.flip()
        clock.tick(60)  # ~60 FPS cap

    pygame.quit()

if __name__ == "__main__":
    main()
