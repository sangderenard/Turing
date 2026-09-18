"""Watch the raincloud through the repository's own volumetric path.

Plays back a recording from chamber_raincloud_record.py in a pygame
OpenGL window, drawn by spectral-analyzer's standard Phong shader
(csrc/shaders/base_material) -- unmodified -- using the field-volume
channel it already has:

    radiance field (nz, ny, nx, 3)  ->  GL_TEXTURE_3D
        ->  BaseGLRenderer.set_field_volume_texture(tex, gain)
        ->  base_material.frag.glsl:  col += texture(uFieldVolume, vPosObj).rgb * uFieldGain

That last line samples the field AT A SURFACE, at the fragment's own
object-space position, and adds it as light.  So the column of air is
drawn as geometry: a stack of thin quads authored directly in [0,1]^3
(which is exactly ``vPosObj``, and therefore exactly the texture's UVW
space), placed in the room by the model matrix.  Every slice fragment
picks up the field at its own position; the sum over slices is the
line integral.  Texture slicing, as it was done before raymarching, on
a shader that was built to be fed a field.

WHAT THE SIM PUBLISHES AND HOW EACH CHANNEL IS SHOWN
----------------------------------------------------
Channels are looked up by the semantic the laws DECLARE
(``LAW_PUBLICATIONS``, carried in the recording), never by guessing.

  cloud_water_content  LWC   ->  extinction (3 LWC / 2 rho_w r_eff), white Mie
  ice_water_content    IWC   ->  extinction, blue-white
  rain_water_content   RWC   ->  extinction at raindrop radius (thin for its mass)
  saturation_ratio     S     ->  faint haze where S -> 1 (the cloud's edge)
  temperature          T     ->  cold tint toward the plate
  dew_film_thickness   h_film   -> the plate's material turns wet: darker, glossier
  frost_thickness      h_frost  -> the plate's material turns frosted: whiter, matte
  wetted_area          A_wet    -> the pool disc's radius
  pool_depth           h_pool   -> the pool's colour deepens

The lighting INSIDE the field is computed here, per frame, because the
Phong shader has no per-fragment extinction (its alpha is a per-material
constant): a Beer-Lambert sweep down the column gives every cell the
light that reaches it, so the cloud's underside is dark and the shaft is
dim.  The room's lights are the shader's own emitter rig and light the
plate, floor and pool the ordinary way.

Slices are composited ADDITIVELY.  The renderer's own blend is
src-alpha, which would turn ninety-six slices of EMPTY column into a
dark haze; additive means nothing-in-the-cell adds nothing, at the
price that a thick cloud brightens rather than occludes.  For a column
in a dark room that is the right trade, and it is stated here rather
than hidden.

    python examples/chamber_raincloud_view.py run.npz [--speed 2] [--gain 1]
        [--snapshot out.png --snapshot-at 30] [--exit-after 40]
    space: pause    left/right: scrub    drag: orbit    wheel: zoom    r: restart
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_SPECTRAL_DIR = HERE.parent.parent / "spectral-analyzer"
if str(_SPECTRAL_DIR) not in sys.path:
    sys.path.insert(0, str(_SPECTRAL_DIR))

import base_gl_renderer as _base_gl_renderer          # noqa: E402
from base_gl_renderer import BaseGLRenderer           # noqa: E402
from OpenGL.GL import (                               # noqa: E402
    GL_ARRAY_BUFFER, GL_STATIC_DRAW, GL_FLOAT, GL_INT, GL_FALSE, GL_ONE,
    GL_TEXTURE_3D, GL_RGB32F, GL_RGB, GL_LINEAR, GL_CLAMP_TO_EDGE,
    GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_WRAP_R,
    GL_DEPTH_TEST, GL_LEQUAL, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_RGBA, GL_UNSIGNED_BYTE,
    glGenVertexArrays, glBindVertexArray, glGenBuffers, glBindBuffer, glBufferData,
    glDeleteVertexArrays, glDeleteBuffers,
    glEnableVertexAttribArray, glVertexAttribPointer, glVertexAttribIPointer,
    glGenTextures, glBindTexture, glTexImage3D, glTexSubImage3D, glTexParameteri,
    glEnable, glDepthFunc, glClearColor, glClear, glViewport, glReadPixels,
)

# -- display optics: real numbers, so the gain is a gain and not a lie --------
RHO_W = 1000.0
R_EFF_CLOUD = 7.0e-6          # m; the droplet the recorder's m_unit is sized for
R_RAIN = 5.0e-4               # m; the recorder's m_unit_r
CLOUD_RGB = np.array([0.96, 0.96, 0.98])
ICE_RGB = np.array([0.86, 0.93, 1.00])
RAIN_RGB = np.array([0.62, 0.68, 0.78])
LIGHT_FROM_ABOVE = np.array([1.0, 0.97, 0.92])
# The ambient term stands in for the multiple scattering a single-scatter
# reduction cannot do -- which is what makes a real cloud white -- so it is
# near-neutral with only a faint blue bias, not a sky colour.  A blue fill
# outweighs the sideways-scattered sun (HG(g=0.62) at 90 deg is ~0.03) and
# tints the whole cloud blue.
SKY_FILL = np.array([0.30, 0.31, 0.34])
G_HG = 0.62


# -- the recording ---------------------------------------------------------------

class Recording:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.meta = json.loads(str(z["__meta__"]))
        self.arrays = {k: z[k] for k in z.files if k != "__meta__"}
        self.nx, self.ny, self.nz = self.meta["shape"]
        self.dx = float(self.meta["dx"])
        self.n_frames = int(self.meta["n_frames"])
        self.t = self.arrays["t"].reshape(-1)
        # semantic -> output name, from the laws' own declaration
        self.by_semantic = {v["semantic"]: k for k, v in self.meta["publications"].items()}

    def channel(self, semantic: str, frame: int, default=None):
        name = self.by_semantic.get(semantic)
        if name is None or name not in self.arrays:
            return default
        return self.arrays[name][frame]

    def field(self, name: str, frame: int):
        """A voxel column (flat x + nx*(y + ny*z)) as (nz, ny, nx)."""
        return self.arrays[name][frame].reshape(self.nz, self.ny, self.nx)

    def frame_at(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.t, t), 0, self.n_frames - 1))


# -- the optical reduction: published state -> radiance field -------------------

def radiance_field(rec: Recording, frame: int) -> np.ndarray:
    """(depth=ny, height=nz, width=nx, 3) float32 -- the GL texture order,
    with the sim's vertical z mapped to the texture's (and the room's) y."""
    lwc = rec.field(rec.by_semantic.get("cloud_water_content", "LWC"), frame)
    iwc = rec.field(rec.by_semantic.get("ice_water_content", "IWC"), frame) if "IWC" in rec.arrays else 0.0 * lwc
    rwc = rec.field(rec.by_semantic.get("rain_water_content", "RWC"), frame) if "RWC" in rec.arrays else 0.0 * lwc
    s = rec.field(rec.by_semantic.get("saturation_ratio", "S"), frame) if "S" in rec.arrays else None

    beta_c = 1.5 * np.maximum(lwc, 0.0) / (RHO_W * R_EFF_CLOUD)
    beta_i = 1.5 * np.maximum(iwc, 0.0) / (RHO_W * R_EFF_CLOUD)
    beta_r = 1.5 * np.maximum(rwc, 0.0) / (RHO_W * R_RAIN)
    beta = beta_c + beta_i + beta_r

    # light enters from the top of the column (the plate side is the lit
    # side of the room); sweep down, one slab at a time -- Beer-Lambert
    trans = np.ones_like(beta)
    carried = np.ones((rec.ny, rec.nx))
    for k in range(rec.nz - 1, -1, -1):
        trans[k] = carried
        carried = carried * np.exp(-beta[k] * rec.dx)

    # a viewer looking horizontally at light coming from above: cos ~ 0
    g2 = G_HG * G_HG
    phase = (1.0 - g2) / (4.0 * math.pi * (1.0 + g2) ** 1.5)

    colour = (beta_c[..., None] * CLOUD_RGB + beta_i[..., None] * ICE_RGB + beta_r[..., None] * RAIN_RGB)
    direct = colour * (trans[..., None] * LIGHT_FROM_ABOVE) * phase
    ambient = colour * SKY_FILL * (0.25 / math.pi)
    out = 0.99 * (direct + ambient)
    if s is not None:
        # the cloud's edge: near-saturated but not yet condensed air scatters faintly
        haze = np.clip((s - 0.97) / 0.03, 0.0, 1.0) * 0.004
        out = out + haze[..., None] * SKY_FILL
    # (nz, ny, nx, 3) -> (ny, nz, nx, 3): sim z becomes texture height (room y)
    return np.ascontiguousarray(np.transpose(out, (1, 0, 2, 3)), dtype=np.float32)


# -- materials: the only contract BaseGLRenderer needs ---------------------------

class RoomMaterials:
    """Rows: 0 slice (black, additive), 1 floor, 2 plate, 3 pool, 4 backdrop."""

    def __init__(self):
        self.rows = [
            dict(albedo=(0.0, 0.0, 0.0), rough=1.0, opacity=1.0, ambient=0.0, spec=0.0, shin=1.0),
            dict(albedo=(0.34, 0.34, 0.36), rough=0.9, opacity=1.0, ambient=0.12, spec=0.05, shin=6.0),
            dict(albedo=(0.70, 0.74, 0.80), rough=0.35, opacity=1.0, ambient=0.08, spec=0.35, shin=48.0),
            dict(albedo=(0.06, 0.08, 0.11), rough=0.10, opacity=1.0, ambient=0.05, spec=0.60, shin=120.0),
            dict(albedo=(0.12, 0.12, 0.13), rough=1.0, opacity=1.0, ambient=0.08, spec=0.0, shin=1.0),
            # 5: a raindrop streak -- a representative, not a medium
            dict(albedo=(0.78, 0.83, 0.92), rough=0.2, opacity=0.65, ambient=0.10, spec=0.55, shin=60.0),
        ]
        self._cache = None

    def set(self, row: int, **kw):
        if any(self.rows[row].get(k) != v for k, v in kw.items()):
            self.rows[row].update(kw)
            self._cache = None

    def build_tensors(self) -> dict:
        if self._cache is not None:
            return self._cache
        n = len(self.rows)
        pbr = np.zeros((n, 16), np.float32)
        phong = np.zeros((n, 8), np.float32)
        enamel = np.zeros((n, 8), np.float32)
        tex = np.zeros((n, 16), np.float32)
        for i, r in enumerate(self.rows):
            a = r["albedo"]
            pbr[i, 0:3] = a; pbr[i, 3] = r["rough"]; pbr[i, 6] = 1.5; pbr[i, 7] = r["opacity"]
            phong[i, 0] = r["ambient"]; phong[i, 1] = r["spec"]; phong[i, 2] = r["shin"]; phong[i, 4:7] = a
            tex[i, 0:4] = -1.0            # no UV layers: emit/color/depth/remit all skipped
            tex[i, 10] = 1.0              # direct_lobe_power (unused, profile 0)
        self._cache = {"pbr": pbr, "phong_compat": phong, "enamel": enamel, "texture_stack": tex,
                       "chunk_strides": {"pbr": 16, "phong": 8}}
        return self._cache


# -- geometry: triangle soup in the shader's VAO layout ----------------------------

class Mesh:
    def __init__(self, pos: np.ndarray, nrm: np.ndarray, mat_id: int):
        pos = np.ascontiguousarray(pos, np.float32).reshape(-1, 3)
        nrm = np.ascontiguousarray(nrm, np.float32).reshape(-1, 3)
        n = pos.shape[0]
        rows = np.ascontiguousarray(np.concatenate([pos, nrm, np.zeros((n, 2), np.float32)], axis=1))
        self.n = n
        self.vao = int(glGenVertexArrays(1))
        bufs = [int(b) for b in glGenBuffers(5)]
        self.bufs = bufs
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, bufs[0])
        glBufferData(GL_ARRAY_BUFFER, rows.nbytes, rows, GL_STATIC_DRAW)
        stride = rows.shape[1] * 4
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1); glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(3); glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))
        ints = (np.full(n, mat_id, np.int32), np.zeros(n, np.int32), np.ones(n, np.int32))
        for buf, loc, vals in zip(bufs[1:4], (2, 4, 5), ints):
            glBindBuffer(GL_ARRAY_BUFFER, buf)
            glBufferData(GL_ARRAY_BUFFER, vals.nbytes, vals, GL_STATIC_DRAW)
            glEnableVertexAttribArray(loc); glVertexAttribIPointer(loc, 1, GL_INT, 4, ctypes.c_void_p(0))
        vel = np.zeros((n, 3), np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, bufs[4])
        glBufferData(GL_ARRAY_BUFFER, vel.nbytes, vel, GL_STATIC_DRAW)
        glEnableVertexAttribArray(6); glVertexAttribPointer(6, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glBindVertexArray(0)

    def delete(self):
        if self.vao:
            glDeleteVertexArrays(1, [self.vao]); glDeleteBuffers(len(self.bufs), self.bufs)
            self.vao = 0


def quad(p0, p1, p2, p3, normal):
    pos = np.array([p0, p1, p2, p0, p2, p3], np.float32)
    nrm = np.tile(np.asarray(normal, np.float32), (6, 1))
    return pos, nrm


def slice_stack(axis: int, n_slices: int):
    """n quads perpendicular to `axis`, spanning [0,1]^3 -- the field's UVW space."""
    pos, nrm = [], []
    for i in range(n_slices):
        u = (i + 0.5) / n_slices
        pts = []
        for a, b in ((0, 0), (1, 0), (1, 1), (0, 1)):
            p = [0.0, 0.0, 0.0]
            others = [k for k in range(3) if k != axis]
            p[axis] = u; p[others[0]] = a; p[others[1]] = b
            pts.append(p)
        normal = [0.0, 0.0, 0.0]; normal[axis] = 1.0
        q_pos, q_nrm = quad(*pts, normal)
        pos.append(q_pos); nrm.append(q_nrm)
    return np.concatenate(pos), np.concatenate(nrm)


def disc(radius: float, y: float, segments: int = 48):
    pos, nrm = [], []
    for i in range(segments):
        a0 = 2 * math.pi * i / segments; a1 = 2 * math.pi * (i + 1) / segments
        pos.append([[0, y, 0], [radius * math.cos(a0), y, radius * math.sin(a0)],
                    [radius * math.cos(a1), y, radius * math.sin(a1)]])
        nrm.append([[0, 1, 0]] * 3)
    return np.array(pos, np.float32).reshape(-1, 3), np.array(nrm, np.float32).reshape(-1, 3)


# -- matrices, ordinary column-vector convention (transposed at the GL call) -----

def look_at(eye, target, up):
    f = target - eye; f = f / np.linalg.norm(f)
    s = np.cross(f, up); s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s; m[1, :3] = u; m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye); m[1, 3] = -np.dot(u, eye); m[2, 3] = np.dot(f, eye)
    return m


def perspective(fov_y, aspect, near, far):
    f = 1.0 / math.tan(fov_y / 2.0)
    m = np.zeros((4, 4), np.float32)
    m[0, 0] = f / aspect; m[1, 1] = f
    m[2, 2] = (far + near) / (near - far); m[2, 3] = 2 * far * near / (near - far); m[3, 2] = -1.0
    return m


def model(scale, translate):
    m = np.eye(4, dtype=np.float32)
    m[0, 0], m[1, 1], m[2, 2] = scale
    m[:3, 3] = translate
    return m


# -- the renderer: base_material, with additive blending for the slice pass -------

class _AdditiveRenderer(BaseGLRenderer):
    """draw_mesh(..., additive=True) swaps the renderer's src-alpha blend for
    ONE/ONE during the super call -- the same swap-during-super idiom
    engine_gl_view uses for _read_glsl.  The shader is untouched."""

    def draw_mesh(self, *args, additive: bool = False, **kwargs):
        if not additive:
            return super().draw_mesh(*args, **kwargs)
        original = _base_gl_renderer.glBlendFunc
        _base_gl_renderer.glBlendFunc = lambda _s, _d: original(GL_ONE, GL_ONE)
        try:
            return super().draw_mesh(*args, **kwargs)
        finally:
            _base_gl_renderer.glBlendFunc = original


class FieldTexture:
    def __init__(self, ny: int, nz: int, nx: int):
        self.tex = int(glGenTextures(1))
        self.shape = (ny, nz, nx)
        glBindTexture(GL_TEXTURE_3D, self.tex)
        for p, v in ((GL_TEXTURE_MIN_FILTER, GL_LINEAR), (GL_TEXTURE_MAG_FILTER, GL_LINEAR),
                     (GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE), (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE),
                     (GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)):
            glTexParameteri(GL_TEXTURE_3D, p, v)
        zeros = np.zeros((ny, nz, nx, 3), np.float32)
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB32F, nx, nz, ny, 0, GL_RGB, GL_FLOAT, zeros)
        glBindTexture(GL_TEXTURE_3D, 0)

    def upload(self, rgb_dhw3: np.ndarray):
        arr = np.ascontiguousarray(rgb_dhw3, np.float32)
        d, h, w = arr.shape[:3]
        glBindTexture(GL_TEXTURE_3D, self.tex)
        glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, w, h, d, GL_RGB, GL_FLOAT, arr)
        glBindTexture(GL_TEXTURE_3D, 0)


class RainRepresentatives:
    """Persistent representative raindrops, driven by what the sim publishes.

    Rain is invisible as a MEDIUM by physics: 1e-4 kg/m^3 of 0.5 mm drops
    has almost no cross-section, so the field texture rightly shows nothing.
    What you see when it rains is drops.  So each cell's published rain
    water content is turned into a COUNT of representatives -- RWC * V /
    m_drop, rounded stochastically to whole drops (the Planck floor of the
    scale seam, applied on the display side) -- kept alive between frames,
    advected at the published fall speed, and drawn as motion-blurred
    streaks.  Persistent rather than resampled per frame, because resampled
    rain reads as noise and falling rain reads as rain.
    """

    def __init__(self, rec: Recording, w_r: float = 4.0, m_drop: float = 5.2e-7, seed: int = 7):
        self.rec, self.w_r, self.m_drop = rec, w_r, m_drop
        self.rng = np.random.default_rng(seed)
        self.pos = np.zeros((0, 3), np.float32)      # room coords: x, y(up), z
        self.mesh: Mesh | None = None

    def update(self, frame: int, dt_weather: float, col_w: float, col_h: float, col_d: float):
        rec, dx = self.rec, self.rec.dx
        rwc = rec.field(rec.by_semantic.get("rain_water_content", "RWC"), frame) if "RWC" in rec.arrays else None
        if rwc is None:
            return
        # advect what exists; whatever reaches the floor has landed
        self.pos[:, 1] -= self.w_r * dt_weather
        self.pos = self.pos[self.pos[:, 1] > 0.0]
        # per-cell target count, stochastically rounded to whole drops
        want = np.floor(rwc * dx ** 3 / self.m_drop + self.rng.random(rwc.shape)).astype(int)
        # per-cell current count from positions (cell index: z vertical)
        have = np.zeros_like(want)
        if len(self.pos):
            ix = np.clip(((self.pos[:, 0] + col_w / 2) / dx).astype(int), 0, rec.nx - 1)
            iy = np.clip(((self.pos[:, 2] + col_d / 2) / dx).astype(int), 0, rec.ny - 1)
            iz = np.clip((self.pos[:, 1] / dx).astype(int), 0, rec.nz - 1)
            np.add.at(have, (iz, iy, ix), 1)
        deficit = want - have
        spawn = []
        for iz, iy, ix in zip(*np.nonzero(deficit > 0)):
            k = int(deficit[iz, iy, ix])
            u = self.rng.random((k, 3))
            spawn.append(np.stack([(ix + u[:, 0]) * dx - col_w / 2, (iz + u[:, 1]) * dx, (iy + u[:, 2]) * dx - col_d / 2], axis=1))
        if spawn:
            self.pos = np.concatenate([self.pos, np.concatenate(spawn).astype(np.float32)])
        # surplus (a cell that rained out): drop a random subset from that cell
        for iz, iy, ix in zip(*np.nonzero(deficit < 0)):
            k = int(-deficit[iz, iy, ix])
            ix_p = np.clip(((self.pos[:, 0] + col_w / 2) / dx).astype(int), 0, rec.nx - 1)
            iy_p = np.clip(((self.pos[:, 2] + col_d / 2) / dx).astype(int), 0, rec.ny - 1)
            iz_p = np.clip((self.pos[:, 1] / dx).astype(int), 0, rec.nz - 1)
            idx = np.nonzero((ix_p == ix) & (iy_p == iy) & (iz_p == iz))[0]
            if len(idx):
                self.pos = np.delete(self.pos, self.rng.choice(idx, min(k, len(idx)), replace=False), axis=0)

    def build_mesh(self, view_dir: np.ndarray, frame_dt: float = 1.0 / 60.0, width: float = 0.003):
        """Camera-facing streaks: each drop swept over one frame of its fall."""
        if self.mesh is not None:
            self.mesh.delete(); self.mesh = None
        n = len(self.pos)
        if n == 0:
            return None
        vd = view_dir / max(np.linalg.norm(view_dir), 1e-9)
        right = np.cross(vd, np.array([0.0, 1.0, 0.0])); right = right / max(np.linalg.norm(right), 1e-9)
        length = self.w_r * frame_dt
        p = self.pos
        a = p - right * (width / 2); b = p + right * (width / 2)
        top = np.array([0.0, length, 0.0], np.float32)
        pos = np.stack([a, b, b + top, a, b + top, a + top], axis=1).reshape(-1, 3)
        nrm = np.tile((-vd).astype(np.float32), (pos.shape[0], 1))
        self.mesh = Mesh(pos, nrm, 5)
        return self.mesh


# -- the window --------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("recording", nargs="?", default=str(HERE / "chamber_raincloud_run.npz"))
    ap.add_argument("--speed", type=float, default=2.0, help="seconds of weather per wall second")
    ap.add_argument("--gain", type=float, default=1.0, help="multiplier on the field's light")
    ap.add_argument("--slices", type=int, default=96)
    ap.add_argument("--size", type=int, nargs=2, default=(960, 720))
    ap.add_argument("--snapshot", type=str, default=None)
    ap.add_argument("--snapshot-at", type=float, default=None, help="weather time to snapshot")
    ap.add_argument("--exit-after", type=float, default=None, help="wall seconds")
    args = ap.parse_args(argv)

    rec = Recording(args.recording)
    nx, ny, nz, dx = rec.nx, rec.ny, rec.nz, rec.dx
    col_w, col_h, col_d = nx * dx, nz * dx, ny * dx          # room metres: x, y(up), z

    import pygame
    from pygame.locals import OPENGL, DOUBLEBUF, QUIT, KEYDOWN, K_SPACE, K_LEFT, K_RIGHT, K_r, K_ESCAPE, \
        MOUSEBUTTONDOWN, MOUSEBUTTONUP, MOUSEMOTION, MOUSEWHEEL
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    W, H = args.size
    pygame.display.set_mode((W, H), OPENGL | DOUBLEBUF)
    pygame.display.set_caption("raincloud -- chamber_raincloud_demo through base_material")

    materials = RoomMaterials()
    renderer = _AdditiveRenderer(materials)
    renderer.init_gl()
    field_tex = FieldTexture(ny, nz, nx)
    # the field is a source term per metre; each slice stands for spacing_m of
    # path, so the shader's gain is the spacing (times the user's multiplier)
    spacing_m = max(col_w, col_h, col_d) / args.slices
    renderer.set_field_volume_texture(field_tex.tex, spacing_m * args.gain)

    stacks = [Mesh(*slice_stack(a, args.slices), 0) for a in range(3)]
    # THE ROOM IS THE CHAMBER: floor and plate are the footprint, the walls
    # are its four sides.  Everything you can see is simulated air.
    hw, hd = col_w / 2, col_d / 2
    floor = Mesh(*quad([-hw, 0, -hd], [hw, 0, -hd], [hw, 0, hd], [-hw, 0, hd], [0, 1, 0]), 1)
    plate = Mesh(*quad([-hw, 0, -hd], [hw, 0, -hd], [hw, 0, hd], [-hw, 0, hd], [0, -1, 0]), 2)
    # (mesh, inward normal, a point on the wall): the two walls facing the
    # camera are skipped each frame so the box is always open toward you
    walls = [
        (Mesh(*quad([-hw, 0, -hd], [hw, 0, -hd], [hw, col_h, -hd], [-hw, col_h, -hd], [0, 0, 1]), 4), np.array([0, 0, 1.0]), np.array([0, 0, -hd])),
        (Mesh(*quad([-hw, 0, hd], [hw, 0, hd], [hw, col_h, hd], [-hw, col_h, hd], [0, 0, -1]), 4), np.array([0, 0, -1.0]), np.array([0, 0, hd])),
        (Mesh(*quad([-hw, 0, -hd], [-hw, 0, hd], [-hw, col_h, hd], [-hw, col_h, -hd], [1, 0, 0]), 4), np.array([1.0, 0, 0]), np.array([-hw, 0, 0])),
        (Mesh(*quad([hw, 0, -hd], [hw, 0, hd], [hw, col_h, hd], [hw, col_h, -hd], [-1, 0, 0]), 4), np.array([-1.0, 0, 0]), np.array([hw, 0, 0])),
    ]
    pool_mesh = {"r": -1.0, "mesh": None}
    rain = RainRepresentatives(rec)

    ident = np.eye(4, dtype=np.float32)
    m_column = model((col_w, col_h, col_d), (-col_w / 2, 0.0, -col_d / 2))
    m_plate = model((1, 1, 1), (0.0, col_h + 0.005, 0.0))
    fov = math.radians(34.0)
    centre = np.array([0.0, col_h * 0.5, 0.0], np.float32)

    size = max(col_w, col_h, col_d)
    az, el, dist = 0.55, 0.36, size * 2.1
    dragging = False
    t_weather, playing = 0.0, True
    snapshot_done = args.snapshot is None
    t_wall0 = time.time(); last = t_wall0
    clock = pygame.time.Clock()

    def gl(m):
        return np.ascontiguousarray(m.T, dtype=np.float32)

    while True:
        for ev in pygame.event.get():
            if ev.type == QUIT or (ev.type == KEYDOWN and ev.key == K_ESCAPE):
                pygame.quit(); return
            if ev.type == KEYDOWN:
                if ev.key == K_SPACE: playing = not playing
                elif ev.key == K_LEFT: t_weather = max(0.0, t_weather - 2.0)
                elif ev.key == K_RIGHT: t_weather = min(rec.t[-1], t_weather + 2.0)
                elif ev.key == K_r: t_weather = 0.0
            if ev.type == MOUSEBUTTONDOWN and ev.button == 1: dragging = True
            if ev.type == MOUSEBUTTONUP and ev.button == 1: dragging = False
            if ev.type == MOUSEMOTION and dragging:
                az += ev.rel[0] * 0.008; el = float(np.clip(el - ev.rel[1] * 0.008, -1.2, 1.2))
            if ev.type == MOUSEWHEEL: dist = float(np.clip(dist * (0.9 if ev.y > 0 else 1.1), size * 0.6, size * 8))

        now = time.time(); dt_wall = now - last; last = now
        if playing:
            t_weather += dt_wall * args.speed
            if t_weather > rec.t[-1]:
                t_weather = 0.0
        frame = rec.frame_at(t_weather)

        # -- the field and the surface materials for this frame --
        field_tex.upload(radiance_field(rec, frame))
        # surface0 is the ceiling plate (build() lists it first); the recorder
        # keys surface channels by index, not by semantic, since there are two
        h_film = float(np.max(rec.arrays["surface0.h_film"][frame])) if "surface0.h_film" in rec.arrays else 0.0
        h_frost = float(np.max(rec.arrays["surface0.h_frost"][frame])) if "surface0.h_frost" in rec.arrays else 0.0
        wet = float(np.clip(h_film / 2e-4, 0, 1)); frost = float(np.clip(h_frost / 5e-4, 0, 1))
        base = np.array([0.70, 0.74, 0.80])
        albedo = base * (1 - 0.35 * wet) * (1 - frost) + np.array([0.93, 0.95, 0.98]) * frost
        materials.set(2, albedo=tuple(float(v) for v in albedo),
                      rough=0.35 * (1 - wet) * (1 - frost) + 0.05 * wet + 0.95 * frost,
                      spec=0.35 + 0.45 * wet - 0.3 * frost, shin=48.0 + 100.0 * wet - 40.0 * frost)
        a_wet = float(rec.arrays["pool0.A_wet"][frame][0]) if "pool0.A_wet" in rec.arrays else 0.0
        h_pool = float(rec.arrays["pool0.h_pool"][frame][0]) if "pool0.h_pool" in rec.arrays else 0.0
        r_pool = math.sqrt(max(a_wet, 0.0) / math.pi)
        if abs(r_pool - pool_mesh["r"]) > 1e-3:
            pool_mesh["r"] = r_pool
            if pool_mesh["mesh"] is not None:
                pool_mesh["mesh"].delete()
            pool_mesh["mesh"] = Mesh(*disc(min(r_pool, min(hw, hd)), 0.002), 3) if r_pool > 1e-3 else None
        rain.update(frame, dt_wall * args.speed if playing else 0.0, col_w, col_h, col_d)
        deep = float(np.clip(h_pool / 0.01, 0, 1))
        materials.set(3, albedo=(0.06 * (1 - deep) + 0.02 * deep, 0.08 * (1 - deep) + 0.05 * deep, 0.11 * (1 - deep) + 0.10 * deep))
        renderer.update_material_ssbo()

        # -- camera and lights --
        eye = centre + dist * np.array([math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)], np.float32)
        view = look_at(eye, centre, np.array([0, 1, 0], np.float32))
        proj = perspective(fov, W / H, 0.05, 40.0)
        scale2 = (col_h * 0.6) ** 2
        lights_pos = np.array([[0.9, col_h + 0.9, 1.1], [-1.0, col_h * 0.6, 0.9], [0.0, col_h + 0.3, -0.6]], np.float32)
        lights_col = np.array([[1.0, 0.97, 0.92], [0.55, 0.62, 0.80], [0.80, 0.85, 1.0]], np.float32)
        lights_int = np.array([scale2 * 14.0, scale2 * 8.0, scale2 * 7.0], np.float32)
        renderer.set_point_lights(lights_pos, lights_col, lights_int)

        glViewport(0, 0, W, H)
        glEnable(GL_DEPTH_TEST); glDepthFunc(GL_LEQUAL)
        glClearColor(0.05, 0.05, 0.06, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # room first, opaque: floor, plate, and only the walls behind the air
        view_dir = centre - eye
        for mesh, m in ((floor, ident), (plate, m_plate)):
            renderer.draw_mesh(mesh.vao, mesh.n, gl(proj @ view @ m), gl(view @ m), enable_blend=False)
        for mesh, inward, point in walls:
            if np.dot(inward, eye - point) > 0.0:          # camera is on the wall's inside
                renderer.draw_mesh(mesh.vao, mesh.n, gl(proj @ view), gl(view), enable_blend=False)
        if pool_mesh["mesh"] is not None:
            renderer.draw_mesh(pool_mesh["mesh"].vao, pool_mesh["mesh"].n, gl(proj @ view), gl(view), enable_blend=False)
        # the rain: representatives, alpha-blended, no depth write
        rain_mesh = rain.build_mesh(view_dir)
        if rain_mesh is not None:
            renderer.draw_mesh(rain_mesh.vao, rain_mesh.n, gl(proj @ view), gl(view), enable_blend=True, depth_write=False)
        # then the air: the slice stack most face-on to the camera, additive, no depth write
        axis = int(np.argmax(np.abs(view_dir)))
        renderer.draw_mesh(stacks[axis].vao, stacks[axis].n, gl(proj @ view @ m_column), gl(view @ m_column),
                           enable_blend=True, depth_write=False, additive=True)

        pygame.display.flip()
        lwc_top = float(rec.arrays["LWC"][frame][-1]) if "LWC" in rec.arrays else float("nan")
        pygame.display.set_caption(
            f"raincloud  t={t_weather:6.1f} s  frame {frame}/{rec.n_frames}  LWC top={lwc_top:.2e} kg/m3  "
            f"film={h_film * 1e6:.0f} um  frost={h_frost * 1e6:.0f} um  pool r={r_pool * 100:.1f} cm  {'||' if not playing else '>'}")

        if not snapshot_done and (args.snapshot_at is None or t_weather >= args.snapshot_at):
            buf = glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE)
            surf = pygame.image.frombuffer(buf, (W, H), "RGBA")
            pygame.image.save(pygame.transform.flip(surf, False, True), args.snapshot)
            print(f"snapshot at t={t_weather:.1f} s -> {args.snapshot}", flush=True)
            snapshot_done = True
        if args.exit_after is not None and time.time() - t_wall0 > args.exit_after:
            pygame.quit(); return
        clock.tick(60)


if __name__ == "__main__":
    main()
