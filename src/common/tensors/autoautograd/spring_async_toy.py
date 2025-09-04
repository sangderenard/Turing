"""
Spring-Repulsor Async Toy (AbstractTensor)
---------------------------------
Minimal, threadful prototype of the "spring–repulsor, multiband gradient acoustics" learner.

- < 20 nodes
- Two roles/threads: Experiencer (ops + impulses) and Reflector (relaxer/integrator)
- Edges store: timestamp, rings (microgradient count), composite spring aggregation
- 3 operators: plus, multiply, gather, and a simple scatter (bonus)
- Local, in-place gradient impulses per edge (requires a residual; we warn if absent)
- Simple DEC/Hodge star stub (*1 = 1.0) with hooks
- Inertial dampener stub using node state histories and windowed spectral mass

Run:
    python spring_async_toy.py

Notes:
- This is a toy: no external deps beyond AbstractTensor. Threading is cooperative and simple.
- Impulses require a residual (sign + magnitude). If no residual is available, the op
  emits **no** impulse and logs a WARNING (clear & explicit).
- The Reflector integrates geometry at steady ticks and periodically "commits" node
  positions back into parameter scalars.
"""
from __future__ import annotations
from .integration.bridge_v2 import (
    push_impulses_from_op_v2,
    batched_forward_v2,
    push_impulses_from_ops_batched,
)
from .whiteboard_cache import WhiteboardCache

import time
import threading
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Callable
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm, colors as mcolors
from ..abstraction import AbstractTensor

L_MIN = 0.05
L_MAX = 3.0
DL_CAP = 0.5
MAGNITUDE_ONLY = False
V_MAX = 2.0
STEP_MAX = 0.2
READOUT_SCALE = 1.0
READOUT_BIAS  = 0.0
W_EPS = 1e-3
W_MIN, W_MAX = 0.25, 4.0 
# --- threshold-pop optimizer knobs ---
POP_FRUSTRATION_TH = 0.10   # how 'annoyed' an edge must be: |L - L*|
POP_AGG_TH         = 0.12   # how large the transient aggregate must be: |sum(contribs)|
POP_QUANTUM        = 0.2   # discrete ΔL moved per pop
POP_MAX_PER_TICK   = 1000      # safety cap per integrator tick

# ----------------------------- Utilities ------------------------------------

def now_s() -> float:
    return time.perf_counter()

def as_x_target(fn, D: int = 3):  # scalar → Dirichlet target on x in D-D
    def _t(t):
        v = AbstractTensor.zeros(D, dtype=float)
        v[0] = float(fn(t))
        return v
    return _t

def as_x_force(fn, D: int = 3):   # scalar → Neumann force on x in D-D
    def _f(t):
        v = AbstractTensor.zeros(D, dtype=float)
        v[0] = float(fn(t))
        return v
    return _f

def interpret_vec(v: AbstractTensor) -> float:
    # scalar meaning of a locational vector (keep it simple: x-component)
    return float(READOUT_SCALE * v[0] + READOUT_BIAS)

def inv_weight(sys: SpringRepulsorSystem, src: int, dst: int) -> float:
    d = float(AbstractTensor.linalg.norm(sys.nodes[dst].p - sys.nodes[src].p))
    w = 1.0 / (W_EPS + d)
    return float(AbstractTensor.clip(w, W_MIN, W_MAX))

def norm_weights(ws):
    s = float(sum(ws))
    if s <= 0.0:
        return [1.0 / max(1, len(ws))] * len(ws)
    return [w / s for w in ws]

def soft_knee(x: float, th: float, ratio: float, knee: float) -> float:
    """Soft-knee compressor/expander on positive x."""
    if x < 0:
        x = -x  # magnitude domain; sign handled elsewhere
    lo, hi = th - knee / 2.0, th + knee / 2.0
    if x < lo:
        return x
    if x > hi:
        return th + (x - th) / max(ratio, 1e-6)
    # Smooth transition within knee
    a = (x - lo) / max(knee, 1e-6)
    return x + ((1.0 / max(ratio, 1e-6) - 1.0) * (x - th)) * a * (2 - a)

def enliven_feature_edges(sys: SpringRepulsorSystem, in_ids: List[int], out_ids: List[int]):
    # seed physical ties so forces/learning can flow
    for i in in_ids:
        for o in out_ids:
            sys.ensure_edge(i, o, "feat")  # op_id tag just to distinguish

# ----------------------------- Core data ------------------------------------

@dataclass
class Node:
    id: int
    # Parameter scalar for this toy; real system could be vector-valued
    theta: float
    # Geometry (2D for easy intuition)
    p: AbstractTensor  # shape (2,)
    v: AbstractTensor  # shape (2,)
    M0: float = 1.0
    last_commit: float = 0.0
    version: int = 0
    # History for inertial dampener (positions)
    hist_p: deque = field(default_factory=lambda: deque(maxlen=128))

    # Mapping pos->parameter (identity on x component for clarity)
    def commit(self):
        self.theta = interpret_vec(self.p)
        self.version += 1



@dataclass
class EdgeBand:
    tau: float  # EMA time constant (s)
    K: float    # stiffness contribution
    kappa: float  # length scaling from impulse magnitude
    th: float   # threshold for knee
    ratio: float  # compression ratio (>1 compresses)
    knee: float
    alpha: float  # spectral inertia scale
    # EMAs
    m: float = 0.0  # magnitude EMA
    s: float = 0.0  # signed EMA (approx sign average)


@dataclass
class CompositeSpringAggregator:
    """Aggregates multiple rest-length suggestions across a fixed edge.

    Maintains a short queue of contributions which decay over time.
    Resulting target rest length L* = l0 + sum(decayed contributions).
    """
    decay: float = 0.98
    maxlen: int = 64
    _contribs: deque = field(default_factory=lambda: deque(maxlen=64))

    def add(self, value: float):
        self._contribs.append(float(value))

    def reduce(self) -> float:
        # Exponentially decay historical contributions and return sum
        total = 0.0
        new = deque(maxlen=self.maxlen)
        scale = 1.0
        for c in self._contribs:
            total += scale * c
            new.append(scale * c)
            scale *= self.decay
        # keep decayed values
        self._contribs = new
        return total


@dataclass
class Edge:
    key: Tuple[int, int, str]  # (i, j, op_id)
    i: int
    j: int
    op_id: str
    hodge1: float = 1.0   # stub for DEC; can be set per-edge
    timestamp: float = field(default_factory=now_s)
    rings: int = 0        # number of microgradients accumulated

    # Spectral bands
    bands: List[EdgeBand] = field(default_factory=list)

    # Composite spring contributions
    spring: CompositeSpringAggregator = field(default_factory=CompositeSpringAggregator)

    # Base rest length l0 (from DEC). Updated on construction.
    l0: float = 1.0

    def ingest_impulse(self, g_scalar: float, dt: float):
        self.timestamp = now_s()
        self.rings += 1
        # Update band EMAs
        for b in self.bands:
            b.m += (abs(g_scalar) - b.m) * (dt / max(b.tau, 1e-6))
            b.s += (AbstractTensor.sign(g_scalar) - b.s) * (dt / max(b.tau, 1e-6))
            # Also push a composite contribution (post-knee)
            # AFTER
            y = soft_knee(b.m, b.th, b.ratio, b.knee)
            sgn = 1.0 if MAGNITUDE_ONLY else AbstractTensor.sign(b.s)
            dl = float(AbstractTensor.clip(b.kappa * sgn * y, -DL_CAP, DL_CAP))
            self.spring.add(dl)



    # --- runtime (internal) ---
    _last_reduce: float = 0.0
    pops: int = 0  # diagnostics

    def target_length(self) -> float:
        # cache current transient aggregate, then clamp total target length
        agg = self.spring.reduce()
        self._last_reduce = float(agg)
        return float(AbstractTensor.clip(self.l0 + agg, L_MIN, L_MAX))

    def maybe_pop(self, L_current: float):
        """
        If both frustration and aggregate exceed thresholds, do 'integer' pops:
          l0 += sgn * POP_QUANTUM
          spring.consume by adding an equal-and-opposite transient (-sgn * POP_QUANTUM)
        """
        # current target uses cached _last_reduce set by target_length() this tick
        L_star = float(AbstractTensor.clip(self.l0 + self._last_reduce, L_MIN, L_MAX))
        frustration = abs(L_current - L_star)
        agg_mag = abs(self._last_reduce)
        if frustration < POP_FRUSTRATION_TH or agg_mag < POP_AGG_TH:
            return  # nothing to do

        # how many quanta could we commit right now?
        want = min(int(agg_mag // POP_QUANTUM), POP_MAX_PER_TICK)
        if want <= 0:
            return

        sgn = 1.0 if self._last_reduce >= 0.0 else -1.0
        for _ in range(want):
            # commit ΔL into the permanent base
            self.l0 += sgn * POP_QUANTUM
            # consume the same amount from the transient buffer immediately
            self.spring.add(-sgn * POP_QUANTUM)
            self.pops += 1

@dataclass
class BoundaryPort:
    nid: int
    alpha: float = 0.0                     # Dirichlet spring strength
    beta: float = 0.0                      # Neumann (traction) gain
    target_fn: Optional[Callable[[float], AbstractTensor]] = None  # t -> R^D
    force_fn: Optional[Callable[[float], AbstractTensor]]  = None  # t -> R^D
    enabled: bool = True


class SpringRepulsorSystem:
    def __init__(self, nodes: List[Node], edges: List[Edge], *,
                 eta: float = 0.1, gamma: float = 0.92, dt: float = 0.02,
                 rep_eps: float = 1e-6):
        self.nodes: Dict[int, Node] = {n.id: n for n in nodes}
        self.edges: Dict[Tuple[int, int, str], Edge] = {e.key: e for e in edges}
        self.eta = eta
        self.gamma = gamma
        self.dt = dt
        self.rep_eps = rep_eps
        # Locks for thread-safety on edges
        self.edge_locks: Dict[Tuple[int, int, str], threading.Lock] = defaultdict(threading.Lock)
        self.boundaries: Dict[int, BoundaryPort] = {}
        self.D = int(next(iter(self.nodes.values())).p.shape[0])

    def add_boundary(self, port: BoundaryPort):
        self.boundaries[port.nid] = port

    def remove_boundary(self, nid: int):
        self.boundaries.pop(nid, None)

    def set_boundary(self, nid: int, **kw):
        b = self.boundaries.get(nid)
        if b:
            for k, v in kw.items():
                setattr(b, k, v)

    # ----------------- Impulse ingestion -----------------
    def ensure_edge(self, i: int, j: int, op_id: str) -> Edge:
        key = (i, j, op_id)
        if key not in self.edges:
            # Default three bands (fast/mid/slow)
            bands = [
                EdgeBand(tau=0.05, K=3.0, kappa=0.15, th=0.05, ratio=3.0, knee=0.05, alpha=1.0),
                EdgeBand(tau=0.25, K=2.0, kappa=0.10, th=0.10, ratio=2.0, knee=0.10, alpha=0.5),
                EdgeBand(tau=1.00, K=1.0, kappa=0.05, th=0.20, ratio=1.2, knee=0.20, alpha=0.2),
            ]
            e = Edge(key=key, i=i, j=j, op_id=op_id, bands=bands, l0=1.0)
            self.edges[key] = e
        return self.edges[key]

    def impulse(self, i: int, j: int, op_id: str, g_scalar: float):
        key = (i, j, op_id)
        e = self.ensure_edge(i, j, op_id)
        with self.edge_locks[key]:
            e.ingest_impulse(g_scalar, self.dt)

    # ----------------- Physics tick -----------------
    def tick(self):
        # Force accumulator
        F: Dict[int, AbstractTensor] = {i: AbstractTensor.zeros(self.D, dtype=float) for i in self.nodes}


        t_now = now_s()
        scale = -0.1
        for key, e in self.edges.items():
            ni, nj = self.nodes[e.i], self.nodes[e.j]
            d = nj.p - ni.p
            L = AbstractTensor.linalg.norm(d)
            if L < 1e-9:
                continue
            u = d / (L + 1e-12)
            Lstar = e.target_length()  # also caches _last_reduce
            Ksum = AbstractTensor.sum(b.K for b in e.bands)
            Fel = Ksum * e.hodge1 * (L - Lstar) * u
            Rep = (self.eta / (self.rep_eps + L * L)) * u
            F[e.i] += (Fel - Rep) * scale
            F[e.j] -= (Fel - Rep) * scale

            # NEW: event-driven discrete commits from stress
            e.maybe_pop(L)

        for b in self.boundaries.values():
            if not b.enabled or b.nid not in self.nodes:
                continue
            n = self.nodes[b.nid]
            # Dirichlet spring toward target
            if b.alpha > 0.0 and b.target_fn is not None:
                tvec = b.target_fn(t_now)
                if AbstractTensor.isfinite(tvec).all():
                    F[n.id] += -b.alpha * (n.p - tvec)
            # Neumann traction force
            if b.beta > 0.0 and b.force_fn is not None:
                fvec = b.force_fn(t_now)
                if AbstractTensor.isfinite(fvec).all():
                    F[n.id] += b.beta * fvec

        # Integrate with heavy damping
        for n in self.nodes.values():
            # spectral response (ND) → smoothing force
            resp, _, _ = self._spectral_inertia(n)
            if not AbstractTensor.isfinite(resp).all():
                resp = AbstractTensor.zeros_like(n.p)
            F[n.id] += -resp
            n.v = self.gamma * n.v + self.dt * F[n.id] / n.M0

            # cap velocity
            speed = AbstractTensor.linalg.norm(n.v)
            if speed > V_MAX:
                n.v *= V_MAX / (speed + 1e-12)

            # cap per-step displacement
            dp = -self.dt * n.v
            step = float(AbstractTensor.linalg.norm(dp))
            if step > STEP_MAX:
                dp *= STEP_MAX / (step + 1e-12)
            n.p = n.p + dp
            n.hist_p.append(n.p.copy())


    def commit_some(self, every_s: float = 0.2):
        t = now_s()
        for n in self.nodes.values():
            if (t - n.last_commit) >= every_s:
                n.commit()
                n.last_commit = t

    # ----------------- Inertial dampener (stub) -----------------
    def _spectral_inertia(self, n: Node):
        """
        Exploratory FFT with adaptive zoom:
        1) coarse FFT to find active bins (remove 'empty' bins)
        2) high-res zero-padded FFT inside each active band
        3) build high-resolution ND rotation response:
            resp = J @ x_t, where J is the aggregated rotation bivector

        Returns:
            resp : (D,) immediate ND response for current state
            J    : (D,D) skew-symmetric rotation generator aggregated over active bands
            bands: list of (w_lo, w_hi, power) tuples for diagnostics
        """
        H = len(n.hist_p)
        if H < 32:
            D = n.p.shape[0]
            return AbstractTensor.zeros(D, float), AbstractTensor.zeros((D, D), float), []

        # --- gather window & detrend ---
        W = min(H, 128)
        xs = AbstractTensor.stack(list(n.hist_p)[-W:])           # (W, D)
        if not AbstractTensor.isfinite(xs).all():
            D = xs.shape[1]
            return AbstractTensor.zeros(D, float), AbstractTensor.zeros((D, D), float), []
        xs = xs - xs.mean(axis=0, keepdims=True)
        # AFTER: xs = xs - xs.mean(...)

        # normalize to avoid huge FFT magnitudes
        scale = max(1.0, float(AbstractTensor.linalg.norm(xs, ord=AbstractTensor.inf)))
        xs = xs / scale

        D = xs.shape[1]
        dt = float(getattr(self, "dt", 1.0))

        # Optional Hann to reduce leakage (pure AbstractTensor)
        w = AbstractTensor.hanning(W) if W > 1 else AbstractTensor.ones(W)
        xw = (w[:, None] * xs)

        # --- 1) coarse FFT ---
        C0 = AbstractTensor.fft.rfft(xw, axis=0)                 # (F0, D)
        w0 = 2.0 * AbstractTensor.pi * AbstractTensor.fft.rfftfreq(W, d=dt)  # (F0,)
        P0 = AbstractTensor.sum(AbstractTensor.abs(C0)**2, axis=1)           # (F0,)
        if P0.sum() <= 1e-12 or len(P0) <= 2:
            return AbstractTensor.zeros(D, float), AbstractTensor.zeros((D, D), float), []

        # prune empty bins: absolute & relative thresholds
        rel = 0.01 * float(P0.max())
        abs_th = max(rel, 1e-12)
        active = P0 > abs_th

        # merge contiguous active bins into bands; drop tiny bands
        bands_idx = []
        i = 0
        while i < len(active):
            if active[i]:
                j = i + 1
                while j < len(active) and active[j]:
                    j += 1
                if (j - i) >= 1:
                    # expand by 1 bin on each side (guard) if in range
                    lo = max(0, i - 1)
                    hi = min(len(active), j + 1)
                    bands_idx.append((lo, hi))
                i = j
            else:
                i += 1
        if not bands_idx:
            return AbstractTensor.zeros(D, float), AbstractTensor.zeros((D, D), float), []

        # --- 2) high-res zoom via zero-padding ---
        Z = 8  # zero-pad factor
        Wz = W * Z
        # zero-pad the windowed signal in time
        xpad = AbstractTensor.pad(xw, ((0, Wz - W), (0, 0)))
        Cz = AbstractTensor.fft.rfft(xpad, axis=0)                 # (Fz, D)
        wz = 2.0 * AbstractTensor.pi * AbstractTensor.fft.rfftfreq(Wz, d=dt)   # (Fz,)

        # helper to map coarse indices to high-res frequency indices
        def coarse_band_to_w(b_lo, b_hi):
            return w0[b_lo], w0[min(b_hi, len(w0)-1)]

        def w_to_hi_idx(wlo, whi):
            i0 = int(AbstractTensor.clip(AbstractTensor.searchsorted(wz, wlo, side="left"), 0, len(wz)-1))
            i1 = int(AbstractTensor.clip(AbstractTensor.searchsorted(wz, whi, side="right"), 0, len(wz)))
            return i0, max(i1, i0+1)

        # --- 3) build high-res rotation bivector across active bands ---
        J = AbstractTensor.zeros((D, D), float)
        bands_meta = []
        total_power = 0.0

        for (blo, bhi) in bands_idx:
            w_lo, w_hi = coarse_band_to_w(blo, bhi)
            hi_lo, hi_hi = w_to_hi_idx(w_lo, w_hi)
            Cz_band = Cz[hi_lo:hi_hi, :]               # (Fb, D)
            if Cz_band.shape[0] < 1:
                continue
            Pw = AbstractTensor.sum(AbstractTensor.abs(Cz_band)**2, axis=1) + 1e-12  # (Fb,)
            if not AbstractTensor.isfinite(Pw).all() or Pw.sum() <= 1e-12:
                continue
            Ww = Pw / Pw.sum()
            wgrid = wz[hi_lo:hi_hi]                    # (Fb,)

            # integrate rotation bivector over refined grid
            for c, wght, omg in zip(Cz_band, Ww, wgrid):
                a = AbstractTensor.real(c)                         # (D,)
                b = AbstractTensor.imag(c)                         # (D,)
                J += wght * omg * (AbstractTensor.outer(a, b) - AbstractTensor.outer(b, a))
            band_power = float(Pw.sum())
            total_power += band_power
            bands_meta.append((w_lo, w_hi, band_power))

        if total_power <= 1e-12:
            return AbstractTensor.zeros(D, float), AbstractTensor.zeros((D, D), float), []

        # immediate ND response on current state
        x_t = xs[-1]                                   # (D,)
        resp = J @ x_t                                 # (D,)

        return resp, J, bands_meta
# liveviz_gl_points.py
# Minimal OpenGL point-field renderer (pygame + PyOpenGL)
# - Keeps node colors from a matplotlib colormap (TwoSlopeNorm around 0)
# - Boundary nodes drawn larger
# - No edges/lines; autoscaled camera; non-blocking window


import pygame
from pygame.locals import DOUBLEBUF, OPENGL, RESIZABLE, VIDEORESIZE, QUIT
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from matplotlib import cm, colors as mcolors
from typing import Any, Tuple

# Expecting SpringRepulsorSystem with:
#   self.nodes: Dict[int, Node] where Node.p is (3,) ndarray-like and Node.theta is scalar
#   self.boundaries: Dict[int, Any] (keys are boundary node ids)
# Edges ignored entirely in this GL version.

class LiveVizGLPoints:
    def __init__(self,
                 sys,
                 node_cmap: str = "coolwarm",
                 base_point_size: float = 6.0,
                 boundary_scale: float = 1.8,
                 bg_color: Tuple[float, float, float] = (0.04, 0.04, 0.06)):
        self.sys = sys
        self.node_cmap = cm.get_cmap(node_cmap)
        self.base_point_size = float(base_point_size)
        self.boundary_scale = float(boundary_scale)
        self.bg_color = bg_color

        # runtime / GL state
        self._win = None
        self._w = 960
        self._h = 720
        self._program = None
        self._vao = None
        self._vbo_pos = None
        self._vbo_col = None
        self._vbo_size = None
        self._num_points = 0

        self._u_mvp = None  # uniform location
        self._mvp = AbstractTensor.eye(4, dtype=AbstractTensor.float32)  # updated each frame

    # ---------- data snapshot ----------
    def _snapshot(self):
        # lock-free minimal copy
        nodes = {i: (AbstractTensor.asarray(n.p, dtype=AbstractTensor.float32).copy(), float(n.theta))
                 for i, n in self.sys.nodes.items()}
        bset = set(self.sys.boundaries.keys())
        return nodes, bset

    # ---------- GL bootstrap ----------
    def _create_window(self):
        pygame.init()
        pygame.display.set_caption("Spring-Repulsor • GL Points")

        # NEW: request modern-ish context + depth buffer
        try:
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        except Exception:
            pass  # not fatal; fall back if unsupported

        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)

        pygame.display.set_mode((self._w, self._h), DOUBLEBUF | OPENGL | RESIZABLE)

        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glDisable(GL_CULL_FACE)
        glEnable(GL_PROGRAM_POINT_SIZE)
        r, g, b = self.bg_color
        glClearColor(r, g, b, 1.0)
        glViewport(0, 0, self._w, self._h)


    def _compile_shaders(self):
        vsrc = """
        #version 330 core
        layout (location = 0) in vec3 in_pos;
        layout (location = 1) in vec3 in_col;
        layout (location = 2) in float in_size;

        uniform mat4 u_mvp;

        out vec3 v_col;

        void main() {
            v_col = in_col;
            gl_Position = u_mvp * vec4(in_pos, 1.0);
            gl_PointSize = in_size;  // requires GL_PROGRAM_POINT_SIZE
        }
        """

        fsrc = """
        #version 330 core
        in vec3 v_col;
        out vec4 FragColor;

        void main() {
            // Circular point sprite mask (optional). Comment out for square points.
            vec2 p = gl_PointCoord * 2.0 - 1.0;
            if (dot(p, p) > 1.0) discard;

            FragColor = vec4(v_col, 1.0);
        }
        """

        self._program = compileProgram(
            compileShader(vsrc, GL_VERTEX_SHADER),
            compileShader(fsrc, GL_FRAGMENT_SHADER)
        )
        self._u_mvp = glGetUniformLocation(self._program, "u_mvp")

    def _create_buffers(self):
        self._vao = glGenVertexArrays(1)
        glBindVertexArray(self._vao)

        self._vbo_pos = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_pos)
        glBufferData(GL_ARRAY_BUFFER, 1, None, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        self._vbo_col = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_col)
        glBufferData(GL_ARRAY_BUFFER, 1, None, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)

        self._vbo_size = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_size)
        glBufferData(GL_ARRAY_BUFFER, 1, None, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(2)

        glBindVertexArray(0)

    # ---------- geometry packing ----------
    def _pack_points(self):
        nodes, bset = self._snapshot()
        ids = AbstractTensor.array(sorted(nodes.keys()))
        if ids.size == 0:
            return (AbstractTensor.zeros((0, 3), AbstractTensor.float32),
                    AbstractTensor.zeros((0, 3), AbstractTensor.float32),
                    AbstractTensor.zeros((0,), AbstractTensor.float32),
                    AbstractTensor.zeros((0, 3), AbstractTensor.float32))

        P = AbstractTensor.stack([nodes[i][0] for i in ids]).astype(AbstractTensor.float32, copy=False)

        # NEW: pad to 3D if needed
        if P.shape[1] == 2:
            P = AbstractTensor.pad(P, ((0,0),(0,1)), constant_values=0.0)

        # NEW: replace NaN/Inf early to avoid NaN bounds
        P = AbstractTensor.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)

        thetas = AbstractTensor.array([nodes[i][1] for i in ids], dtype=AbstractTensor.float32)
        is_b = AbstractTensor.array([i in bset for i in ids], dtype=bool)

        # Color map around 0.0 with TwoSlopeNorm
        vmin = float(AbstractTensor.min(thetas))
        vmax = float(AbstractTensor.max(thetas))
        if vmax <= vmin:
            vmax = vmin + 1e-6
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        C = self.node_cmap(norm(thetas))[:, :3].astype(AbstractTensor.float32)  # RGB

        # Point sizes (boundary nodes larger)
        sizes = AbstractTensor.full(ids.shape, self.base_point_size, dtype=AbstractTensor.float32)
        sizes[is_b] *= self.boundary_scale

        return P, C, sizes, P  # return P twice; last is for autoscale

    def _update_buffers(self):
        P, C, S, P_for_bounds = self._pack_points()
        self._num_points = P.shape[0]

        glBindVertexArray(self._vao)

        # positions
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_pos)
        glBufferData(GL_ARRAY_BUFFER, P.nbytes, P, GL_DYNAMIC_DRAW)

        # colors
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_col)
        glBufferData(GL_ARRAY_BUFFER, C.nbytes, C, GL_DYNAMIC_DRAW)

        # sizes
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_size)
        glBufferData(GL_ARRAY_BUFFER, S.nbytes, S, GL_DYNAMIC_DRAW)

        glBindVertexArray(0)

        # update camera
        self._compute_mvp(P_for_bounds)

    # ---------- camera / MVP ----------
    @staticmethod
    def _look_at(eye, center, up) -> AbstractTensor:
        f = center - eye
        f = f / (AbstractTensor.linalg.norm(f) + 1e-12)
        upn = up / (AbstractTensor.linalg.norm(up) + 1e-12)
        s = AbstractTensor.cross(f, upn)
        s = s / (AbstractTensor.linalg.norm(s) + 1e-12)
        u = AbstractTensor.cross(s, f)

        M = AbstractTensor.eye(4, dtype=AbstractTensor.float32)
        M[0, :3] = s
        M[1, :3] = u
        M[2, :3] = -f
        T = AbstractTensor.eye(4, dtype=AbstractTensor.float32)
        T[:3, 3] = -eye
        return M @ T

    @staticmethod
    def _perspective(fovy_deg, aspect, znear, zfar) -> AbstractTensor:
        f = 1.0 / AbstractTensor.tan(AbstractTensor.deg2rad(fovy_deg) / 2.0)
        M = AbstractTensor.zeros((4, 4), dtype=AbstractTensor.float32)
        M[0, 0] = f / max(aspect, 1e-6)
        M[1, 1] = f
        M[2, 2] = (zfar + znear) / (znear - zfar)
        M[2, 3] = (2 * zfar * znear) / (znear - zfar)
        M[3, 2] = -1.0
        return M

    def _compute_mvp(self, P: AbstractTensor):
        if P.size == 0:
            self._mvp = AbstractTensor.eye(4, dtype=AbstractTensor.float32)
            return
        P = AbstractTensor.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)  # NEW

        lo = AbstractTensor.min(P, axis=0)
        hi = AbstractTensor.max(P, axis=0)
        ctr = 0.5 * (lo + hi)
        extent_t = AbstractTensor.get_tensor(AbstractTensor.max(hi - lo))
        if not extent_t.isfinite().item() or extent_t.item() <= 1e-6:
            extent_t = AbstractTensor.get_tensor(1.0)  # NEW: avoid zero/NaN extent
        extent = float(extent_t.item())

        rad = extent * 0.6 + 1e-3
        eye = ctr + AbstractTensor.array([rad * 1.6, rad * 1.6, rad * 1.6], dtype=AbstractTensor.float32)
        up  = AbstractTensor.array([0.0, 1.0, 0.0], dtype=AbstractTensor.float32)

        V  = self._look_at(eye, ctr, up)
        aspect = self._w / max(self._h, 1)
        znear  = max(rad * 0.05, 1e-3)
        zfar   = max(rad * 20.0, znear + 1.0)
        Pm = self._perspective(45.0, aspect, znear, zfar)
        self._mvp = Pm @ V


    # ---------- public API ----------
    def launch(self, width: int = 960, height: int = 720):
        self._w, self._h = int(width), int(height)
        self._create_window()
        self._compile_shaders()
        self._create_buffers()
        self._update_buffers()  # initial fill
    
    def _rebuild_gl_objects(self):
        # delete old objects if they exist
        try:
            if self._vbo_pos:  glDeleteBuffers(1, [self._vbo_pos])
            if self._vbo_col:  glDeleteBuffers(1, [self._vbo_col])
            if self._vbo_size: glDeleteBuffers(1, [self._vbo_size])
            if self._vao:      glDeleteVertexArrays(1, [self._vao])
            if self._program:  glDeleteProgram(self._program)
        except Exception:
            pass
        self._program = None
        self._vao = self._vbo_pos = self._vbo_col = self._vbo_size = None

        self._compile_shaders()
        self._create_buffers()

    def _handle_events(self):
        for evt in pygame.event.get():
            if evt.type == QUIT:
                self.close()
            elif evt.type == VIDEORESIZE:
                self._w, self._h = evt.w, evt.h
                pygame.display.set_mode((self._w, self._h), DOUBLEBUF | OPENGL | RESIZABLE)
                glViewport(0, 0, self._w, self._h)

                # NEW: context was recreated → rebuild program & buffers
                self._rebuild_gl_objects()
                self._update_buffers()

    def _draw(self):
        r, g, b = self.bg_color
        glClearColor(r, g, b, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(self._program)
        # CHANGED: upload transpose so GLSL sees the right matrix
        glUniformMatrix4fv(self._u_mvp, 1, GL_FALSE, self._mvp.T)

        glBindVertexArray(self._vao)
        glDrawArrays(GL_POINTS, 0, self._num_points)
        glBindVertexArray(0)

        pygame.display.flip()


    def step(self, _dt: float = 0.0):
        """Call from your main loop (non-blocking)."""
        if self._program is None:
            # if user forgot to launch, do it lazily
            self.launch(self._w, self._h)
        self._handle_events()
        self._update_buffers()
        self._draw()

    def close(self):
        try:
            if self._vbo_pos:
                glDeleteBuffers(1, [self._vbo_pos])
            if self._vbo_col:
                glDeleteBuffers(1, [self._vbo_col])
            if self._vbo_size:
                glDeleteBuffers(1, [self._vbo_size])
            if self._vao:
                glDeleteVertexArrays(1, [self._vao])
            if self._program:
                glDeleteProgram(self._program)
        except Exception:
            pass
        finally:
            try:
                pygame.display.quit()
                pygame.quit()
            except Exception:
                pass
            self._program = None

class LiveViz3D:
    def __init__(self, sys: SpringRepulsorSystem,
                 edge_cmap="viridis", node_cmap="coolwarm",
                 interval_ms: int = 60):
        self.sys = sys
        self.edge_cmap = plt.colormaps.get_cmap(edge_cmap)
        self.node_cmap = plt.colormaps.get_cmap(node_cmap)
        self.norm_nodes = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
        self.norm_edges = mcolors.Normalize(vmin=0.0, vmax=0.5)  # auto-updated
        self.interval_ms = interval_ms

        self.fig = None
        self.ax = None
        self.scat_nodes = None
        self.scat_bounds = None
        self.edge_artists = []
        self.anim = None

    def _snapshot(self):
        # cheap, lock-free snapshot
        nodes = {i: (n.p.copy(), float(n.theta)) for i, n in self.sys.nodes.items()}
        edges = list(self.sys.edges.values())
        bset = set(self.sys.boundaries.keys())
        return nodes, edges, bset

    def _compute_edge_energy(self, e, nodes):
        pi = nodes[e.i][0]; pj = nodes[e.j][0]
        d = pj - pi
        L = float(AbstractTensor.linalg.norm(d) + 1e-12)
        Lstar = e.target_length()
        Ksum = sum(b.K for b in e.bands)
        return 0.5 * Ksum * (L - Lstar) ** 2, (pi, pj)

    def _init_fig(self):
        self.fig = plt.figure("Spring-Repulsor • Live 3D", figsize=(7, 6))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_box_aspect((1, 1, 1))

    def _init_artists(self):
        nodes, edges, bset = self._snapshot()
        P = AbstractTensor.stack([nodes[i][0] for i in sorted(nodes.keys())])
        ids = AbstractTensor.array(sorted(nodes.keys()))
        is_b = AbstractTensor.array([i in bset for i in ids])

        # split boundary vs non-boundary
        Pn = P[~is_b]; Pb = P[is_b]
        thetas = AbstractTensor.array([nodes[i][1] for i in ids])
        self.norm_nodes = mcolors.TwoSlopeNorm(vmin=AbstractTensor.min(thetas), vcenter=0.0, vmax=AbstractTensor.max(thetas))

        if Pn.size:
            self.scat_nodes = self.ax.scatter(Pn[:,0], Pn[:,1], Pn[:,2],
                                              s=40, marker="o",
                                              c=self.node_cmap(self.norm_nodes(thetas[~is_b])),
                                              depthshade=False, linewidths=0.5, edgecolors="k")
        if Pb.size:
            self.scat_bounds = self.ax.scatter(Pb[:,0], Pb[:,1], Pb[:,2],
                                               s=90, marker="s",
                                               c=self.node_cmap(self.norm_nodes(thetas[is_b])),
                                               depthshade=False, linewidths=1.0, edgecolors="black")

        # edges (draw once; will be recreated each frame)
        for L in self.edge_artists:
            L.remove()
        self.edge_artists = []
        U_vals = []
        segs = []
        for e in edges:
            U, (pi, pj) = self._compute_edge_energy(e, nodes)
            U_vals.append(U); segs.append((pi, pj))
        if U_vals:
            U_vals = AbstractTensor.array(U_vals)
            self.norm_edges = mcolors.Normalize(vmin=float(AbstractTensor.percentile(U_vals, 5)),
                                                vmax=float(AbstractTensor.percentile(U_vals, 95)))
            for (pi, pj), U in zip(segs, U_vals):
                col = self.edge_cmap(self.norm_edges(U))
                art, = self.ax.plot([pi[0], pj[0]], [pi[1], pj[1]], [pi[2], pj[2]],
                                    lw=1.5, color=col, alpha=0.9)
                self.edge_artists.append(art)

        # axes limits
        self._autoscale(P)

    def _autoscale(self, P):
        lo = AbstractTensor.min(P, axis=0); hi = AbstractTensor.max(P, axis=0)
        ctr = 0.5 * (lo + hi)
        rad = float(AbstractTensor.max(hi - lo) * 0.6 + 1e-3)
        self.ax.set_xlim(ctr[0] - rad, ctr[0] + rad)
        self.ax.set_ylim(ctr[1] - rad, ctr[1] + rad)
        self.ax.set_zlim(ctr[2] - rad, ctr[2] + rad)

    def _update(self, _frame):
        nodes, edges, bset = self._snapshot()
        ids = AbstractTensor.array(sorted(nodes.keys()))
        P = AbstractTensor.stack([nodes[i][0] for i in ids])
        thetas = AbstractTensor.array([nodes[i][1] for i in ids])
        is_b = AbstractTensor.array([i in bset for i in ids])

        # update node colors/positions
        self.norm_nodes.vmin = min(self.norm_nodes.vmin, float(AbstractTensor.min(thetas)))
        self.norm_nodes.vmax = max(self.norm_nodes.vmax, float(AbstractTensor.max(thetas)))
        C_all = self.node_cmap(self.norm_nodes(thetas))

        Pn = P[~is_b]; Cn = C_all[~is_b]
        Pb = P[is_b];  Cb = C_all[is_b]

        if self.scat_nodes is not None:
            self.scat_nodes._offsets3d = (Pn[:,0], Pn[:,1], Pn[:,2]) if Pn.size else ([], [], [])
            self.scat_nodes.set_facecolors(Cn if Pn.size else [])
        if self.scat_bounds is not None:
            self.scat_bounds._offsets3d = (Pb[:,0], Pb[:,1], Pb[:,2]) if Pb.size else ([], [], [])
            self.scat_bounds.set_facecolors(Cb if Pb.size else [])

        # redraw edges fresh (edge set can change)
        for L in self.edge_artists:
            L.remove()
        self.edge_artists = []
        U_vals = []
        segs = []
        for e in edges:
            U, (pi, pj) = self._compute_edge_energy(e, nodes)
            U_vals.append(U); segs.append((pi, pj))
        if U_vals:
            U_vals = AbstractTensor.array(U_vals)
            lo = float(AbstractTensor.percentile(U_vals, 5)); hi = float(AbstractTensor.percentile(U_vals, 95))
            if hi <= lo: hi = lo + 1e-12
            self.norm_edges.vmin = lo; self.norm_edges.vmax = hi
            for (pi, pj), U in zip(segs, U_vals):
                col = self.edge_cmap(self.norm_edges(U))
                art, = self.ax.plot([pi[0], pj[0]], [pi[1], pj[1]], [pi[2], pj[2]],
                                    lw=1.5, color=col, alpha=0.9)
                self.edge_artists.append(art)

        self._autoscale(P)
        return self.edge_artists + ([self.scat_nodes] if self.scat_nodes is not None else []) + \
               ([self.scat_bounds] if self.scat_bounds is not None else [])

    def launch(self):
        if self.fig is None:
            self._init_fig()
            self._init_artists()
            # Single persistent window; timer-driven updates.
            self.anim = FuncAnimation(self.fig, self._update, interval=self.interval_ms, blit=False)
            plt.show(block=False)

    def step(self, dt: float = 0.001):
        # keep GUI responsive from your main loop
        plt.pause(dt)

    def close(self):
        try:
            plt.close(self.fig)
        except Exception:
            pass

def fwd_plus(sys, a, b, out):
    pa, pb = sys.nodes[a].p, sys.nodes[b].p
    wa, wb = norm_weights([inv_weight(sys, a, out), inv_weight(sys, b, out)])
    yv = wa * pa + wb * pb
    return yv, interpret_vec(yv), (wa, wb)

def fwd_mul(sys, a, b, out):
    pa, pb = sys.nodes[a].p, sys.nodes[b].p
    wa, wb = inv_weight(sys, a, out), inv_weight(sys, b, out)
    pa_w, pb_w = wa * pa, wb * pb
    yv = pa_w * pb_w  # elementwise
    return yv, interpret_vec(yv), (pa_w, pb_w)

def fwd_gather(sys, srcs, out):
    ps = [sys.nodes[s].p for s in srcs]
    ws = norm_weights([inv_weight(sys, s, out) for s in srcs])
    yv = sum(w * p for w, p in zip(ws, ps))
    return yv, interpret_vec(yv), ws

def fwd_scatter(sys, src, outs):
    p = sys.nodes[src].p
    ws = [inv_weight(sys, src, o) for o in outs]
    wn = norm_weights(ws)
    yvs = [w * p for w in wn]
    ys  = [interpret_vec(v) for v in yvs]
    return yvs, ys, wn

# ----------------------------- Operators ------------------------------------
# Each op consumes source nodes and writes a single output node's scalar.
# It also emits per-edge impulses proportional to local derivative * residual.
# If residual is None, we print a WARNING and emit no impulses.

class Ops:
    _cache = WhiteboardCache()
    @staticmethod
    def _need_residual_warn(op_name: str):
        print(f"[WARNING] {op_name}: residual required for impulse; skipping impulses.")

    @staticmethod
    def call(sys, op_name: str, src_ids, out_id, *, residual=None, scale=1.0,
             write_out: bool = True, weight: str = "none"):
        y = push_impulses_from_op_v2(
            sys,
            op_name,
            src_ids,
            out_id,
            residual=residual,
            scale=scale,
            weight=weight,
            cache=Ops._cache,
        )
        if write_out and out_id in sys.nodes:
            sys.nodes[out_id].theta = y
        return y

# ----------------------------- Threads --------------------------------------

class Reflector(threading.Thread):
    def __init__(self, sys: SpringRepulsorSystem, stop: threading.Event,
                 tick_hz: float = 50.0, commit_every_s: float = 0.25):
        super().__init__(daemon=True)
        self.sys = sys
        self.stop = stop
        self.tick_dt = 1.0 / tick_hz
        self.commit_every_s = commit_every_s

    def run(self):
        t_last = now_s()
        t_commit = now_s()
        while not self.stop.is_set():
            t0 = now_s()
            self.sys.tick()
            if (t0 - t_commit) >= self.commit_every_s:
                self.sys.commit_some(self.commit_every_s)
                t_commit = t0
            # Sleep to maintain approx tick rate
            elapsed = now_s() - t0
            to_sleep = max(0.0, self.tick_dt - elapsed)
            #print("Reflector tick")
            time.sleep(to_sleep)


class Experiencer(threading.Thread):
    def __init__(self, sys: SpringRepulsorSystem, stop: threading.Event,
                 outputs: Dict[int, Callable[[float], float]],
                 schedule_hz: float = 30.0,
                 ops_program: Optional[List[Tuple[str, List[int], int]]] = None):
        super().__init__(daemon=True)
        self.sys = sys
        self.stop = stop
        self.dt = 1.0 / schedule_hz
        self.outputs = outputs
        # If none provided, fall back to the tiny demo. (We’ll pass one in.)
        self.ops_program = ops_program

    def _residual_for_out(self, out_id: int, y_val: float, t: float) -> Optional[float]:
        if out_id not in self.outputs:
            return None
        target = self.outputs[out_id](t)
        return y_val - target

    def run(self):
        t0 = now_s()
        while not self.stop.is_set():
            t = now_s() - t0
            # 1) Update intermediate mul ops (write outputs), no impulses
            for (name, srcs, out) in self.ops_program:
                if str(name).lower() in ("mul", "prod", "mul2", "prod_k"):
                    _ = Ops.call(self.sys, name, srcs, out, residual=None, write_out=True, scale=0.0)

            # 2) Batched forward + impulses for outputs with targets
            out_specs: list[tuple[str, list[int], int]] = []
            for (name, srcs, out) in self.ops_program:
                if out in self.outputs:
                    out_specs.append((name, srcs, out))
            if out_specs:
                ys_hat = batched_forward_v2(self.sys, out_specs, weight=None, scale=0.1)
                residuals: list[float] = []
                for (name, srcs, out), y_hat in zip(out_specs, ys_hat):
                    target = self.outputs.get(out)
                    r = float(y_hat) - float(target(t)) if target is not None else 0.0
                    residuals.append(r)
                ys = push_impulses_from_ops_batched(self.sys, out_specs, residuals, weight=None, scale=0.1)
                for (name, srcs, out), y in zip(out_specs, ys):
                    self.sys.nodes[out].theta = float(y)
            time.sleep(self.dt)


# ----------------------------- Demo -----------------------------------------

# ----- linear_block_factory.py -----
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

@dataclass
class LinearBlock:
    base_id: int
    in_ids: List[int]
    out_ids: List[int]
    w_ids: Dict[Tuple[int, int], int]   # (i,j) -> node id
    b_ids: Dict[int, int]               # j -> node id
    mid_ids: Dict[Tuple[int, int], int] # (i,j) -> node id
    nodes: List[Node]
    edges: List[Edge]
    ops: List[Tuple[str, List[int], int]]  # (op_name, src_ids, out_id)

class LinearBlockFactory:
    def __init__(self, n_in: int, n_out: int, *, spacing: float = 0.35, seed: int = 0):
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.spacing = float(spacing)
        self.rng = AbstractTensor.random.default_rng(seed)

    def _mk_edge(self, i, j, op):
        bands = [
            EdgeBand(tau=0.05, K=3.0, kappa=0.12, th=0.05, ratio=3.0, knee=0.05, alpha=1.0),
            EdgeBand(tau=0.25, K=2.0, kappa=0.08, th=0.10, ratio=2.0, knee=0.10, alpha=0.5),
            EdgeBand(tau=1.00, K=1.0, kappa=0.04, th=0.20, ratio=1.2, knee=0.20, alpha=0.2),
        ]
        return Edge(key=(i, j, op), i=i, j=j, op_id=op, bands=bands, l0=1.0)

    def build(self, start_id: int = 0, *, z_level: float = 0.0) -> LinearBlock:
        nid = int(start_id)
        nodes: List[Node] = []
        edges: List[Edge] = []
        ops: List[Tuple[str, List[int], int]] = []

        # --- allocate IDs ---
        in_ids  = [nid + k for k in range(self.n_in)]; nid += self.n_in
        out_ids = [nid + k for k in range(self.n_out)]; nid += self.n_out
        # weights (i,j)
        w_ids: Dict[Tuple[int,int], int] = {}
        for j in range(self.n_out):
            for i in range(self.n_in):
                w_ids[(i, j)] = nid; nid += 1
        # biases (j)
        b_ids: Dict[int, int] = {j: (nid + j) for j in range(self.n_out)}
        nid += self.n_out
        # intermediates m_ij = in_i * w_ij
        mid_ids: Dict[Tuple[int,int], int] = {}
        for j in range(self.n_out):
            for i in range(self.n_in):
                mid_ids[(i, j)] = nid; nid += 1

        # --- place nodes in 3D for clarity (inputs left, mids center slab, outs right) ---
        def jitter(s=0.07): return self.rng.uniform(-s, s, size=(3,))
        # inputs
        for k, i_id in enumerate(in_ids):
            x = -2.0; y = (k - 0.5*(self.n_in-1)) * self.spacing; z = z_level
            nodes.append(Node(id=i_id, theta=self.rng.uniform(-0.1,0.1),
                              p=AbstractTensor.array([x,y,z]) + jitter(), v=AbstractTensor.zeros(3)))
        # outputs
        for k, o_id in enumerate(out_ids):
            x = +2.0; y = (k - 0.5*(self.n_out-1)) * self.spacing; z = z_level
            nodes.append(Node(id=o_id, theta=self.rng.uniform(-0.1,0.1),
                              p=AbstractTensor.array([x,y,z]) + jitter(), v=AbstractTensor.zeros(3)))
        # weights near the mids
        for (i,j), w_id in w_ids.items():
            x = -0.6; y = (j - 0.5*(self.n_out-1)) * self.spacing + 0.05*self.rng.standard_normal()
            z = z_level + 0.15*self.rng.standard_normal()
            nodes.append(Node(id=w_id, theta=self.rng.uniform(-0.3,0.3),
                              p=AbstractTensor.array([x,y,z]) + jitter(), v=AbstractTensor.zeros(3)))
        # biases near output column
        for j, b_id in b_ids.items():
            x = +1.1; y = (j - 0.5*(self.n_out-1)) * self.spacing + 0.03*self.rng.standard_normal()
            z = z_level + 0.15*self.rng.standard_normal()
            nodes.append(Node(id=b_id, theta=self.rng.uniform(-0.1,0.1),
                              p=AbstractTensor.array([x,y,z]) + jitter(), v=AbstractTensor.zeros(3)))
        # intermediates column
        for (i,j), m_id in mid_ids.items():
            x = +0.3; y = (j - 0.5*(self.n_out-1)) * self.spacing + 0.02*(i - 0.5*(self.n_in-1))
            z = z_level + 0.05*self.rng.standard_normal()
            nodes.append(Node(id=m_id, theta=0.0,
                              p=AbstractTensor.array([x,y,z]) + jitter(), v=AbstractTensor.zeros(3)))

        # --- wire edges & ops: m_ij = in_i * w_ij ; out_j = sum_i m_ij + b_j ---
        # mul edges + ops
        for j in range(self.n_out):
            for i in range(self.n_in):
                ii = in_ids[i]; wij = w_ids[(i,j)]; mij = mid_ids[(i,j)]
                edges.append(self._mk_edge(ii,  mij, "mul"))
                edges.append(self._mk_edge(wij, mij, "mul"))
                ops.append(("mul", [ii, wij], mij))
        # gather edges + ops per output j
        for j in range(self.n_out):
            srcs = [mid_ids[(i,j)] for i in range(self.n_in)] + [b_ids[j]]
            oj = out_ids[j]
            for s in srcs:
                edges.append(self._mk_edge(s, oj, "gather_and"))
            ops.append(("gather_and", srcs, oj))

        return LinearBlock(
            base_id=start_id,
            in_ids=in_ids,
            out_ids=out_ids,
            w_ids=w_ids,
            b_ids=b_ids,
            mid_ids=mid_ids,
            nodes=nodes,
            edges=edges,
            ops=ops,
        )

# ---------- convenience: make constant byte targets for a phrase ----------
def ascii_targets_for(phrase: str, out_ids: List[int]) -> Dict[int, Callable[[float], float]]:
    # scale bytes [0..255] → [-1,1] for your scalar residuals
    vals = [ord(c) for c in phrase]
    def scale(v): return (v / 127.5) - 1.0
    return {oid: (lambda t, v=scale(vals[k % len(vals)]): float(v)) 
            for k, oid in enumerate(out_ids)}



def build_toy_system(seed=0):
    rng = AbstractTensor.random.default_rng(seed)
    nodes = []   # <- list, not dict
    edges = []   # <- list, not dict
    outputs = {}

    TEXT = "I am one million monkeys typing on a keyboard"

    lb = LinearBlockFactory(
        n_in=8,
        n_out=len(TEXT),
        spacing=0.28,
        seed=123
    ).build(start_id=0, z_level=0.0)

    # Install the block
    nodes.extend(lb.nodes)
    edges.extend(lb.edges)

    sys = SpringRepulsorSystem(nodes, edges, eta=0.08, gamma=0.93, dt=0.02)

    # Drive inputs
    def sin_at(freq, amp=0.4):
        def _s(t, f=freq, a=amp):
            return float((AbstractTensor.get_tensor(f * t).sin() * a).item())
        return _s
    freqs = AbstractTensor.linspace(0.3, 1.1, len(lb.in_ids))
    for nid, f in zip(lb.in_ids, freqs):
        sys.add_boundary(BoundaryPort(nid=nid, beta=0.8, force_fn=as_x_force(sin_at(f), D=sys.D)))

    # ASCII targets
    byte_targets = ascii_targets_for(TEXT, lb.out_ids)
    outputs.update(byte_targets)

    # Wire the op program
    sys.ops_program = lb.ops

    return sys, outputs



def main(duration_s: float = 8.0):
    sys, outputs = build_toy_system(seed=42)

    stop = threading.Event()
    refl = Reflector(sys, stop, tick_hz=30.0, commit_every_s=1.00)
    expr = Experiencer(sys, stop, outputs, schedule_hz=60.0, ops_program=sys.ops_program)

    print("[INFO] Starting threads…")
    refl.start(); expr.start()
    t0 = now_s()
#    viz = LiveViz3D(sys)
#    viz.launch()
    viz = LiveVizGLPoints(sys, node_cmap="coolwarm", base_point_size=6.0)
    viz.launch()              # once

    try:
        t0 = now_s()
        while (now_s() - t0) < duration_s:
            t = now_s() - t0
            # sample first few outputs
            sample = list(outputs.keys())
            errors = [sys.nodes[oid].theta - outputs[oid](t) for oid in sample]
            mae = AbstractTensor.mean([abs(err) for err in errors])
            error_str = ''.join([f"{chr(int((err + 1) * 127.5))}" for err in errors])
            print(f"[DBG] outputs MAE (first {len(sample)} chars): {mae: .3f}, Error Phrase: {error_str}")
            viz.step(0.5)

    finally:
        stop.set()
        expr.join(timeout=2.0)
        refl.join(timeout=2.0)
        print("[INFO] Stopped.")
        viz.close()

if __name__ == "__main__":
    main(600.0)
