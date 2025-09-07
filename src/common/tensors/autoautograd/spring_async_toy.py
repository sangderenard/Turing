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
    push_impulses_from_ops_batched,
)
from .whiteboard_cache import WhiteboardCache

import time
import threading
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Callable
import matplotlib

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import colors as mcolors
from ..abstraction import AbstractTensor
from ..filtered_poisson import filtered_poisson

L_MIN = 0.05
L_MAX = 3.0
DL_CAP = 0.5
MAGNITUDE_ONLY = False
V_MAX = 2.0
STEP_MAX = 10.2
READOUT_SCALE = 1.0
READOUT_BIAS  = 0.0
W_EPS = 1e-3
W_MIN, W_MAX = 0.25, 4.0 
# --- threshold-pop optimizer knobs ---
POP_FRUSTRATION_TH = 1e-6   # how 'annoyed' an edge must be: |L - L*|
POP_AGG_TH         = 1e-6   # how large the transient aggregate must be: |sum(contribs)|
POP_QUANTUM        = 1e-2   # discrete ΔL moved per pop
POP_MAX_PER_TICK   = 1e10      # safety cap per integrator tick

# ----------------------------- Utilities ------------------------------------

def now_s() -> float:
    return time.perf_counter()

def as_axis_target(fn, axis: int, D: int = 3):
    """Scalar → Dirichlet target on ``axis`` in D-D space."""

    def _t(t):
        v = AbstractTensor.zeros(D, dtype=float)
        v[axis] = float(fn(t))
        return v

    return _t


def as_axis_force(fn, axis: int, D: int = 3):
    """Scalar → Neumann force on ``axis`` in D-D space."""

    def _f(t):
        v = AbstractTensor.zeros(D, dtype=float)
        v[axis] = float(fn(t))
        return v

    return _f


def as_x_target(fn, D: int = 3):
    return as_axis_target(fn, 0, D)


def as_x_force(fn, D: int = 3):
    return as_axis_force(fn, 0, D)

def inv_weight(sys: SpringRepulsorSystem, src: int, dst: int) -> float:
    d = float(AbstractTensor.linalg.norm(sys.nodes[dst].p - sys.nodes[src].p))
    w = 1.0 / (W_EPS + d)
    return float(AbstractTensor.get_tensor(w).clip(W_MIN, W_MAX))

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

# ----------------------------- Boundary helpers ------------------------------

def attach_dirichlet(
    sys: 'SpringRepulsorSystem',
    nid: int,
    value_fn: Callable[[float], float],
    *,
    D: Optional[int] = None,
    alpha: float = 2.0,
    axis: int = 0,
) -> None:
    """Clamp only the chosen axis of ``nid`` toward value_fn(t); leave other axes untouched."""
    D = sys.D if D is None else int(D)

    def _target_vec(t, _sys=sys, _nid=nid, _axis=axis):
        # copy current position so non-target axes see zero Dirichlet force
        p_now = _sys.nodes[_nid].p.clone()
        p_now[_axis] = float(value_fn(t))
        return p_now

    sys.add_boundary(BoundaryPort(nid=nid, alpha=alpha, target_fn=_target_vec))


def attach_neumann_noop(
    sys: 'SpringRepulsorSystem',
    nid: int,
    *,
    D: Optional[int] = None,
    beta: float = 1.0,
    axis: int = 0,
) -> None:
    """Attach a traction-capable boundary that exerts no force (pass-through)."""

    D = sys.D if D is None else int(D)
    sys.add_boundary(
        BoundaryPort(nid=nid, beta=beta, force_fn=as_axis_force(lambda _t: 0.0, axis, D))
    )

def _fresh_node_id(sys: 'SpringRepulsorSystem') -> int:
    return (max(sys.nodes.keys()) + 1) if sys.nodes else 0

def enliven_feature_edges(sys: SpringRepulsorSystem, in_ids: List[int], out_ids: List[int]):
    # seed physical ties so forces/learning can flow
    for i in in_ids:
        for o in out_ids:
            sys.ensure_edge(i, o, "feat")  # op_id tag just to distinguish
def empty_param_generator():
    while True:
        yield AbstractTensor.get_tensor([0.0, 1.0, 0.0])

def wire_input_chain(
    sys: "SpringRepulsorSystem",
    system_nid: int,
    sample_fn: Callable[[float], float],
    *,
    D: Optional[int] = None,
    alpha: float = 2.0,
    beta: float = 1.0,
) -> int:
    """Create Dirichlet→Neumann chain feeding into an existing system node."""
    AT = AbstractTensor
    D = sys.D if D is None else int(D)
    d = _fresh_node_id(sys)
    p = AT.zeros(D, dtype=float)
    param_generator = empty_param_generator()
    param = param_generator.__next__()
    sys.nodes[d] = Node(
        id=d,
        param=param,
        p=p,
        v=AT.zeros(D, dtype=float),
        sphere=AbstractTensor.concat([p, param], dim=0),
    )
    attach_dirichlet(sys, d, sample_fn, D=D, alpha=alpha)
    n = _fresh_node_id(sys)
    p = AT.zeros(D, dtype=float)
    param = param_generator.__next__()
    sys.nodes[n] = Node(
        id=n,
        param=param,
        p=p,
        v=AT.zeros(D, dtype=float),
        sphere=AbstractTensor.concat([p, param], dim=0),
    )
    attach_neumann_noop(sys, n, D=D, beta=beta)
    sys.ensure_edge(d, n, "in_link")
    sys.ensure_edge(n, system_nid, "in_link")
    return n


def wire_output_chain(
    sys: "SpringRepulsorSystem",
    system_nid: int,
    target_fn: Callable[[float], float],
    *,
    D: Optional[int] = None,
    beta: float = 1.0,
    alpha: float = 2.0,
) -> int:
    """Create system_output→Neumann→Dirichlet chain anchored to `target_fn`."""
    AT = AbstractTensor
    D = sys.D if D is None else int(D)
    n = _fresh_node_id(sys)
    p = AT.zeros(D, dtype=float)
    param_generator = empty_param_generator()
    param = param_generator.__next__()
    sys.nodes[n] = Node(
        id=n,
        param=param,
        p=p,
        v=AT.zeros(D, dtype=float),
        sphere=AbstractTensor.concat([p, param], dim=0),
    )
    attach_neumann_noop(sys, n, D=D, beta=beta)
    sys.ensure_edge(system_nid, n, "out_link")
    d = _fresh_node_id(sys)
    p = AT.zeros(D, dtype=float)
    param = param_generator.__next__()
    sys.nodes[d] = Node(
        id=d,
        param=param,
        p=p,
        v=AT.zeros(D, dtype=float),
        sphere=AbstractTensor.concat([p, param], dim=0),
    )
    attach_dirichlet(sys, d, target_fn, D=D, alpha=alpha)
    sys.ensure_edge(n, d, "readout")
    return n

# ----------------------------- Core data ------------------------------------

@dataclass
class Node:
    id: int
    param: AbstractTensor  # scalar parameter
    p: AbstractTensor  # shape (D,)
    v: AbstractTensor  # shape (D,)
    sphere: AbstractTensor
    M0: float = 10.0
    last_commit: float = 0.0
    version: int = 0
    hist_p: deque = field(default_factory=lambda: deque(maxlen=128))

    def commit(self, param_residuals=None):
        self.sphere = AbstractTensor.concat([self.p, self.param], dim=0)
        if param_residuals is not None:
            self.param += param_residuals
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
            b.s += (AbstractTensor.sign(AbstractTensor.get_tensor(g_scalar)) - b.s) * (dt / max(b.tau, 1e-6))
            # Also push a composite contribution (post-knee)
            # AFTER
            y = soft_knee(b.m, b.th, b.ratio, b.knee)
            sgn = 1.0 if MAGNITUDE_ONLY else AbstractTensor.sign(b.s)
            dl = float(AbstractTensor.get_tensor(b.kappa * sgn * y).clip(-DL_CAP, DL_CAP))
            self.spring.add(dl)



    # --- runtime (internal) ---
    _last_reduce: float = 0.0
    pops: int = 0  # diagnostics

    def target_length(self) -> float:
        # cache current transient aggregate, then clamp total target length
        agg = self.spring.reduce()
        self._last_reduce = float(agg)
        return float(AbstractTensor.get_tensor(self.l0 + agg).clip(L_MIN, L_MAX))

    def maybe_pop(self, L_current: float, quantized=False):
        """
        If both frustration and aggregate exceed thresholds, do 'integer' pops:
          l0 += sgn * POP_QUANTUM
          spring.consume by adding an equal-and-opposite transient (-sgn * POP_QUANTUM)
        """
        # current target uses cached _last_reduce set by target_length() this tick
        L_star = float(AbstractTensor.get_tensor(self.l0 + self._last_reduce).clip(L_MIN, L_MAX))
        frustration = abs(L_current - L_star)
        agg_mag = abs(self._last_reduce)
        if frustration < POP_FRUSTRATION_TH or agg_mag < POP_AGG_TH:
            #return  # nothing to do
            pass #diagnostic

        # how many quanta could we commit right now?
        want = min(int(agg_mag // POP_QUANTUM), POP_MAX_PER_TICK)
        if want <= 0:
            #return
            pass #diagnostic

        sgn = 1.0 if self._last_reduce >= 0.0 else -1.0
        for _ in range(1):#want):
            if False and quantized:
                # commit ΔL into the permanent base
                self.l0 += sgn * POP_QUANTUM
                # consume the same amount from the transient buffer immediately
                self.spring.add(-sgn * POP_QUANTUM)
            else:
                # fractional pop (for testing)
                delta = sgn * agg_mag #min(POP_QUANTUM, agg_mag)
                self.l0 += delta
                #self.spring.add(-delta)
                agg_mag -= delta
                if agg_mag <= 0.0:
                    break
            
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
        # loop-back loss plumbing (optional) – support multiple edges
        self.feedback_edges: List[Tuple[int, int, str]] = []

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

    def impulse_batch(self, src_ids, dst_id: int, op_id: str, g_scalars):
        g_vals = AbstractTensor.get_tensor(g_scalars).reshape(-1)
        for i, g in zip(src_ids, g_vals):
            try:
                g_val = float(getattr(g, "item", lambda: g)())
            except Exception:
                g_val = float(g)
            self.impulse(int(i), int(dst_id), op_id, g_val)

    def add_feedback_edge(self, src: int, dst: int, op_id: str = "loss_fb") -> None:
        """Register an edge that will receive global loss impulses."""
        self.feedback_edges.append((int(src), int(dst), str(op_id)))

    # Backwards compatibility helper
    def set_feedback_edge(self, src: int, dst: int, op_id: str = "loss_fb") -> None:
        """Reset feedback edges to a single connection."""
        self.feedback_edges = [(int(src), int(dst), str(op_id))]

    # ----------------- Physics tick -----------------
    def tick(self):
        # Force accumulator
        F: Dict[int, AbstractTensor] = {i: AbstractTensor.zeros(self.D, dtype=float) for i in self.nodes}


        t_now = now_s()
        scale = -1.0
        for key, e in self.edges.items():
            ni, nj = self.nodes[e.i], self.nodes[e.j]
            d = nj.p - ni.p
            L = AbstractTensor.linalg.norm(d)
            if L < 1e-9:
                continue
            u = d / (L + 1e-12)
            Lstar = e.target_length()  # also caches _last_reduce
            Ksum = AbstractTensor.get_tensor([b.K for b in e.bands]).sum()
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
        xs = xs - xs.mean(dim=0, keepdim=True)
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
        # Frequency bins for real FFT with sample step dt, matched to xs backend
        w0 = 2.0 * AbstractTensor.pi() * AbstractTensor.fft.rfftfreq(int(W), d=dt, like=xs)  # (F0,)
        P0 = AbstractTensor.sum(AbstractTensor.abs(C0)**2, dim=1)           # (F0,)
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
        # Pad along time axis (axis 0) by Wz - W zeros at the end
        # Note: numpy backend pad_ consumes flattened pads in reverse axis order
        # so we pass (0,0, 0,Wz-W) to target axis0 only.
        xpad = AbstractTensor.pad(xw, (0, 0, 0, Wz - W))
        Cz = AbstractTensor.fft.rfft(xpad, axis=0)                 # (Fz, D)
        wz = 2.0 * AbstractTensor.pi() * AbstractTensor.fft.rfftfreq(Wz, d=dt, like=xs)   # (Fz,)

        # helper to map coarse indices to high-res frequency indices
        def coarse_band_to_w(b_lo, b_hi):
            return w0[b_lo], w0[min(b_hi, len(w0)-1)]

        def w_to_hi_idx(wlo, whi):
            i0 = AbstractTensor.get_tensor(AbstractTensor.searchsorted(wz, wlo, side="left")).clip(0, len(wz)-1)
            i1 = AbstractTensor.get_tensor(AbstractTensor.searchsorted(wz, whi, side="right")).clip(0, len(wz))
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
            Pw = AbstractTensor.sum(AbstractTensor.abs(Cz_band)**2, dim=1) + 1e-12  # (Fb,)
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
# - Edges rendered as GL_LINES with spring energy colormap
# - Autoscaled camera; non-blocking window


import math
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, RESIZABLE, VIDEORESIZE, QUIT, KEYDOWN, K_SPACE
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from matplotlib import cm, colors as mcolors
from typing import Any, Tuple
# spring_async_toy.py (top-level, before creating LiveVizGLPoints)
from ..pyopengl_handler import install_pyopengl_handlers
install_pyopengl_handlers()

# Expecting SpringRepulsorSystem with:
#   self.nodes: Dict[int, Node] where Node.p is (3,) ndarray-like and Node.param is scalar
#   self.boundaries: Dict[int, Any] (keys are boundary node ids)
#   self.edges: Dict[Tuple[int, int, str], Edge] spring edges rendered as GL_LINES

class LiveVizGLPoints:
    def __init__(self,
                 sys,
                 node_cmap: str = "coolwarm",
                 edge_cmap: str = "coolwarm",
                 base_point_size: float = 6.0,
                 boundary_scale: float = 1.3,
                 bg_color: Tuple[float, float, float] = (0.04, 0.04, 0.06)):
        self.sys = sys
        self.node_cmap = matplotlib.colormaps.get_cmap(node_cmap)
        self.edge_cmap = matplotlib.colormaps.get_cmap(edge_cmap)
        self.base_point_size = float(base_point_size)
        self.role_palette = {
            "input":    mcolors.to_rgb("#14b8a6"),  # teal-500
            "neumann":  mcolors.to_rgb("#f59e0b"),  # amber-500
            "output":   mcolors.to_rgb("#d946ef"),  # fuchsia-500
            "anchor":   mcolors.to_rgb("#10b981"),  # emerald-500
            "dirichlet":mcolors.to_rgb("#10b981"),
        }
        self.role_size_gain = {
            "input":   1.3,
            "neumann": 1.6,
            "output":  1.8,
            "anchor":  2.2,
            "dirichlet": 2.0,
        }
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
        self._edge_vao = None
        self._edge_vbo_pos = None
        self._edge_vbo_col = None
        self._edge_vbo_size = None
        self._num_edge_vertices = 0
        self._cap_pos = self._cap_col = self._cap_size = 0
        self._edge_cap_pos = self._edge_cap_col = self._edge_cap_size = 0

        self._u_mvp = None  # uniform location
        tensor = AbstractTensor.get_tensor(0)
        dtype = tensor.get_dtype()
        self._mvp = AbstractTensor.eye(4, dtype=dtype)  # updated each frame

        # optional wandering camera
        self._auto_rotate = False
        self._rot_theta = 0.0
        self._rot_phi = 0.0
        self._rot_dtheta = 0.002
        self._rot_dphi = 0.0015
    def _upload(self, vbo, arr, cap_attr, usage=GL_DYNAMIC_DRAW):
        import numpy as np
        # bytes & pointer for numpy vs AbstractTensor
        if isinstance(arr, np.ndarray):
            nbytes = int(arr.nbytes); ptr = arr
        else:
            nbytes = int(arr.nbytes()); ptr = arr.data

        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        cap = getattr(self, cap_attr, 0)

        # grow (and orphan) only when needed
        if nbytes > cap:
            new_cap = max(int(nbytes * 1.5), 256)
            glBufferData(GL_ARRAY_BUFFER, new_cap, None, usage)  # allocate/orphan
            setattr(self, cap_attr, new_cap)

        glBufferSubData(GL_ARRAY_BUFFER, 0, nbytes, ptr)         # upload bytes

    # ---------- data snapshot ----------
    def _snapshot(self):
        # lock-free minimal copy
        nodes = {
            i: (n.p.clone(), n.param.clone())
            for i, n in self.sys.nodes.items()
        }
        edges = list(self.sys.edges.values())
        bset = set(self.sys.boundaries.keys())
        return nodes, edges, bset

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

        # edge buffers
        self._edge_vao = glGenVertexArrays(1)
        glBindVertexArray(self._edge_vao)

        self._edge_vbo_pos = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._edge_vbo_pos)
        glBufferData(GL_ARRAY_BUFFER, 1, None, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        self._edge_vbo_col = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._edge_vbo_col)
        glBufferData(GL_ARRAY_BUFFER, 1, None, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)

        self._edge_vbo_size = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._edge_vbo_size)
        glBufferData(GL_ARRAY_BUFFER, 1, None, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(2)

        glBindVertexArray(0)

    # ---------- geometry packing ----------
    def _pack_points(self):
        nodes, _, _ = self._snapshot()
        ids = AbstractTensor.get_tensor(sorted(nodes.keys()))
        if ids.shape == (0,):
            return (AbstractTensor.zeros((0, 3), ids.float_dtype),
                    AbstractTensor.zeros((0, 3), ids.float_dtype),
                    AbstractTensor.zeros((0,), ids.float_dtype),
                    AbstractTensor.zeros((0, 3), ids.float_dtype))

        P = AbstractTensor.stack([nodes[i][0] for i in ids])
        # NEW: pad to 3D if needed
        if P.shape[1] == 2:
            # Add one column (axis 1) of zeros to promote (N,2) -> (N,3)
            # Backend expects flattened pads in reverse axis order: (axis1, axis0)
            P = AbstractTensor.pad(P, (0, 1, 0, 0), value=0.0)

        # NEW: replace NaN/Inf early to avoid NaN bounds
        P = AbstractTensor.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)

        # --- existing params & base colormap (KEEP this: learning nodes stay coolwarm) ---
        params = AbstractTensor.get_tensor([nodes[i][1][1] for i in ids])
        
        vmin = AbstractTensor.min(params)
        vmax = AbstractTensor.max(params)
        if vmin > 0.0:
            vmin = 0.0 - 1e-6
        if vmax <= vmin:
            vmax = vmin + 1e-6
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        import numpy as np
        C = self.node_cmap(norm(params))[:, :3].astype(np.float32)  # (N,3) RGB

        # --- sizes (base) ---
        sizes = AbstractTensor.full(
            ids.shape,
            self.base_point_size,
            dtype=ids.float_dtype,
            device=ids.get_device(),
            cls=type(ids),
        )

        # --- role inference ---
        roles_map = getattr(self.sys, "roles", {})
        roles = [roles_map.get(int(i), "") for i in ids]

        # auto-tag boundary *type* if no explicit role:
        btypes = {}
        for nid, port in self.sys.boundaries.items():
            if port.enabled:
                if port.beta > 0.0 and port.alpha <= 0.0:
                    btypes[nid] = "neumann"
                elif port.alpha > 0.0 and port.beta <= 0.0:
                    btypes[nid] = "dirichlet"
                elif port.alpha > 0.0 and port.beta > 0.0:
                    btypes[nid] = "robin"

        # --- apply role color/size overrides ---
        for idx, nid in enumerate(ids):
            rid = int(nid)
            role = roles[idx] or btypes.get(rid, "")
            if role in self.role_palette:
                C[idx] = np.array(self.role_palette[role], dtype=np.float32)
            if role in self.role_size_gain:
                sizes[idx] *= float(self.role_size_gain[role])

        # (Optional) still emphasize anything marked as a boundary, lightly
        bset = set(self.sys.boundaries.keys())
        is_b = AbstractTensor.get_tensor([int(i) in bset for i in ids], dtype=ids.bool_dtype, like=ids)
        is_b = is_b.to_dtype("bool")
        sizes[is_b] *= self.boundary_scale

        return P, C, sizes, P  # return P twice; last is for autoscale

    def _compute_edge_energy(self, e, nodes):
        pi = nodes[e.i][0]
        pj = nodes[e.j][0]
        d = pj - pi
        L = float(AbstractTensor.linalg.norm(d) + 1e-12)
        Lstar = e.target_length()
        Ksum = sum(b.K for b in e.bands)
        return 0.5 * Ksum * (L - Lstar) ** 2, (pi, pj)

    def _pack_edges(self):
        nodes, edges, _ = self._snapshot()
        if not edges:
            AT = AbstractTensor
            return AT.zeros((0, 3), float), AT.zeros((0, 3), float), AT.zeros((0,), float)

        P_segs = []
        U_vals = []
        for e in edges:
            U, (pi, pj) = self._compute_edge_energy(e, nodes)
            P_segs.extend([pi, pj])
            U_vals.append(U)

        AT = AbstractTensor
        P = AT.stack(P_segs).astype(float)
        if P.shape[1] == 2:
            P = AT.pad(P, (0, 1, 0, 0), value=0.0)
        P = AT.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)

        U_vals = AT.get_tensor(U_vals, dtype=float)
        lo = float(AT.percentile(U_vals, 5))
        hi = float(AT.percentile(U_vals, 95))
        if hi <= lo:
            hi = lo + 1e-12
        norm = mcolors.Normalize(vmin=lo, vmax=hi)
        import numpy as np
        colors = self.edge_cmap(norm(U_vals))[:, :3].astype(np.float32)
        C = np.repeat(colors, 2, axis=0)

        S = AT.full((P.shape[0],), 1.0, dtype=float)
        return P, C, S

    def _update_buffers(self):
        P, C, S, P_for_bounds = self._pack_points()
        self._num_points = P.shape[0]
        # points
        self._upload(self._vbo_pos,  P, "_cap_pos")
        self._upload(self._vbo_col,  C, "_cap_col")
        self._upload(self._vbo_size, S, "_cap_size")



        glBindVertexArray(0)

        #PE, CE, SE = self._pack_edges()
        #self._num_edge_vertices = PE.shape[0]

        # edges
        #self._upload(self._edge_vbo_pos, PE, "_edge_cap_pos")
        #self._upload(self._edge_vbo_col, CE, "_edge_cap_col")
        #self._upload(self._edge_vbo_size, SE, "_edge_cap_size")

        #glBindVertexArray(0)

        # update camera
        self._compute_mvp(P_for_bounds)

    # ---------- camera / MVP ----------

    @staticmethod
    def _perspective(fovy_deg, aspect, znear, zfar) -> AbstractTensor:
        f = 1.0 / AbstractTensor.tan(AbstractTensor.deg2rad(fovy_deg) / 2.0)
        M = AbstractTensor.zeros((4, 4), dtype=f.float_dtype)
        M[0, 0] = f / max(aspect, 1e-6)
        M[1, 1] = f
        M[2, 2] = (zfar + znear) / (znear - zfar)
        M[2, 3] = (2 * zfar * znear) / (znear - zfar)
        M[3, 2] = -1.0
        return M


    @staticmethod
    def _look_at(eye, center, up) -> AbstractTensor:
        AT = AbstractTensor

        def _vec3(x):
            t = x.clone()
            # Flatten anything like (1,3), (3,1), (3,3) into (3,)
            if hasattr(t, "shape"):
                if len(t.shape) > 1:
                    t = t.reshape((t.shape[-1],))
            return t

        eye    = _vec3(eye)
        center = _vec3(center)
        up     = _vec3(up)

        f = center - eye
        f = f / (AT.linalg.norm(f) + 1e-12)
        upn = up / (AT.linalg.norm(up) + 1e-12)

        # Manual cross products → guaranteed (3,)
        fx, fy, fz = f[0], f[1], f[2]
        ux, uy, uz = upn[0], upn[1], upn[2]
        s = AT.get_tensor([fy*uz - fz*uy, fz*ux - fx*uz, fx*uy - fy*ux])
        s = s / (AT.linalg.norm(s) + 1e-12)

        sx, sy, sz = s[0], s[1], s[2]
        u = AT.get_tensor([sy*fz - sz*fy, sz*fx - sx*fz, sx*fy - sy*fx])

        # NOTE: pass an int to eye()
        M = AT.eye(4, dtype=s.float_dtype)
        M[0, :3] = s
        M[1, :3] = u
        M[2, :3] = -f

        T = AT.eye(4, dtype=s.float_dtype)
        T[:3, 3] = -eye
        return M @ T

    def _compute_mvp(self, P: AbstractTensor):
        if P.shape == (0,):
            self._mvp = AbstractTensor.eye(4, dtype=P.float_dtype)
            return
        P = AbstractTensor.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)

        lo = AbstractTensor.min(P, dim=0)
        hi = AbstractTensor.max(P, dim=0)
        ctr = 0.5 * (lo + hi)

        # safer extent
        extent_t = AbstractTensor.get_tensor(hi - lo).max()
        try:
            finite = AbstractTensor.get_tensor(extent_t).isfinite()
        except Exception:
            finite = True
        if (not finite) or float(extent_t) <= 1e-6:
            extent_t = AbstractTensor.get_tensor(1.0)

        rad = extent_t * 0.6 + 1e-3
        r = float(rad) * 1.6
        if self._auto_rotate:
            x = r * math.sin(self._rot_phi) * math.cos(self._rot_theta)
            y = r * math.cos(self._rot_phi)
            z = r * math.sin(self._rot_phi) * math.sin(self._rot_theta)
            eye = ctr + AbstractTensor.get_tensor([x, y, z], dtype=P.float_dtype)
        else:
            eye = ctr + AbstractTensor.get_tensor([r, r, r], dtype=P.float_dtype)
        up  = AbstractTensor.get_tensor([0.0, 1.0, 0.0], dtype=P.float_dtype)

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
            if self._edge_vbo_pos: glDeleteBuffers(1, [self._edge_vbo_pos])
            if self._edge_vbo_col: glDeleteBuffers(1, [self._edge_vbo_col])
            if self._edge_vbo_size: glDeleteBuffers(1, [self._edge_vbo_size])
            if self._edge_vao: glDeleteVertexArrays(1, [self._edge_vao])
            if self._program:  glDeleteProgram(self._program)
        except Exception:
            pass
        self._program = None
        self._vao = self._vbo_pos = self._vbo_col = self._vbo_size = None
        self._edge_vao = self._edge_vbo_pos = self._edge_vbo_col = self._edge_vbo_size = None
        self._num_edge_vertices = 0

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
            elif evt.type == KEYDOWN and evt.key == K_SPACE:
                self._auto_rotate = not self._auto_rotate

    def _draw(self):
        r, g, b = self.bg_color
        glClearColor(r, g, b, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(self._program)
        # CHANGED: upload transpose so GLSL sees the right matrix
        import numpy as np
        glUniformMatrix4fv(self._u_mvp, 1, GL_FALSE, np.array(self._mvp.T(), dtype=np.float32))

        glBindVertexArray(self._vao)
        glDrawArrays(GL_POINTS, 0, self._num_points)
        glBindVertexArray(0)

        glBindVertexArray(self._edge_vao)
        glDrawArrays(GL_LINES, 0, self._num_edge_vertices)
        glBindVertexArray(0)

        pygame.display.flip()


    def _update_rotation(self):
        if not self._auto_rotate:
            return
        self._rot_theta += self._rot_dtheta
        self._rot_phi += self._rot_dphi
        if self._rot_phi < 0.0 or self._rot_phi > math.pi:
            self._rot_dphi = -self._rot_dphi
            self._rot_phi = max(0.0, min(math.pi, self._rot_phi))
        self._rot_dtheta += 0.00005 * math.sin(self._rot_phi * 1.7)
        self._rot_dphi   += 0.00005 * math.cos(self._rot_theta * 1.3)
        self._rot_dtheta *= 0.9995
        self._rot_dphi   *= 0.9995

    def step(self, _dt: float = 0.0):
        """Call from your main loop (non-blocking)."""
        if self._program is None:
            # if user forgot to launch, do it lazily
            self.launch(self._w, self._h)
        self._handle_events()
        self._update_rotation()
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
            if self._edge_vbo_pos:
                glDeleteBuffers(1, [self._edge_vbo_pos])
            if self._edge_vbo_col:
                glDeleteBuffers(1, [self._edge_vbo_col])
            if self._edge_vbo_size:
                glDeleteBuffers(1, [self._edge_vbo_size])
            if self._edge_vao:
                glDeleteVertexArrays(1, [self._edge_vao])
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
        nodes = {
            i: (n.p.copy(), float(AbstractTensor.get_tensor(n.param)))
            for i, n in self.sys.nodes.items()
        }
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
        params = AbstractTensor.array([nodes[i][1] for i in ids])
        self.norm_nodes = mcolors.TwoSlopeNorm(vmin=AbstractTensor.min(params), vcenter=0.0, vmax=AbstractTensor.max(params))

        if Pn.size:
            self.scat_nodes = self.ax.scatter(Pn[:,0], Pn[:,1], Pn[:,2],
                                              s=40, marker="o",
                                              c=self.node_cmap(self.norm_nodes(params[~is_b])),
                                              depthshade=False, linewidths=0.5, edgecolors="k")
        if Pb.size:
            self.scat_bounds = self.ax.scatter(Pb[:,0], Pb[:,1], Pb[:,2],
                                               s=90, marker="s",
                                               c=self.node_cmap(self.norm_nodes(params[is_b])),
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
        lo = AbstractTensor.min(P, dim=0); hi = AbstractTensor.max(P, dim=0)
        ctr = 0.5 * (lo + hi)
        rad = float(AbstractTensor.max(hi - lo) * 0.6 + 1e-3)
        self.ax.set_xlim(ctr[0] - rad, ctr[0] + rad)
        self.ax.set_ylim(ctr[1] - rad, ctr[1] + rad)
        self.ax.set_zlim(ctr[2] - rad, ctr[2] + rad)

    def _update(self, _frame):
        nodes, edges, bset = self._snapshot()
        ids = AbstractTensor.array(sorted(nodes.keys()))
        P = AbstractTensor.stack([nodes[i][0] for i in ids])
        params = AbstractTensor.array([nodes[i][1] for i in ids])
        is_b = AbstractTensor.array([i in bset for i in ids])

        # update node colors/positions
        self.norm_nodes.vmin = min(self.norm_nodes.vmin, float(AbstractTensor.min(params)))
        self.norm_nodes.vmax = max(self.norm_nodes.vmax, float(AbstractTensor.max(params)))
        C_all = self.node_cmap(self.norm_nodes(params))

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
             write_out: bool = False, weight: str = "none"):
        y, _ = push_impulses_from_op_v2(
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
            node = sys.nodes[out_id]
            y_t = AbstractTensor.get_tensor(0.0) if y is None else AbstractTensor.get_tensor(y)
            mean_val = y_t.mean() if getattr(y_t, "ndim", 0) > 0 else y_t
            node.p[2] = float(mean_val)
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
    def __init__(
        self,
        sys: SpringRepulsorSystem,
        stop: threading.Event,
        outputs: Dict[int, Callable[[float], AbstractTensor]],
        schedule_hz: float = 30.0,
        ops_program: Optional[
            List[Tuple[str, List[int], int, Optional[Tuple[Any, ...]], Optional[Dict[str, Any]]]]
        ] = None,
    ):
        super().__init__(daemon=True)
        self.sys = sys
        self.stop = stop
        self.dt = 1.0 / schedule_hz
        self.outputs = outputs
        # If none provided, fall back to the tiny demo. (We’ll pass one in.)
        self.ops_program = ops_program

    def _residual_for_out(
        self, out_id: int, y_val: AbstractTensor, t: float
    ) -> Optional[AbstractTensor]:
        if out_id not in self.outputs or y_val is None:
            return None
        target_vec = self.outputs[out_id](t)
        return y_val - target_vec

    def run(self):
        t0 = now_s()
        while not self.stop.is_set():
            t = now_s() - t0
            
            # 2) Batched forward + gradients for outputs with targets
            out_specs: list[tuple[str, list[int], int, Optional[Tuple[Any, ...]], Optional[Dict[str, Any]]]] = []
            for (name, srcs, out, args, kwargs) in self.ops_program:
                if out in self.outputs:
                    out_specs.append((name, srcs, out, args, kwargs))
            residual_map: Dict[int, AbstractTensor] = {}
            if out_specs:
                ys, grads, _ = push_impulses_from_ops_batched(
                    self.sys, out_specs, weight=None, scale=1.0
                )
                for spec, y, g_list in zip(out_specs, ys, grads):
                    name, srcs, out, args, kwargs = spec
                    r = self._residual_for_out(out, y, t)
                    if r is None:
                        continue
                    residual_map[out] = r

            # Residuals for any other nodes with targets (e.g. standalone boundaries)
            for nid, target_fn in self.outputs.items():
                if nid in residual_map:
                    continue
                node = self.sys.nodes.get(nid)
                if node is None:
                    continue
                cur = getattr(node, "param", None)
                if cur is None:
                    cur = getattr(node, "p", None)
                if cur is None:
                    continue
                residual_map[nid] = AbstractTensor.get_tensor(cur) - target_fn(t)

            # Build adjacency among nodes with residuals
            if residual_map:
                nids = list(residual_map.keys())
                idx = {nid: i for i, nid in enumerate(nids)}
                N = len(nids)
                adjacency = AbstractTensor.zeros((N, N), dtype=float)
                for e in self.sys.edges.values():
                    if e.i in idx and e.j in idx:
                        ii, jj = idx[e.i], idx[e.j]
                        adjacency[ii, jj] = adjacency[jj, ii] = 1.0

                # Stack residuals and smooth per-parameter via filtered Poisson
                R = AbstractTensor.stack([residual_map[n] for n in nids], dim=0)

                if True or adjacency.sum() <= 0.0:
                    R_sm = R
                elif R.ndim == 1:
                    # 1D case: smooth directly
                    R_sm = filtered_poisson(R, iterations=20, adjacency=adjacency)
                else:
                    F = int(R.shape[1])
                    if F == 0:
                        # Nothing to smooth — keep originals; do not overwrite with empties
                        R_sm = R
                    else:
                        cols = []
                        for k in range(F):
                            col = filtered_poisson(R[:, k], iterations=20, adjacency=adjacency)
                            cols.append(col)
                        # Ensure row-major (N, F)
                        R_sm = AbstractTensor.stack(cols, dim=1)

                # Only overwrite residual_map if shapes align and we actually have features
                if getattr(R_sm, "ndim", 0) >= 1 and int(R_sm.shape[0]) == len(nids) and (R_sm.ndim == 1 or int(getattr(R_sm, "shape", (0, 0))[1]) > 0):
                    for nid, r_sm in zip(nids, R_sm):
                        residual_map[nid] = r_sm
                # else: keep the pre-smoothing residuals


                # Impulses and param updates for ops whose outputs have residuals
                if out_specs:
                    for (name, srcs, out, args, kwargs), y, g_list in zip(
                        out_specs, ys, grads
                    ):
                        r = residual_map.get(out, None)
                        if r is None:
                            Ops._need_residual_warn(name)
                            continue
                        if y is None or g_list is None:
                            Ops._need_residual_warn(name)
                            continue
                        g_stack = AbstractTensor.stack(list(g_list), dim=0)  # (S, C) e.g. (65, 3)
                        r_tensor = AbstractTensor.get_tensor(r)

                        # --- normalize r_tensor BEFORE any elementwise ops (avoid nan_to_num on empties) ---
                        C = int(g_stack.shape[1]) if getattr(g_stack, "ndim", 0) >= 2 else 1

                        if getattr(r_tensor, "ndim", 0) >= 1:
                            # flatten; handle empty; dimension mismatch -> reduce to scalar
                            try:
                                r_tensor = r_tensor.reshape(-1)
                            except Exception:
                                r_tensor = AbstractTensor.get_tensor(r_tensor).reshape(-1)
                            nfeat = int(getattr(r_tensor, "shape", (0,))[0])
                            if nfeat == 0:
                                r_tensor = AbstractTensor.get_tensor(0.0)     # neutral; no impulses this tick
                            elif nfeat != C:
                                r_tensor = r_tensor.mean()                    # broadcastable scalar

                        # now safe to sanitize NaN/Inf
                        try:
                            r_tensor = AbstractTensor.nan_to_num(r_tensor, nan=0.0, posinf=0.0, neginf=0.0)
                        except Exception:
                            # backend 'where' can be picky — fall back to scalar/mean path
                            if getattr(r_tensor, "ndim", 0) == 0:
                                r_tensor = AbstractTensor.get_tensor(float(r_tensor))
                            else:
                                r_tensor = r_tensor  # already shaped or zeroed above

                        prod = g_stack * r_tensor  # r_tensor is scalar or (C,)

                        g_scalars = prod.reshape(len(srcs), -1).sum(dim=1)
                        self.sys.impulse_batch(srcs, out, name, -g_scalars)

                        param_nodes = []
                        param_idx = []
                        for idx_i, i in enumerate(srcs):
                            node = self.sys.nodes.get(int(i))
                            if node is not None and hasattr(node, "param"):
                                param_nodes.append(node)
                                param_idx.append(idx_i)
                        if param_nodes:
                            params = AbstractTensor.stack([n.param for n in param_nodes], dim=0)
                            upd_full = prod[param_idx]
                            if getattr(params, "shape", ()) == getattr(upd_full, "shape", () ):
                                params = params + upd_full
                            else:
                                extra_dims = tuple(range(1, getattr(upd_full, "ndim", 1)))
                                params = params + upd_full.sum(dim=extra_dims).reshape(params.shape)
                            for node, new_param in zip(param_nodes, params):
                                node.param = new_param

            if self.sys.feedback_edges and residual_map:
                L = AbstractTensor.get_tensor(0.0)
                for r in residual_map.values():
                    L += 0.5 * (r * r).sum()
                L_scalar = (
                    float(getattr(L, "item_", lambda: L)())
                    if hasattr(L, "item_")
                    else float(L)
                )
                for i, j, op_id in self.sys.feedback_edges:
                    try:
                        self.sys.impulse(i, j, op_id, g_scalar=L_scalar)
                    except Exception:
                        pass

            time.sleep(self.dt)



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
    row_ids: List[int]                  # intermediate row nodes
    nodes: List[Node]
    edges: List[Edge]
    ops: List[Tuple[str, List[int], int]]  # (op_name, src_ids, out_id)

class LinearBlockFactory:
    def __init__(self, n_in: int, n_out: int, *, spacing: float = 0.35, seed: int = 0, rows=1, hidden_dim: int = 64, gather_operators=["add", "mul"]):
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.spacing = float(spacing)
        self.rng = AbstractTensor.random
        self.rng.set_seed(int(seed))
        self.rows = int(rows)
        self.hidden_dim = int(hidden_dim)
        self.gather_operators = list(gather_operators)

    def _mk_edge(self, i, j, op):
        bands = [
            EdgeBand(tau=0.05, K=3.0, kappa=0.12, th=0.05, ratio=3.0, knee=0.05, alpha=1.0),
            EdgeBand(tau=0.25, K=2.0, kappa=0.08, th=0.10, ratio=2.0, knee=0.10, alpha=0.5),
            EdgeBand(tau=1.00, K=1.0, kappa=0.04, th=0.20, ratio=1.2, knee=0.20, alpha=0.2),
        ]
        return Edge(key=(i, j, op), i=i, j=j, op_id=op, bands=bands, l0=1.0)

    def build(self, start_id: int = 0, *, z_level: float = 0.0) -> LinearBlock:
        from .integration.bridge_v2 import _op_apply_factory

        nid = int(start_id)
        nodes: List[Node] = []
        edges: List[Edge] = []
        ops: List[
            Tuple[
                str,
                List[int],
                int,
                Optional[Tuple[Any, ...]],
                Optional[Dict[str, Any]],
            ]
        ] = []

        # --- allocate IDs ---
        in_ids  = [nid + k for k in range(self.n_in)]; nid += self.n_in
        out_ids = [nid + k for k in range(self.n_out)]; nid += self.n_out

        # biases (j) -> give each output its own bias node ID
        b_ids: Dict[int, int] = {j: (nid + j) for j in range(self.n_out)}
        nid += self.n_out

        # inner grid (rows x hidden_dim)
        # FIX: stride row bases by hidden_dim so IDs don't overlap across rows
        row_ids: List[int] = [nid + r * self.hidden_dim for r in range(self.rows)]
        grid_ids: List[List[int]] = [
            [row_ids[r] + c for c in range(self.hidden_dim)] for r in range(self.rows)
        ]
        nid += self.rows * self.hidden_dim

        # --- place nodes in 3D (inputs left, grid center slab, outs right) ---
        def jitter(s=0.07):
            random_tensor = AbstractTensor.random_tensor(size=(3,), scope=(-s, s))
            return random_tensor

        # inputs
        for k, i_id in enumerate(in_ids):
            x = -2.0; y = (k - 0.5*(self.n_in-1)) * self.spacing; z = z_level
            p = AbstractTensor.get_tensor([x, y, z]) + jitter()
            param_generator = empty_param_generator()
            param = param_generator.__next__()
            nodes.append(
                Node(
                    id=i_id,
                    param=param,
                    p=p,
                    v=AbstractTensor.zeros(3),
                    sphere=AbstractTensor.concat([p, param], dim=0),
                )
            )

        # outputs
        for k, o_id in enumerate(out_ids):
            x = +2.0; y = (k - 0.5*(self.n_out-1)) * self.spacing; z = z_level
            p = AbstractTensor.get_tensor([x, y, z]) + jitter()
            param = param_generator.__next__()
            nodes.append(
                Node(
                    id=o_id,
                    param=param,
                    p=p,
                    v=AbstractTensor.zeros(3),
                    sphere=AbstractTensor.concat([p, param], dim=0),
                )
            )

        # biases: place near outputs but a bit inward
        for j in range(self.n_out):
            b_id = b_ids[j]
            x = +1.2
            y = (j - 0.5*(self.n_out-1)) * self.spacing
            z = z_level
            p = AbstractTensor.get_tensor([x, y, z]) + jitter()
            param = param_generator.__next__()
            nodes.append(
                Node(
                    id=b_id,
                    param=param,
                    p=p,
                    v=AbstractTensor.zeros(3),
                    sphere=AbstractTensor.concat([p, param], dim=0),
                )
            )

        # inner grid nodes
        for r in range(self.rows):
            for c in range(self.hidden_dim):
                n_id = grid_ids[r][c]
                x = 0.0 + (c - 0.5*(self.hidden_dim-1)) * self.spacing * 0.6
                y = (r - 0.5*(self.rows-1)) * self.spacing
                z = z_level + 0.05 * self.rng.standard_normal()
                p = AbstractTensor.get_tensor([x, y, z]) + jitter()
                param = param_generator.__next__()
                nodes.append(
                    Node(
                        id=n_id,
                        param=param,
                        p=p,
                        v=AbstractTensor.zeros(3),
                        sphere=AbstractTensor.concat([p, param], dim=0),
                    )
                )

        # common agg fn for gather_and
        agg_fn = _op_apply_factory(["__add__", "__mul__"], [(0,), (1,)])

        # --- connect: inputs -> first row (fully connected) ---
        if self.rows > 0:
            for tgt in grid_ids[0]:
                srcs = list(in_ids)
                for s in srcs:
                    edges.append(self._mk_edge(s, tgt, "gather_and"))
                ops.append((
                    "gather_and",
                    srcs,
                    tgt,
                    None,
                    {"indices": srcs, "dim": 0, "fn": agg_fn},
                ))

        # --- connect: row r -> row r+1 (fully connected between consecutive rows) ---
        for r in range(self.rows - 1):
            prev_row = grid_ids[r]
            next_row = grid_ids[r + 1]
            for tgt in next_row:
                srcs = list(prev_row)
                for s in srcs:
                    edges.append(self._mk_edge(s, tgt, "gather_and"))
                ops.append((
                    "gather_and",
                    srcs,
                    tgt,
                    None,
                    {"indices": srcs, "dim": 0, "fn": agg_fn},
                ))

        # --- connect: last row -> outputs (+ bias per output) ---
        last_sources = grid_ids[-1] if self.rows > 0 else list(in_ids)
        for j, oj in enumerate(out_ids):
            srcs = list(last_sources) + [b_ids[j]]
            for s in srcs:
                edges.append(self._mk_edge(s, oj, "gather_and"))
            ops.append((
                "gather_and",
                srcs,
                oj,
                None,
                {"indices": srcs, "dim": 0, "fn": agg_fn},
            ))

        return LinearBlock(
            base_id=start_id,
            in_ids=in_ids,
            out_ids=out_ids,
            w_ids={},
            b_ids=b_ids,
            row_ids=row_ids,   # row bases (first column of each row); full grid is internal
            nodes=nodes,
            edges=edges,
            ops=ops,
        )

# ---------- convenience: make constant byte targets for a phrase ----------
def ascii_targets_for(phrase: str, out_ids: List[int]) -> Dict[int, Callable[[float], float]]:
    # scale bytes [0..255] → [-1,1] for your scalar residuals
    vals = [ord(c) for c in phrase]
    def impulse_scale(v): return ((v / 127.5) - 1.0)
    return {oid: (lambda t, v=impulse_scale(vals[k % len(vals)]): float(v))
            for k, oid in enumerate(out_ids)}



def build_toy_system(seed=0, *, batch_size: int = 4096, batch_refresh_hz: float = 20.0):
    """Build a toy spring–repulsor system where inputs are drawn from a large
    random batch tensor.

    - Generates X ~ U(-1,1) with shape (batch_size, n_in).
    - Each input node i gets a Neumann force function that selects the current
      sample s = floor(t * batch_refresh_hz) % batch_size and emits X[s, i].
    - As before, inputs are also clamped by a Dirichlet spring to the live
      batch mean across all input features for that sample.
    """
    rng = AbstractTensor.random.set_seed(seed)
    nodes = []   # <- list, not dict
    edges = []   # <- list, not dict
    outputs = {}

    TEXT = "I am one million monkeys typing on a keyboard"

    lb = LinearBlockFactory(
        n_in=len(TEXT),
        n_out=len(TEXT),
        spacing=0.28,
        rows=3,
        gather_operators=["fused_add_mul"],
        seed=seed
    ).build(start_id=0, z_level=0.0)

    # Install the block
    nodes.extend(lb.nodes)
    edges.extend(lb.edges)

    sys = SpringRepulsorSystem(nodes, edges, eta=0.0, gamma=0.3, dt=0.02)

    # Tag roles for visualization
    sys.roles = {}
    for i in lb.in_ids:
        sys.roles[i] = "input"
    for o in lb.out_ids:
        sys.roles[o] = "output"

    # ---------------- Random batch-driven inputs ----------------
    B = int(max(1, batch_size))
    N_in = int(len(lb.in_ids))
    # X ~ U(-1,1) of shape (B, N_in)
    X = AbstractTensor.random_tensor(size=(B, N_in), scope=(-1.0, 1.0))

    def make_batch_fn(col: int):
        def _f(t, c=col):
            # Select sample index based on wall time (shared across inputs)
            s_idx = int((t * batch_refresh_hz)) % B
            val = AbstractTensor.get_tensor(X[s_idx, c])
            # Robust scalar extract across backends
            v = float(getattr(val, "item_", getattr(val, "item", lambda: val))())
            return v
        return _f

    input_force_fns: Dict[int, Callable[[float], float]] = {}
    for nid, col in zip(lb.in_ids, range(N_in)):
        fn = make_batch_fn(col)
        input_force_fns[nid] = fn
        # Neumann traction with random sample value
        sys.add_boundary(BoundaryPort(nid=nid, beta=0.8, force_fn=as_x_force(fn, D=sys.D)))

    # ASCII targets (constant data fns for outputs)
    byte_targets = ascii_targets_for(TEXT, lb.out_ids)
    outputs.update(byte_targets)

    # Dirichlet and Neumann/Robin boundaries for inputs and outputs.
    # Inputs clamp to the live mean of their batch data; outputs track their
    # individual byte targets.
    def group_data_mean_fn(fn_map):
        def _mean(t, fn_map=fn_map):
            vals = [fn(t) for fn in fn_map.values()]
            return float(AbstractTensor.get_tensor(vals).mean()) if vals else 0.0
        return _mean

    in_mean_fn = group_data_mean_fn(input_force_fns)

    for idx, nid in enumerate(lb.in_ids):
        attach_dirichlet(sys, nid, in_mean_fn, axis=0)
        # The demo previously attached extra Neumann "noop" boundaries to many
        # hidden row nodes.  Commenting this out reduces visual clutter and
        # keeps role tagging focused on true inputs/outputs.
        # if lb.row_ids:
        #     rid = lb.row_ids[idx % len(lb.row_ids)]
        #     attach_neumann_noop(sys, rid)
    for idx, oid in enumerate(lb.out_ids):
        target_fn = byte_targets[oid]
        attach_dirichlet(sys, oid, lambda t, fn=target_fn: fn(t), axis=2)
        # if lb.row_ids:
        #     rid = lb.row_ids[idx % len(lb.row_ids)]
        #     attach_neumann_noop(sys, rid)

    # Wire the op program
    sys.ops_program = lb.ops

    return sys, outputs



def main(duration_s: float = 8.0):
    # Default to a large random batch driving the inputs
    sys, outputs = build_toy_system(seed=42, batch_size=10000, batch_refresh_hz=15.0)

    stop = threading.Event()
    refl = Reflector(sys, stop, tick_hz=30.0, commit_every_s=5.0)
    expr = Experiencer(sys, stop, outputs, schedule_hz=60.0, ops_program=sys.ops_program)

    print("[INFO] Starting threads…")
    refl.start(); expr.start()
    t0 = now_s()
#    viz = LiveViz3D(sys)
#    viz.launch()
    viz = LiveVizGLPoints(sys, node_cmap="coolwarm", base_point_size=6.0)
    viz.launch()              # once
    from collections import deque

    BATCH = 16  # batch/window size for the running mean
    batch_actual_q = deque(maxlen=BATCH)
    batch_target_q = deque(maxlen=BATCH)

    sample = [oid for oid in outputs.keys() if oid in sys.nodes]  # fixed feature order

    def _to_byte(x):
        y = (float(x) + 1.0) * 127.5
        return max(0, min(255, round(y)))

    def mse_batch(a, b):
        d = a - b
        return float(((d * d).mean()).item())

    t0 = now_s()
    try:
        LOG_EVERY = 2.0  # seconds
        next_log = now_s()
        t0 = now_s()
        while (now_s() - t0) < duration_s:
            t = now_s() - t0


            # (F,) feature rows for this tick
            actual_row = AbstractTensor.stack([
                AbstractTensor.cat([sys.nodes[oid].p.clone(), sys.nodes[oid].param.clone()]) if sys.nodes.get(oid) is not None else AbstractTensor.get_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                for oid in sample
            ])

            target_row = AbstractTensor.get_tensor([outputs[oid](t)     for oid in sample])

            # accumulate batch
            batch_actual_q.append(actual_row)
            batch_target_q.append(target_row)

            # (B', F) batches (B' ≤ BATCH while warming up)
            actual_batch = AbstractTensor.get_tensor([row for row in batch_actual_q])
            target_batch = AbstractTensor.get_tensor([row for row in batch_target_q])

            if now_s() >= next_log:
                mean_vals  = actual_batch.mean(dim=0)
                mean_tgts  = target_batch.mean(dim=0)
                output_str = ''.join(chr(_to_byte(v[2])) for v in mean_vals)
                target_str = ''.join(chr(_to_byte(v))    for v in mean_tgts)
                error = mse_batch(actual_batch[..., 2], target_batch)
                print(f"[DBG] outputs→batch-mean Error: {error: .3f} | Out: {output_str} | Tgt: {target_str}")
                next_log += LOG_EVERY

            viz.step(0.0)           # render without forcing a big sleep
            time.sleep(0.005)       # tiny yield so we don’t busy-wait



    finally:
        stop.set()
        expr.join(timeout=2.0)
        refl.join(timeout=2.0)
        print("[INFO] Stopped.")
        viz.close()

if __name__ == "__main__":
    main(float('inf'))
