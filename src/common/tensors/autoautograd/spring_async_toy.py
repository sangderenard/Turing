"""
Spring-Repulsor Async Toy (AbstractTensor)
---------------------------------
Minimal, threadful prototype of the "spring–repulsor, multiband gradient acoustics"
learner.  It sketches the dual data/geometry domains from ``../THEORETIC.md`` where
nodes shuttle signals through x/z terminals while edges and faces carry the discrete
exterior calculus (DEC) geometry.  Nodes and edges each expose two parameter sets:

* ``phys`` – per-node Dirichlet targets ``[x, y, z]`` describing the geometric
  clamps.  The **x channel is always read from upstream** while **z writes to
  downstream**; both are refreshed every tick.  The y entry alone is left to evolve
  under the physics worker.
* ``ctrl`` – data‑path scalars ``[alpha, w, b]`` (alpha, weight and bias) committed
  alongside geometry to drive the learning rule.

Edges mirror this split with their own ``ctrl`` block and geometric parameters:

* ``phys`` – ``[l0, k, face_weight, curl]`` where rest length ``l0`` and stiffness
  ``k`` anchor the spring, ``face_weight`` scales relative to edge distance via a
  provisional face stencil (its angle is assumed rather than solved), and ``curl``
  captures the local face curvature.  Introducing the stencil adds temporary points
  in space so faces can exist without fixing global geometry.

All of these geometric terms are learnable by gradient descent much like the
data‑path scalars, letting a purely message‑passing process coax a graph‑based
geometry and spatial field into existence.  Operator outputs continue to live in
the data domain.

The toy keeps the asynchronous ``Experiencer`` / ``Reflector`` split but still deviates
from the plan in several ways:

- Uses an explicit time-step integrator instead of the proposed timeless, impulse-only
  update (see ``tick``).
- Because x and z are Dirichlet inputs they must be refreshed on every update; the
  y channel alone is free to integrate.
- Caching, batching and stress-based updates are stubs.

Current features:

- < 20 nodes
- Two roles/threads: Experiencer (ops + impulses) and Reflector (relaxer/integrator)
- Edges store: timestamp, rings (microgradient count), geometry and control scalars
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
  positions back into control scalars.
"""
from __future__ import annotations
from .integration.bridge_v2 import (
    push_impulses_from_op_v2,
    push_impulses_from_ops_batched,
)
from .whiteboard_cache import WhiteboardCache
from .residual_store import ResidualStore, Space

import time
import threading
import math
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Callable

# Optional heavy visualization deps (lazy-loaded via CLI flags)
matplotlib = None  # type: ignore
plt = None  # type: ignore
FuncAnimation = None  # type: ignore
mcolors = None  # type: ignore
GL_DYNAMIC_DRAW = GL_ARRAY_BUFFER = GL_DEPTH_TEST = GL_LEQUAL = GL_CULL_FACE = None  # type: ignore
GL_PROGRAM_POINT_SIZE = GL_COLOR_BUFFER_BIT = GL_DEPTH_BUFFER_BIT = GL_POINTS = GL_LINES = None  # type: ignore
GL_VERTEX_SHADER = GL_FRAGMENT_SHADER = GL_FLOAT = GL_FALSE = None  # type: ignore

from ..abstraction import AbstractTensor
from ..filtered_poisson import filtered_poisson
from .node_tensor import NodeAttrView
from ...dt_system.spectral_dampener import spectral_inertia
from ...dt_system.curvature import hex_face_curvature_batch
from ...dt_system.integrator.integrator import Integrator
from ...dt_system.dt_scaler import Metrics
from ...dt_system.dt_controller import STController, Targets
from ...dt_system.dt_graph import (
    AdvanceNode,
    ControllerNode,
    MetaLoopRunner,
    RoundNode,
    StateNode,
)
from ...dt_system.dt import SuperstepPlan
from ...dt_system.roundnode_engine import RoundNodeEngine
from ...dt_system.state_table import StateTable
from ...dt_system.threaded_system import ThreadedSystemEngine
from .fluxspring.fs_spec_builder import (
    FluxSpringSpec,
    NodeSpec,
    EdgeSpec,
    DECSpec,
)
V_MAX = 2.0
STEP_MAX = 10.2
READOUT_SCALE = 1.0
READOUT_BIAS  = 0.0
W_EPS = 1e-3
W_MIN, W_MAX = 0.25, 4.0

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


def capture_node_positions(sys: 'SpringRepulsorSystem'):
    """Return a capture function yielding node positions.

    The returned callable packs a mapping ``{"positions": [[x, y, z], ...]}``
    suitable for ThreadedSystemEngine's ``capture`` hook, enabling optional
    visualisation of the current node positions.
    """

    def _cap() -> Dict[str, List[List[float]]]:
        pos = [
            AbstractTensor.get_tensor(n.p).tolist()  # type: ignore[call-arg]
            for n in sys.nodes.values()
        ]
        return {"positions": pos}

    return _cap


def _stack_grads_per_source(
    name: str, out: int, srcs, g_list
) -> Tuple[AbstractTensor, int]:
    """Stack gradients per source ensuring equal width and count.

    Parameters
    ----------
    name:
        Operator name for error messages.
    out:
        Output node identifier for error messages.
    srcs:
        Sequence of source node identifiers.
    g_list:
        Gradients aligned to ``srcs``.

    Returns
    -------
    Tuple[AbstractTensor, int]
        Stacked gradients and inferred width ``C``.

    Raises
    ------
    ValueError
        If any gradient is missing or has a mismatched width. The error
        message includes ``name``, ``out`` and the offending source index.
    """
    print("_stack_grads_per_source:", name, out, srcs, g_list)
    tensors: List[AbstractTensor] = []
    C = None
    for i, _ in enumerate(srcs):
        try:
            g = g_list[i]
        except IndexError:
            raise ValueError(
                f"{name}: missing gradient for source index {i} (output {out})"
            )
        if g is None:
            raise ValueError(
                f"{name}: missing gradient for source index {i} (output {out})"
            )
        t = AbstractTensor.get_tensor(g)
        width = t.shape[-1] if getattr(t, "ndim", 0) > 0 else 1
        if C is None:
            C = width
        elif width != C:
            raise ValueError(
                f"{name}: grad width {width} for source index {i} mismatches {C} (output {out})"
            )
        tensors.append(t)

    if len(g_list) != len(srcs):
        raise ValueError(
            f"{name}: expected {len(srcs)} gradients, got {len(g_list)} (output {out})"
        )

    g_stack = AbstractTensor.stack(tensors, dim=0)
    return g_stack, C or 0

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
    """Clamp only the chosen axis of ``nid`` toward value_fn(t); leave other axes untouched.

    ``THEORETIC.md`` envisions x and z as fixed Dirichlet channels.  This helper
    now installs an explicit per-axis mask so the integrator can respect that
    convention.
    """
    D = sys.D if D is None else int(D)

    def _target_vec(t, _sys=sys, _nid=nid, _axis=axis):
        # copy current position so non-target axes see zero Dirichlet force
        p_now = _sys.nodes[_nid].p.clone()
        p_now[_axis] = float(value_fn(t))
        return p_now
    mask = AbstractTensor.zeros(D, dtype=float)
    mask[axis] = 1.0
    sys.add_boundary(
        BoundaryPort(nid=nid, alpha=alpha, target_fn=_target_vec, axis_mask=mask)
    )


def attach_neumann_noop(
    sys: 'SpringRepulsorSystem',
    nid: int,
    *,
    D: Optional[int] = None,
    beta: float = 1.0,
    axis: int = 0,
) -> None:
    """Attach a traction-capable boundary that exerts no force (pass-through).

    The mask matches ``attach_dirichlet`` so axes can be toggled free vs. fixed.
    """

    D = sys.D if D is None else int(D)
    mask = AbstractTensor.zeros(D, dtype=float)
    mask[axis] = 1.0
    sys.add_boundary(
        BoundaryPort(
            nid=nid,
            beta=beta,
            force_fn=as_axis_force(lambda _t: 0.0, axis, D),
            axis_mask=mask,
        )
    )

def _fresh_node_id(sys: 'SpringRepulsorSystem') -> int:
    return (max(sys.nodes.keys()) + 1) if sys.nodes else 0

def enliven_feature_edges(sys: SpringRepulsorSystem, in_ids: List[int], out_ids: List[int]):
    # seed physical ties so forces/learning can flow
    for i in in_ids:
        for o in out_ids:
            sys.ensure_edge(i, o, "feat")  # op_id tag just to distinguish
def _default_ctrl() -> AbstractTensor:
    """Return the default control triple `(alpha, w, b)`."""
    return AbstractTensor.get_tensor([0.0, 1.0, 0.0])


def _phys_from_p(p: AbstractTensor) -> AbstractTensor:
    """Extract `[x0, 0.0, z0]` from a position tensor ``p``."""
    try:
        x0 = float(AbstractTensor.get_tensor(p[0]))
    except Exception:
        x0 = 0.0
    try:
        z0 = float(AbstractTensor.get_tensor(p[2]))
    except Exception:
        z0 = 0.0
    return AbstractTensor.get_tensor([x0, 0.0, z0])

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
    mean_in = float(getattr(sys, "prev_in_mean", 0.0))
    mean_out = float(getattr(sys, "prev_out_mean", 0.0))

    d = _fresh_node_id(sys)
    p = AT.zeros(D, dtype=float)
    if D > 0:
        p[0] = mean_in
    if D > 2:
        p[2] = mean_out
    phys = _phys_from_p(p)
    ctrl = _default_ctrl()
    node_d = Node(
        id=d,
        phys=phys,
        ctrl=ctrl,
        p=p,
        v=AT.zeros(D, dtype=float),
        geom_mask=AbstractTensor.tensor([0.0, 1.0, 0.0]),
        sphere=AbstractTensor.concat([p, phys, ctrl], dim=0),
    )
    node_d.commit()
    node_d.hist_p.append(node_d.p.clone())
    sys.nodes[d] = node_d
    attach_dirichlet(sys, d, sample_fn, D=D, alpha=alpha)
    n = _fresh_node_id(sys)
    p = AT.zeros(D, dtype=float)
    if D > 0:
        p[0] = mean_in
    if D > 2:
        p[2] = mean_out
    phys = _phys_from_p(p)
    ctrl = _default_ctrl()
    node_n = Node(
        id=n,
        phys=phys,
        ctrl=ctrl,
        p=p,
        v=AT.zeros(D, dtype=float),
        geom_mask=AbstractTensor.tensor([0.0, 1.0, 0.0]),
        sphere=AbstractTensor.concat([p, phys, ctrl], dim=0),
    )
    node_n.commit()
    node_n.hist_p.append(node_n.p.clone())
    sys.nodes[n] = node_n
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
    mean_in = float(getattr(sys, "prev_in_mean", 0.0))
    mean_out = float(getattr(sys, "prev_out_mean", 0.0))

    n = _fresh_node_id(sys)
    p = AT.zeros(D, dtype=float)
    if D > 0:
        p[0] = mean_in
    if D > 2:
        p[2] = mean_out
    phys = _phys_from_p(p)
    ctrl = _default_ctrl()
    node_n = Node(
        id=n,
        phys=phys,
        ctrl=ctrl,
        p=p,
        v=AT.zeros(D, dtype=float),
        geom_mask=AbstractTensor.tensor([0.0, 1.0, 0.0]),
        sphere=AbstractTensor.concat([p, phys, ctrl], dim=0),
    )
    node_n.commit()
    node_n.hist_p.append(node_n.p.clone())
    sys.nodes[n] = node_n
    attach_neumann_noop(sys, n, D=D, beta=beta)
    sys.ensure_edge(system_nid, n, "out_link")
    d = _fresh_node_id(sys)
    p = AT.zeros(D, dtype=float)
    if D > 0:
        p[0] = mean_in
    if D > 2:
        p[2] = mean_out
    phys = _phys_from_p(p)
    ctrl = _default_ctrl()
    node_d = Node(
        id=d,
        phys=phys,
        ctrl=ctrl,
        p=p,
        v=AT.zeros(D, dtype=float),
        geom_mask=AbstractTensor.tensor([0.0, 1.0, 0.0]),
        sphere=AbstractTensor.concat([p, phys, ctrl], dim=0),
    )
    node_d.commit()
    node_d.hist_p.append(node_d.p.clone())
    sys.nodes[d] = node_d
    attach_dirichlet(sys, d, target_fn, D=D, alpha=alpha)
    sys.ensure_edge(n, d, "readout")
    return n

# ----------------------------- Core data ------------------------------------



@dataclass
class BoundaryPort:
    nid: int
    alpha: float = 0.0                     # Dirichlet spring strength
    beta: float = 0.0                      # Neumann (traction) gain
    target_fn: Optional[Callable[[float], AbstractTensor]] = None  # t -> R^D
    force_fn: Optional[Callable[[float], AbstractTensor]]  = None  # t -> R^D
    # Optional per-axis mask selecting which components the boundary acts on.
    axis_mask: Optional[AbstractTensor] = None
    enabled: bool = True


class ParamLogger:
    """
    Periodically prints a per-node table:
      nid | role | P | ctrl[0:K] | Δp1 | Δp2 | resF | resG | impulses | deg(in/out) | pops | credit | status

    - Δp1/Δp2: L1/L2 delta since last log
    - resF/resG: counts of residual items present this tick (Feature / Geometry)
    - impulses: sum of edge ring deltas incident to node since last log
    - pops/credit: summed pops and mean credit_ema across incident edges
    - status: DEAD if no movement & no residuals & no impulses
    """
    def __init__(self, sys, *, interval_s=5.0, dead_tol=1e-8, preview_k=3, sort_by="Δp2"):
        self.sys = sys
        self.interval_s = float(interval_s)
        self.dead_tol = float(dead_tol)
        self.preview_k = int(preview_k)
        self.sort_by = sort_by  # "Δp2" or "nid"

        self._t_last = 0.0
        self._prev_ctrl = {}
        self._prev_rings = {}
        self._prime()

    def _prime(self):
        # cheap, lock-free snapshots
        self._prev_ctrl = {
            i: n.ctrl.clone() if hasattr(n.ctrl, "clone") else AbstractTensor.get_tensor(n.ctrl)
            for i, n in self.sys.nodes.items()
        }
        self._prev_rings = {k: int(e.rings) for k, e in self.sys.edges.items()}
        self._t_last = now_s()

    def _fmt_vec(self, v):
        # preview first K components, robust to any width/backend
        t = AbstractTensor.get_tensor(v).reshape(-1)
        k = min(self.preview_k, int(t.shape[0])) if getattr(t, "shape", None) else 1
        vals = [float(getattr(t[i], "item", lambda: t[i])()) for i in range(k)]
        tail = "" if int(getattr(t, "shape", [k])[0]) <= k else "…"
        return "[" + ", ".join(f"{x:+.4f}" for x in vals) + tail + "]"

    def _norms(self, v):
        t = AbstractTensor.get_tensor(v).reshape(-1)
        l1 = float(AbstractTensor.sum(AbstractTensor.abs(t)))
        l2 = float(AbstractTensor.linalg.norm(t))
        return l1, l2

    def _rings_delta_per_node(self, prev_rings):
        # sum incident edge ring deltas per node
        delta_by_node = {i: 0 for i in self.sys.nodes}
        for (i, j, _), e in self.sys.edges.items():
            d = int(e.rings) - int(prev_rings.get((i, j, _), int(e.rings)))
            if d != 0:
                delta_by_node[i] = delta_by_node.get(i, 0) + d
                delta_by_node[j] = delta_by_node.get(j, 0) + d
        return delta_by_node

    def _deg_in_out(self):
        din = {i: 0 for i in self.sys.nodes}
        dout = {i: 0 for i in self.sys.nodes}
        for (i, j, _op) in self.sys.edges.keys():
            dout[i] = dout.get(i, 0) + 1
            din[j]  = din.get(j, 0) + 1
        return din, dout

    def tick(self, *, residuals=None, touched_srcs=None, force=False):
        """Call once per Experiencer iteration."""
        t = now_s()
        if not force and (t - self._t_last) < self.interval_s:
            return

        # --- snapshots
        roles = getattr(self.sys, "roles", {})
        ctrl_now = {
            i: n.ctrl.clone() if hasattr(n.ctrl, "clone") else AbstractTensor.get_tensor(n.ctrl)
            for i, n in self.sys.nodes.items()
        }
        rings_delta = self._rings_delta_per_node(self._prev_rings)
        din, dout = self._deg_in_out()

        # residual presence maps (counts per node)
        resF_ct = {i: 0 for i in self.sys.nodes}
        resG_ct = {i: 0 for i in self.sys.nodes}
        if residuals is not None:
            try:
                bucketF = residuals.get_bucket(Space.F) or {}
                for nid, items in bucketF.items():
                    resF_ct[int(nid)] = len(items)
            except Exception:
                pass
            try:
                bucketG = residuals.get_bucket(Space.G) or {}
                for nid, items in bucketG.items():
                    resG_ct[int(nid)] = len(items)
            except Exception:
                pass

        # pops / credit summaries per node from incident edges
        pops_sum = {i: 0 for i in self.sys.nodes}
        credit_mean = {i: 0.0 for i in self.sys.nodes}
        credit_cnt = {i: 0 for i in self.sys.nodes}
        for (i, j, _), e in self.sys.edges.items():
            for n in (i, j):
                pops_sum[n] = pops_sum.get(n, 0) + int(getattr(e, "pops", 0))
                credit_mean[n] = credit_mean.get(n, 0.0) + float(getattr(e, "credit_ema", 0.0))
                credit_cnt[n] = credit_cnt.get(n, 0) + 1
        for n in credit_mean:
            c = credit_cnt.get(n, 0)
            if c > 0:
                credit_mean[n] = credit_mean[n] / c

        # table rows
        rows = []
        for nid, p_now in ctrl_now.items():
            p_prev = self._prev_ctrl.get(nid, AbstractTensor.zeros_like(p_now))
            dp = p_now - p_prev
            l1, l2 = self._norms(dp)
            role = roles.get(int(nid), "")
            P = int(getattr(p_now, "shape", (1,))[-1]) if hasattr(p_now, "shape") else 1
            status = "DEAD" if (l2 <= self.dead_tol and resF_ct[nid] == 0 and resG_ct[nid] == 0 and rings_delta.get(nid, 0) == 0) else ""
            rows.append({
                "nid": int(nid),
                "role": role,
                "P": P,
                "ctrl": self._fmt_vec(p_now),
                "Δp1": l1,
                "Δp2": l2,
                "resF": int(resF_ct[nid]),
                "resG": int(resG_ct[nid]),
                "imp": int(rings_delta.get(nid, 0)),
                "deg": f"{din.get(nid,0)}/{dout.get(nid,0)}",
                "pops": int(pops_sum.get(nid, 0)),
                "credit": float(credit_mean.get(nid, 0.0)),
                "status": status,
            })

        # sort (smallest movement first helps spot dead regions)
        if self.sort_by == "Δp2":
            rows.sort(key=lambda r: (r["Δp2"], r["nid"]))
        elif self.sort_by == "nid":
            rows.sort(key=lambda r: r["nid"])

        # print
        print("\n[PARAMS] t=%.2fs (Δ since last=%.2fs)" % (t, t - self._t_last))
        print(" nid | role      |  P | ctrl[0:K]              |   Δp1    |   Δp2    | rF | rG | imp | deg(in/out) | pops |  credit  | status")
        print("-----+-----------+----+------------------------+----------+----------+----+----+-----+-------------+------+----------+--------")
        import random as rnd
        for r in rnd.sample(rows, min(len(rows), 20)) if len(rows) > 20 else rows:
            print(f"{r['nid']:>4d} | {r['role'][:9]:<9} | {r['P']:>2d} | {r['ctrl']:<22} | "
                  f"{r['Δp1']:>8.2e} | {r['Δp2']:>8.2e} | {r['resF']:>2d} | {r['resG']:>2d} | "
                  f"{r['imp']:>3d} | {r['deg']:^11} | {r['pops']:>4d} | {r['credit']:>8.2e} | {r['status']}")
        print(flush=True)

        # rollover
        self._prev_ctrl = ctrl_now
        self._prev_rings = {k: int(e.rings) for k, e in self.sys.edges.items()}
        self._t_last = t


class SpringRepulsorSystem:
    def __init__(
        self,
        nodes: List[Node],
        edges: List[Edge],
        faces: Optional[List[Face]] = None,
        *,
        eta: float = 0.1,
        gamma: float = 0.92,
        dt: float = 0.02,
        rep_eps: float = 1e-6,
    ):
        self.nodes: Dict[int, Node] = {n.id: n for n in nodes}
        for n in self.nodes.values():
            n.commit()
            n.hist_p.append(n.p.clone())
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

        # Running batch means for geometry embedding
        self.prev_in_mean = 0.0
        self.prev_out_mean = 0.0
        self.in_mean_fn: Optional[Callable[[float], float]] = None
        self.out_mean_fn: Optional[Callable[[float], float]] = None
        self.prev_mean_time = now_s()

        # ----- Geometry assembly -----
        self.node_ids = [n.id for n in nodes]
        node_index = {nid: i for i, nid in enumerate(self.node_ids)}
        self.node_index = node_index
        self.edge_list = list(edges)
        E = len(self.edge_list)
        N = len(self.node_ids)
        l0_vals = []
        k_vals = []
        self.edge_src_idx: List[int] = []
        self.edge_dst_idx: List[int] = []
        D0_rows: List[AbstractTensor] = []
        for e in self.edge_list:
            i_idx = node_index[e.i]
            j_idx = node_index[e.j]
            self.edge_src_idx.append(i_idx)
            self.edge_dst_idx.append(j_idx)
            row = AbstractTensor.zeros(N, dtype=float)
            row[i_idx] = -1.0
            row[j_idx] = 1.0
            D0_rows.append(row)
            l0_vals.append(AbstractTensor.get_tensor(e.l0).reshape(()))
            k_vals.append(AbstractTensor.get_tensor(e.k).reshape(()))
        self.D0 = (
            AbstractTensor.stack(D0_rows, dim=0) if D0_rows else AbstractTensor.zeros((0, N), dtype=float)
        )
        self.l0 = (
            AbstractTensor.stack(l0_vals, dim=0) if l0_vals else AbstractTensor.zeros(0, dtype=float)
        )
        self.k = (
            AbstractTensor.stack(k_vals, dim=0) if k_vals else AbstractTensor.zeros(0, dtype=float)
        )

        self.faces: List[Face] = list(faces) if faces else []
        F = len(self.faces)
        alpha_vals = []
        c_vals = []
        D1_rows: List[AbstractTensor] = []
        for face in self.faces:
            alpha_vals.append(AbstractTensor.get_tensor(face.alpha).reshape(()))
            c_vals.append(AbstractTensor.get_tensor(face.c).reshape(()))
            row = AbstractTensor.zeros(E, dtype=float)
            for e_oriented in face.edges:
                e_id = abs(int(e_oriented)) - 1
                if 0 <= e_id < E:
                    sign = 1.0 if e_oriented > 0 else -1.0
                    row[e_id] = sign
            D1_rows.append(row)
        self.D1 = (
            AbstractTensor.stack(D1_rows, dim=0) if D1_rows else AbstractTensor.zeros((0, E), dtype=float)
        )
        self.alpha_face = (
            AbstractTensor.stack(alpha_vals, dim=0)
            if alpha_vals
            else AbstractTensor.zeros(0, dtype=float)
        )
        self.c = (
            AbstractTensor.stack(c_vals, dim=0) if c_vals else AbstractTensor.zeros(0, dtype=float)
        )

    def add_boundary(self, port: BoundaryPort):
        self.boundaries[port.nid] = port

    def remove_boundary(self, nid: int):
        self.boundaries.pop(nid, None)

    def set_boundary(self, nid: int, **kw):
        b = self.boundaries.get(nid)
        if b:
            for k, v in kw.items():
                setattr(b, k, v)

    # ----------------- Geometry helpers (Guiding Theory §4) -----------------
    def edge_strain(self, y: AbstractTensor) -> AbstractTensor:
        """Edge strain ``g = D0 @ y - l0``."""
        return self.D0 @ AbstractTensor.get_tensor(y) - self.l0

    def face_curvature(self, g: AbstractTensor) -> AbstractTensor:
        """Discrete curvature ``z = D1 @ g``."""
        return self.D1 @ AbstractTensor.get_tensor(g)

    def curvature_activation(self, z: AbstractTensor) -> tuple[AbstractTensor, AbstractTensor]:
        """Curvature activation ``Φα(z)`` and its derivative.

        Parameters
        ----------
        z:
            Curvature values ``z``.

        Returns
        -------
        Tuple[AbstractTensor, AbstractTensor]
            ``u = Φα(z)`` and derivative ``Φα'(z)``.
        """
        alpha = self.alpha_face
        z_t = AbstractTensor.get_tensor(z)
        t = AbstractTensor.tanh(z_t)
        u = (1.0 - alpha) * z_t + alpha * t
        dphi = (1.0 - alpha) + alpha * (1.0 - t * t)
        return u, dphi

    def energy(self, y: AbstractTensor) -> AbstractTensor:
        """Total energy ``E(y)``."""
        g = self.edge_strain(y)
        z = self.face_curvature(g)
        u, _ = self.curvature_activation(z)
        E_edges = 0.5 * AbstractTensor.sum(self.k * g * g)
        E_faces = 0.5 * AbstractTensor.sum(self.c * u * u)
        return E_edges + E_faces

    def energy_grad(self, y: AbstractTensor) -> AbstractTensor:
        """Gradient ``∇E(y)``."""
        g = self.edge_strain(y)
        z = self.face_curvature(g)
        u, dphi = self.curvature_activation(z)
        Kg = self.k * g
        r = u * dphi
        inner = Kg + self.D1.T() @ (self.c * r)
        return self.D0.T() @ inner

    # ----------------- Impulse ingestion -----------------
    def ensure_edge(self, i: int, j: int, op_id: str) -> Edge:
        key = (i, j, op_id)
        if key not in self.edges:
            ctrl = AbstractTensor.zeros(3, dtype=float)
            l0 = AbstractTensor.tensor(1.0)
            k = AbstractTensor.tensor(1.0)
            e = Edge(key=key, i=i, j=j, op_id=op_id, ctrl=ctrl, l0=l0, k=k)
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

    # ----------------- Force assembly -----------------
    def assemble_forces(self, dt: float) -> Dict[int, AbstractTensor]:
        """Assemble spring, repulsor and boundary forces.

        Parameters
        ----------
        dt:
            Time step used for spectral inertia.
        """
        F: Dict[int, AbstractTensor] = {
            i: AbstractTensor.zeros(self.D, dtype=float) for i in self.nodes
        }

        t_now = now_s()
        scale = -1.0
        for key, e in self.edges.items():
            ni, nj = self.nodes[e.i], self.nodes[e.j]
            d = nj.p - ni.p
            L = AbstractTensor.linalg.norm(d)
            if L < 1e-9:
                continue
            u = d / (L + 1e-12)
            Lstar = e.target_length()
            k = AbstractTensor.get_tensor(e.k)
            Fel = k * e.hodge1 * (L - Lstar) * u
            Rep = (self.eta / (self.rep_eps + L * L)) * u
            curv = AbstractTensor.get_tensor(e.curvature)
            F[e.i] += (Fel - Rep + curv * u) * scale
            F[e.j] -= (Fel - Rep + curv * u) * scale

        for n in self.nodes.values():
            mask = getattr(n, "geom_mask", None)
            if mask is None:
                mask = AbstractTensor.ones_like(n.p)
            F[n.id] *= mask

        for b in self.boundaries.values():
            if not b.enabled or b.nid not in self.nodes:
                continue
            n = self.nodes[b.nid]
            mask = (
                b.axis_mask
                if b.axis_mask is not None
                else AbstractTensor.ones(self.D, dtype=float)
            )
            if b.alpha > 0.0 and b.target_fn is not None:
                tvec = b.target_fn(t_now)
                if AbstractTensor.isfinite(tvec).all():
                    F[n.id] += mask * (-b.alpha * (n.p - tvec))
            if b.beta > 0.0 and b.force_fn is not None:
                fvec = b.force_fn(t_now)
                if AbstractTensor.isfinite(fvec).all():
                    F[n.id] += mask * (b.beta * fvec)

        for n in self.nodes.values():
            mask = getattr(n, "geom_mask", None)
            if mask is None:
                mask = AbstractTensor.ones_like(n.p)
            resp, _, _ = spectral_inertia(n.hist_p, dt)
            if not AbstractTensor.isfinite(resp).all():
                resp = AbstractTensor.zeros_like(n.p)
            F[n.id] += -resp * mask
        return F


class SpringDtEngine(Integrator):
    """dt-system engine driving a :class:`SpringRepulsorSystem`."""

    def __init__(self, sys: SpringRepulsorSystem, *, algorithm: str = "rk4"):
        self.sys = sys
        self.node_ids = list(sys.node_ids)
        self.D = sys.D
        self.current_dt = getattr(sys, "dt", 0.01)
        self._last_forces: Dict[int, AbstractTensor] = {
            nid: AbstractTensor.zeros(self.D, dtype=float) for nid in self.node_ids
        }

        def _dynamics(t, state_vec):
            N = len(self.node_ids)
            D = self.D
            pos = state_vec[: N * D].reshape((N, D))
            vel = state_vec[N * D :].reshape((N, D))
            for idx, nid in enumerate(self.node_ids):
                node = self.sys.nodes[nid]
                node.p = AbstractTensor.get_tensor(pos[idx])
                node.v = AbstractTensor.get_tensor(vel[idx])
            F = self.sys.assemble_forces(self.current_dt)
            self._last_forces = F
            acc_list = []
            for nid in self.node_ids:
                n = self.sys.nodes[nid]
                acc = F[nid] / n.M0 + ((self.sys.gamma - 1.0) / self.current_dt) * n.v
                acc_list.append(acc)
            vel_arr = AbstractTensor.stack([self.sys.nodes[nid].v for nid in self.node_ids], dim=0)
            acc_arr = AbstractTensor.stack(acc_list, dim=0)
            dpdt = -vel_arr.reshape(-1)
            dvdt = acc_arr.reshape(-1)
            return AbstractTensor.concat([dpdt, dvdt], dim=0)

        super().__init__(dynamics=_dynamics, algorithm=algorithm)
        self._state_vec = self._pack_state()
        self._state = self._state_vec

    def _pack_state(self) -> AbstractTensor:
        pos = [self.sys.nodes[nid].p for nid in self.node_ids]
        vel = [self.sys.nodes[nid].v for nid in self.node_ids]
        pos_vec = AbstractTensor.stack(pos, dim=0).reshape(-1)
        vel_vec = AbstractTensor.stack(vel, dim=0).reshape(-1)
        return AbstractTensor.concat([pos_vec, vel_vec], dim=0)

    def _unpack_state(self, state_vec: AbstractTensor) -> None:
        N = len(self.node_ids)
        D = self.D
        pos = state_vec[: N * D].reshape((N, D))
        vel = state_vec[N * D :].reshape((N, D))
        for idx, nid in enumerate(self.node_ids):
            n = self.sys.nodes[nid]
            n.p = AbstractTensor.get_tensor(pos[idx])
            n.v = AbstractTensor.get_tensor(vel[idx])
            n.hist_p.append(n.p.copy())
            n.commit()
        self._state_vec = state_vec

    def step(self, dt: float, state_table=None):  # type: ignore[override]
        self.current_dt = dt
        ok, _, new_state = super().step(dt, self._state_vec, state_table=state_table)
        self._unpack_state(new_state)
        max_vel = max(
            float(AbstractTensor.linalg.norm(self.sys.nodes[nid].v)) for nid in self.node_ids
        )
        max_flux = max(
            float(AbstractTensor.linalg.norm(f)) for f in self._last_forces.values()
        )
        metrics = Metrics(
            max_vel=max_vel,
            max_flux=max_flux,
            div_inf=0.0,
            mass_err=0.0,
        )
        return ok, metrics, new_state

    def get_state(self, state=None) -> object:  # type: ignore[override]
        return self._state_vec
# liveviz_gl_points.py
# Minimal OpenGL point-field renderer (pygame + PyOpenGL)
# - Keeps node colors from a matplotlib colormap (TwoSlopeNorm around 0)
# - Boundary nodes drawn larger
# - Edges rendered as GL_LINES with spring energy colormap
# - Autoscaled camera; non-blocking window


from typing import Any, Tuple

# Expecting SpringRepulsorSystem with:
#   self.nodes: Dict[int, Node] where Node.p is (3,) ndarray-like and Node.ctrl is scalar
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
        t = AbstractTensor.get_tensor(arr)
        nbytes_attr = getattr(t, "nbytes", None)
        nbytes = int(nbytes_attr() if callable(nbytes_attr) else nbytes_attr)
        ptr = getattr(t, "data", t)

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
            i: (n.p.clone(), n.ctrl.clone())
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

        # --- existing ctrls & base colormap (KEEP this: learning nodes stay coolwarm) ---
        ctrl_vals = AbstractTensor.get_tensor([nodes[i][1][1] for i in ids])

        vmin = AbstractTensor.min(ctrl_vals)
        vmax = AbstractTensor.max(ctrl_vals)
        if vmin > 0.0:
            vmin = 0.0 - 1e-6
        if vmax <= vmin:
            vmax = vmin + 1e-6
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        C = AbstractTensor.get_tensor(self.node_cmap(norm(ctrl_vals)), dtype="float32")[:, :3]

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
                C[idx] = AbstractTensor.get_tensor(self.role_palette[role], dtype="float32")
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
        k = float(AbstractTensor.get_tensor(e.k))
        return 0.5 * k * (L - Lstar) ** 2, (pi, pj)

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
        colors = AbstractTensor.get_tensor(self.edge_cmap(norm(U_vals)), dtype="float32")[:, :3]
        C = AbstractTensor.repeat(colors, 2, axis=0)

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
        mvp = AbstractTensor.get_tensor(self._mvp.T(), dtype="float32")
        glUniformMatrix4fv(self._u_mvp, 1, GL_FALSE, getattr(mvp, "data", mvp))

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
        self.edge_cmap = matplotlib.colormaps.get_cmap(edge_cmap)
        self.node_cmap = matplotlib.colormaps.get_cmap(node_cmap)
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
            i: (n.p.copy(), float(AbstractTensor.get_tensor(n.ctrl[1])))
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
        k = float(AbstractTensor.get_tensor(e.k))
        return 0.5 * k * (L - Lstar) ** 2, (pi, pj)

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
        ctrl_vals = AbstractTensor.array([nodes[i][1] for i in ids])
        self.norm_nodes = mcolors.TwoSlopeNorm(vmin=AbstractTensor.min(ctrl_vals), vcenter=0.0, vmax=AbstractTensor.max(ctrl_vals))

        if Pn.size:
            self.scat_nodes = self.ax.scatter(Pn[:,0], Pn[:,1], Pn[:,2],
                                              s=40, marker="o",
                                              c=self.node_cmap(self.norm_nodes(ctrl_vals[~is_b])),
                                              depthshade=False, linewidths=0.5, edgecolors="k")
        if Pb.size:
            self.scat_bounds = self.ax.scatter(Pb[:,0], Pb[:,1], Pb[:,2],
                                               s=90, marker="s",
                                               c=self.node_cmap(self.norm_nodes(ctrl_vals[is_b])),
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
        ctrl_vals = AbstractTensor.array([nodes[i][1] for i in ids])
        is_b = AbstractTensor.array([i in bset for i in ids])

        # update node colors/positions
        self.norm_nodes.vmin = min(self.norm_nodes.vmin, float(AbstractTensor.min(ctrl_vals)))
        self.norm_nodes.vmax = max(self.norm_nodes.vmax, float(AbstractTensor.max(ctrl_vals)))
        C_all = self.node_cmap(self.norm_nodes(ctrl_vals))

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
        self.logger = ParamLogger(sys, interval_s=5.0, dead_tol=1e-8, preview_k=3, sort_by="Δp2")
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
            print("Experiencer step")
            t = now_s() - t0
            
            # --- Forward sweep: evaluate all ops once ---
            specs = self.ops_program
            ys, grads, _ = push_impulses_from_ops_batched(
                self.sys, specs, weight=None, scale=1.0
            )

            # Map output id -> (spec, grads) for later reverse pass
            # map by output if needed later (currently unused)

            # Seed residuals at supervised outputs
            residuals = ResidualStore()
            for spec, y in zip(specs, ys):
                name, srcs, out, args, kwargs = spec
                if out not in self.outputs:
                    continue
                r = self._residual_for_out(out, y, t)
                if r is not None:
                    width = r.shape[-1] if getattr(r, "ndim", 0) > 0 else 1
                    residuals.add(int(out), r, space=Space.F, width=width)

            # Dirichlet and Neumann boundaries (axis-wise)
            for nid, port in getattr(self.sys, "boundaries", {}).items():
                if port is None or not getattr(port, "enabled", False):
                    continue
                node = self.sys.nodes.get(nid)
                if node is None:
                    continue
                if port.target_fn is not None and port.alpha > 0.0:
                    tvec = port.target_fn(t)
                    if AbstractTensor.isfinite(tvec).all():
                        rb = node.p - tvec
                        if getattr(rb, "ndim", 0) == 0:
                            residuals.put(nid, rb, space=Space.G, width=1, axis=0)
                        else:
                            for axis in range(rb.shape[-1]):
                                residuals.put(
                                    nid,
                                    rb[axis],
                                    space=Space.G,
                                    width=1,
                                    axis=axis,
                                )

                if port.force_fn is not None and getattr(port, "beta", 0.0) > 0.0:
                    fvec = port.force_fn(t)
                    if AbstractTensor.isfinite(fvec).all():
                        if getattr(fvec, "ndim", 0) == 0:
                            residuals.put(nid, fvec, space=Space.G, width=1, axis=0)
                        else:
                            for axis in range(fvec.shape[-1]):
                                residuals.put(
                                    nid,
                                    fvec[axis],
                                    space=Space.G,
                                    width=1,
                                    axis=axis,
                                )

            # --- Reverse sweep: propagate residuals upstream ---
            # --- Reverse sweep: propagate residuals upstream ---
            for space in Space:
                bucket = residuals.get_bucket(space)
                if not bucket:
                    print(f"[WARNING] No residuals in space {space}; skipping impulse pass.")
                    continue

                # specs, grads, ys are aligned
                for (name, srcs, out, args, kwargs), g_list, y in reversed(list(zip(specs, grads, ys))):
                    items = bucket.get(int(out))
                    if items is None:
                        print(f"[WARNING] No items found in bucket for output {out}; skipping.")
                        continue
                    if g_list is None or (isinstance(g_list, (list, tuple, AbstractTensor)) and len(g_list) == 0):
                        print(f"[WARNING] No gradients for op {name} output {out}; skipping.")
                        continue

                    # Consume tensor of per-source grads directly
                    g_stack = AbstractTensor.get_tensor(g_list)
                    if g_stack.shape[0] != len(srcs):
                        raise ValueError(
                            f"{name}: expected {len(srcs)} gradients, got {g_stack.shape[0]} (output {out})"
                        )
                    C = g_stack.shape[-1] if getattr(g_stack, "ndim", 0) > 0 else 1

                    for item in list(items.values()):
                        # 1) Ensure residual width matches the operator channel width
                        #    - If explicitly mismatched in Feature space, expand zero-width to C zeros
                        if space is Space.F and item.width != C:
                            if item.width == 0:
                                # promote empty residual to a zero vector of width C
                                item.value = AbstractTensor.zeros(C, dtype=float)
                                item.width = C
                            else:
                                raise ValueError(f"{name}: feature residual width {item.width} mismatches grad width {C}")

                        rF = AbstractTensor.get_tensor(item.value)  # shape (C,) for F, or scalar for axis-G

                        # 2) Local J^T * r_out product (never collapse feature width)
                        if item.axis is not None:
                            # Axis-specific residual (Geometry space): map back into C channels by
                            # selecting that channel. prod_axis has shape (S,)
                            prod_axis = g_stack[:, item.axis] * rF
                            # Expand into (S, C) by placing on the chosen axis (vector-valued update)
                            prod = AbstractTensor.zeros_like(g_stack)
                            prod[:, item.axis] = prod_axis
                        else:
                            # Full feature residual: elementwise (S,C) * (C,) → (S,C)
                            prod = g_stack * rF

                        # 3) Emit impulses (still per-source, vector aware; batch API will scalarize per edge)
                        if space is Space.F:
                            # Reduce per source to a scalar impulse if your edge scalar expects magnitude.
                            # If you want pure vector impulses, adapt impulse_batch to accept (S,C).
                            g_scalars = prod.sum(dim=1)  # (S,)
                            self.sys.impulse_batch(srcs, out, name, g_scalars)

                        # 4) Control updates for source nodes — preserve ctrl width P
                        ctrl_nodes = []
                        ctrl_idx = []
                        for idx_i, i in enumerate(srcs):
                            node = self.sys.nodes.get(int(i))
                            if node is not None and hasattr(node, "ctrl"):
                                ctrl_nodes.append(node)
                                ctrl_idx.append(idx_i)

                        if ctrl_nodes:
                            # ctrls: (M, P)
                            ctrls = AbstractTensor.stack([n.ctrl for n in ctrl_nodes], dim=0)
                            P = ctrls.shape[-1]
                            # upd_full: (M, C) pulled out of prod by source index
                            upd_full = prod[ctrl_idx]

                            if upd_full.ndim == 1:
                                # Shouldn't happen now (we expanded axis to C), but defend anyway
                                # broadcast to (M, P)
                                upd_ctrl = upd_full[:, None].repeat(P, axis=1)
                            else:
                                if C == P:
                                    upd_ctrl = upd_full  # shape matches
                                else:
                                    # If you want a learned projector, apply it here:
                                    #   upd_ctrl = (Proj @ upd_full.T).T  # Proj: (P, C)
                                    # For now, fail loudly so we don’t silently scramble shapes.
                                    raise ValueError(
                                        f"{name}: update width C={C} does not match ctrl width P={P} "
                                        f"for sources {ctrl_idx}; provide a projector or align widths."
                                    )

                            ctrls = ctrls + upd_ctrl
                            for node, new_ctrl in zip(ctrl_nodes, ctrls):
                                node.ctrl = new_ctrl

                        # 5) Transport residuals upstream (keep vector width)
                        for idx_i, src in enumerate(srcs):
                            r_in_full = prod[idx_i]                # (C,)
                            node = self.sys.nodes.get(int(src))
                            # choose a base to shape against; prefer ctrl (P) then p (D)
                            base = getattr(node, "ctrl", None) if node is not None else None
                            if base is None:
                                base = getattr(node, "p", None) if node is not None else None

                            if base is not None and getattr(base, "ndim", 0) > 0:
                                target_w = base.shape[-1]
                                if r_in_full.shape[-1] == target_w:
                                    r_in = r_in_full
                                elif r_in_full.shape[-1] == 1:
                                    r_in = AbstractTensor.repeat(r_in_full, target_w, axis=-1)
                                else:
                                    # Last-chance projector spot if C != target_w
                                    raise ValueError(
                                        f"{name}: cannot map residual width {r_in_full.shape[-1]} "
                                        f"to base width {target_w} at src {src}; add a projector."
                                    )
                            else:
                                r_in = r_in_full  # fallback

                            width = r_in.shape[-1] if getattr(r_in, "ndim", 0) > 0 else 1
                            residuals.add(int(src), r_in, space=space, width=width)
            print("Post-op residuals:", residuals)
            print("Post-op residual buckets:", {space: len(residuals.get_bucket(space)) for space in Space})
            print("----")
            print("")
            # --- Optional smoothing pass on residuals ---
            print("Smoothing residuals...")

            # Optional Poisson redistribution across all residual nodes
            for space in Space:
                continue
                bucket = residuals.get_bucket(space)
                if not bucket:
                    continue
                groups: Dict[int, List[Tuple[int, ResidualItem]]] = {}
                for nid, items in bucket.items():
                    for item in items.values():
                        groups.setdefault(item.width, []).append((nid, item))
                for pairs in groups.values():
                    if not pairs:
                        continue
                    nids = [nid for nid, _ in pairs]
                    idx: Dict[int, List[int]] = {}
                    for i, nid in enumerate(nids):
                        idx.setdefault(nid, []).append(i)
                    N = len(nids)
                    adjacency = AbstractTensor.zeros((N, N), dtype=float)
                    for e in self.sys.edges.values():
                        for i in idx.get(e.i, []):
                            for j in idx.get(e.j, []):
                                adjacency[i, j] = adjacency[j, i] = 1.0

                    R = AbstractTensor.stack([item.value for _, item in pairs], dim=0)
                    print(f"Smoothing {len(pairs)} residuals of shape {R.shape if R.ndim > 1 else 1} in space {space}...")
                    print("R:")
                    print(R)
                    print("adjacency:")
                    print(adjacency)
                    if len(R) != N:
                        raise ValueError(f"Residual smoothing: expected {N} residuals, got {len(R)}")
                    if R.ndim == 1:
                        R_sm = filtered_poisson(R, iterations=20, adjacency=adjacency)
                    else:
                        cols = []
                        for k in range(R.shape[1]):
                            col = filtered_poisson(R[:, k], iterations=20, adjacency=adjacency)
                            cols.append(col)
                        R_sm = AbstractTensor.stack(cols, dim=1)
                    for (nid, item), r_sm in zip(pairs, R_sm):
                        item.value = r_sm
            print("Residuals:", residuals)
            print("Residual buckets:", {space: len(residuals.get_bucket(space)) for space in Space})
            print("----")
            print("")
            print("Feedback edges:", self.sys.feedback_edges)
            # Global feedback edges seeded from aggregate residuals
            if self.sys.feedback_edges and residuals.any():
                L = AbstractTensor.get_tensor(0.0)
                for r in residuals.iter_values():
                    L += 0.5 * (r * r).sum()
                L_scalar = (
                    float(getattr(L, "item_", lambda: L)())
                    if hasattr(L, "item_")
                    else float(L)
                )
                for i, j, op_id in self.sys.feedback_edges:
                    try:
                        key = (int(i), int(j), op_id)
                        e = self.sys.ensure_edge(int(i), int(j), op_id)
                        with self.sys.edge_locks[key]:
                            e.update_credit(L_scalar, self.dt)
                            e.ingest_impulse(L_scalar, self.dt)
                    except Exception:
                        print(f"[ERROR] Failed to update feedback edge {key}: {e}")
                        pass
            self.logger.tick(residuals=residuals)



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
        ctrl = AbstractTensor.zeros(3, dtype=float)
        l0 = AbstractTensor.tensor(1.0)
        k = AbstractTensor.tensor(1.0)
        return Edge(key=(i, j, op), i=i, j=j, op_id=op, ctrl=ctrl, l0=l0, k=k)

    def build(self, start_id: int = 0, *, z_level: float = 0.0) -> LinearBlock:
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
            phys = _phys_from_p(p)
            ctrl = _default_ctrl()
            nodes.append(
                Node(
                    id=i_id,
                    phys=phys,
                    ctrl=ctrl,
                    p=p,
                    v=AbstractTensor.zeros(3),
                    geom_mask=AbstractTensor.tensor([0.0, 1.0, 0.0]),
                    sphere=AbstractTensor.concat([p, phys, ctrl], dim=0),
                )
            )

        # outputs
        for k, o_id in enumerate(out_ids):
            x = +2.0; y = (k - 0.5*(self.n_out-1)) * self.spacing; z = z_level
            p = AbstractTensor.get_tensor([x, y, z]) + jitter()
            phys = _phys_from_p(p)
            ctrl = _default_ctrl()
            nodes.append(
                Node(
                    id=o_id,
                    phys=phys,
                    ctrl=ctrl,
                    p=p,
                    v=AbstractTensor.zeros(3),
                    geom_mask=AbstractTensor.tensor([0.0, 1.0, 0.0]),
                    sphere=AbstractTensor.concat([p, phys, ctrl], dim=0),
                )
            )

        # biases: place near outputs but a bit inward
        for j in range(self.n_out):
            b_id = b_ids[j]
            x = +1.2
            y = (j - 0.5*(self.n_out-1)) * self.spacing
            z = z_level
            p = AbstractTensor.get_tensor([x, y, z]) + jitter()
            phys = _phys_from_p(p)
            ctrl = _default_ctrl()
            nodes.append(
                Node(
                    id=b_id,
                    phys=phys,
                    ctrl=ctrl,
                    p=p,
                    v=AbstractTensor.zeros(3),
                    geom_mask=AbstractTensor.tensor([0.0, 1.0, 0.0]),
                    sphere=AbstractTensor.concat([p, phys, ctrl], dim=0),
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
                phys = _phys_from_p(p)
                ctrl = _default_ctrl()
                nodes.append(
                    Node(
                        id=n_id,
                        phys=phys,
                        ctrl=ctrl,
                        p=p,
                        v=AbstractTensor.zeros(3),
                        geom_mask=AbstractTensor.tensor([0.0, 1.0, 0.0]),
                        sphere=AbstractTensor.concat([p, phys, ctrl], dim=0),
                    )
                )

        # chain for gather_and: multiply then add using per-node ctrls
        fn_specs = [
            (AbstractTensor.__mul__, slice(1, None, 3)),
            (AbstractTensor.__add__, slice(2, None, 3)),
        ]

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
                    (list(range(len(srcs))), fn_specs),   # drop positional dim
                    {"dim": 0},                           # set as kwarg so batching can offset
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
                    (list(range(len(srcs))), fn_specs),   # drop positional dim
                    {"dim": 0},                           # set as kwarg so batching can offset
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
                (list(range(len(srcs))), fn_specs),   # drop positional dim
                {"dim": 0},                           # set as kwarg so batching can offset
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

    # Build a FluxSpringSpec describing the same topology
    AT = AbstractTensor
    fs_nodes: List[NodeSpec] = []
    for n in nodes:
        fs_nodes.append(
            NodeSpec(
                id=n.id,
                p=n.p,
                v=n.v,
                phys=n.phys,
                mask=n.geom_mask,
                ctrl=n.ctrl,
                mass=AT.get_tensor(1.0),
                in_value=AT.get_tensor(0.0),
                out_value=AT.get_tensor(0.0),
                in_target=AT.get_tensor(0.0),
                out_target=AT.get_tensor(0.0),
            )
        )

    fs_edges: List[EdgeSpec] = []
    for r, e in enumerate(edges):
        fs_edges.append(
            EdgeSpec(
                eid_1=r + 1,
                row_idx=r,
                src=e.i,
                dst=e.j,
                k=e.k,
                l0=e.l0,
                h1=AT.get_tensor(1.0),
                ctrl=e.ctrl,
                flux=AT.get_tensor(0.0),
                op=e.op_id,
            )
        )

    N = len(fs_nodes)
    E = len(fs_edges)
    F = 0
    D0 = AT.zeros((E, N))
    for r, e in enumerate(fs_edges):
        D0[r, e.src] = -1.0
        D0[r, e.dst] = +1.0
    D1 = AT.zeros((F, E))
    H0 = AT.ones((N,))
    H1 = AT.ones((E,))
    H2 = AT.ones((F,))
    S_fe = AT.zeros((F, E))
    dec = DECSpec(
        D0=D0,
        D1=D1,
        H0=H0,
        H1=H1,
        H2=H2,
        S_fe=S_fe,
        node_rows=[n.id for n in fs_nodes],
    )
    spec = FluxSpringSpec(D=sys.D, nodes=fs_nodes, edges=fs_edges, dec=dec)

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
    out_mean_fn = group_data_mean_fn(outputs)

    sys.in_mean_fn = in_mean_fn
    sys.out_mean_fn = out_mean_fn
    now = now_s()
    sys.prev_in_mean = float(in_mean_fn(now))
    sys.prev_out_mean = float(out_mean_fn(now))
    sys.prev_mean_time = now

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

    return sys, outputs, spec


def build_round_node(sys: SpringRepulsorSystem, dt: float, table: StateTable) -> RoundNode:
    """Construct a RoundNode tree with curvature and spectral dampening."""

    spring_engine = SpringDtEngine(sys)
    targets = Targets(cfl=1.0, div_max=1.0, mass_max=1.0)
    ctrl = STController(dt_min=1e-6)
    state = StateNode(state=None)
    controller = ControllerNode(ctrl=ctrl, targets=targets, dx=1.0)

    def _curv(state_obj, dt, *, realtime: bool = False, state_table=None):
        pos_view = NodeAttrView(sys.nodes, "p", indices=sys.node_ids).build()
        pos = pos_view.tensor
        src = sys.edge_src_idx
        dst = sys.edge_dst_idx
        curv = hex_face_curvature_batch(pos[src], pos[dst])
        edge_view = NodeAttrView(sys.edge_list, "curvature").build()
        with edge_view.editing() as C:
            C[:] = curv
        return True, Metrics(0.0, 0.0, 0.0, 0.0), state_obj

    def _spectral(state_obj, dt, *, realtime: bool = False, state_table=None):
        v_view = NodeAttrView(sys.nodes, "v", indices=sys.node_ids).build()
        with v_view.editing() as V:
            for i, nid in enumerate(sys.node_ids):
                n = sys.nodes[nid]
                resp, _, _ = spectral_inertia(n.hist_p, dt)
                V[i] += -resp
        return True, Metrics(0.0, 0.0, 0.0, 0.0), state_obj

    def _spring(state_obj, dt, *, realtime: bool = False, state_table=None):
        ok, m, new_state = spring_engine.step(dt, state_table=state_table)
        return ok, m, new_state

    plan = SuperstepPlan(round_max=float(dt), dt_init=float(dt))
    children = [
        AdvanceNode(advance=_curv, state=state, label="advance:curvature"),
        AdvanceNode(advance=_spectral, state=state, label="advance:spectral"),
        AdvanceNode(advance=_spring, state=state, label="advance:spring"),
    ]
    return RoundNode(plan=plan, controller=controller, children=children, state_table=table)



def main(duration_s: float = 8.0, viz_mode: str = "none"):
    # Default to a large random batch driving the inputs
    sys, outputs, _spec = build_toy_system(seed=42, batch_size=10, batch_refresh_hz=15.0)

    stop = threading.Event()
    tick_hz = 30.0
    tick_dt = 1.0 / tick_hz

    table = StateTable()
    round_node = build_round_node(sys, tick_dt, table)
    round_engine = RoundNodeEngine(inner=round_node, runner=MetaLoopRunner(state_table=table))
    worker = ThreadedSystemEngine(round_engine, capture=capture_node_positions(sys))

    def _drive() -> None:
        while not stop.is_set():
            worker.step(tick_dt, state_table=table)

    drive_thread = threading.Thread(target=_drive, daemon=True)
    expr = Experiencer(sys, stop, outputs, schedule_hz=60.0, ops_program=sys.ops_program)

    print("[INFO] Starting threads…")
    drive_thread.start(); expr.start()
    t0 = now_s()

    if viz_mode == "gl":
        global matplotlib, mcolors, pygame, DOUBLEBUF, OPENGL, RESIZABLE, VIDEORESIZE, QUIT, KEYDOWN, K_SPACE
        global compileProgram, compileShader
        import matplotlib  # type: ignore
        from matplotlib import colors as mcolors  # type: ignore
        import pygame  # type: ignore
        from pygame.locals import DOUBLEBUF, OPENGL, RESIZABLE, VIDEORESIZE, QUIT, KEYDOWN, K_SPACE  # type: ignore
        import OpenGL.GL as _GL  # type: ignore
        from OpenGL.GL import shaders as _shaders  # type: ignore
        globals().update({k: getattr(_GL, k) for k in dir(_GL) if not k.startswith("_")})
        compileProgram = _shaders.compileProgram
        compileShader = _shaders.compileShader
        from ..pyopengl_handler import install_pyopengl_handlers
        install_pyopengl_handlers()
        viz = LiveVizGLPoints(sys, node_cmap="coolwarm", base_point_size=6.0)
        viz.launch()
    elif viz_mode == "mpl":
        global matplotlib, plt, FuncAnimation, mcolors
        import matplotlib  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.animation import FuncAnimation  # type: ignore
        from matplotlib import colors as mcolors  # type: ignore
        viz = LiveViz3D(sys)
        viz.launch()
    else:
        class _NullViz:
            def launch(self, *args, **kwargs):
                return None
            def step(self, _dt: float = 0.0):
                return None
            def close(self):
                return None
        viz = _NullViz()

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
                AbstractTensor.cat([
                    sys.nodes[oid].p.clone(),
                    sys.nodes[oid].phys.clone(),
                    sys.nodes[oid].ctrl.clone(),
                ]) if sys.nodes.get(oid) is not None else AbstractTensor.get_tensor([0.0] * 9)
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



    finally:
        stop.set()
        expr.join(timeout=2.0)
        drive_thread.join(timeout=2.0)
        worker.stop()
        print("[INFO] Stopped.")
        viz.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Spring-Repulsor Async Toy")
    parser.add_argument("--duration", type=float, default=float("inf"), help="run duration in seconds")
    parser.add_argument(
        "--viz",
        choices=["none", "mpl", "gl"],
        default="none",
        help="optional visualization backend",
    )
    args = parser.parse_args()
    main(duration_s=args.duration, viz_mode=args.viz)
