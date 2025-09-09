# fluxspring_spec_tensor_min.py
# Spec-only: AbstractTensor fields, no helpers, no integrator, no faces.

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from ...abstraction import AbstractTensor as AT

# ----------------------------- Nodes -----------------------------------------

@dataclass
class NodeSpec:
    id: int
    p: AT            # (D,)
    v: AT            # (D,)
    phys: AT         # (D,) Dirichlet targets
    mask: AT         # (D,) geometry mask
    ctrl: AT         # (P,) data-path scalars (free width)
    mass: AT         # scalar tensor

    # per-node IO mirrors (scalar tensors)
    in_value: AT
    out_value: AT
    in_target: AT
    out_target: AT

    role: str = ""   # optional tag

    def sync_io(self) -> None:
        """Refresh per-node IO mirrors from p/phys (x=0, z=2 if present)."""
        D = int(getattr(self.p, "shape", (0,))[0]) if hasattr(self.p, "shape") else 0
        if D > 0:
            self.in_value  = self.p[0]
            self.in_target = self.phys[0]
        if D > 2:
            self.out_value  = self.p[2]
            self.out_target = self.phys[2]

# ----------------------------- Edges -----------------------------------------

@dataclass
class EdgeSpec:
    # Row-aligned with D0 (E x N)
    eid_1: int       # 1..E (human-friendly id)
    row_idx: int     # 0..E-1 (D0 row index)
    src: int
    dst: int

    # geometric / control
    k: AT            # scalar tensor
    l0: AT           # scalar tensor
    h1: AT           # scalar tensor (Hodge-1 weight)
    ctrl: AT         # (Q,) edge data-path scalars (free width)

    # learnable parameters and optional resonance state
    theta: AT        # raw parameter for kappa
    kappa: AT        # non-negative spring gain
    x: Optional[AT] = None  # optional resonance state

    op: str = ""     # optional operator tag

# ----------------------------- DEC -------------------------------------------

@dataclass
class DECSpec:
    D0: AT                  # (E, N) edge–node incidence
    D1: AT                  # (F, E) face–edge incidence
    H0: AT                  # (N,) Hodge-0 diagonal (node volumes)
    H1: AT                  # (E,) Hodge-1 diagonal (edge lengths)
    H2: AT                  # (F,) Hodge-2 diagonal (face areas)
    S_fe: AT                # (F, E) face→edge stencil (flux region per face over edges)
    node_rows: List[int]    # node id order matching D0 columns

# ----------------------------- Top-level Spec --------------------------------

@dataclass
class FluxSpringSpec:
    D: int
    nodes: List[NodeSpec]
    edges: List[EdgeSpec]
    dec: DECSpec

# ----------------------------- Minimal, non-redundant builder ----------------

def make_classifier_spec(
    name: str,
    M: int,          # input feature count
    H: int,          # hidden width
    C: int,          # output count (classes)
    *,
    D_geom: int = 3  # e.g., keep x/z scripted, y free
) -> FluxSpringSpec:
    N = M + H + C

    # Nodes (no helpers; constant-time defaults)
    nodes: List[NodeSpec] = []
    for i in range(N):
        p    = AT.zeros((D_geom,))
        v    = AT.zeros((D_geom,))
        phys = AT.zeros((D_geom,))
        mask = AT.ones((D_geom,))
        ctrl = AT.get_tensor([0.0])     # (P,)= (1,) default; free width upstream
        mass = AT.get_tensor(1.0)       # scalar
        z    = AT.get_tensor(0.0)       # scalar mirrors init
        nodes.append(NodeSpec(
            id=i, p=p, v=v, phys=phys, mask=mask, ctrl=ctrl, mass=mass,
            in_value=z, out_value=z, in_target=z, out_target=z, role=""
        ))

    # Id bands
    in_ids  = list(range(0, M))
    hid_ids = list(range(M, M + H))
    out_ids = list(range(M + H, N))

    # Edges (fully connect M→H and H→C; no nested loop boilerplate kept)
    edges: List[EdgeSpec] = []
    # M → H
    for r, (i, j) in enumerate((ii, jj) for jj in hid_ids for ii in in_ids):
        theta = AT.get_tensor(0.0)
        kappa = AT.softplus(theta) if hasattr(AT, 'softplus') else (theta.exp() + 1.0).log()
        edges.append(EdgeSpec(
            eid_1=r + 1, row_idx=r, src=i, dst=j,
            k=AT.get_tensor(1.0), l0=AT.get_tensor(1.0),
            h1=AT.get_tensor(1.0), ctrl=AT.get_tensor([0.0]),
            theta=theta, kappa=kappa, x=None, op="",
        ))
    # H → C
    base = len(edges)
    for t, (j, k_) in enumerate((jj, kk) for kk in out_ids for jj in hid_ids):
        r = base + t
        theta = AT.get_tensor(0.0)
        kappa = AT.softplus(theta) if hasattr(AT, 'softplus') else (theta.exp() + 1.0).log()
        edges.append(EdgeSpec(
            eid_1=r + 1, row_idx=r, src=j, dst=k_,
            k=AT.get_tensor(1.0), l0=AT.get_tensor(1.0),
            h1=AT.get_tensor(1.0), ctrl=AT.get_tensor([0.0]),
            theta=theta, kappa=kappa, x=None, op="",
        ))

    # DEC (exactly what was requested; faces not used here → F=0)
    E = len(edges)
    F = 0  # no faces in this minimal classifier graph

    # D0: (E, N)
    D0 = AT.zeros((E, N))
    for r, e in enumerate(edges):
        D0[r, e.src] = -1.0
        D0[r, e.dst] = +1.0

    # D1: (F, E) = (0, E)
    D1 = AT.zeros((F, E))

    # Hodge stars: (N,), (E,), (F,)
    H0 = AT.ones((N,))
    H1 = AT.ones((E,))
    H2 = AT.ones((F,))  # empty if F==0

    # S_fe: (F, E)
    S_fe = AT.zeros((F, E))

    dec = DECSpec(
        D0=D0, D1=D1,
        H0=H0, H1=H1, H2=H2,
        S_fe=S_fe,
        node_rows=[n.id for n in nodes],
    )

    return FluxSpringSpec(D=D_geom, nodes=nodes, edges=edges, dec=dec)
