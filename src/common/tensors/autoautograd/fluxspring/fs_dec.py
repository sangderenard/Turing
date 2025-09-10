# -*- coding: utf-8 -*-
"""AbstractTensor-based DEC helpers and transport updates for FluxSpring.

No torch usage here.
"""
from __future__ import annotations
from typing import Tuple, Dict, Optional
from ...abstraction import AbstractTensor as AT
from .fs_types import FluxSpringSpec

# --------- Incidence & validators ---------
def incidence_tensors_AT(spec: FluxSpringSpec):
    D0 = AT.get_tensor(spec.dec.D0).astype(float)  # (E,N)
    D1 = AT.get_tensor(spec.dec.D1).astype(float)  # (F,E)
    return D0, D1

def validate_boundary_of_boundary_AT(spec: FluxSpringSpec, tol: float = 1e-9) -> float:
    D0, D1 = incidence_tensors_AT(spec)
    bdry = D1 @ D0  # (F,N)
    nrm = float(AT.linalg.norm(bdry))
    if nrm > tol:
        raise ValueError(f"DEC violation: ||D1@D0||={nrm:.3e} > {tol}")
    return nrm

# --------- Geometry primitives ---------
def _edge_indices(spec: FluxSpringSpec):
    idx_i = AT.get_tensor([e.src for e in spec.edges], dtype=int)
    idx_j = AT.get_tensor([e.dst for e in spec.edges], dtype=int)
    return idx_i, idx_j

def edge_vectors_AT(P: AT.Tensor, spec: FluxSpringSpec) -> AT.Tensor:
    """
    P: (N,D) node positions
    returns d: (E,D) with order aligned to spec.edges
    """
    idx_i, idx_j = _edge_indices(spec)
    Pi = P.index_select(0, idx_i)
    Pj = P.index_select(0, idx_j)
    return Pj - Pi

def edge_params_AT(spec: FluxSpringSpec):
    k = AT.get_tensor([
        e.transport.k if e.transport.k is not None else AT.tensor(1.0)
        for e in spec.edges
    ]).astype(float)  # (E,)
    l0 = AT.get_tensor([
        e.transport.l0 if e.transport.l0 is not None else AT.tensor(1.0)
        for e in spec.edges
    ]).astype(float)  # (E,)
    return k, l0

def face_params_AT(spec: FluxSpringSpec):
    alpha = AT.get_tensor([fc.alpha for fc in spec.faces]).astype(float)  # (F,)
    c     = AT.get_tensor([fc.c     for fc in spec.faces]).astype(float)  # (F,)
    return alpha, c

def edge_strain_AT(P: AT.Tensor, spec: FluxSpringSpec, l0: AT.Tensor) -> AT.Tensor:
    d = edge_vectors_AT(P, spec)                      # (E,D)
    L = AT.linalg.norm(d, dim=1) + 1e-12              # (E,)
    return L - l0

def face_flux_AT(g: AT.Tensor, spec: FluxSpringSpec) -> AT.Tensor:
    D0, D1 = incidence_tensors_AT(spec)
    return D1 @ g  # (F,)

def curvature_activation_AT(z: AT.Tensor, alpha_face: AT.Tensor):
    t = AT.tanh(z)
    u = (1.0 - alpha_face) * z + alpha_face * t
    dphi = (1.0 - alpha_face) + alpha_face * (1.0 - t * t)
    return u, dphi

# --------- Energies ---------
def edge_energy_AT(P: AT.Tensor, spec: FluxSpringSpec, k: AT.Tensor, l0: AT.Tensor) -> AT.Tensor:
    g = edge_strain_AT(P, spec, l0)
    return 0.5 * AT.sum(k * g * g)


def face_energy_from_strain_AT(
    g: AT.Tensor, spec: FluxSpringSpec, alpha_face: AT.Tensor, c: AT.Tensor
):
    z = face_flux_AT(g, spec)
    u, dphi = curvature_activation_AT(z, alpha_face)
    E_face = 0.5 * AT.sum(c * u * u)
    return E_face, z, u, dphi


def total_energy_AT(P: AT.Tensor, spec: FluxSpringSpec, *, return_parts: bool = False):
    k, l0 = edge_params_AT(spec)
    alpha, c = face_params_AT(spec)
    g = edge_strain_AT(P, spec, l0)
    E_edge = 0.5 * AT.sum(k * g * g)
    z = face_flux_AT(g, spec)
    u, _ = curvature_activation_AT(z, alpha)
    E_face = 0.5 * AT.sum(c * u * u)
    E = E_edge + E_face
    if return_parts:
        return E, {"edge": E_edge, "face": E_face, "g": g, "z": z, "u": u}
    return E


def dec_energy_and_gradP_AT(P: AT.Tensor, spec: FluxSpringSpec):
    """Analytic gradient of total energy wrt node positions ``P``."""

    k, l0 = edge_params_AT(spec)
    alpha, c = face_params_AT(spec)
    idx_i, idx_j = _edge_indices(spec)
    N = len(spec.nodes)

    # edge geometry
    d = edge_vectors_AT(P, spec)                       # (E,D)
    L = AT.linalg.norm(d, dim=1) + 1e-12               # (E,)
    uhat = d / L[:, None]                              # (E,D)
    g = L - l0                                         # (E,)

    # face flux & backpressure
    D0, D1 = incidence_tensors_AT(spec)
    z = D1 @ g                                         # (F,)
    u, dphi = curvature_activation_AT(z, alpha)        # (F,)
    r = k * g + (D1.T() @ (c * u * dphi))              # (E,)

    # accumulate to node grads
    gradP = AT.zeros_like(P)
    for eidx in range(len(spec.edges)):
        i = int(idx_i[eidx])
        j = int(idx_j[eidx])
        gradP[i] += -r[eidx] * uhat[eidx]
        gradP[j] += +r[eidx] * uhat[eidx]
    E = 0.5 * AT.sum(k * g * g) + 0.5 * AT.sum(c * u * u)
    return E, gradP


def path_edge_energy_AT(
    P: AT.Tensor, spec: FluxSpringSpec, edge_indices_1based: list[int]
) -> AT.Tensor:
    k, l0 = edge_params_AT(spec)
    g_all = edge_strain_AT(P, spec, l0)
    idx = AT.get_tensor([i - 1 for i in edge_indices_1based], dtype=int)
    g = g_all.index_select(0, idx)
    k_sel = k.index_select(0, idx)
    return 0.5 * AT.sum(k_sel * g * g)


# --------- Transport ---------
def transport_tick(
    psi: AT.Tensor,
    spec: FluxSpringSpec,
    *,
    eta: float,
    P: AT.Tensor | None = None,
    phi=AT.tanh,
) -> Tuple[AT.Tensor, Dict[str, AT.Tensor]]:
    """Advance node potentials with transport params and geometry coupling."""

    D0, _ = incidence_tensors_AT(spec)
    dpsi = D0 @ psi  # (E,)

    kappa = (
        AT.get_tensor([e.transport.kappa for e in spec.edges])
        .astype(float)
        .reshape(-1)
    )  # (E,)

    if P is not None:
        k = (
            AT.get_tensor([
                e.transport.k if e.transport.k is not None else AT.tensor(0.0)
                for e in spec.edges
            ])
            .astype(float)
            .reshape(-1)
        )
        l0 = (
            AT.get_tensor([
                e.transport.l0 if e.transport.l0 is not None else AT.tensor(0.0)
                for e in spec.edges
            ])
            .astype(float)
            .reshape(-1)
        )
        lambda_s = (
            AT.get_tensor([
                e.transport.lambda_s
                if e.transport.lambda_s is not None
                else AT.tensor(0.0)
                for e in spec.edges
            ])
            .astype(float)
            .reshape(-1)
        )

        g = edge_strain_AT(P, spec, l0)
        G = lambda_s * k * g
    else:
        G = AT.zeros_like(kappa)

    x = (
        AT.get_tensor([
            e.transport.x if e.transport.x is not None else AT.tensor(0.0)
            for e in spec.edges
        ])
        .astype(float)
        .reshape(-1)
    )

    gamma = AT.get_tensor(spec.gamma).astype(float).reshape(-1)

    R = gamma * x

    delta = dpsi + G + R
    q = kappa * phi(delta)

    s = D0.T() @ q  # node balances
    psi_next = psi + eta * s

    x_new = x + eta * q
    for e, x_val in zip(spec.edges, x_new):
        e.transport.x = AT.get_tensor(float(x_val))

    stats = {"q": q, "dpsi": dpsi, "G": G, "R": R, "s": s}
    return psi_next, stats


# --------- Data pump ---------
def pump_tick(
    psi: AT.Tensor,
    spec: FluxSpringSpec,
    *,
    eta: float,
    phi=AT.tanh,
    external: Optional[Dict[int, AT.Tensor]] = None,
) -> Tuple[AT.Tensor, Dict[str, AT.Tensor]]:
    """Advance node potentials via data-path control parameters.

    Parameters
    ----------
    psi : (N,) ``AT``
        Node potentials for the graph.  Gradients may flow back to ``psi``.
    spec : ``FluxSpringSpec``
        Graph specification providing control parameters.
    eta : float
        Step size for the Euler update.
    phi : callable, optional
        Elementwise activation used on both edges and nodes.
    external : dict, optional
        Mapping of ``node_id → value`` used to inject fresh inputs before the
        tick.  Values are written via :func:`AT.scatter_row` so they remain on
        the autograd tape.
    """

    # Inject fresh external inputs before computing edge potentials. ``ids`` is
    # (K,) and ``vals`` is (K,); scatter_row preserves autograd history.
    if external:
        ids = list(external.keys())
        vals = AT.stack([external[i] for i in ids]).reshape(-1)
        psi = AT.scatter_row(psi.clone(), ids, vals, dim=0)

    D0, _ = incidence_tensors_AT(spec)
    dpsi = D0 @ psi  # (E,) edge potential differences

    # Edge controls (E,)
    alpha_e = AT.stack([e.ctrl.alpha for e in spec.edges]).reshape(-1)
    # hard-bound to [0,1]; if AT.clip is unavailable, replace with max(min(...))
    alpha_e = AT.clip(alpha_e, 0.0, 1.0)
    w_e     = AT.stack([e.ctrl.w     for e in spec.edges]).reshape(-1)
    b_e     = AT.stack([e.ctrl.b     for e in spec.edges]).reshape(-1)
    edge_raw = dpsi + b_e
    # wet mix: (1-α)*raw + α*φ(raw)
    edge_u   = (1.0 - alpha_e) * edge_raw + alpha_e * phi(edge_raw)
    q        = w_e * edge_u
    s = D0.T() @ q  # (N,) accumulated edge contributions

    # Node controls (N,)
    alpha_n = AT.stack([n.ctrl.alpha for n in spec.nodes]).reshape(-1)
    alpha_n = AT.clip(alpha_n, 0.0, 1.0)
    w_n     = AT.stack([n.ctrl.w     for n in spec.nodes]).reshape(-1)
    b_n     = AT.stack([n.ctrl.b     for n in spec.nodes]).reshape(-1)
    node_raw = s + b_n
    node_u   = (1.0 - alpha_n) * node_raw + alpha_n * phi(node_raw)
    delta    = w_n * node_u

    psi_next = psi + eta * delta  # (N,) updated node potentials
    stats = {
        "dpsi": dpsi, "q": q, "s": s, "delta": delta,
        "edge_raw": edge_raw, "edge_u": edge_u,
        "node_raw": node_raw, "node_u": node_u,
        "alpha_e": alpha_e, "alpha_n": alpha_n,
    }

    # Maintain ring buffers for spectral analysis.
    if spec.spectral.enabled:
        for n, val in zip(spec.nodes, psi_next):
            if n.ring is not None:
                r = AT.get_tensor(n.ring)
                D = int(r.shape[1]) if r.ndim == 2 else 1
                n.push_ring(AT.ones(D, dtype=float) * val)
        for e, q_val in zip(spec.edges, q):
            if e.ring is not None:
                e.push_ring(q_val)

    return psi_next, stats
