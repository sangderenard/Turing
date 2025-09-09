# -*- coding: utf-8 -*-
"""
AbstractTensor-based DEC helpers & energies for FluxSpring.
No torch usage here. Analytic ∂E/∂P provided.
"""
from __future__ import annotations
from typing import Tuple, List, Dict
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
    k  = AT.get_tensor([e.hooke.k for e in spec.edges]).astype(float)   # (E,)
    l0 = AT.get_tensor([e.hooke.l0 for e in spec.edges]).astype(float)  # (E,)
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

def face_energy_from_strain_AT(g: AT.Tensor, spec: FluxSpringSpec, alpha_face: AT.Tensor, c: AT.Tensor):
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
    """
    Analytic gradient of total energy wrt node positions P (N,D).
    Matches the form used in your engine:
      ∂E/∂p_i = -Σ_{e=(i->j)} r_e u_e + Σ_{e=(k->i)} r_e u_e
      where r = k*g + D1^T ( c * u * dphi )
    """
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
        i = int(idx_i[eidx]); j = int(idx_j[eidx])
        gradP[i] += -r[eidx] * uhat[eidx]
        gradP[j] += +r[eidx] * uhat[eidx]
    E = 0.5 * AT.sum(k * g * g) + 0.5 * AT.sum(c * u * u)
    return E, gradP

def path_edge_energy_AT(P: AT.Tensor, spec: FluxSpringSpec, edge_indices_1based: List[int]) -> AT.Tensor:
    k, l0 = edge_params_AT(spec)
    g_all = edge_strain_AT(P, spec, l0)
    # convert to 0-based
    idx = AT.get_tensor([i-1 for i in edge_indices_1based], dtype=int)
    g = g_all.index_select(0, idx)
    k_sel = k.index_select(0, idx)
    return 0.5 * AT.sum(k_sel * g * g)
