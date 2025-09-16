# -*- coding: utf-8 -*-
"""
FluxSpring spec builders for data-network experiments.

- make_classifier_spec(...) : one hidden slab, fully connected
  Inputs:  M spectral features (input nodes)
  Hidden:  H intermediate nodes
  Outputs: C class logits (output nodes)
  Edges:   fully-connected M→H and H→C
  DEC:     D0 from (src,dst), faces empty (D1 has 0 rows) ⇒ D1@D0 = 0
"""
from __future__ import annotations
from typing import List

from .fs_types import (
    FluxSpringSpec, NodeSpec, EdgeSpec, FaceSpec,
    NodeCtrl, EdgeCtrl, EdgeTransport, EdgeTransportLearn,
    LearnCtrl, DECSpec, DirichletCfg, RegCfg
)
from ...abstraction import AbstractTensor as AT

def _node(idx: int, D: int, scripted_axes=(0, 2)) -> NodeSpec:
    p0 = AT.zeros(D)
    v0 = AT.zeros(D)
    ctrl = NodeCtrl(
        alpha=AT.tensor(0.15),
        w=AT.tensor(1.0),
        b=AT.tensor(0.0),
        learn=LearnCtrl(True, True, True),
    )
    return NodeSpec(
        id=idx,
        p0=p0,
        v0=v0,
        mass=AT.tensor(1.0),
        ctrl=ctrl,
        scripted_axes=list(scripted_axes),
        temperature=AT.tensor(0.0),
        exclusive=False,
    )

def _edge(i: int, j: int) -> EdgeSpec:
    transport = EdgeTransport(
        kappa=AT.tensor(1.0),
        learn=EdgeTransportLearn(kappa=True, k=False, l0=False, lambda_s=False, x=False),
    )
    ctrl = EdgeCtrl(
        alpha=AT.tensor(0.15),
        w=AT.tensor(1.0),
        b=AT.tensor(0.0),
        learn=LearnCtrl(True, True, True),
    )
    return EdgeSpec(src=i, dst=j, transport=transport, ctrl=ctrl, temperature=AT.tensor(0.0), exclusive=False)

def make_classifier_spec(
    name: str,
    M: int,          # input feature count
    H: int,          # hidden width
    C: int,          # classes (output nodes = logits)
    *,
    D_geom: int = 3  # keep x/z scripted, y free if you later run physics
) -> FluxSpringSpec:
    version = f"{name}-fs-1.0"
    N = M + H + C
    nodes: List[NodeSpec] = [_node(i, D_geom) for i in range(N)]

    # Edges: fully connect M→H and H→C
    edges: List[EdgeSpec] = []
    in_ids   = list(range(0, M))
    hid_ids  = list(range(M, M+H))
    out_ids  = list(range(M+H, N))
    for j in hid_ids:
        for i in in_ids:
            edges.append(_edge(i, j))
    for k in out_ids:
        for j in hid_ids:
            edges.append(_edge(j, k))

    # Faces: none (you can add oriented triplets if you want curvature penalties)
    faces: List[FaceSpec] = []

    # DEC: build D0 from edges, and D1 with 0 rows (trivially satisfies ∂∘∂=0)
    E = len(edges)
    D0 = [[0.0]*N for _ in range(E)]
    for eidx, e in enumerate(edges):
        D0[eidx][e.src] = -1.0
        D0[eidx][e.dst] = +1.0
    D1: List[List[float]] = []  # 0×E

    dec = DECSpec(D0=D0, D1=D1)
    dirichlet = DirichletCfg(window=None, ema_beta=0.9, gain=3.0)
    regs = RegCfg(lambda_phi=0.0, mu_smooth=0.0, lambda_l0=0.0,
                  lambda_b=1e-6, lambda_c=0.0, lambda_w=1e-6)

    spec = FluxSpringSpec(
        version=version, D=D_geom,
        nodes=nodes, edges=edges, faces=faces,
        dec=dec, dirichlet=dirichlet, regularizers=regs,
    )
    return spec

def io_summary(spec: FluxSpringSpec) -> str:
    N = len(spec.nodes); E = len(spec.edges); F = len(spec.faces)
    M = "unknown"; H = "unknown"; C = "unknown"
    # best-effort split (only correct for make_classifier_spec layout)
    # try to infer by degree pattern
    indeg = [0]*N; outdeg = [0]*N
    for e in spec.edges:
        outdeg[e.src] += 1; indeg[e.dst] += 1
    in_nodes  = [i for i in range(N) if outdeg[i] > 0 and indeg[i] == 0]
    out_nodes = [i for i in range(N) if indeg[i] > 0 and outdeg[i] == 0]
    hid_nodes = [i for i in range(N) if i not in in_nodes and i not in out_nodes]
    if in_nodes and out_nodes:
        M, H, C = len(in_nodes), len(hid_nodes), len(out_nodes)
    return f"FluxSpringSpec[{spec.version}]: N={N} (M/H/C={M}/{H}/{C}), E={E}, F={F}, D={spec.D}"
