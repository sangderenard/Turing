from __future__ import annotations
from typing import List

from .fs_types import (
    FluxSpringSpec,
    NodeSpec,
    EdgeSpec,
    DECSpec,
    NodeCtrl,
    EdgeCtrl,
    EdgeTransport,
    EdgeTransportLearn,
    LearnCtrl,
)
from ...abstraction import AbstractTensor as AT


def make_classifier_spec(
    name: str,
    M: int,
    H: int,
    C: int,
    *,
    D_geom: int = 3,
) -> FluxSpringSpec:
    """Build a minimal FluxSpringSpec for a one-hidden-layer classifier."""
    version = f"{name}-fs-1.0"
    N = M + H + C

    nodes: List[NodeSpec] = []
    for i in range(N):
        ctrl = NodeCtrl(
            alpha=AT.tensor(0.0),
            w=AT.tensor(1.0),
            b=AT.tensor(0.0),
            learn=LearnCtrl(True, True, True),
        )
        nodes.append(
            NodeSpec(
                id=i,
                p0=AT.zeros(D_geom),
                v0=AT.zeros(D_geom),
                mass=AT.tensor(1.0),
                ctrl=ctrl,
                scripted_axes=[0, 2] if D_geom >= 3 else list(range(min(2, D_geom))),
                temperature=AT.tensor(0.0),
                exclusive=False,
            )
        )

    def _edge(i: int, j: int) -> EdgeSpec:
        transport = EdgeTransport(
            kappa=AT.tensor(1.0),
            k=AT.tensor(1.0),
            l0=AT.tensor(1.0),
            learn=EdgeTransportLearn(kappa=True, k=False, l0=False, lambda_s=False, x=False),
        )
        ctrl = EdgeCtrl(
            alpha=AT.tensor(0.0),
            w=AT.tensor(1.0),
            b=AT.tensor(0.0),
            learn=LearnCtrl(True, True, True),
        )
        return EdgeSpec(src=i, dst=j, transport=transport, ctrl=ctrl, temperature=AT.tensor(0.0), exclusive=False)

    edges: List[EdgeSpec] = []
    in_ids = list(range(0, M))
    hid_ids = list(range(M, M + H))
    out_ids = list(range(M + H, N))
    for j in hid_ids:
        for i in in_ids:
            edges.append(_edge(i, j))
    for k in out_ids:
        for j in hid_ids:
            edges.append(_edge(j, k))

    E = len(edges)
    D0 = [[0.0] * N for _ in range(E)]
    for r, e in enumerate(edges):
        D0[r][e.src] = -1.0
        D0[r][e.dst] = 1.0
    D1: List[List[float]] = []

    dec = DECSpec(D0=D0, D1=D1)

    spec = FluxSpringSpec(
        version=version,
        D=D_geom,
        nodes=nodes,
        edges=edges,
        faces=[],
        dec=dec,
    )
    return spec

