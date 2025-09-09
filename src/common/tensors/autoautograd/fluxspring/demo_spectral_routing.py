"""FluxSpring spectral routing demonstration.

This script showcases how spectral features extracted from a time
domain buffer can drive a FluxSpring data graph.  Three sine bands are
analysed via :func:`compute_metrics` and the resulting band powers are
fed into a small FluxSpring spec consisting of two identity stacks
bracketing a mixing layer.  The graph is executed directly using
AbstractTensor operations to apply the edge weights encoded in the spec.
"""

from __future__ import annotations

from ...abstraction import AbstractTensor as AT
from .spectral_readout import compute_metrics
from .fs_types import (
    DECSpec,
    EdgeCtrl,
    EdgeSpec,
    EdgeTransport,
    EdgeTransportLearn,
    FluxSpringSpec,
    LearnCtrl,
    NodeCtrl,
    NodeSpec,
    SpectralCfg,
    SpectralMetrics,
)
from .fs_io import validate_fluxspring


def _node(idx: int) -> NodeSpec:
    """Create a frozen linear node."""

    ctrl = NodeCtrl(
        alpha=AT.tensor(0.0),
        w=AT.tensor(1.0),
        b=AT.tensor(0.0),
        learn=LearnCtrl(False, False, False),
    )
    return NodeSpec(
        id=idx,
        p0=AT.zeros(3),
        v0=AT.zeros(3),
        mass=AT.tensor(1.0),
        ctrl=ctrl,
        scripted_axes=[0, 2],
        temperature=AT.tensor(0.0),
        exclusive=False,
    )


def _edge(i: int, j: int, w: float) -> EdgeSpec:
    """Create a frozen linear edge with weight ``w``."""

    ctrl = EdgeCtrl(
        alpha=AT.tensor(0.0),
        w=AT.tensor(w),
        b=AT.tensor(0.0),
        learn=LearnCtrl(False, False, False),
    )
    transport = EdgeTransport(
        kappa=AT.tensor(1.0),
        learn=EdgeTransportLearn(kappa=False, k=False, l0=False, lambda_s=False, x=False),
    )
    return EdgeSpec(src=i, dst=j, transport=transport, ctrl=ctrl, temperature=AT.tensor(0.0), exclusive=False)


def build_spec(spectral: SpectralCfg) -> FluxSpringSpec:
    """Construct the demo FluxSpringSpec.

    The graph has six layers (input, two pre-mix identity layers, a
    mixing layer and two post-mix identity layers) with three nodes per
    layer.  Only the central layer mixes features; all other edges carry
    identity weights.
    """

    layers = 6
    nodes = [_node(i) for i in range(3 * layers)]
    edges: list[EdgeSpec] = []

    def add_identity(src_start: int, dst_start: int) -> None:
        for k in range(3):
            edges.append(_edge(src_start + k, dst_start + k, 1.0))

    # Input → pre-mix stacks
    add_identity(0, 3)
    add_identity(3, 6)

    # Mixing layer
    Wmid = [
        [1.0, 0.5, 0.0],
        [0.0, 1.0, 0.5],
        [0.5, 0.0, 1.0],
    ]
    for i in range(3):
        for j in range(3):
            edges.append(_edge(6 + i, 9 + j, Wmid[i][j]))

    # Post-mix stacks
    add_identity(9, 12)
    add_identity(12, 15)

    E = len(edges)
    N = len(nodes)
    D0 = [[0.0] * N for _ in range(E)]
    for r, e in enumerate(edges):
        D0[r][e.src] = -1.0
        D0[r][e.dst] = 1.0
    dec = DECSpec(D0=D0, D1=[])

    spec = FluxSpringSpec(
        version="spectral-demo-fs-1.0",
        D=3,
        nodes=nodes,
        edges=edges,
        faces=[],
        dec=dec,
        spectral=spectral,
    )
    validate_fluxspring(spec)
    return spec


def main() -> None:
    tick_hz = 400.0
    N = 400
    t = AT.arange(N, dtype=float) / tick_hz
    freqs = [40.0, 80.0, 160.0]
    sig = AT.stack([(2 * AT.pi() * f * t).sin() for f in freqs], dim=1)

    spectral_cfg = SpectralCfg(
        enabled=True,
        tick_hz=tick_hz,
        win_len=N,
        hop_len=N,
        window="hann",
        metrics=SpectralMetrics(bands=[[30, 50], [70, 90], [150, 170]]),
    )
    metrics = compute_metrics(sig, spectral_cfg)
    feats = metrics["bandpower"]

    spec = build_spec(spectral_cfg)

    # Run a simple forward pass using the edge weights in the spec.
    activ = AT.zeros(len(spec.nodes), dtype=float)
    for i, val in enumerate(feats):
        activ[i] = AT.tensor(val)
    for e in spec.edges:
        w = float(AT.get_tensor(e.ctrl.w).data.item())
        b = float(AT.get_tensor(e.ctrl.b).data.item())
        activ[e.dst] = activ[e.dst] + w * activ[e.src] + b
    out = activ[15:18]

    print("Band powers:", feats)
    print("Routed output:", [float(AT.get_tensor(o).data.item()) for o in out])


if __name__ == "__main__":
    main()

