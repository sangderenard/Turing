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
from .spectral_readout import gather_ring_metrics
from . import fs_dec
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
    win = 400
    frames = 5

    bands = [[30, 50], [70, 90], [150, 170]]
    spectral_cfg = SpectralCfg(
        enabled=True,
        tick_hz=tick_hz,
        win_len=win,
        hop_len=win,
        window="hann",
        metrics=SpectralMetrics(bands=bands),
    )

    spec = build_spec(spectral_cfg)
    psi = AT.zeros(len(spec.nodes), dtype=float)
    routed = []

    def band_noise(lo: float, hi: float) -> AT:
        t = AT.arange(win, dtype=float) / tick_hz
        freqs = AT.linspace(lo, hi, steps=3)
        n = AT.zeros(win, dtype=float)
        for f in freqs:
            n += (2 * AT.pi() * f * t).sin()
        return n + 0.1 * AT.randn(win)

    for _ in range(frames):
        chunks = [band_noise(lo, hi) for lo, hi in bands]
        for k in range(win):
            for i in range(3):
                psi[i] = chunks[i][k]
            psi, _ = fs_dec.pump_tick(psi, spec, eta=0.1, phi=AT.tanh)

        ring_stats = gather_ring_metrics(spec)
        feats = [ring_stats[i]["bandpower"][i] for i in range(3)]
        for i, val in enumerate(feats):
            psi[i] = AT.tensor(val)
        psi, _ = fs_dec.pump_tick(psi, spec, eta=0.1, phi=AT.tanh)
        out = [psi[15], psi[16], psi[17]]
        routed.append([float(AT.get_tensor(o).data.item()) for o in out])

    print("Routed output:", routed)


if __name__ == "__main__":
    main()

