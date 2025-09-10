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
from ...autograd import autograd
from .spectral_readout import gather_ring_metrics, _window
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


def _ring_signal(n: NodeSpec) -> AT:
    """Return node ``n``'s ring buffer ordered by insertion time."""

    buf = n.ring[:, 0] if AT.get_tensor(n.ring).ndim == 2 else n.ring
    N = int(buf.shape[0])
    idx = n.ring_idx % N
    if idx == 0:
        return buf
    return AT.cat([buf[idx:], buf[:idx]], dim=0)


def _bandpower_fft(buffer: AT, cfg: SpectralCfg) -> AT:
    """Return band powers for ``buffer`` using the backend FFT."""

    x = buffer if buffer.ndim == 1 else buffer[:, 0]
    N = int(x.shape[0])
    w = _window(cfg.window, N)
    xw = w * x
    C = AT.rfft(xw, axis=0)
    real = AT.real(C)
    imag = AT.imag(C)
    power = real**2 + imag**2
    freqs = AT.rfftfreq(N, d=1.0 / cfg.tick_hz, like=xw)
    band_vals = []
    for lo, hi in cfg.metrics.bands:
        mask = ((freqs >= lo) & (freqs <= hi)).astype(float)
        band_vals.append(AT.sum(power * mask))
    return AT.stack(band_vals)


def _node(idx: int) -> NodeSpec:
    """Create a frozen linear node."""

    ctrl = NodeCtrl(
        alpha=AT.tensor(0.0),
        w=AT.tensor(1.0, requires_grad=True),
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
        w=AT.tensor(w, requires_grad=True),
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
    mixing layer and two post-mix identity layers).  Each layer contains
    one node per configured spectral band.  Only the central layer mixes
    features; all other edges carry identity weights.
    """

    layers = 6
    B = len(spectral.metrics.bands)
    nodes = [_node(i) for i in range(B * layers)]
    edges: list[EdgeSpec] = []

    def add_identity(src_start: int, dst_start: int) -> None:
        for k in range(B):
            edges.append(_edge(src_start + k, dst_start + k, 1.0))

    # Input → pre-mix stacks
    add_identity(0, B)
    add_identity(B, 2 * B)

    # Mixing layer
    Wmid = [[1.0 if i == j else 0.5 for j in range(B)] for i in range(B)]
    for i in range(B):
        for j in range(B):
            edges.append(_edge(2 * B + i, 3 * B + j, Wmid[i][j]))

    # Post-mix stacks
    add_identity(3 * B, 4 * B)
    add_identity(4 * B, 5 * B)

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

    bands = [[20, 40], [40, 60], [60, 80], [80, 100], [100, 120], [120, 140], [140, 160], [160, 180]]
    spectral_cfg = SpectralCfg(
        enabled=True,
        tick_hz=tick_hz,
        win_len=win,
        hop_len=win,
        window="hann",
        metrics=SpectralMetrics(bands=bands),
    )

    spec = build_spec(spectral_cfg)
    B = len(bands)
    psi = AT.zeros(len(spec.nodes), dtype=float)
    routed = []

    # Track node and edge weights for gradient logging
    edge_params = [e.ctrl.w for e in spec.edges]
    node_params = [n.ctrl.w for n in spec.nodes]
    params = edge_params + node_params
    out_start = 5 * B
    tick = 0

    # Frequency targets for middle layer histogram loss
    hist_targets: dict[int, AT] = {}
    for j, nid in enumerate(range(3 * B, 4 * B)):
        tvec = AT.zeros(B, dtype=float)
        tvec[j] = 1.0
        hist_targets[nid] = tvec

    def log_grads() -> None:
        e0 = float(AT.get_tensor(edge_params[0].grad).data.item())
        n0 = float(AT.get_tensor(node_params[0].grad).data.item())
        print(f"tick {tick}: edge0.w.grad={e0:.4f} node0.w.grad={n0:.4f}")

    def pump_with_loss(state: AT.Tensor, target_out: AT.Tensor) -> AT.Tensor:
        nonlocal tick
        state, _ = fs_dec.pump_tick(state, spec, eta=0.1, phi=AT.tanh)
        loss_out = ((state[out_start : out_start + B] - target_out) ** 2).mean()
        hist_loss = AT.tensor(0.0)
        for nid, targ in hist_targets.items():
            node = spec.nodes[nid]
            if node.ring is not None:
                buf = _ring_signal(node)
                bp = _bandpower_fft(buf, spectral_cfg)
                bp = bp / (AT.sum(bp) + 1e-12)
                hist_loss = hist_loss + ((bp - targ) ** 2).mean()
        loss = loss_out + hist_loss
        
        autograd.grad(loss, params, retain_graph=False, allow_unused=True)
        loss.zero_grad()
        log_grads()
        tick += 1
        return state.detach()

    centers = [(lo + hi) / 2.0 for lo, hi in bands]

    def band_noise(lo: float, hi: float) -> AT:
        t = AT.arange(win, dtype=float) / tick_hz
        freqs = AT.linspace(lo, hi, steps=3)
        n = AT.zeros(win, dtype=float)
        for f in freqs:
            n += (2 * AT.pi() * f * t).sin()
        return n + 0.1 * AT.randn((win,))

    sine_chunks = []
    t = AT.arange(win, dtype=float) / tick_hz
    for c in centers:
        sine_chunks.append((2 * AT.pi() * c * t).sin())

    for _ in range(frames):
        chunks = [band_noise(lo, hi) for lo, hi in bands]
        for k in range(win):
            for i in range(B):
                psi[i] = chunks[i][k]
            target_out = AT.stack([sine_chunks[i][k] for i in range(B)])
            psi = pump_with_loss(psi, target_out)

        ring_stats = gather_ring_metrics(spec)
        feats = [ring_stats[i]["bandpower"][i] for i in range(B)]
        for i, val in enumerate(feats):
            psi[i] = AT.tensor(val)
        psi = pump_with_loss(psi, AT.zeros(B, dtype=float))
        out = [psi[out_start + i] for i in range(B)]
        routed.append([float(AT.get_tensor(o).data.item()) for o in out])

    print("Routed output:", routed)


if __name__ == "__main__":
    main()

