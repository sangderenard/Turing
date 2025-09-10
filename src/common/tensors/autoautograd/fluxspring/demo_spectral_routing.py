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
from .spectral_readout import compute_metrics
from . import fs_dec, register_learnable_params
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
import numpy as np


def _node(idx: int) -> NodeSpec:
    """Create a frozen linear node."""

    ctrl = NodeCtrl(
        alpha=AT.tensor(0.0),
        w=AT.tensor(1.0),
        b=AT.tensor(0.0),
        # Enable learning for all ctrl parameters so gradients propagate to
        # alpha, weight and bias alike.
        learn=LearnCtrl(True, True, True),
    )
    return NodeSpec(
        id=idx,
        p0=AT.zeros(3),
        v0=AT.zeros(3),
        mass=AT.get_tensor(1.0),
        ctrl=ctrl,
        scripted_axes=[0, 2],
        temperature=AT.get_tensor(0.0),
        exclusive=False,
    )


def _edge(i: int, j: int, w: float) -> EdgeSpec:
    """Create a frozen linear edge with weight ``w``."""

    ctrl = EdgeCtrl(
        alpha=AT.tensor(0.0),
        w=AT.tensor(w),
        b=AT.tensor(0.0),
        # Train all ctrl parameters on edges as well.
        learn=LearnCtrl(True, True, True),
    )
    transport = EdgeTransport(
        kappa=AT.get_tensor(1.0, requires_grad=True),
        learn=EdgeTransportLearn(kappa=True, k=True, l0=True, lambda_s=True, x=True),
    )
    return EdgeSpec(src=i, dst=j, transport=transport, ctrl=ctrl, temperature=AT.get_tensor(0.0), exclusive=False)


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
    win = 40
    frames = 50

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
    params = register_learnable_params(spec)
    B = len(bands)
    psi = AT.zeros(len(spec.nodes), dtype=float)
    routed = []
    out_start = 5 * B
    tick = 0

    # Frequency targets for middle layer histogram loss
    hist_targets: dict[int, AT] = {}
    for j, nid in enumerate(range(3 * B, 4 * B)):
        tvec = AT.zeros(B, dtype=float)
        tvec[j] = 1.0
        hist_targets[nid] = tvec

    def log_grads() -> None:
        grads_np = []
        for idx, p in enumerate(params):
            for attr in ("alpha", "w", "b"):
                if hasattr(p, attr):
                    pa = getattr(p, attr)
                    if pa.grad is None:
                        print(f"tick {tick}: param{idx}.{attr} no grad")
                        continue
                    ga = AT.get_tensor(pa.grad)
                    ga_np = AT.to_numpy(ga)
                    grads_np.append(ga_np.reshape(-1))
                    va_np = AT.to_numpy(AT.get_tensor(pa))
                    print(
                        f"tick {tick}: param{idx}.{attr} value shape={va_np.shape} value={va_np} grad shape={ga_np.shape} grad={ga_np}"
                    )
        if grads_np:
            all_grads = np.concatenate(grads_np)
            g_min = all_grads.min()
            g_max = all_grads.max()
            g_mean = all_grads.mean()
            g_std = all_grads.std()
            print(
                f"tick {tick}: grad stats min={g_min} max={g_max} mean={g_mean} std={g_std}"
            )



    # --- MOD: pump_with_loss (replace your existing one) --------------------------

    def node_bandpower(nid: int) -> AT | None:
        n = spec.nodes[nid]
        if getattr(n, "ring", None) is None:
            return None
        buf = n.ring[:, 0] if AT.get_tensor(n.ring).ndim == 2 else n.ring
        R = int(buf.shape[0])
        idx = n.ring_idx % R if R > 0 else 0
        if R > 0 and idx != 0:
            buf = AT.cat([buf[idx:], buf[:idx]], dim=0)
        m = compute_metrics(buf, spectral_cfg)
        return m.get("bandpower")

    def pump_with_loss(state: AT.Tensor, target_out: AT.Tensor) -> AT.Tensor:
        nonlocal tick
        state, _ = fs_dec.pump_tick(state, spec, eta=0.1, phi=AT.tanh, norm="all")

        # output loss
        loss_out = ((state[out_start: out_start + B] - target_out) ** 2).mean()

        # histogram loss over middle layer, with per-node latest windows (no cross-node sync)
        mids = list(range(3 * B, 4 * B))
        bp_rows: list[AT] = []
        kept_ids: list[int] = []
        for nid in mids:
            bp = node_bandpower(nid)
            if bp is not None:
                bp_rows.append(bp)
                kept_ids.append(nid)
        if bp_rows:
            bp_mat = AT.stack(bp_rows)
            targ_mat = AT.stack([hist_targets[nid] for nid in kept_ids])
            hist_loss = ((bp_mat - targ_mat) ** 2).mean()
        else:
            hist_loss = AT.tensor(0.0)

        loss = loss_out + hist_loss

        grads = autograd.grad(loss, params, retain_graph=False, allow_unused=True)
        print(grads)
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

        # Preserve tensor metrics so they can influence the subsequent loss
        for nid in range(B):
            bp = node_bandpower(nid)
            if bp is not None:
                psi[nid] = bp[nid]
        psi = pump_with_loss(psi, AT.zeros(B, dtype=float))
        out = [psi[out_start + i] for i in range(B)]
        routed.append(AT.get_tensor(out))

    print("Routed output:", routed)


if __name__ == "__main__":
    main()

