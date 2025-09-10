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
from .spectral_readout import (
    gather_recent_windows,
    batched_bandpower_from_windows,
)
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

def generate_signals(
    bands: list[list[float]],
    win: int,
    tick_hz: float,
    frames: int,
    seed: int = 0,
) -> tuple[list[AT.Tensor], list[list[AT.Tensor]]]:
    """Generate deterministic sine and noise signals for each band."""

    rng = np.random.default_rng(seed)
    centers = [(lo + hi) / 2.0 for lo, hi in bands]
    t = AT.arange(win, dtype=float) / tick_hz
    sine_chunks = [(2 * AT.pi() * c * t).sin() for c in centers]

    noise_frames: list[list[AT.Tensor]] = []
    for _ in range(frames):
        frame_chunks: list[AT.Tensor] = []
        for lo, hi in bands:
            freqs = AT.linspace(lo, hi, steps=3)
            n = AT.zeros(win, dtype=float)
            for f in freqs:
                n += (2 * AT.pi() * f * t).sin()
            noise = AT.tensor(rng.standard_normal(win))
            frame_chunks.append(n + 0.1 * noise)
        noise_frames.append(frame_chunks)
    return sine_chunks, noise_frames


def train_routing(
    spec: FluxSpringSpec,
    spectral_cfg: SpectralCfg,
    sine_chunks: list[AT.Tensor],
    noise_frames: list[list[AT.Tensor]],
) -> list[AT.Tensor]:
    """Run the spectral routing training loop."""

    params = register_learnable_params(spec)
    B = len(spectral_cfg.metrics.bands)
    psi = AT.zeros(len(spec.nodes), dtype=float)
    routed: list[AT.Tensor] = []
    out_start = 5 * B

    hist_targets: dict[int, AT.Tensor] = {}
    for j, nid in enumerate(range(3 * B, 4 * B)):
        tvec = AT.zeros(B, dtype=float)
        tvec[j] = 1.0
        hist_targets[nid] = tvec

    previous_grads = None
    def pump_with_loss(state: AT.Tensor, target_out: AT.Tensor) -> AT.Tensor:
        for p in params:
            if hasattr(p, "zero_grad"):
                p.zero_grad()
        state, _ = fs_dec.pump_tick(state, spec, eta=0.1, phi=AT.tanh, norm="all")
        loss_out = ((state[out_start : out_start + B] - target_out) ** 2).mean()
        mids = list(range(3 * B, 4 * B))
        window_matrix, kept_ids = gather_recent_windows(spec, mids, spectral_cfg)
        if window_matrix is not None and len(kept_ids) > 0:
            bp = batched_bandpower_from_windows(window_matrix, spectral_cfg)
            targ_mat = AT.stack([hist_targets[nid] for nid in kept_ids])
            hist_loss = ((bp - targ_mat) ** 2).mean()
        else:
            hist_loss = AT.tensor(0.0)
        loss = loss_out + hist_loss
        loss.backward()
        grads = []
        for p in params:
            if hasattr(p, "grad"):
                grads = grads + [p.grad]
        nonlocal previous_grads
        if previous_grads is not None:
            changed = False
            for idx, (g, pg) in enumerate(zip(grads, previous_grads)):
                if g is None and pg is not None:
                    changed = True
                    print(f"[Gradients] param {idx} lost gradient")
                elif g is not None and pg is None:
                    changed = True
                    print(f"[Gradients] param {idx} gained gradient")

                if g != pg:
                    changed = True
                    print(f"[Gradients] param {idx} gradient changed")
            if changed:
                print(f"[Gradients] previous: {previous_grads}")
                print(f"[Gradients] current:  {grads}")
        else:
            print(f"[Gradients] initial gradients: {grads}")
        previous_grads = grads
        print(f"loss: {loss_out.item():.6f}, hist_loss: {hist_loss.item():.6f}")
        return state

    win = sine_chunks[0].shape[0]
    B = len(sine_chunks)
    for frame_chunks in noise_frames:
        for k in range(win):
            for i in range(B):
                psi[i] = frame_chunks[i][k]
            target_out = AT.stack([sine_chunks[i][k] for i in range(B)])
            psi = pump_with_loss(psi, target_out)

        window_matrix, kept = gather_recent_windows(spec, list(range(B)), spectral_cfg)
        if window_matrix is not None and kept:
            bp = batched_bandpower_from_windows(window_matrix, spectral_cfg)
            for row, nid in enumerate(kept):
                psi[nid] = bp[row, nid]
        psi = pump_with_loss(psi, AT.zeros(B, dtype=float))
        out = [psi[out_start + i] for i in range(B)]
        routed.append(AT.get_tensor(out))

    # Report gradient status for all learnable parameters once outputs have been
    # produced.  This helps diagnose dead graphs where ``loss.backward`` fails
    # to populate ``grad`` fields.
    for idx, p in enumerate(params):
        grad = getattr(p, "grad", None)
        if grad is None:
            print(f"[Gradients] param {idx} missing gradient")
        else:
            g = AT.get_tensor(grad)
            if np.allclose(g, 0.0):
                print(f"[Gradients] param {idx} gradient is zero: {g}")
            else:
                print(f"[Gradients] param {idx} grad: {g}")

    return routed


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
    sine_chunks, noise_frames = generate_signals(bands, win, tick_hz, frames)
    routed = train_routing(spec, spectral_cfg, sine_chunks, noise_frames)
    print("Routed output:", routed)


if __name__ == "__main__":
    main()

