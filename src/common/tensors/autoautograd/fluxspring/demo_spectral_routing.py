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
        # Train all ctrl parameters on edges as well.
        learn=LearnCtrl(True, True, True),
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
            if p.grad is None:
                continue
            g = AT.get_tensor(p.grad)
            g_np = AT.to_numpy(g)
            grads_np.append(g_np.reshape(-1))
            val_np = AT.to_numpy(AT.get_tensor(p))
            print(
                f"tick {tick}: param{idx} value shape={val_np.shape} value={val_np} grad shape={g_np.shape} grad={g_np}"
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

    # --- NEW: per-node aligned window gather -------------------------------------

    def _gather_recent_windows(spec: FluxSpringSpec, node_ids: list[int], cfg: SpectralCfg) -> tuple[AT | None, list[int]]:
        """
        Returns (W, kept) where W has shape (M, Nw) with each row the most-recent,
        time-contiguous window for that node (i.e., ended at that node's own ring_idx).
        Uses _ring_signal(n) to eliminate the discontinuity at the ring wrap.
        """
        Nw = int(cfg.win_len)
        wins: list[AT] = []
        kept: list[int] = []

        for nid in node_ids:
            n = spec.nodes[nid]
            if getattr(n, "ring", None) is None:
                continue
            # _ring_signal returns the buffer in chronological order (oldest→newest)
            buf = _ring_signal(n)  # (R,)
            R = int(buf.shape[0])
            if R >= Nw:
                win = buf[-Nw:]     # latest Nw samples for THIS node (no discontinuity)
            else:
                # ring smaller than window: left-pad to fixed length for stable FFT
                pad = Nw - R
                win = AT.cat([AT.zeros(pad, dtype=AT.get_tensor(buf).dtype), buf], dim=0)
            wins.append(win)
            kept.append(nid)

        if not wins:
            return None, []
        return AT.stack(wins), kept  # (M, Nw), node ids


    def _batched_bandpower_from_windows(window_matrix: AT, cfg: SpectralCfg) -> AT:
        """
        window_matrix: (M, Nw) most-recent, per-node aligned windows
        returns: (M, B) normalized band powers
        """
        Nw = int(window_matrix.shape[-1])
        w = _window(cfg.window, Nw)                  # (Nw,)
        xw = window_matrix * w                       # (M, Nw)

        C = AT.rfft(xw, axis=-1)                     # (M, F), F = Nw//2 + 1
        real, imag = AT.real(C), AT.imag(C)
        power = real**2 + imag**2                    # (M, F)

        freqs = AT.rfftfreq(Nw, d=1.0 / cfg.tick_hz, like=xw)  # (F,)
        band_rows = [(((freqs >= lo) & (freqs <= hi)).astype(float)) for lo, hi in cfg.metrics.bands]
        mask_FB = AT.stack(band_rows).transpose(0, 1)          # (F, B)

        band_powers = AT.matmul(power, mask_FB)      # (M, B)
        band_powers = band_powers / (AT.sum(band_powers, dim=1, keepdim=True) + 1e-12)
        return band_powers


    # --- MOD: pump_with_loss (replace your existing one) --------------------------

    def pump_with_loss(state: AT.Tensor, target_out: AT.Tensor) -> AT.Tensor:
        nonlocal tick
        state, _ = fs_dec.pump_tick(state, spec, eta=0.1, phi=AT.tanh)

        # output loss
        loss_out = ((state[out_start : out_start + B] - target_out) ** 2).mean()

        # histogram loss over middle layer, with per-node latest windows (no cross-node sync)
        mids = list(range(3 * B, 4 * B))
        window_matrix, kept_ids = _gather_recent_windows(spec, mids, spectral_cfg)  # (M, Nw)
        if window_matrix is not None and len(kept_ids) > 0:
            bp = _batched_bandpower_from_windows(window_matrix, spectral_cfg)       # (M, B)
            targ_mat = AT.stack([hist_targets[nid] for nid in kept_ids])            # (M, B)
            hist_loss = ((bp - targ_mat) ** 2).mean()
        else:
            hist_loss = AT.tensor(0.0)

        loss = loss_out + hist_loss

        #loss.zero_grad()
        autograd.grad(loss, params, retain_graph=False, allow_unused=True)

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
        routed.append(AT.get_tensor(out))

    print("Routed output:", routed)


if __name__ == "__main__":
    main()

