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
from .fs_harness import RingHarness, LineageLedger
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
from src.common.tensors.autograd_probes import (
    set_strict_mode,
    annotate_params,
    probe_losses,
)
from ..whiteboard_runtime import run_batched_vjp
from ...abstract_nn.train import rebind
from types import SimpleNamespace
import numpy as np
from typing import Callable, Optional


# Special node IDs for harness-managed gradient buffers.
OUT_FEAT_ID = -1
OUT_TARG_ID = -2
OUT_IDS_ID = -3
HIST_FEAT_ID = -4
HIST_TARG_ID = -5
HIST_IDS_ID = -6


class TensorRingBuffer:
    """AbstractTensor-based ring buffer for per-tick logs."""

    def __init__(
        self,
        capacity: int,
        flush_hook: Callable[[dict[str, AT.Tensor]], None] | None = None,
    ) -> None:
        self.capacity = capacity
        self.flush_hook = flush_hook
        self._buf: dict[str, AT.Tensor] | None = None
        self._len = 0

    def append(self, entry: dict[str, AT.Tensor]) -> None:
        if self._buf is None:
            self._buf = {k: v[None, ...] for k, v in entry.items()}
            self._len = 1
            return
        for k, v in entry.items():
            self._buf[k] = AT.cat([self._buf[k], v[None, ...]], dim=0)
        self._len += 1
        if self._len > self.capacity:
            overflow = self._len - self.capacity
            if self.flush_hook is not None:
                flushed = {k: v[:overflow] for k, v in self._buf.items()}
                self.flush_hook(flushed)
            for k in self._buf:
                self._buf[k] = self._buf[k][overflow:]
            self._len = self.capacity

    def snapshot(self) -> dict[str, AT.Tensor]:
        if self._buf is None:
            return {}
        return {k: v.clone() for k, v in self._buf.items()}

    def flush_all(self) -> None:
        if self.flush_hook is not None and self._buf is not None and self._len > 0:
            self.flush_hook({k: v.clone() for k, v in self._buf.items()})
        self._buf = None
        self._len = 0


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
    *,
    flush_hook: Optional[Callable[[dict[str, AT.Tensor]], None]] = None,
    log_capacity: Optional[int] = None,
) -> tuple[list[AT.Tensor], dict[str, AT.Tensor]]:
    """Run the spectral routing training loop.

    Logs are stored in an :class:`AbstractTensor` ring buffer so unresolved
    backward references stay valid. When the buffer fills, the oldest entries
    are optionally flushed via ``flush_hook``.
    """
    patience = 10
    params = register_learnable_params(spec)
    set_strict_mode(True)
    annotate_params(params)
    B = len(spectral_cfg.metrics.bands)
    psi = AT.zeros(len(spec.nodes), dtype=float)
    routed: list[AT.Tensor] = []
    out_start = 5 * B
    layers = max(1, len(spec.nodes) // B)
    ring_capacity = log_capacity or max(1, layers - 1)
    log_buf = TensorRingBuffer(ring_capacity, flush_hook)
    harness = RingHarness(default_size=spectral_cfg.win_len if spectral_cfg.enabled else None)
    ledger = LineageLedger()

    hist_targets: dict[int, AT.Tensor] = {}
    for j, nid in enumerate(range(3 * B, 4 * B)):
        tvec = AT.zeros(B, dtype=float)
        tvec[j] = 1.0
        hist_targets[nid] = tvec

    previous_grads = None
    mix_buf: dict[int, AT.Tensor] = {}
    hist_buf: dict[int, AT.Tensor] = {}

    def try_backward(lin: int) -> None:
        nonlocal previous_grads, patience, log_buf
        line = (lin,)
        rb_out_feat = harness.get_node_ring(OUT_FEAT_ID, lineage=line)
        rb_out_targ = harness.get_node_ring(OUT_TARG_ID, lineage=line)
        rb_out_ids = harness.get_node_ring(OUT_IDS_ID, lineage=line)
        rb_hist_feat = harness.get_node_ring(HIST_FEAT_ID, lineage=line)
        rb_hist_targ = harness.get_node_ring(HIST_TARG_ID, lineage=line)
        rb_hist_ids = harness.get_node_ring(HIST_IDS_ID, lineage=line)
        if None in (
            rb_out_feat,
            rb_out_targ,
            rb_out_ids,
            rb_hist_feat,
            rb_hist_targ,
            rb_hist_ids,
        ):
            return
        out_feat = rb_out_feat.buf[0]
        out_targ = rb_out_targ.buf[0]
        hist_ids = rb_hist_ids.buf[0]
        B = int(out_feat.shape[0])
        M = int(hist_ids.shape[0])
        if (
            int(rb_hist_feat.buf.shape[1]) != M * B
            or int(rb_hist_targ.buf.shape[1]) != M * B
        ):
            return
        hist_feat = rb_hist_feat.buf[0].reshape(M, B)
        hist_targ = rb_hist_targ.buf[0].reshape(M, B)
        mix_residual = out_feat - out_targ
        hist_residual = hist_feat - hist_targ
        hist_residual_summary = hist_residual.mean(0)
        hist_loss = (hist_residual ** 2).mean()
        loss_out = (mix_residual ** 2).mean()
        losses = {"loss_out": loss_out, "hist_loss": hist_loss}
        probe_losses(losses, params)
        mix_buf[lin] = mix_residual
        hist_buf[lin] = hist_residual_summary
        if lin not in mix_buf or lin not in hist_buf:
            return
        mix_seed = mix_buf.pop(lin)
        hist_seed = hist_buf.pop(lin)
        seed_val = float((mix_seed.mean() + hist_seed.mean()).item())
        sys = SimpleNamespace(
            nodes={i: SimpleNamespace(sphere=p) for i, p in enumerate(params)}
        )
        jobs = [
            SimpleNamespace(job_id=f"p{i}", op="__neg__", src_ids=(i,), residual=seed_val)
            for i in range(len(params))
        ]
        batch = run_batched_vjp(sys=sys, jobs=jobs)
        grads = []
        if batch.grads_per_source_tensor is not None:
            g_tensor = AT.get_tensor(batch.grads_per_source_tensor)
            for idx, p in enumerate(params):
                grad = -g_tensor[idx]
                grads.append(grad)
                new_p = p - 0.01 * grad
                params[idx] = rebind(f"param[{idx}]", new_p)
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
                patience -= 1
                if patience <= 0:
                    print("[Gradients] no changes in gradients, stopping early")
                    exit(0)
        else:
            print(f"[Gradients] initial gradients: {grads}")
        previous_grads = grads
        print(f"loss: {loss_out.item():.6f}, hist_loss: {hist_loss.item():.6f}")
        tick_idx = ledger.tick_of_lid[lin]
        log_entry = {
            "tick": AT.tensor([float(tick_idx)]),
            "out_feat": out_feat.clone(),
            "out_targ": out_targ.clone(),
            "mix_residual": mix_seed.clone(),
            "hist_residual": hist_seed.clone(),
            "param_grad": AT.stack(grads) if grads else AT.zeros(len(params)),
        }
        log_buf.append(log_entry)
        for key_id in (
            OUT_FEAT_ID,
            OUT_TARG_ID,
            OUT_IDS_ID,
            HIST_FEAT_ID,
            HIST_TARG_ID,
            HIST_IDS_ID,
        ):
            k = harness._key(key_id, line)
            harness.node_rings.pop(k, None)
        for n in spec.nodes:
            k = harness._key(n.id, line)
            harness.node_rings.pop(k, None)
        for idx in range(len(spec.edges)):
            k = harness._key(idx, line)
            harness.edge_rings.pop(k, None)
        ledger.purge_through_lid(lin)

    def pump_with_loss(state: AT.Tensor, target_out: AT.Tensor) -> AT.Tensor:
        lid = ledger.ingest()
        state, _ = fs_dec.pump_tick(
            state,
            spec,
            eta=0.1,
            phi=AT.tanh,
            norm="all",
            harness=harness,
            lineage_id=lid,
        )
        out_feat = state[out_start : out_start + B].clone()
        harness.push_node(OUT_FEAT_ID, out_feat, lineage=(lid,), size=1)
        harness.push_node(OUT_TARG_ID, target_out.clone(), lineage=(lid,), size=1)
        out_ids = AT.arange(out_start, out_start + B, dtype=float)
        harness.push_node(OUT_IDS_ID, out_ids, lineage=(lid,), size=1)

        mids = list(range(3 * B, 4 * B))
        win_map, kept_map = gather_recent_windows(spec, mids, spectral_cfg, harness, ledger)
        for lin, W in win_map.items():
            complete = True
            for nid in kept_map[lin]:
                rb = harness.get_node_ring(nid, lineage=(lin,))
                if rb is None or rb.idx < spectral_cfg.win_len:
                    complete = False
                    break
            if not complete:
                continue
            bp = batched_bandpower_from_windows(W, spectral_cfg)
            targ_mat = AT.stack([hist_targets[nid] for nid in kept_map[lin]])
            harness.push_node(
                HIST_FEAT_ID,
                bp.flatten(),
                lineage=(lin,),
                size=1,
            )
            harness.push_node(
                HIST_TARG_ID,
                targ_mat.flatten(),
                lineage=(lin,),
                size=1,
            )
            harness.push_node(
                HIST_IDS_ID,
                AT.tensor(kept_map[lin], dtype=float),
                lineage=(lin,),
                size=1,
            )
            try_backward(lin)

        try_backward(lid)
        return state

    win = sine_chunks[0].shape[0]
    B = len(sine_chunks)
    for frame_chunks in noise_frames:
        for k in range(win):
            for i in range(B):
                psi[i] = frame_chunks[i][k]
            target_out = AT.stack([sine_chunks[i][k] for i in range(B)]).flatten()
            psi = pump_with_loss(psi, target_out)

        win_map, kept_map = gather_recent_windows(
            spec, list(range(B)), spectral_cfg, harness, ledger
        )
        if win_map:
            for lin, W in win_map.items():
                bp = batched_bandpower_from_windows(W, spectral_cfg)
                for row, nid in enumerate(kept_map[lin]):
                    psi[nid] = bp[row, nid]
        psi = pump_with_loss(psi, AT.zeros(B, dtype=float))
        out = [psi[out_start + i] for i in range(B)]
        routed.append(AT.stack(out))

    for lid in list(ledger.tick_of_lid.keys()):
        try_backward(lid)

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

    remaining_logs = log_buf.snapshot()
    log_buf.flush_all()

    return routed, remaining_logs


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
    routed, logs = train_routing(spec, spectral_cfg, sine_chunks, noise_frames)
    print("Routed output:", routed)
    ticks = logs.get("tick")
    print("Collected ticks:", int(ticks.shape[0]) if ticks is not None else 0)


if __name__ == "__main__":
    main()

