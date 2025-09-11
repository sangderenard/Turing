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
import logging
import numpy as np
from typing import Callable, Optional


# Module logger setup (configured in main if not already configured)
logger = logging.getLogger(__name__)


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
        self._buf: dict[str, list[AT.Tensor]] | None = None
        self._idx = 0
        self._len = 0

    def append(self, entry: dict[str, AT.Tensor]) -> None:
        if self._buf is None:
            self._buf = {k: [None] * self.capacity for k in entry}
            logger.debug(
                "TensorRingBuffer: initialized with first entry keys=%s",
                list(entry.keys()),
            )
        if self._len == self.capacity and self.flush_hook is not None:
            flushed = {k: self._buf[k][self._idx][None, ...] for k in self._buf}
            logger.debug(
                "TensorRingBuffer: capacity=%d flushing index=%d",
                self.capacity,
                self._idx,
            )
            self.flush_hook(flushed)
        for k, v in entry.items():
            assert self._buf is not None  # for type checkers
            self._buf[k][self._idx] = v.clone()
        self._idx = (self._idx + 1) % self.capacity
        if self._len < self.capacity:
            self._len += 1

    def snapshot(self) -> dict[str, AT.Tensor]:
        if self._buf is None or self._len == 0:
            return {}
        snap: dict[str, AT.Tensor] = {}
        for k, lst in self._buf.items():
            if self._len == self.capacity:
                ordered = lst[self._idx :] + lst[: self._idx]
            else:
                ordered = lst[: self._len]
            snap[k] = AT.stack([t.clone() for t in ordered], dim=0)
        logger.debug(
            "TensorRingBuffer: snapshot taken keys=%s length=%d",
            list(snap.keys()),
            self._len,
        )
        return snap

    def flush_all(self) -> None:
        if self.flush_hook is not None and self._buf is not None and self._len > 0:
            logger.debug(
                "TensorRingBuffer: flush_all called with len=%d keys=%s",
                self._len,
                list(self._buf.keys()),
            )
            self.flush_hook(self.snapshot())
        self._buf = None
        self._idx = 0
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

    # Band-power logging branch
    band_start = B * layers
    for i in range(B):
        nodes.append(_node(band_start + i))
        edges.append(_edge(3 * B + i, band_start + i, 1.0))

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
    logger.debug(
        "Spec built: layers=6 bands=%d nodes=%d edges=%d",
        B,
        len(nodes),
        len(edges),
    )
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
    max_lineage_backlog: int = 1024,
) -> tuple[list[AT.Tensor], dict[str, AT.Tensor]]:
    """Run the spectral routing training loop.

    Logs are stored in an :class:`AbstractTensor` ring buffer so unresolved
    backward references stay valid. When the buffer fills, the oldest entries
    are optionally flushed via ``flush_hook``.  ``max_lineage_backlog`` provides
    a safety net: if lineage cleanup fails and the number of outstanding
    lineages grows beyond this threshold, the oldest entries are purged along
    with any cached ring data to avoid unbounded memory use.
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
    logger.debug(
        "train_routing: start B=%d layers=%d out_start=%d ring_capacity=%d",
        B,
        layers,
        out_start,
        ring_capacity,
    )

    hist_targets: dict[int, AT.Tensor] = {}
    band_start = len(spec.nodes) - B
    for j, nid in enumerate(range(band_start, band_start + B)):
        tvec = AT.zeros(B, dtype=float)
        tvec[j] = 1.0
        hist_targets[nid] = tvec
        logger.debug("hist_target for node %d set as one-hot idx=%d", nid, j)

    previous_grads = None
    mix_buf: dict[int, AT.Tensor] = {}
    hist_buf: dict[int, AT.Tensor] = {}

    def purge_lineage_backlog(max_pending: int) -> None:
        """Drop the oldest lineages when backlog exceeds ``max_pending``."""
        active = ledger.lineages()
        if len(active) <= max_pending:
            return
        stale = active[:-max_pending]
        logger.warning(
            "purge_lineage_backlog: dropping %d stale lineages (keeping %d)",
            len(stale),
            max_pending,
        )
        for lid in stale:
            line = (lid,)
            for key_id in (
                OUT_FEAT_ID,
                OUT_TARG_ID,
                OUT_IDS_ID,
                HIST_FEAT_ID,
                HIST_TARG_ID,
                HIST_IDS_ID,
            ):
                harness.node_rings.pop(harness._key(key_id, line), None)
            for n in spec.nodes:
                harness.node_rings.pop(harness._key(n.id, line), None)
            for idx in range(len(spec.edges)):
                harness.edge_rings.pop(harness._key(idx, line), None)
        ledger.purge_through_lid(stale[-1])

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
            logger.debug(
                "try_backward(lin=%d): missing rings — skipping (out_feat=%s out_targ=%s out_ids=%s hist_feat=%s hist_targ=%s hist_ids=%s)",
                lin,
                rb_out_feat is not None,
                rb_out_targ is not None,
                rb_out_ids is not None,
                rb_hist_feat is not None,
                rb_hist_targ is not None,
                rb_hist_ids is not None,
            )
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
            logger.debug(
                "try_backward(lin=%d): histogram shape mismatch M=%d B=%d (feat=%s targ=%s) — skipping",
                lin,
                M,
                B,
                tuple(rb_hist_feat.buf.shape) if rb_hist_feat is not None else None,
                tuple(rb_hist_targ.buf.shape) if rb_hist_targ is not None else None,
            )
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
        logger.debug(
            "try_backward(lin=%d): loss_out=%.6f hist_loss=%.6f",
            lin,
            float(loss_out.item()),
            float(hist_loss.item()),
        )
        mix_buf[lin] = mix_residual
        hist_buf[lin] = hist_residual_summary
        if lin not in mix_buf or lin not in hist_buf:
            return
        mix_seed = mix_buf.pop(lin)
        hist_seed = hist_buf.pop(lin)
        seed_val = float((mix_seed.mean() + hist_seed.mean()).item())
        logger.debug(
            "try_backward(lin=%d): batching VJP with seed_val=%.6f params=%d",
            lin,
            seed_val,
            len(params),
        )
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
        logger.debug(
            "try_backward(lin=%d): computed grads for %d params",
            lin,
            len(grads),
        )
        if previous_grads is not None:
            changed = False
            for idx, (g, pg) in enumerate(zip(grads, previous_grads)):
                if g is None and pg is not None:
                    changed = True
                    logger.debug("[Gradients] param %d lost gradient", idx)
                elif g is not None and pg is None:
                    changed = True
                    logger.debug("[Gradients] param %d gained gradient", idx)
                if g != pg:
                    changed = True
                    logger.debug("[Gradients] param %d gradient changed", idx)
            if changed:
                logger.debug("[Gradients] previous: %s", previous_grads)
                logger.debug("[Gradients] current:  %s", grads)
            else:
                patience -= 1
                if patience <= 0:
                    logger.debug("[Gradients] no changes in gradients, stopping early")
                    exit(0)
        else:
            logger.debug("[Gradients] initial gradients: %s", grads)
        previous_grads = grads
        logger.debug(
            "loss: %.6f, hist_loss: %.6f",
            float(loss_out.item()),
            float(hist_loss.item()),
        )
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
        logger.debug(
            "try_backward(lin=%d): logged tick=%d keys=%s",
            lin,
            int(float(tick_idx)),
            list(log_entry.keys()),
        )
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
        logger.debug("try_backward(lin=%d): cleared transient node rings", lin)
        for n in spec.nodes:
            k = harness._key(n.id, line)
            harness.node_rings.pop(k, None)
        for idx in range(len(spec.edges)):
            k = harness._key(idx, line)
            harness.edge_rings.pop(k, None)
        ledger.purge_through_lid(lin)
        logger.debug("try_backward(lin=%d): purged lineage from ledger", lin)

    def pump_with_loss(state: AT.Tensor, target_out: AT.Tensor) -> AT.Tensor:
        lid = ledger.ingest()
        logger.debug("pump_with_loss: ingest lid=%d", lid)
        state, _ = fs_dec.pump_tick(
            state,
            spec,
            eta=0.1,
            phi=AT.tanh,
            norm="all",
            harness=harness,
            lineage_id=lid,
        )
        logger.debug("pump_with_loss: post pump lid=%d state_len=%d", lid, int(state.shape[0]))
        out_feat = state[out_start : out_start + B].clone()
        harness.push_node(OUT_FEAT_ID, out_feat, lineage=(lid,), size=1)
        harness.push_node(OUT_TARG_ID, target_out.clone(), lineage=(lid,), size=1)
        out_ids = AT.arange(out_start, out_start + B, dtype=float)
        harness.push_node(OUT_IDS_ID, out_ids, lineage=(lid,), size=1)
        logger.debug(
            "pump_with_loss: pushed OUT_* rings lid=%d out_ids=[%d..%d)",
            lid,
            out_start,
            out_start + B,
        )

        mids = list(range(band_start, band_start + B))
        win_map, kept_map = gather_recent_windows(spec, mids, spectral_cfg, harness, ledger)
        logger.debug(
            "pump_with_loss: gather_recent_windows mids=%d returned lineages=%d",
            len(mids),
            len(win_map),
        )
        for lin, W in win_map.items():
            complete = True
            for nid in kept_map[lin]:
                rb = harness.get_node_ring(nid, lineage=(lin,))
                if rb is None or rb.idx < spectral_cfg.win_len:
                    complete = False
                    break
            if not complete:
                logger.debug(
                    "pump_with_loss: lineage %d windows incomplete — skipping hist compute",
                    lin,
                )
                continue
            bp = batched_bandpower_from_windows(W, spectral_cfg)
            bp_map = {nid: bp[row] for row, nid in enumerate(kept_map[lin])}
            targ_map = {nid: hist_targets[nid] for nid in kept_map[lin]}
            feat_mat = AT.stack([bp_map[nid] for nid in kept_map[lin]])
            targ_mat = AT.stack([targ_map[nid] for nid in kept_map[lin]])
            harness.push_node(
                HIST_FEAT_ID,
                feat_mat.flatten(),
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
            logger.debug(
                "pump_with_loss: lineage %d pushed HIST_* (rows=%d B=%d)",
                lin,
                int(bp.shape[0]),
                B,
            )
            try_backward(lin)

        try_backward(lid)
        purge_lineage_backlog(max_lineage_backlog)
        return state

    win = sine_chunks[0].shape[0]
    B = len(sine_chunks)
    for frame_idx, frame_chunks in enumerate(noise_frames):
        logger.debug("train_routing: processing frame %d/%d", frame_idx + 1, len(noise_frames))
        for k in range(win):
            for i in range(B):
                psi[i] = frame_chunks[i][k]
            target_out = AT.stack([sine_chunks[i][k] for i in range(B)]).flatten()
            psi = pump_with_loss(psi, target_out)

        win_map, kept_map = gather_recent_windows(
            spec, list(range(B)), spectral_cfg, harness, ledger
        )
        if win_map:
            logger.debug(
                "train_routing: post-frame gather_recent_windows at inputs returned %d lineages",
                len(win_map),
            )
            for lin, W in win_map.items():
                bp = batched_bandpower_from_windows(W, spectral_cfg)
                for row, nid in enumerate(kept_map[lin]):
                    psi[nid] = bp[row, nid]
        psi = pump_with_loss(psi, AT.zeros(B, dtype=float))
        out = [psi[out_start + i] for i in range(B)]
        routed.append(AT.stack(out))
        logger.debug("train_routing: appended routed output for frame %d", frame_idx + 1)

    for lid in list(ledger.tick_of_lid.keys()):
        try_backward(lid)
    purge_lineage_backlog(max_lineage_backlog)

    # Report gradient status for all learnable parameters once outputs have been
    # produced.  This helps diagnose dead graphs where ``loss.backward`` fails
    # to populate ``grad`` fields.
    for idx, p in enumerate(params):
        grad = getattr(p, "grad", None)
        if grad is None:
            logger.debug("[Gradients] param %d missing gradient", idx)
        else:
            g = AT.get_tensor(grad)
            if np.allclose(g, 0.0):
                logger.debug("[Gradients] param %d gradient is zero: %s", idx, g)
            else:
                logger.debug("[Gradients] param %d grad: %s", idx, g)

    remaining_logs = log_buf.snapshot()
    log_buf.flush_all()

    return routed, remaining_logs


def main() -> None:
    # Configure logging if not configured by the application/test harness
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    logger.debug("demo_spectral_routing: starting main()")
    tick_hz = 400.0
    win = 40
    frames = 50
    max_lineage_backlog = win * 2
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
    routed, logs = train_routing(spec, spectral_cfg, sine_chunks, noise_frames, max_lineage_backlog=max_lineage_backlog)
    logger.debug("Routed output: %s", routed)
    ticks = logs.get("tick")
    logger.debug(
        "Collected ticks: %d",
        int(ticks.shape[0]) if ticks is not None else 0,
    )


if __name__ == "__main__":
    main()

