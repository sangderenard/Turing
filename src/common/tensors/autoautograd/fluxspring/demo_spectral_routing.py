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
from . import fs_dec, register_param_wheels, ParamWheel
from .fs_harness import RingHarness, RingBuffer
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
)
from ..slot_backprop import SlotBackpropQueue
from ..whiteboard_runtime import _WBJob
from types import SimpleNamespace
import logging
import numpy as np
from typing import Callable, Optional, Sequence
from dataclasses import dataclass


# Module logger setup (configured in main if not already configured)
logger = logging.getLogger(__name__)

FLUX_PARAM_SCHEMA = ("p",)


def _vectorize_wheel_params_to_1d(wheels: Sequence[ParamWheel]) -> None:
    """Ensure each parameter slot is at least 1-D for autograd."""
    for w_idx, w in enumerate(wheels):
        for p_idx, p in enumerate(w.params):
            pt = AT.get_tensor(p)
            logger.debug(
                "vectorize_wheel: wheel=%d param=%d before shape=%s",
                w_idx,
                p_idx,
                getattr(pt, "shape", None),
            )
            reshaped = pt.reshape(1) if getattr(pt, "ndim", 0) == 0 else pt.reshape(-1)
            logger.debug(
                "vectorize_wheel: wheel=%d param=%d after shape=%s",
                w_idx,
                p_idx,
                getattr(reshaped, "shape", None),
            )
            w.params[p_idx] = reshaped


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
    band_bounds = AT.tensor(bands, dtype=float)

    t = AT.arange(win, dtype=float)[None, :] / tick_hz
    centers = band_bounds.mean(dim=1, keepdim=True)
    sine_matrix = (2 * AT.pi() * centers * t).sin()
    sine_chunks = [sine_matrix[i] for i in range(len(bands))]

    lo = band_bounds[:, 0:1]
    hi = band_bounds[:, 1:2]
    freq_grid = AT.linspace(lo, hi, steps=3)
    phase = 2 * AT.pi() * freq_grid[..., None] * t
    sinusoid_sum = phase.sin().sum(dim=1)

    noise_frames: list[list[AT.Tensor]] = []
    for _ in range(frames):
        noise = AT.tensor(rng.standard_normal((len(bands), win)))
        mix = sinusoid_sum + 0.1 * noise
    noise_frames.append([mix[i] for i in range(len(bands))])
    return sine_chunks, noise_frames


@dataclass
class RoutingState:
    spec: FluxSpringSpec
    harness: RingHarness
    wheels: list[ParamWheel]
    log_buf: TensorRingBuffer
    out_buf: RingBuffer
    tgt_buf: RingBuffer
    hist_buf: RingBuffer
    bp_queue: SlotBackpropQueue


def initialize_signal_state(
    spec: FluxSpringSpec, spectral_cfg: SpectralCfg
) -> tuple[AT.Tensor, dict[int, AT.Tensor], int, int]:
    """Prepare the initial node state and histogram targets."""
    B = len(spectral_cfg.metrics.bands)
    psi = AT.zeros(len(spec.nodes), dtype=float)
    hist_targets: dict[int, AT.Tensor] = {}
    band_start = len(spec.nodes) - B
    for j, nid in enumerate(range(band_start, band_start + B)):
        tvec = AT.zeros(B, dtype=float)
        tvec[j] = 1.0
        hist_targets[nid] = tvec
        logger.debug("hist_target for node %d set as one-hot idx=%d", nid, j)
    return psi, hist_targets, band_start, B


def log_param_gradients(params: list[AT.Tensor]) -> None:
    """Report gradient status for learnable parameters."""
    for idx, p in enumerate(params):
        value = AT.get_tensor(p)
        requires_grad = getattr(p, "requires_grad", False)
        grad = getattr(p, "grad", None)
        if grad is None:
            logger.debug(
                "[Gradients] param %d value=%s requires_grad=%s missing gradient",
                idx,
                value,
                requires_grad,
            )
        else:
            g = AT.get_tensor(grad)
            if np.allclose(g, 0.0):
                logger.debug(
                    "[Gradients] param %d value=%s requires_grad=%s gradient is zero: %s",
                    idx,
                    value,
                    requires_grad,
                    g,
                )
            else:
                logger.debug(
                    "[Gradients] param %d value=%s requires_grad=%s grad: %s",
                    idx,
                    value,
                    requires_grad,
                    g,
                )
def pump_with_loss(
    ctx: RoutingState,
    state: AT.Tensor,
    target_out: AT.Tensor,
    spectral_cfg: SpectralCfg,
    hist_targets: dict[int, AT.Tensor],
    band_start: int,
    out_start: int,
    B: int,
    tick: int,
) -> AT.Tensor:
    """Advance the system by one tick and queue residuals and jobs."""

    state, _ = fs_dec.pump_tick(
        state,
        ctx.spec,
        eta=0.1,
        phi=AT.tanh,
        norm="all",
        harness=ctx.harness,
        wheels=ctx.wheels,
        tick=tick,
        update_fn=lambda p, g: p,
    )

    out_feat = state[out_start : out_start + B].clone()
    ctx.out_buf.push(out_feat)
    ctx.tgt_buf.push(target_out.clone())
    hist_residual_summary = None

    mids = list(range(band_start, band_start + B))
    W, kept = gather_recent_windows(mids, spectral_cfg, ctx.harness)
    if len(kept) == len(mids):
        bp = batched_bandpower_from_windows(W, spectral_cfg)
        targ_mat = AT.stack([hist_targets[nid] for nid in kept])
        hist_residual = bp - targ_mat
        hist_residual_summary = hist_residual.mean(0)
        ctx.hist_buf.push(hist_residual_summary)
    fft_tick = tick - (spectral_cfg.win_len - 1)
    if fft_tick >= 0:
        idx = (ctx.out_buf.idx - spectral_cfg.win_len) % spectral_cfg.win_len
        delayed_out = ctx.out_buf.buf[idx]
        delayed_tgt = ctx.tgt_buf.buf[idx]
        main_residual = delayed_out - delayed_tgt
        ctx.bp_queue.add_residual(
            tick=fft_tick,
            main=main_residual.mean(),
            spectral=hist_residual_summary.mean() if hist_residual_summary is not None else None,
        )
        src_ids = tuple(range(len(ctx.wheels)))

        def _route_fn(_p: AT.Tensor) -> AT.Tensor:
            psi_tmp, _ = fs_dec.pump_tick(
                state.clone(),
                ctx.spec,
                eta=0.1,
                phi=AT.tanh,
                norm="all",
                harness=ctx.harness,
                wheels=ctx.wheels,
                tick=fft_tick,
                update_fn=lambda p, g: p,
            )
            out_tmp = psi_tmp[out_start : out_start + B]
            return out_tmp.mean()

        job_route = _WBJob(
            job_id=f"route:{fft_tick}",
            op=None,
            src_ids=src_ids,
            residual=None,
            fn=_route_fn,
        )
        ctx.bp_queue.queue_job(None, job_route, tick=fft_tick, kind="main")

        def _fft_fn(_p: AT.Tensor) -> AT.Tensor:
            mids_local = list(range(band_start, band_start + B))
            W_loc, kept_loc = gather_recent_windows(mids_local, spectral_cfg, ctx.harness)
            if len(kept_loc) == len(mids_local):
                bp_loc = batched_bandpower_from_windows(W_loc, spectral_cfg)
                targ_mat = AT.stack([hist_targets[nid] for nid in kept_loc])
                return (bp_loc - targ_mat).mean()
            return AT.tensor(0.0)

        job_fft = _WBJob(
            job_id=f"fft:{fft_tick}",
            op=None,
            src_ids=src_ids,
            residual=None,
            fn=_fft_fn,
        )
        ctx.bp_queue.queue_job(None, job_fft, tick=fft_tick, kind="spectral")

    return state


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
    wheels = register_param_wheels(spec)
    _vectorize_wheel_params_to_1d(wheels)
    for w in wheels:
        w.rotate(); w.bind_slot()
    set_strict_mode(True)
    annotate_params([v for w in wheels for v in w.versions()])
    logger.debug(
        "train_routing: parameter wheels sized to %d slots", len(wheels[0].versions()) if wheels else 0
    )

    psi, hist_targets, band_start, B = initialize_signal_state(spec, spectral_cfg)
    routed: list[AT.Tensor] = []
    out_start = 5 * B
    layers = max(1, len(spec.nodes) // B)
    ring_capacity = (
        spectral_cfg.win_len
        if spectral_cfg.enabled
        else (log_capacity or max(1, layers - 1))
    )
    log_buf = TensorRingBuffer(ring_capacity, flush_hook)
    harness = RingHarness(
        default_size=spectral_cfg.win_len if spectral_cfg.enabled else None
    )
    logger.debug(
        "train_routing: start B=%d layers=%d out_start=%d ring_capacity=%d win_len=%d",
        B,
        layers,
        out_start,
        ring_capacity,
        spectral_cfg.win_len,
    )

    out_buf = RingBuffer(AT.zeros((spec.spectral.win_len, B), dtype=float))
    tgt_buf = RingBuffer(AT.zeros((spec.spectral.win_len, B), dtype=float))
    hist_buf = RingBuffer(AT.zeros((spec.spectral.win_len, B), dtype=float))
    bp_queue = SlotBackpropQueue(wheels)
    if bp_queue.slots != spectral_cfg.win_len:
        raise ValueError(
            "train_routing: bp_queue.slots %d does not match spectral_cfg.win_len %d"
            % (bp_queue.slots, spectral_cfg.win_len)
        )
    ctx = RoutingState(
        spec=spec,
        harness=harness,
        wheels=wheels,
        log_buf=log_buf,
        out_buf=out_buf,
        tgt_buf=tgt_buf,
        hist_buf=hist_buf,
        bp_queue=bp_queue,
    )

    win = sine_chunks[0].shape[0]
    tick = 0
    for frame_idx, frame_chunks in enumerate(noise_frames):
        logger.debug(
            "train_routing: processing frame %d/%d",
            frame_idx + 1,
            len(noise_frames),
        )
        for k in range(win):
            for i in range(B):
                psi[i] = frame_chunks[i][k]
            target_out = AT.stack([sine_chunks[i][k] for i in range(B)]).flatten()
            psi = pump_with_loss(
                ctx,
                psi,
                target_out,
                spectral_cfg,
                hist_targets,
                band_start,
                out_start,
                B,
                tick,
            )
            mature_tick = tick - (spectral_cfg.win_len - 1)
            if mature_tick >= 0:
                mature_slot = mature_tick % ctx.bp_queue.slots
                sys = SimpleNamespace(
                    nodes={i: SimpleNamespace(p=w.params[mature_slot]) for i, w in enumerate(ctx.wheels)}
                )
                res = ctx.bp_queue.process_slot(mature_slot, sys=sys, node_attrs=FLUX_PARAM_SCHEMA)
                if res is not None:
                    ctx.log_buf.append({"tick": AT.tensor([float(mature_tick)])})
            tick += 1

        W, kept = gather_recent_windows(list(range(B)), spectral_cfg, harness)
        if len(kept) == B:
            logger.debug(
                "train_routing: post-frame gather_recent_windows at inputs returned %d windows",
                len(kept),
            )
            bp = batched_bandpower_from_windows(W, spectral_cfg)
            rows = AT.arange(len(kept), dtype=int)
            cols = AT.tensor(kept, dtype=int)
            psi[cols] = bp[rows, cols]
        psi = pump_with_loss(
            ctx,
            psi,
            AT.zeros(B, dtype=float),
            spectral_cfg,
            hist_targets,
            band_start,
            out_start,
            B,
            tick,
        )
        mature_tick = tick - (spectral_cfg.win_len - 1)
        if mature_tick >= 0:
            mature_slot = mature_tick % ctx.bp_queue.slots
            sys = SimpleNamespace(
                nodes={i: SimpleNamespace(p=w.params[mature_slot]) for i, w in enumerate(ctx.wheels)}
            )
            res = ctx.bp_queue.process_slot(mature_slot, sys=sys, node_attrs=FLUX_PARAM_SCHEMA)
            if res is not None:
                ctx.log_buf.append({"tick": AT.tensor([float(mature_tick)])})
        tick += 1
        out = [psi[out_start + i] for i in range(B)]
        routed.append(AT.stack(out))
        logger.debug("train_routing: appended routed output for frame %d", frame_idx + 1)

    for _ in range(spectral_cfg.win_len - 1):
        mature_tick = tick - (spectral_cfg.win_len - 1)
        if mature_tick >= 0:
            mature_slot = mature_tick % ctx.bp_queue.slots
            sys = SimpleNamespace(
                nodes={i: SimpleNamespace(p=w.params[mature_slot]) for i, w in enumerate(ctx.wheels)}
            )
            res = ctx.bp_queue.process_slot(mature_slot, sys=sys, node_attrs=FLUX_PARAM_SCHEMA)
            if res is not None:
                ctx.log_buf.append({"tick": AT.tensor([float(mature_tick)])})
        tick += 1

    log_param_gradients([p for w in ctx.wheels for p in w.versions()])
    remaining_logs = log_buf.snapshot()
    log_buf.flush_all()

    return routed, remaining_logs


def main(log_file: str = "fluxspring_debug.log") -> None:
    # Configure logging to a file (force replaces any existing console handlers)
    logging.basicConfig(
        level=logging.DEBUG,
        filename=log_file,
        filemode="w",
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    logger.debug("demo_spectral_routing: starting main(), logging to %s", log_file)
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
    logger.debug("Routed output: %s", routed)
    ticks = logs.get("tick")
    logger.debug(
        "Collected ticks: %d",
        int(ticks.shape[0]) if ticks is not None else 0,
    )


if __name__ == "__main__":
    main()

