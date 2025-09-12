"""Spectral metrics utilities for FluxSpring.

This module exposes a ``compute_metrics`` function that operates on
:class:`~src.common.tensors.abstraction.AbstractTensor` buffers.  It
computes power spectra using the backend's FFT implementation when
available and falls back to explicit sine/cosine bases otherwise.  The
function supports several common spectral metrics that are useful for
training FluxSpring graphs with frequency‑selective objectives.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import logging

import numpy as np

from ...abstraction import AbstractTensor as AT
from .fs_types import FluxSpringSpec, SpectralCfg
from .fs_harness import RingHarness, LineageLedger, RingBuffer

logger = logging.getLogger(__name__)


def _rfft_real_imag(x: AT, tick_hz: float) -> Tuple[AT, AT, AT]:
    """Return real/imag parts of the FFT and the frequency grid.

    Uses the backend FFT when available.  If missing, a direct DFT is
    computed using sine/cosine bases.  This avoids constructing complex
    tensors manually and stays within the AbstractTensor API.
    """

    N = int(x.shape[0])
    try:
        C = AT.rfft(x, axis=0)
        freqs = AT.rfftfreq(N, d=1.0 / tick_hz, like=x)
        return AT.real(C), AT.imag(C), freqs
    except Exception:
        t = AT.arange(N, dtype=float)
        k = AT.arange(N // 2 + 1, dtype=float)
        ang = (2.0 * AT.pi() * t[:, None] * k[None, :]) / float(N)
        cos_b = ang.cos()
        sin_b = ang.sin()
        c_real = cos_b.T() @ x
        c_imag = -sin_b.T() @ x
        freqs = k * (tick_hz / float(N))
        return c_real, c_imag, freqs


def _window(name: str, N: int) -> AT:
    n = name.lower()
    if n in ("hann", "hanning"):
        return AT.hanning(N)
    if n == "hamming":
        return AT.hamming(N)
    if n == "zeros":
        return AT.zeros(N, dtype=float)
    return AT.ones(N, dtype=float)


def compute_metrics(buffer: AT, cfg: SpectralCfg, *, return_tensor: bool = True) -> Dict[str, Any]:
    """Compute spectral metrics for ``buffer`` according to ``cfg``.

    Parameters
    ----------
    buffer:
        Input signal to analyse.
    cfg:
        Spectral configuration describing the analysis window and metrics.
    return_tensor:
        When ``True`` (default), values in the returned dictionary remain as
        backend tensors.  If ``False``, results are converted to Python
        ``float`` objects for compatibility with callers expecting host types.
    """

    if buffer.ndim == 1:
        x = buffer[:, None]
    else:
        x = buffer

    N = int(x.shape[0])
    w = _window(cfg.window, N)
    xw = w[:, None] * x

    real, imag, freqs = _rfft_real_imag(xw, cfg.tick_hz)
    power = real**2 + imag**2
    logger.debug(
        "compute_metrics: N=%d window=%s power_shape=%s",
        int(xw.shape[0]),
        cfg.window,
        tuple(getattr(power, "shape", ())),
    )

    metrics: Dict[str, Any] = {}
    m = cfg.metrics

    if m.bands:
        band_vals: List[Any] = []
        for lo, hi in m.bands:
            mask = (freqs >= lo) & (freqs <= hi)
            bw = AT.sum(power * mask[:, None])
            if return_tensor:
                band_vals.append(bw)
            else:
                band_vals.append(float(AT.get_tensor(bw).data.item()))
        metrics["bandpower"] = AT.stack(band_vals) if return_tensor else band_vals

    if m.centroid:
        total = AT.sum(power) + 1e-12
        cent = AT.sum(freqs * AT.sum(power, dim=1)) / total
        metrics["centroid"] = cent if return_tensor else float(AT.get_tensor(cent).data.item())

    if m.flatness:
        logp = AT.log(power + 1e-12)
        gm = AT.exp(AT.mean(logp))
        am = AT.mean(power)
        flat = gm / (am + 1e-12)
        metrics["flatness"] = flat if return_tensor else float(AT.get_tensor(flat).data.item())

    if m.coherence and xw.shape[1] >= 2:
        r0, i0 = real[:, 0], imag[:, 0]
        r1, i1 = real[:, 1], imag[:, 1]
        pxx = r0**2 + i0**2
        pyy = r1**2 + i1**2
        pxy_r = r0 * r1 + i0 * i1
        pxy_i = i0 * r1 - r0 * i1
        den = pxx * pyy
        coh = (pxy_r**2 + pxy_i**2) / (den + 1e-12)
        mask = den > 1e-12
        coh_masked = AT.where(mask, coh, AT.zeros_like(coh))
        mean_coh = AT.sum(coh_masked) / (AT.sum(mask) + 1e-12)
        metrics["coherence"] = mean_coh if return_tensor else float(AT.get_tensor(mean_coh).data.item())

    return metrics


def gather_ring_metrics(
    spec: FluxSpringSpec,
    harness: RingHarness,
    *,
    return_tensor: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """Compile spectral metrics for node ring buffers managed by ``harness``."""

    cfg = spec.spectral
    if not cfg.enabled:
        return {}
    stats: Dict[int, Dict[str, Any]] = {}
    for n in spec.nodes:
        rb = harness.get_node_ring(n.id)
        if rb is None:
            continue
        buf = rb.buf[:, 0] if AT.get_tensor(rb.buf).ndim == 2 else rb.buf
        stats[n.id] = compute_metrics(buf, cfg, return_tensor=return_tensor)
    return stats


def gather_recent_windows(
    node_ids: List[int],
    cfg: SpectralCfg,
    harness: RingHarness,
    ledger: LineageLedger,
) -> Tuple[Dict[int, AT], Dict[int, List[int]]]:
    """Return recent raw windows for specified node IDs grouped by lineage.

    For each lineage tracked in ``ledger`` this function attempts to retrieve
    the pre‑mix ring buffer for every node in ``node_ids``.  When a ring
    contains at least ``cfg.win_len`` samples the most recent ``win_len`` values
    are stacked into a window.  The returned ``win_map`` associates each
    lineage identifier with a tensor of shape ``(M, win_len)`` where ``M`` is the
    number of nodes that produced a valid window.  ``kept_map`` stores the node
    identifiers corresponding to each row in the window tensor.
    """

    win_map: Dict[int, AT] = {}
    kept_map: Dict[int, List[int]] = {}

    def _ordered(rb: RingBuffer) -> AT:
        buf = rb.buf
        R = int(buf.shape[0])
        idx = rb.idx
        if idx < R:
            return buf[:idx]
        start = idx % R
        if start == 0:
            return buf
        return AT.cat([buf[start:], buf[:start]], dim=0)

    for lin in ledger.lineages():
        windows: List[AT] = []
        kept: List[int] = []
        for nid in node_ids:
            rb = harness.get_premix_ring(nid, lineage=(lin,))
            if rb is None:
                logger.debug(
                    "gather_recent_windows: lin=%d nid=%d premix ring missing",
                    lin,
                    nid,
                )
                continue
            ordered = _ordered(rb)
            if int(ordered.shape[0]) < cfg.win_len:
                logger.debug(
                    "gather_recent_windows: lin=%d nid=%d insufficient samples %d < win_len %d",
                    lin,
                    nid,
                    int(ordered.shape[0]),
                    cfg.win_len,
                )
                continue
            windows.append(ordered[-cfg.win_len :].reshape(-1))
            kept.append(nid)
        if windows:
            win_map[lin] = AT.stack(windows)
            kept_map[lin] = kept
            logger.debug(
                "gather_recent_windows: lin=%d windows=%d kept=%s",
                lin,
                len(windows),
                kept,
            )

    return win_map, kept_map


def quantile_band_targets(
    node_ids: List[int], cfg: SpectralCfg, harness: RingHarness
) -> Dict[int, Tuple[float, float]]:
    """Return quantile frequency bands from node pre-mix histories.

    For the provided ``node_ids``, the most recent ``cfg.win_len`` samples are
    gathered from each node's pre-mix ring buffer.  Their FFT power spectra are
    summed to produce an aggregate frequency distribution.  The cumulative power
    distribution is then divided into ``len(node_ids)`` quantile bins, assigning
    each middle node a target frequency range with equal mass.
    """

    if not node_ids:
        return {}

    freqs_ref: AT | None = None
    agg_power: AT | None = None

    def _ordered(rb: RingBuffer) -> AT:
        buf = rb.buf
        R = int(buf.shape[0])
        idx = rb.idx
        if idx < R:
            return buf[:idx]
        start = idx % R
        if start == 0:
            return buf
        return AT.cat([buf[start:], buf[:start]], dim=0)

    for nid in node_ids:
        rb = harness.get_premix_ring(nid)
        if rb is None:
            continue
        ordered = _ordered(rb)
        if int(ordered.shape[0]) < cfg.win_len:
            continue
        win = ordered[-cfg.win_len :]
        real, imag, freqs = _rfft_real_imag(win, cfg.tick_hz)
        power = real**2 + imag**2
        if freqs_ref is None:
            freqs_ref = freqs
            agg_power = power
        else:
            agg_power = agg_power + power  # type: ignore[operator]

    if freqs_ref is None or agg_power is None:
        return {}

    f_np = np.asarray(AT.get_tensor(freqs_ref), dtype=float).reshape(-1)
    p_np = np.asarray(AT.get_tensor(agg_power), dtype=float).reshape(-1)
    total = p_np.sum()
    if total <= 0:
        return {}
    cdf = np.cumsum(p_np) / total
    quant = np.linspace(0.0, 1.0, len(node_ids) + 1)
    edges = np.interp(quant, np.concatenate(([0.0], cdf)), np.concatenate(([f_np[0]], f_np)))

    targets: Dict[int, Tuple[float, float]] = {}
    for i, nid in enumerate(sorted(node_ids)):
        targets[nid] = (float(edges[i]), float(edges[i + 1]))
    return targets


def batched_bandpower_from_windows(window_matrix: AT, cfg: SpectralCfg) -> AT:
    """Compute normalized band powers for a batch of windows.

    Parameters
    ----------
    window_matrix:
        Tensor of shape ``(M, Nw)`` containing aligned windows.
    cfg:
        Spectral configuration specifying analysis bands.

    Returns
    -------
    AT
        Tensor of shape ``(M, B)`` with each row the normalized band powers
        for the corresponding window.
    """

    Nw = int(window_matrix.shape[-1])
    w = _window(cfg.window, Nw)
    xw = window_matrix * w

    C = AT.rfft(xw, axis=-1)
    real, imag = AT.real(C), AT.imag(C)
    power = real**2 + imag**2

    freqs = AT.rfftfreq(Nw, d=1.0 / cfg.tick_hz, like=xw)
    mask_FB = AT.stack(
        [
            AT.get_tensor((freqs >= lo) & (freqs <= hi), dtype=float)
            for lo, hi in cfg.metrics.bands
        ]
    ).transpose(0, 1)

    band_powers = AT.matmul(power, mask_FB)
    band_powers = band_powers / (AT.sum(band_powers, dim=1, keepdim=True) + 1e-12)
    return band_powers


def premix_histogram_loss(
    rb: RingBuffer, *, band_idx: int, total_bands: int, tick_hz: float
) -> AT.Tensor:
    """Return histogram loss for a raw pre-mix node history.

    The buffer stored in ``rb`` is windowed with a Hann window, transformed via
    an ``rfft`` and its power integrated into ``total_bands`` equal-frequency
    bands.  The resulting normalized histogram is compared against a one-hot
    target with index ``band_idx`` using squared error.
    """

    buf = rb.buf[:, 0] if AT.get_tensor(rb.buf).ndim == 2 else rb.buf
    N = int(buf.shape[0])
    w = _window("hann", N)
    xw = w * buf

    real, imag, freqs = _rfft_real_imag(xw, tick_hz)
    power = real**2 + imag**2

    nyq = tick_hz / 2.0
    edges = AT.linspace(0.0, nyq, steps=total_bands + 1)
    bands: List[AT] = []
    for b in range(total_bands):
        lo, hi = edges[b], edges[b + 1]
        mask = (freqs >= lo) & (freqs <= hi if b == total_bands - 1 else freqs < hi)
        bands.append(AT.sum(power * mask))
    hist = AT.stack(bands)
    hist = hist / (AT.sum(hist) + 1e-12)

    target = AT.zeros(total_bands, dtype=float)
    target_idx = int(band_idx)
    if 0 <= target_idx < total_bands:
        target[target_idx] = 1.0
    return AT.sum((hist - target) ** 2)
