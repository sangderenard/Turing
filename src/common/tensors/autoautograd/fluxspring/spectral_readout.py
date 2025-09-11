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

from ...abstraction import AbstractTensor as AT
from .fs_types import FluxSpringSpec, SpectralCfg
from .fs_harness import RingHarness, LineageLedger, RingBuffer


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


_fft_bands_rb: RingBuffer | None = None
_fft_lids_rb: RingBuffer | None = None
_fft_count: int = 0


def update_fft_window(lineage: int, band: AT, cfg: SpectralCfg) -> None:
    """Record ``band`` and its ``lineage`` in a global tensor ring.

    The ring buffer stores the most recent ``cfg.win_len`` entries using
    :class:`RingBuffer` so gradients remain connected without manual tensor
    slicing.  ``band`` is promoted to a 1‑D row to fit the ring layout.
    """

    global _fft_bands_rb, _fft_lids_rb, _fft_count

    row = band if getattr(AT.get_tensor(band), "ndim", 0) > 0 else band[None]

    if _fft_bands_rb is None:
        D = int(row.shape[0])
        zeros = AT.zeros((int(cfg.win_len), D), dtype=AT.get_tensor(row).dtype)
        _fft_bands_rb = RingBuffer(zeros)
        _fft_lids_rb = RingBuffer(AT.zeros(int(cfg.win_len), dtype=float))

    _fft_bands_rb.push(row)
    _fft_lids_rb.push(AT.tensor(lineage, dtype=float))
    _fft_count += 1


def _ordered_ring(rb: RingBuffer, count: int) -> AT:
    buf = rb.buf
    R = int(buf.shape[0])
    c = count if count < R else R
    if count < R:
        return buf[:c]
    start = count % R
    if start == 0:
        return buf
    return AT.cat([buf[start:], buf[:start]], dim=0)


def current_fft_window() -> Tuple[AT, AT]:
    """Return the stacked bands and their lineage identifiers."""

    if _fft_bands_rb is None or _fft_lids_rb is None or _fft_count == 0:
        return AT.zeros(0, dtype=float), AT.zeros(0, dtype=float)
    bands = _ordered_ring(_fft_bands_rb, _fft_count)
    lids = _ordered_ring(_fft_lids_rb, _fft_count)
    return bands, lids


def reset_fft_windows() -> None:
    """Clear stored window history for all lineages."""

    global _fft_bands_rb, _fft_lids_rb, _fft_count
    _fft_bands_rb = None
    _fft_lids_rb = None
    _fft_count = 0


def gather_recent_windows(
    spec: FluxSpringSpec,
    node_ids: List[int],
    cfg: SpectralCfg,
    harness: RingHarness,
    ledger: LineageLedger,
) -> Tuple[Dict[int, AT], Dict[int, List[int]]]:
    """Compatibility shim returning the global FFT window keyed by lineage.

    Each lineage present in the window is mapped to its band tensor.  ``node_ids``
    are echoed back in ``kept_map`` so legacy callers can associate results with
    originating nodes even though per-lineage history is no longer stored.
    """

    win_map: Dict[int, AT] = {}
    kept_map: Dict[int, List[int]] = {}

    bands, lids = current_fft_window()
    if int(bands.shape[0]) == 0:
        return win_map, kept_map

    lid_list = [int(x) for x in AT.get_tensor(lids).flatten().tolist()]
    for i, lin in enumerate(lid_list):
        win_map[lin] = bands[i : i + 1]
        kept_map[lin] = list(node_ids)
    return win_map, kept_map


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
