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


def gather_ring_metrics(spec: FluxSpringSpec, *, return_tensor: bool = True) -> Dict[int, Dict[str, Any]]:
    """Compile spectral metrics for node ring buffers in ``spec``.

    Returns a mapping from node id to the computed metrics.  Only nodes with
    allocated ring buffers are analysed.
    """

    cfg = spec.spectral
    if not cfg.enabled:
        return {}
    stats: Dict[int, Dict[str, Any]] = {}
    for n in spec.nodes:
        if n.ring is None:
            continue
        buf = n.ring[:, 0] if AT.get_tensor(n.ring).ndim == 2 else n.ring
        stats[n.id] = compute_metrics(buf, cfg, return_tensor=return_tensor)
    return stats


def gather_recent_windows(
    spec: FluxSpringSpec, node_ids: List[int], cfg: SpectralCfg
) -> Tuple[AT | None, List[int]]:
    """Return the latest time‑contiguous window for each node.

    Parameters
    ----------
    spec:
        FluxSpring specification containing the nodes.
    node_ids:
        Node ids to sample.
    cfg:
        Spectral configuration providing the window length and tapers.

    Returns
    -------
    window_matrix, kept_ids:
        ``window_matrix`` has shape ``(M, Nw)`` with each row holding the most
        recent ``Nw`` samples for a node.  ``kept_ids`` lists the node ids for
        which a window was gathered.  Nodes lacking a ring buffer are skipped.
    """

    Nw = int(cfg.win_len)
    wins: List[AT] = []
    kept: List[int] = []

    for nid in node_ids:
        n = spec.nodes[nid]
        if getattr(n, "ring", None) is None:
            continue

        buf = n.ring[:, 0] if AT.get_tensor(n.ring).ndim == 2 else n.ring
        R = int(buf.shape[0])
        idx = n.ring_idx % R if R > 0 else 0
        if R > 0 and idx != 0:
            buf = AT.cat([buf[idx:], buf[:idx]], dim=0)

        if R >= Nw:
            win = buf[-Nw:]
        else:
            pad = Nw - R
            win = AT.cat([AT.zeros(pad, dtype=AT.get_tensor(buf).dtype), buf], dim=0)

        wins.append(win)
        kept.append(nid)

    if not wins:
        return None, []
    return AT.stack(wins), kept


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
        [(((freqs >= lo) & (freqs <= hi)).astype(float)) for lo, hi in cfg.metrics.bands]
    ).transpose(0, 1)

    band_powers = AT.matmul(power, mask_FB)
    band_powers = band_powers / (AT.sum(band_powers, dim=1, keepdim=True) + 1e-12)
    return band_powers
