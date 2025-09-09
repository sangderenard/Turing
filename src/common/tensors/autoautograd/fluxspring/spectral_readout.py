# -*- coding: utf-8 -*-
"""
Spectral feature extraction utilities for FluxSpring.

``compute_metrics`` accepts a ``buffer`` and configuration describing the sample
rate, FFT parameters and which metrics to compute.  All numeric work uses the
:class:`AbstractTensor` API so callers can remain backend agnostic.  If a backend
lacks FFT primitives the implementation falls back to a sine/cosine basis DFT.
"""
from __future__ import annotations

from typing import Callable, Dict, Any, List
import math
import numpy as np

from ...abstraction import AbstractTensor as AT

# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------

def _hann_window(n: int) -> AT:
    k = np.arange(n, dtype=float)
    w = 0.5 - 0.5 * np.cos(2.0 * math.pi * k / (n - 1))
    return AT.get_tensor(w)

def _rect_window(n: int) -> AT:
    return AT.get_tensor(np.ones(n, dtype=float))

_WINDOW_REGISTRY: Dict[str, Callable[[int], AT]] = {
    "hann": _hann_window,
    "rect": _rect_window,
}

# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def _apply_window(x: AT, win_fn: Callable[[int], AT]) -> AT:
    if win_fn is None:
        return x
    w = win_fn(x.shape[-1])
    return x * w

def _fft_power(x: AT, n: int, tick_hz: float) -> tuple[AT, AT, bool, List[AT], List[AT]]:
    """Return (power, freqs, used_fft, cos_parts, sin_parts)."""
    try:
        fft = AT.fft.rfft(x, n=n, axis=-1)
        power = AT.real(fft) ** 2 + AT.imag(fft) ** 2
        freqs = AT.fft.rfftfreq(n, d=1.0 / tick_hz)
        return power, freqs, True, [], []
    except Exception:
        # Manual DFT using cosine/sine basis
        k = AT.arange(0, n // 2 + 1, dtype=AT.float_dtype_)
        t = AT.arange(0, n, dtype=AT.float_dtype_)
        angles = 2.0 * math.pi * k[:, None] * t[None, :] / float(n)
        cos_term = AT.cos(angles)
        sin_term = -AT.sin(angles)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        # Broadcast multiplication then sum over time axis
        cos_part = (x[:, None, :] * cos_term[None, :, :]).sum(-1)
        sin_part = (x[:, None, :] * sin_term[None, :, :]).sum(-1)
        power = cos_part**2 + sin_part**2
        freqs = k * (tick_hz / n)
        cos_list = [cos_part[i] for i in range(cos_part.shape[0])]
        sin_list = [sin_part[i] for i in range(sin_part.shape[0])]
        return power, freqs, False, cos_list, sin_list

def compute_metrics(buffer: Any, config: Dict[str, Any]) -> Dict[str, AT]:
    """Compute spectral metrics over ``buffer`` according to ``config``.

    Parameters
    ----------
    buffer:
        1-D or 2-D array-like.  ``AbstractTensor`` instances are used directly,
        otherwise they are converted via :func:`AbstractTensor.get_tensor`.
    config:
        Dictionary with keys:
        ``tick_hz`` (float), ``win_len`` (int), ``window_fn`` (callable or str)
        and ``metrics`` (dict) specifying which metrics to compute.  Supported
        metrics: ``bands`` (list of [lo,hi]), ``centroid`` (bool), ``flatness``
        (bool), ``coherence`` (bool).
    """
    x = buffer if isinstance(buffer, AT) else AT.get_tensor(buffer)
    x = x.float()
    n = int(config.get("win_len", x.shape[-1]))
    x = x[..., -n:]

    win_fn = config.get("window_fn")
    if isinstance(win_fn, str):
        win_fn = _WINDOW_REGISTRY.get(win_fn, _hann_window)
    elif win_fn is None:
        name = config.get("window", "hann")
        win_fn = _WINDOW_REGISTRY.get(name, _hann_window)
    x = _apply_window(x, win_fn)

    tick_hz = float(config.get("tick_hz", 1.0))
    power, freqs, used_fft, cos_parts, sin_parts = _fft_power(x, n, tick_hz)

    metrics_cfg = config.get("metrics", {})
    out: Dict[str, AT] = {}

    bands = metrics_cfg.get("bands", [])
    if bands:
        band_vals: List[AT] = []
        for lo, hi in bands:
            mask = ((freqs >= lo) & (freqs < hi)).float()
            band_vals.append((power * mask).sum(dim=-1))
        out["bandpower"] = AT.stack(band_vals, dim=-1)

    if metrics_cfg.get("centroid"):
        num = (power * freqs).sum(dim=-1)
        den = power.sum(dim=-1) + 1e-12
        out["centroid"] = num / den

    if metrics_cfg.get("flatness"):
        p = power + 1e-12
        geo = AT.log(p).mean(dim=-1).exp()
        arith = p.mean(dim=-1)
        out["flatness"] = geo / arith

    if metrics_cfg.get("coherence"):
        if x.ndim >= 2 and x.shape[0] >= 2:
            if used_fft:
                ffts = AT.fft.rfft(x[:2], n=n, axis=-1)
                r0, i0 = AT.real(ffts[0]), AT.imag(ffts[0])
                r1, i1 = AT.real(ffts[1]), AT.imag(ffts[1])
            else:
                r0, i0 = cos_parts[0], sin_parts[0]
                r1, i1 = cos_parts[1], sin_parts[1]
            Sxx = r0**2 + i0**2
            Syy = r1**2 + i1**2
            cross_real = r0 * r1 + i0 * i1
            cross_imag = i0 * r1 - r0 * i1
            Sxy2 = cross_real**2 + cross_imag**2
            coh = (Sxy2 / (Sxx * Syy + 1e-12)).mean(dim=-1)
            out["coherence"] = coh
    return out
