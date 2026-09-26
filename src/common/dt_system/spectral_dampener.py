"""FFT-based inertial dampener utilities for dt_system.

This module provides a reusable routine for estimating a smoothing force
from a history of node positions.  The implementation originated in the
``autoautograd`` toy spring system but has been generalised for the dt
runtime so other engines can reuse it.

The routine analyses recent motion using a windowed FFT, focuses on the
energetic frequency bands and synthesises an immediate rotation response.
The response can then be used as a damping term in an integrator.

The physics here is experimental and intentionally lightweight – it treats
node history as a flat sequence and does not attempt to model a full mesh.
A more faithful implementation would thread the spectral estimate through
proper DEC operators.  For now it acts as a non-linear low‑pass filter.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np

# Minimum number of history samples required before performing any FFT based
# analysis.  Below this threshold the routine returns a zero response so callers
# can treat the result as a no-op and simply pass the state through unchanged.
_MIN_FFT_WINDOW = 32


def spectral_inertia(history: Iterable[np.ndarray], dt: float) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float, float]]]:
    """Estimate a spectral inertia response from ``history``.

    Parameters
    ----------
    history:
        Iterable of past position vectors ordered from oldest to newest.
    dt:
        Sampling interval between successive entries in ``history``.

    Returns
    -------
    resp:
        Immediate ND response vector acting opposite to rapid oscillations.
    J:
        Skew-symmetric rotation generator aggregated over energetic bands.
    bands:
        Metadata describing analysed frequency bands ``(w_lo, w_hi, power)``.
    """
    # This is integrator-local physics over NumPy position histories. Routing
    # it through AbstractTensor made an unrelated globally selected compiler
    # backend responsible for FFT complex values (Nodus cannot store them),
    # allowing compilation policy to crash the visualization physics.
    hist = [np.asarray(value, dtype=float) for value in history]
    H = len(hist)
    if H < _MIN_FFT_WINDOW:
        D = hist[-1].shape[0] if H else 0
        return (
            np.zeros(D, dtype=float),
            np.zeros((D, D), dtype=float),
            [],
        )

    W = min(H, 128)
    xs = np.stack(hist[-W:])  # (W, D)
    if not np.isfinite(xs).all():
        D = xs.shape[1]
        return (
            np.zeros(D, dtype=float),
            np.zeros((D, D), dtype=float),
            [],
        )

    xs = xs - xs.mean(axis=0, keepdims=True)
    scale = max(1.0, float(np.linalg.norm(xs, ord=np.inf)))
    xs = xs / scale

    D = xs.shape[1]
    w = np.hanning(W) if W > 1 else np.ones(W)
    xw = w[:, None] * xs

    C0 = np.fft.rfft(xw, axis=0)  # (F0, D)
    w0 = 2.0 * np.pi * np.fft.rfftfreq(int(W), d=dt)
    P0 = np.sum(np.abs(C0) ** 2, axis=1)
    if P0.sum() <= 1e-12 or len(P0) <= 2:
        return (
            np.zeros(D, dtype=float),
            np.zeros((D, D), dtype=float),
            [],
        )

    rel = 0.01 * float(P0.max())
    abs_th = max(rel, 1e-12)
    active = P0 > abs_th

    bands_idx = []
    i = 0
    while i < len(active):
        if active[i]:
            j = i + 1
            while j < len(active) and active[j]:
                j += 1
            if (j - i) >= 1:
                lo = max(0, i - 1)
                hi = min(len(active), j + 1)
                bands_idx.append((lo, hi))
            i = j
        else:
            i += 1
    if not bands_idx:
        return (
            np.zeros(D, dtype=float),
            np.zeros((D, D), dtype=float),
            [],
        )

    Z = 8
    Wz = W * Z
    xpad = np.pad(xw, ((0, Wz - W), (0, 0)))
    Cz = np.fft.rfft(xpad, axis=0)
    wz = 2.0 * np.pi * np.fft.rfftfreq(Wz, d=dt)

    def coarse_band_to_w(b_lo, b_hi):
        return w0[b_lo], w0[min(b_hi, len(w0) - 1)]

    def w_to_hi_idx(wlo, whi):
        i0 = int(np.clip(np.searchsorted(wz, wlo, side="left"), 0, len(wz) - 1))
        i1 = int(np.clip(np.searchsorted(wz, whi, side="right"), 0, len(wz)))
        return i0, max(i1, i0 + 1)

    J = np.zeros((D, D), dtype=float)
    bands_meta: List[Tuple[float, float, float]] = []
    total_power = 0.0

    for (blo, bhi) in bands_idx:
        w_lo, w_hi = coarse_band_to_w(blo, bhi)
        hi_lo, hi_hi = w_to_hi_idx(w_lo, w_hi)
        Cz_band = Cz[hi_lo:hi_hi, :]
        if Cz_band.shape[0] < 1:
            continue
        Pw = np.sum(np.abs(Cz_band) ** 2, axis=1) + 1e-12
        if not np.isfinite(Pw).all() or Pw.sum() <= 1e-12:
            continue
        Ww = Pw / Pw.sum()
        wgrid = wz[hi_lo:hi_hi]
        for c, wght, omg in zip(Cz_band, Ww, wgrid):
            a = np.real(c)
            b = np.imag(c)
            J += wght * omg * (np.outer(a, b) - np.outer(b, a))
        band_power = float(Pw.sum())
        total_power += band_power
        bands_meta.append((w_lo, w_hi, band_power))

    if total_power <= 1e-12:
        return (
            np.zeros(D, dtype=float),
            np.zeros((D, D), dtype=float),
            [],
        )

    x_t = xs[-1]
    resp = J @ x_t
    return resp, J, bands_meta
