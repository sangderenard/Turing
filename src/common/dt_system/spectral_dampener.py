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

from ..tensors.abstraction import AbstractTensor


def spectral_inertia(history: Iterable[AbstractTensor], dt: float) -> Tuple[AbstractTensor, AbstractTensor, List[Tuple[float, float, float]]]:
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
    hist = list(history)
    H = len(hist)
    if H < 32:
        if H == 0:
            D = 0
        else:
            D = hist[0].shape[0]
        return (
            AbstractTensor.zeros(D, float),
            AbstractTensor.zeros((D, D), float),
            [],
        )

    W = min(H, 128)
    xs = AbstractTensor.stack(hist[-W:])  # (W, D)
    if not AbstractTensor.isfinite(xs).all():
        D = xs.shape[1]
        return (
            AbstractTensor.zeros(D, float),
            AbstractTensor.zeros((D, D), float),
            [],
        )

    xs = xs - xs.mean(dim=0, keepdim=True)
    scale = max(1.0, float(AbstractTensor.linalg.norm(xs, ord=AbstractTensor.inf)))
    xs = xs / scale

    D = xs.shape[1]
    w = AbstractTensor.hanning(W) if W > 1 else AbstractTensor.ones(W)
    xw = w[:, None] * xs

    C0 = AbstractTensor.fft.rfft(xw, axis=0)  # (F0, D)
    w0 = 2.0 * AbstractTensor.pi() * AbstractTensor.fft.rfftfreq(int(W), d=dt, like=xs)
    P0 = AbstractTensor.sum(AbstractTensor.abs(C0) ** 2, dim=1)
    if P0.sum() <= 1e-12 or len(P0) <= 2:
        return (
            AbstractTensor.zeros(D, float),
            AbstractTensor.zeros((D, D), float),
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
            AbstractTensor.zeros(D, float),
            AbstractTensor.zeros((D, D), float),
            [],
        )

    Z = 8
    Wz = W * Z
    xpad = AbstractTensor.pad(xw, (0, 0, 0, Wz - W))
    Cz = AbstractTensor.fft.rfft(xpad, axis=0)
    wz = 2.0 * AbstractTensor.pi() * AbstractTensor.fft.rfftfreq(Wz, d=dt, like=xs)

    def coarse_band_to_w(b_lo, b_hi):
        return w0[b_lo], w0[min(b_hi, len(w0) - 1)]

    def w_to_hi_idx(wlo, whi):
        i0 = AbstractTensor.get_tensor(AbstractTensor.searchsorted(wz, wlo, side="left")).clip(0, len(wz) - 1)
        i1 = AbstractTensor.get_tensor(AbstractTensor.searchsorted(wz, whi, side="right")).clip(0, len(wz))
        return i0, max(i1, i0 + 1)

    J = AbstractTensor.zeros((D, D), float)
    bands_meta: List[Tuple[float, float, float]] = []
    total_power = 0.0

    for (blo, bhi) in bands_idx:
        w_lo, w_hi = coarse_band_to_w(blo, bhi)
        hi_lo, hi_hi = w_to_hi_idx(w_lo, w_hi)
        Cz_band = Cz[hi_lo:hi_hi, :]
        if Cz_band.shape[0] < 1:
            continue
        Pw = AbstractTensor.sum(AbstractTensor.abs(Cz_band) ** 2, dim=1) + 1e-12
        if not AbstractTensor.isfinite(Pw).all() or Pw.sum() <= 1e-12:
            continue
        Ww = Pw / Pw.sum()
        wgrid = wz[hi_lo:hi_hi]
        for c, wght, omg in zip(Cz_band, Ww, wgrid):
            a = AbstractTensor.real(c)
            b = AbstractTensor.imag(c)
            J += wght * omg * (AbstractTensor.outer(a, b) - AbstractTensor.outer(b, a))
        band_power = float(Pw.sum())
        total_power += band_power
        bands_meta.append((w_lo, w_hi, band_power))

    if total_power <= 1e-12:
        return (
            AbstractTensor.zeros(D, float),
            AbstractTensor.zeros((D, D), float),
            [],
        )

    x_t = xs[-1]
    resp = J @ x_t
    return resp, J, bands_meta
