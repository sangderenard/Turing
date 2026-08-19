"""Influence propagation on a circulant graph: spectral heat, phasic wave,
and the blend between them.

Continues ``tests/test_dec_fft_continuous.py``'s pinned identity set (graph
Laplacian == DEC == continuous Laplace-Beltrami on a periodic domain, with
its spectrum obtainable by FFT). This module tests
``src/common/tensors/abstract_convolution/spectral_propagator.py``'s two
regimes and, concretely, what "phasic" buys over ordinary diffusion: a
travel distance recoverable from phase, not merely a spread fraction.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.common.tensors import AbstractTensor as AT
from src.common.tensors.abstract_convolution.spectral_propagator import (
    circulant_spectrum,
    jump_length,
    propagate,
)


def _ring_laplacian(n: int):
    L = np.zeros((n, n))
    for i in range(n):
        L[i, i] = 2.0
        L[i, (i + 1) % n] = -1.0
        L[i, (i - 1) % n] = -1.0
    return AT.get_tensor(L)


def _np(t) -> np.ndarray:
    return np.asarray(t.data if hasattr(t, "data") else t)


# --------------------------------------------------------------------------
# the spectrum itself matches the identity already pinned elsewhere
# --------------------------------------------------------------------------

def test_circulant_spectrum_matches_the_pinned_ring_formula():
    n = 12
    spectrum = _np(circulant_spectrum(_ring_laplacian(n)))
    analytic = 4.0 * np.sin(np.pi * np.arange(n) / n) ** 2
    assert np.allclose(spectrum, analytic, atol=1e-10)


# --------------------------------------------------------------------------
# phasic=0 recovers the ordinary heat kernel exactly, real-only
# --------------------------------------------------------------------------

def test_pure_heat_limit_matches_independent_fft_computation():
    n = 10
    L = _ring_laplacian(n)
    lam = _np(circulant_spectrum(L))
    damping, t = 0.2, 1.7

    u0 = AT.get_tensor(np.eye(n)[0])
    real, imag = propagate(L, u0, t, damping=damping, phasic=0.0)

    expected = np.fft.ifft(np.exp(-damping * t * lam))
    assert np.allclose(_np(real), expected.real, atol=1e-10)
    assert np.max(np.abs(_np(imag))) < 1e-10


def test_pure_heat_conserves_total_mass():
    """The DC mode always has eigenvalue 0, so its amplitude never decays
    regardless of damping -- total influence is conserved even as it spreads."""
    n = 9
    L = _ring_laplacian(n)
    u0 = AT.get_tensor(np.eye(n)[3])  # impulse at an arbitrary node
    real, _ = propagate(L, u0, t=5.0, damping=0.8, phasic=0.0)
    assert float(np.sum(_np(real))) == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------
# damping=0, phasic=1 is the discrete wave equation: unit-modulus gain
# --------------------------------------------------------------------------

def test_pure_wave_limit_is_energy_preserving_per_mode():
    """No damping means every mode's gain has magnitude exactly 1 -- energy
    moves through phase, not amplitude decay."""
    n = 11
    L = _ring_laplacian(n)
    u0 = AT.get_tensor(np.eye(n)[0])
    real, imag = propagate(L, u0, t=3.3, damping=0.0, phasic=1.0)

    # Parseval: total energy in real space equals total energy in mode
    # space, which is n (an impulse's spectrum has magnitude 1 per bin,
    # unit-modulus gain leaves it that way) divided by n^2 from ifft's
    # normalisation -- rather than re-derive it, cross-check against an
    # independent numpy computation of the same unit-modulus propagator.
    lam = _np(circulant_spectrum(L))
    gain = np.exp(1j * 3.3 * np.sqrt(lam))
    expected = np.fft.ifft(np.fft.fft(_np(u0)) * gain)
    assert np.allclose(_np(real) + 1j * _np(imag), expected, atol=1e-9)


def test_wave_dispersion_matches_the_pinned_spectrum_square_root():
    """omega_m = sqrt(lambda_m); lambda_m is already pinned as
    4*sin^2(pi*m/n) elsewhere, so omega_m = 2*|sin(pi*m/n)| follows directly
    -- no new analytic derivation, full reuse of the existing identity."""
    n = 14
    lam = _np(circulant_spectrum(_ring_laplacian(n)))
    omega = np.sqrt(lam)
    analytic = 2.0 * np.abs(np.sin(np.pi * np.arange(n) / n))
    assert np.allclose(omega, analytic, atol=1e-10)


# --------------------------------------------------------------------------
# the concrete claim: a single mode travels a specific, computable number
# of graph hops -- not merely "spreads" -- under the phasic propagator
# --------------------------------------------------------------------------

@pytest.mark.parametrize("mode,hops", [(1, 3), (2, 5), (1, -2)])
def test_single_mode_travels_exactly_the_predicted_number_of_hops(
    mode: int, hops: int,
) -> None:
    """A pure spectral bin is a traveling wave under the phasic propagator:
    choosing t so the phase advances by an exact multiple of the mode's
    spatial period must land the signal on an EXACT circular shift by that
    many hops -- the literal meaning of "jump length travel accumulation"
    recovered from a purely spectral computation."""

    n = 12
    L = _ring_laplacian(n)
    lam = _np(circulant_spectrum(L))
    omega_m = math.sqrt(lam[mode % n])
    # A spatial shift of one hop is a phase step of 2*pi*mode/n (the
    # mode's own spatial frequency), not 2*pi/n -- solving for t must
    # scale by mode, exactly as jump_length's own formula does.
    t = 2.0 * math.pi * mode * hops / (n * omega_m)

    x = np.arange(n)
    u0_np = np.exp(1j * 2 * np.pi * mode * x / n)
    u0 = AT.complex(AT.get_tensor(u0_np.real), AT.get_tensor(u0_np.imag))

    real, imag = propagate(L, u0, t, damping=0.0, phasic=1.0)
    produced = _np(real) + 1j * _np(imag)
    expected = np.roll(u0_np, -hops)

    assert np.allclose(produced, expected, atol=1e-8)
    assert jump_length(L, mode, t) == pytest.approx(float(hops), abs=1e-8)


def test_jump_length_refuses_the_dc_mode():
    L = _ring_laplacian(8)
    with pytest.raises(ValueError, match="mode=0"):
        jump_length(L, 0, 1.0)


# --------------------------------------------------------------------------
# a blended regime is neither pure heat nor pure wave, and is well-formed
# --------------------------------------------------------------------------

def test_blended_regime_decays_and_carries_phase_simultaneously():
    n = 10
    L = _ring_laplacian(n)
    lam = _np(circulant_spectrum(L))
    damping, phasic, t = 0.1, 0.5, 2.0

    u0 = AT.get_tensor(np.eye(n)[0])
    real, imag = propagate(L, u0, t, damping=damping, phasic=phasic)

    gain = np.exp(-damping * t * lam) * np.exp(1j * phasic * t * np.sqrt(lam))
    expected = np.fft.ifft(np.fft.fft(_np(u0)) * gain)

    assert np.allclose(_np(real) + 1j * _np(imag), expected, atol=1e-9)
    # It neither collapses to the pure-heat (all-real) nor the pure-wave
    # (unit-modulus) special case: it genuinely decays AND carries phase.
    assert np.max(np.abs(_np(imag))) > 1e-6
    assert float(np.sum(_np(real) ** 2 + _np(imag) ** 2)) < float(np.sum(_np(u0) ** 2))
