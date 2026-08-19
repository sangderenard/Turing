"""Influence propagation on a circulant graph, spectral AND phasic.

Continues the identity set pinned in ``tests/test_dec_fft_continuous.py``
(``BuildGraphLaplace`` combinatorial ``D - A`` == the DEC incidence assembly
== the continuous Laplace-Beltrami operator on a periodic domain, and its
spectrum is obtainable by FFT instead of a dense eigensolve) and the axes
argued for in ``DTYPE_AND_SPECTRAL_DOMAIN_MANIFESTO.md``.

Ordinary graph diffusion (the heat kernel ``exp(-t L)``) is purely spectral:
it says how much influence has spread, never how far or in which direction --
every mode only decays, it never carries a phase. The same eigenbasis that
diagonalizes the Laplacian also diagonalizes the discrete WAVE equation
``d^2u/dt^2 = -L u``, whose modes oscillate at ``omega_m = sqrt(lambda_m)``
instead of merely decaying. :func:`propagate` interpolates between the two:
every mode still decays under the graph's diffusive spectrum (``damping``),
but can simultaneously carry a coherent phase (``phasic``) whose rate of
change is a genuine travel distance in graph hops, not just a spread
fraction -- see :func:`jump_length` and the module tests for what that means
concretely.

Complex arithmetic is intentionally never used for the per-mode gain
multiply. Elementwise ``*`` between two complex ``AbstractTensor``\\ s still
raises ``NodusUnsupported`` on the default backend today (manifesto sections
1.2 and 4.1 -- complex rung-2 realization is explicitly open, unlanded work).
``_complex_mul`` expands the product by hand into real add/sub/mul, which is
unconditionally free on every backend, and is *exactly* the rung-2 lowering
the manifesto specifies a compiled complex multiply should reduce to -- so it
also stands as a working reference for whatever eventually compiles that
cell natively.
"""
from __future__ import annotations

import math
from typing import Tuple

from ..abstraction import AbstractTensor


def circulant_spectrum(L) -> AbstractTensor:
    """Eigenvalues of a circulant Laplacian: the (real) DFT of its first row.

    The same identity ``tests/test_dec_fft_continuous.py`` pins against the
    dense eigensolve and the analytic ring/torus spectra
    (``test_ring_spectrum_via_fft_matches_analytic``,
    ``test_ring_spectrum_via_fft_matches_eigh``). Kept as a one-line public
    function here, rather than duplicated privately in every module that
    needs a circulant graph's spectrum, of which this is now the second.
    """

    L = AbstractTensor.get_tensor(L)
    return AbstractTensor.real(L[0].fft())


def _complex_mul(
    ar: AbstractTensor, ai: AbstractTensor,
    br: AbstractTensor, bi: AbstractTensor,
) -> Tuple[AbstractTensor, AbstractTensor]:
    """``(ar + i*ai)(br + i*bi)``, expanded into real ops only. See module
    docstring for why this is not simply ``AT.complex(...) * AT.complex(...)``.
    """

    real = ar * br - ai * bi
    imag = ar * bi + ai * br
    return real, imag


def propagate(
    L,
    u0,
    t: float,
    *,
    damping: float = 1.0,
    phasic: float = 0.0,
) -> Tuple[AbstractTensor, AbstractTensor]:
    """Propagate a signal through the eigenbasis that diagonalizes ``L``.

    ``L`` is a circulant graph Laplacian (e.g. from ``BuildGraphLaplace`` on
    a ring or torus adjacency); ``u0`` a signal on its nodes, real or
    complex; ``t`` the propagation time.

    The per-mode gain is
    ``exp(-damping * t * lambda_m) * exp(i * phasic * t * sqrt(lambda_m))``:

    * ``phasic=0`` is the ordinary heat kernel. Pure amplitude decay, real
      output, isotropic spreading -- spectral, not phasic.
    * ``damping=0, phasic=1`` is the discrete wave equation on the same
      graph: unit-modulus gain per mode (energy-preserving), phase
      accumulating at each mode's own dispersion ``omega_m = sqrt(lambda_m)``.
      A single-mode input travels a specific, computable number of graph
      hops in time ``t`` -- see :func:`jump_length` and
      ``tests/test_spectral_propagator.py`` for an exact demonstration.
    * Intermediate values blend the two: damped, phase-coherent propagation
      (a discrete telegrapher's/Klein-Gordon-style equation) -- influence
      that both spreads AND carries a directional, accumulating phase.

    Returns ``(real_part, imag_part)`` of the propagated signal -- two
    ordinary real ``AbstractTensor``\\ s, never a native complex product.
    """

    lam = circulant_spectrum(L)

    amplitude = (-float(damping) * float(t) * lam).exp()
    phase_angle = float(phasic) * float(t) * lam.sqrt()
    gain_real = amplitude * phase_angle.cos()
    gain_imag = amplitude * phase_angle.sin()

    u0 = AbstractTensor.get_tensor(u0)
    spectrum = u0.fft()
    ur, ui = AbstractTensor.real(spectrum), AbstractTensor.imag(spectrum)

    yr, yi = _complex_mul(ur, ui, gain_real, gain_imag)
    # The one unavoidable ifft call multiplies nothing complex-by-complex --
    # AT.complex only assembles a carrier from two already-real parts -- so
    # it does not reach the still-open gap this module otherwise routes
    # around.
    propagated = AbstractTensor.complex(yr, yi).ifft()
    return AbstractTensor.real(propagated), AbstractTensor.imag(propagated)


def jump_length(L, mode: int, t: float) -> float:
    """How many graph hops mode ``mode``'s phase front has advanced by ``t``.

    Under the pure-phasic (``damping=0``) propagator, a single spectral bin
    ``m`` evolves as ``exp(i * (2*pi*m*x/n + omega_m * t))`` -- a traveling
    wave whose points of constant phase move at ``omega_m * n / (2*pi*m)``
    graph hops per unit time (``omega_m = sqrt(lambda_m)``, the graph's own
    wave dispersion). This is the "specific jump length" a mode's influence
    has traveled: not how far it spread (the heat kernel's question), but
    how far its phase front moved, which recovers a genuine travel distance
    from a purely spectral quantity.

    Raises ``ValueError`` for ``mode=0`` (the DC mode never travels; its
    eigenvalue and frequency are both zero).
    """

    if mode == 0:
        raise ValueError("the DC mode (mode=0) has no phase velocity")
    lam = circulant_spectrum(L)
    n = int(lam.shape[0])
    # Scalar indexing collapses an AbstractTensor to a bare backend scalar
    # (not an AbstractTensor), so this reaches for math.sqrt rather than a
    # tensor method -- there is nothing left to dispatch through.
    omega_m = math.sqrt(float(lam[mode % n]))
    return omega_m * n * float(t) / (2.0 * math.pi * mode)


__all__ = ["circulant_spectrum", "propagate", "jump_length"]
