"""Spectrum to colour by colorimetry, not by a hue dial.

The influence field carries a real spectral power distribution: a set of
(frequency, weight) lines. Turning that into a colour by mapping the mean
frequency onto a hue is a convention, and it is not colour science -- two
different distributions with the same mean become the same colour, which
is exactly the case a reader most needs to distinguish.

This module does the standard thing instead. A distribution is integrated
against the CIE 1931 2-degree colour matching functions to give tristimulus
XYZ, which is then taken to sRGB through the IEC 61966-2-1 matrix and
transfer function. That is defensible to someone who does this for a
living, and it has the property the hue dial lacks: metamerism is
represented correctly, because it is a projection of the whole spectrum
rather than of one moment of it.

The matching functions
----------------------
The tabulated CIE functions are data, not closed form. The analytic fits
used here are the multi-lobe piecewise Gaussians of Wyman, Sloan and
Shirley, *Simple Analytic Approximations to the CIE XYZ Color Matching
Functions*, Journal of Computer Graphics Techniques 2 (2), 2013, which
reproduce the tabulated functions to about one percent of peak -- far
inside the error introduced by quantising to eight bits per channel.

They are written as SymPy expressions because they ARE the science: an
expression can be differentiated, integrated, series-expanded, checked
against the tables, or lowered by this compiler like any other authored
mathematics. A float function hides all of that.

    g(x; mu, s1, s2) = exp(-(x - mu)^2 / (2 * s^2)),  s = s1 if x < mu else s2

Evaluation runs through ``AbstractTensor`` so the same expressions serve a
diagnostic in Python and a lowered kernel, rather than existing twice.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import sympy

# Visible band. The influence field allocates frequencies on a normalised
# arc; `wavelength_of` maps that arc onto real nanometres so the
# colorimetry below operates on physical quantities rather than on an
# index that merely looks like one.
VIOLET_NM = 380.0
RED_NM = 700.0

#: sRGB primaries and D65 white, IEC 61966-2-1. Linear-light XYZ to linear
#: RGB. Rows are R, G, B.
XYZ_TO_LINEAR_SRGB = (
    (3.2406255, -1.5372080, -0.4986286),
    (-0.9689307, 1.8757561, 0.0415175),
    (0.0557101, -0.2040211, 1.0569959),
)

#: sRGB transfer function breakpoint and coefficients, same standard.
SRGB_LINEAR_CUTOFF = 0.0031308
SRGB_LINEAR_SLOPE = 12.92
SRGB_GAMMA = 2.4
SRGB_SCALE = 1.055
SRGB_OFFSET = 0.055

#: Wyman/Sloan/Shirley multi-lobe fits: (amplitude, mu, sigma_low, sigma_high)
#: per lobe, wavelengths in nanometres.
XBAR_LOBES = (
    (1.056, 599.8, 37.9, 31.0),
    (0.362, 442.0, 16.0, 26.7),
    (-0.065, 501.1, 20.4, 26.2),
)
YBAR_LOBES = (
    (0.821, 568.8, 46.9, 40.5),
    (0.286, 530.9, 16.3, 31.1),
)
ZBAR_LOBES = (
    (1.217, 437.0, 11.8, 36.0),
    (0.681, 459.0, 26.0, 13.8),
)

WAVELENGTH = sympy.Symbol("lambda_nm", real=True, positive=True)


def _lobe(amplitude: float, mu: float, low: float, high: float) -> Any:
    """One piecewise-Gaussian lobe as a SymPy expression.

    The two sigmas are what make the fit good: the CIE functions are
    markedly asymmetric about each peak, and a symmetric Gaussian cannot
    follow both flanks at once.
    """
    spread = sympy.Piecewise(
        (sympy.Float(low), WAVELENGTH < sympy.Float(mu)),
        (sympy.Float(high), True),
    )
    offset = WAVELENGTH - sympy.Float(mu)
    return sympy.Float(amplitude) * sympy.exp(
        -(offset ** 2) / (2 * spread ** 2)
    )


def matching_functions() -> tuple[Any, Any, Any]:
    """(xbar, ybar, zbar) as SymPy expressions in ``WAVELENGTH``."""
    return tuple(
        sum(_lobe(*lobe) for lobe in lobes)
        for lobes in (XBAR_LOBES, YBAR_LOBES, ZBAR_LOBES)
    )


def wavelength_of(frequency: float, arc_end: float = 0.75) -> float:
    """Map a field frequency on its normalised arc to nanometres.

    The arc is the field's own allocation space and stops short of a full
    turn on purpose, so this is a linear placement onto the visible band
    rather than a wrap. A frequency outside the arc is clamped rather than
    folded: folding would map two distinct origins onto one wavelength and
    silently create a metamer that the field did not contain.
    """
    position = min(max(float(frequency) / float(arc_end), 0.0), 1.0)
    return VIOLET_NM + position * (RED_NM - VIOLET_NM)


def tristimulus(lines: Iterable[tuple[float, float]], *, arc_end: float = 0.75):
    """Integrate a line spectrum against the matching functions -> XYZ.

    A line spectrum is a sum of weighted deltas, so the integral collapses
    to a weighted sum of the matching functions sampled at each line. No
    quadrature is involved and none should be: sampling a delta comb on a
    wavelength grid is where spectral renderers lose energy.
    """
    from ..common.tensors.abstraction import AbstractTensor

    entries = [
        (wavelength_of(frequency, arc_end), float(weight))
        for frequency, weight in lines
    ]
    if not entries:
        return AbstractTensor.zeros((3,), dtype=float)

    xbar, ybar, zbar = matching_functions()
    kernel = sympy.lambdify(WAVELENGTH, [xbar, ybar, zbar], "math")

    wavelengths = AbstractTensor.tensor([row[0] for row in entries])
    weights = AbstractTensor.tensor([row[1] for row in entries])
    samples = AbstractTensor.tensor([
        list(kernel(float(value))) for value in wavelengths.tolist()
    ])
    # weights (n,) against samples (n, 3): one weighted sum per channel.
    scaled = samples * weights.reshape(-1, 1)
    return scaled.sum(dim=0)


def xyz_to_srgb(xyz, *, normalise: bool = True):
    """Linear XYZ to display sRGB, matrix then transfer function.

    ``normalise`` divides by the largest channel before encoding. The
    field's weights are influence, not radiometric power, so their
    absolute scale has no photometric meaning; keeping the ratios and
    fixing the magnitude is the honest reading, and luminance is supplied
    separately from the spectrum's own power.
    """
    from ..common.tensors.abstraction import AbstractTensor

    matrix = AbstractTensor.tensor([list(row) for row in XYZ_TO_LINEAR_SRGB])
    linear = (matrix * xyz.reshape(1, -1)).sum(dim=1)
    values = [max(0.0, float(channel)) for channel in linear.tolist()]
    if normalise:
        peak = max(values)
        if peak > 0.0:
            values = [channel / peak for channel in values]
    encoded = [
        SRGB_LINEAR_SLOPE * channel if channel <= SRGB_LINEAR_CUTOFF
        else SRGB_SCALE * channel ** (1.0 / SRGB_GAMMA) - SRGB_OFFSET
        for channel in values
    ]
    return AbstractTensor.tensor([
        min(1.0, max(0.0, channel)) for channel in encoded
    ])


def spectrum_rgb(
    lines: Sequence[tuple[float, float]], *, arc_end: float = 0.75,
) -> tuple[float, float, float]:
    """The whole projection: line spectrum -> XYZ -> sRGB, in [0, 1]."""
    xyz = tristimulus(lines, arc_end=arc_end)
    if float(xyz.sum().item()) <= 0.0:
        return (0.0, 0.0, 0.0)
    return tuple(float(channel) for channel in xyz_to_srgb(xyz).tolist())


__all__ = [
    "WAVELENGTH", "VIOLET_NM", "RED_NM", "XYZ_TO_LINEAR_SRGB",
    "matching_functions", "wavelength_of", "tristimulus", "xyz_to_srgb",
    "spectrum_rgb",
]
