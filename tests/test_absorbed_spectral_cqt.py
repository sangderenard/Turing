"""The absorbed constant-Q primitives must agree with their numpy originals.

This is the test ``common.tensors.absorbed`` requires before a translation
counts as absorbed.  "It compiled" is explicitly not evidence in this tree --
the same investigation that produced this package found a path where every
stage reported success and the program computed nothing -- so each entry point
is executed here and compared elementwise against the original math.

Inputs are chosen to be non-degenerate on purpose.  A geometric ladder makes
log-spacing uniform, which would make a spacing-derived quantity constant and
agree even if the translation were wrong; the spacing cases below use
irregular values so a wrong answer is a different answer.
"""
from __future__ import annotations

import numpy as np

from src.common.tensors import AbstractTensor
from src.common.tensors.absorbed import spectral_cqt as absorbed


def _np(value) -> np.ndarray:
    return np.asarray(value.data if hasattr(value, "data") else value)


def test_provenance_is_recorded():
    record = absorbed.ABSORPTION
    assert record.source_repository == "spectral-analyzer"
    assert record.source_symbols
    assert record.verified_against
    assert "auto-ported" in record.describe()


def test_cqt_frequencies_matches_numpy():
    """The geometric ladder: centre k at fmin * 2**(k/bins_per_octave)."""

    got = absorbed.sa__cqt_frequencies(24, 32.7, 12)
    expected = 32.7 * (2.0 ** (np.arange(24) / 12))
    assert np.allclose(_np(got), expected)


def test_cqt_frequencies_spans_the_expected_octaves():
    """Two bands per octave apart must differ by exactly a factor of two."""

    ladder = _np(absorbed.sa__cqt_frequencies(25, 55.0, 12))
    assert np.isclose(ladder[12] / ladder[0], 2.0)
    assert np.isclose(ladder[24] / ladder[0], 4.0)


def test_relative_bandwidth_matches_numpy_on_irregular_spacing():
    spacing = AbstractTensor.get_tensor(np.array([0.5, 1.0, 2.0, 3.5]))
    got = absorbed.sa__relative_bandwidth_from_spacing(spacing)
    ratio = 2.0 ** (2.0 / np.array([0.5, 1.0, 2.0, 3.5]))
    expected = (ratio - 1.0) / (ratio + 1.0)
    assert np.allclose(_np(got), expected)


def test_relative_bandwidth_is_a_fraction_between_zero_and_one():
    """alpha is a bandwidth as a fraction of centre, so it must stay in (0, 1)."""

    spacing = AbstractTensor.get_tensor(np.array([0.25, 1.0, 4.0, 16.0]))
    alpha = _np(absorbed.sa__relative_bandwidth_from_spacing(spacing))
    assert np.all(alpha > 0.0) and np.all(alpha < 1.0)


def test_quality_factor_matches_numpy():
    alpha = np.array([0.1, 0.25, 0.5])
    got = absorbed.sa__quality_factor(AbstractTensor.get_tensor(alpha), 1.0)
    assert np.allclose(_np(got), 1.0 / alpha)


def test_wavelet_lengths_matches_numpy():
    freqs = np.array([32.7, 65.4, 130.8])
    alpha = np.array([0.1, 0.2, 0.4])
    got = absorbed.sa__wavelet_lengths(
        AbstractTensor.get_tensor(freqs), 44100.0, 1.0,
        AbstractTensor.get_tensor(alpha),
    )
    expected = (1.0 / alpha) * 44100.0 / freqs
    assert np.allclose(_np(got), expected)


def test_hann_window_matches_numpy_and_is_a_real_window():
    index = np.arange(8, dtype=float)
    got = _np(absorbed.sa__hann_window(AbstractTensor.get_tensor(index), 8.0))
    expected = 0.5 - 0.5 * np.cos(2.0 * np.pi * index / 8.0)
    assert np.allclose(got, expected)
    # A periodic Hann starts at zero and peaks at the centre of its period.
    assert np.isclose(got[0], 0.0)
    assert np.isclose(got[4], 1.0)


def test_log_spacing_matches_numpy():
    hi = np.array([3.0, 7.0, 8.5])
    lo = np.array([0.0, 1.0, 3.0])
    got = absorbed.sa__log_spacing(
        AbstractTensor.get_tensor(hi), AbstractTensor.get_tensor(lo)
    )
    assert np.allclose(_np(got), 2.0 / (hi - lo))


def test_the_ladder_and_bandwidth_compose_into_a_constant_q_bank():
    """The primitives together are the thing they were absorbed for.

    On a geometric ladder every band sees the same log-spacing, so every
    band's Q comes out the same -- which is exactly what "constant-Q" names.
    """

    bins_per_octave = 12
    ladder = _np(absorbed.sa__cqt_frequencies(13, 55.0, bins_per_octave))
    log_ladder = np.log2(ladder)

    spacing = _np(absorbed.sa__log_spacing(
        AbstractTensor.get_tensor(log_ladder[2:]),
        AbstractTensor.get_tensor(log_ladder[:-2]),
    ))
    alpha = _np(absorbed.sa__relative_bandwidth_from_spacing(
        AbstractTensor.get_tensor(spacing)
    ))
    quality = _np(absorbed.sa__quality_factor(
        AbstractTensor.get_tensor(alpha), 1.0
    ))

    assert np.allclose(quality, quality[0])
    assert quality[0] > 1.0
