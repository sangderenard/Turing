"""Constant-Q filterbank primitives, auto-ported from spectral-analyzer.

MACHINE-TRANSLATED. Nothing in this file was written by hand. It is the
output of ``tools/absorb_source.py`` -- the numpy originals compiled through
``lower_ast_source_to_ssa`` and re-emitted by ``ssa_python_materializer`` with
``tensor_vocabulary=True`` -- and it is kept verbatim, including the SSA's own
temporary names and its region/aggregate calling convention. See
``common.tensors.absorbed`` for why it is not tidied.

These are the primitives a constant-Q bank is built from: a geometric ladder
of centres, the fractional bandwidth that makes neighbouring bands meet, the
quality factor, the filter length, and a Hann window. Their purpose here is
spectral GRAPH analysis: a constant-Q bank over a Laplacian's eigenvalues is a
spectral graph wavelet bank, geometric scales with bandwidth proportional to
centre, which is the same construction read on graph frequency instead of
audio frequency.

Each ``*__planned_region_0`` is the compiler's own region function; the
matching bare name is its entry point and is what callers use.
"""
from __future__ import annotations

from src.common.tensors import AbstractTensor
from src.common.tensors.absorbed.provenance import Absorption

ABSORPTION = Absorption(
    source_repository="spectral-analyzer",
    source_path="torch_cqt_new.py",
    source_symbols=(
        "_cqt_frequencies",
        "_relative_bandwidth",
        "_wavelet_lengths",
        "_periodic_window",
    ),
    entrypoints=(
        "cqt_frequencies",
        "relative_bandwidth_from_spacing",
        "quality_factor",
        "wavelet_lengths",
        "hann_window",
        "log_spacing",
    ),
    verified_against=(
        "each entry point executed and compared elementwise against the numpy "
        "original on non-degenerate input; see "
        "tests/test_absorbed_spectral_cqt.py"
    ),
    caveats=(
        "The originals are torch functions; the numpy form of the same math "
        "was what was compiled, since the math and not the framework is what "
        "transfers.",
        "_relative_bandwidth's own body reads its neighbours by slicing to "
        "build the spacing; only the spacing->alpha step is absorbed here, "
        "with the spacing passed in.",
    ),
)


def sa__cqt_frequencies__planned_region_0(t1, t2, t0):
    t4 = 2.0
    t5 = AbstractTensor.arange(t1)
    t6 = t5 / t2
    t7 = t4 ** t6
    t8 = t0 * t7
    return (t8, t4, t5, t6, t7)

def sa__cqt_frequencies(n_bins, fmin, bins_per_octave):
    t9 = sa__cqt_frequencies__planned_region_0(n_bins, bins_per_octave, fmin)
    t18 = 4
    t19 = t9[4]
    t7 = t19
    t16 = 3
    t17 = t9[3]
    t6 = t17
    t14 = 2
    t15 = t9[2]
    t5 = t15
    t12 = 1
    t13 = t9[1]
    t4 = t13
    t10 = 0
    t11 = t9[0]
    t8 = t11
    return t8

def sa__relative_bandwidth_from_spacing__planned_region_0(t0):
    t2 = 2.0
    t1 = 2.0
    t5 = 1.0
    t7 = 1.0
    t3 = t2 / t0
    t4 = t1 ** t3
    t6 = t4 - t5
    t8 = t4 + t7
    t9 = t6 / t8
    return (t9, t1, t2, t3, t4, t5, t6, t7, t8)

def sa__relative_bandwidth_from_spacing(spacing):
    t10 = sa__relative_bandwidth_from_spacing__planned_region_0(spacing)
    t27 = 8
    t28 = t10[8]
    t8 = t28
    t25 = 7
    t26 = t10[7]
    t7 = t26
    t23 = 6
    t24 = t10[6]
    t6 = t24
    t21 = 5
    t22 = t10[5]
    t5 = t22
    t19 = 4
    t20 = t10[4]
    t4 = t20
    t17 = 3
    t18 = t10[3]
    t3 = t18
    t15 = 2
    t16 = t10[2]
    t2 = t16
    t13 = 1
    t14 = t10[1]
    t1 = t14
    t11 = 0
    t12 = t10[0]
    t9 = t12
    return t9

def sa__quality_factor__planned_region_0(t0, t1):
    t2 = t0 / t1
    return (t2,)

def sa__quality_factor(alpha, filter_scale):
    t3 = sa__quality_factor__planned_region_0(filter_scale, alpha)
    t4 = 0
    t5 = t3[0]
    t2 = t5
    return t2

def sa__wavelet_lengths__planned_region_0(t0, t1, t2, t3):
    t4 = t0 / t1
    t5 = t4 * t2
    t6 = t5 / t3
    return (t6, t4, t5)

def sa__wavelet_lengths(freqs, sr, filter_scale, alpha):
    t7 = sa__wavelet_lengths__planned_region_0(filter_scale, alpha, sr, freqs)
    t12 = 2
    t13 = t7[2]
    t5 = t13
    t10 = 1
    t11 = t7[1]
    t4 = t11
    t8 = 0
    t9 = t7[0]
    t6 = t9
    return t6

def sa__hann_window__planned_region_0(t1, t2):
    t7 = 6.283185307179586
    t5 = 0.5
    t4 = 0.5
    t8 = t7 * t1
    t9 = t8 / t2
    t10 = t9.cos()
    t11 = t5 * t10
    t12 = t4 - t11
    return (t12, t4, t5, t7, t8, t9, t10, t11)

def sa__hann_window(n_index, n_total):
    t13 = sa__hann_window__planned_region_0(n_index, n_total)
    t28 = 7
    t29 = t13[7]
    t11 = t29
    t26 = 6
    t27 = t13[6]
    t10 = t27
    t24 = 5
    t25 = t13[5]
    t9 = t25
    t22 = 4
    t23 = t13[4]
    t8 = t23
    t20 = 3
    t21 = t13[3]
    t7 = t21
    t18 = 2
    t19 = t13[2]
    t5 = t19
    t16 = 1
    t17 = t13[1]
    t4 = t17
    t14 = 0
    t15 = t13[0]
    t12 = t15
    return t12

def sa__log_spacing__planned_region_0(t0, t1):
    t2 = 2.0
    t3 = t0 - t1
    t4 = t2 / t3
    return (t4, t2, t3)

def sa__log_spacing(logf_hi, logf_lo):
    t5 = sa__log_spacing__planned_region_0(logf_hi, logf_lo)
    t10 = 2
    t11 = t5[2]
    t3 = t11
    t8 = 1
    t9 = t5[1]
    t2 = t9
    t6 = 0
    t7 = t5[0]
    t4 = t7
    return t4
