"""Auto-porting numpy source into AbstractTensor Python, through the compiler.

This is the round trip used as a *translator* rather than as a diagnostic:
authored numpy goes in, ``lower_ast_source_to_ssa`` normalises it into the
canonical tensor vocabulary, and ``ssa_python_materializer`` spells that
vocabulary back out as AbstractTensor Python.  Nothing here is hand-ported --
the emitted source is whatever the compiler produced, and the assertions run
it and compare against the numpy original.

The material is real: these are the constant-Q filterbank primitives from the
sibling ``spectral-analyzer`` repository (``torch_cqt_new._cqt_frequencies``
and ``_wavelet_lengths``, themselves clones of librosa's), written in the
plain numpy the math is actually expressed in.
"""
from __future__ import annotations

import ast
import warnings

import numpy as np
import pytest

from src.common.tensors import AbstractTensor
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_python_materializer import materialize_ir_module


def _auto_port(source: str, entrypoint: str, name: str):
    """Compile ``source`` and return (emitted_python, namespace, skipped)."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module, outputs, _exports = lower_ast_source_to_ssa(
            source, entrypoint, name=name
        )
    assert outputs.get(f"{name}__{entrypoint}"), (
        f"{entrypoint} lowered to no outputs at all -- the statement was "
        "dropped rather than compiled"
    )
    emitted, skipped = materialize_ir_module(module)
    code = ast.unparse(emitted)
    namespace = {"AbstractTensor": AbstractTensor}
    exec(compile(code, f"<auto-ported {name}>", "exec"), namespace)
    return code, namespace, skipped


def _np(value):
    return np.asarray(value.data if hasattr(value, "data") else value)


CQT_FREQUENCIES = """
import numpy as np
def cqt_frequencies(n_bins, fmin, bins_per_octave):
    return fmin * (2.0 ** (np.arange(n_bins) / bins_per_octave))
"""

WAVELET_LENGTHS = """
import numpy as np
def wavelet_lengths(freqs, sr, filter_scale, alpha):
    Q = filter_scale / alpha
    return Q * sr / freqs
"""


def test_numpy_arange_is_emitted_as_an_abstract_tensor_call():
    """The whole point: numpy went in, AbstractTensor came out."""

    code, _namespace, skipped = _auto_port(
        CQT_FREQUENCIES, "cqt_frequencies", "sa"
    )
    assert skipped == {}, f"materializer refused part of the program: {skipped}"
    assert "AbstractTensor.arange(" in code
    assert "np." not in code and "numpy" not in code


def test_auto_ported_cqt_ladder_matches_numpy():
    _code, namespace, skipped = _auto_port(
        CQT_FREQUENCIES, "cqt_frequencies", "sa2"
    )
    assert skipped == {}
    got = namespace["sa2__cqt_frequencies"](12, 32.7, 12)
    expected = 32.7 * (2.0 ** (np.arange(12) / 12))
    assert np.allclose(_np(got), expected)


def test_auto_ported_wavelet_lengths_matches_numpy():
    _code, namespace, skipped = _auto_port(
        WAVELET_LENGTHS, "wavelet_lengths", "wl"
    )
    assert skipped == {}
    freqs = np.array([32.7, 65.4, 130.8])
    alpha = np.array([0.1, 0.2, 0.4])
    got = namespace["wl__wavelet_lengths"](
        AbstractTensor.get_tensor(freqs), 44100.0, 1.0,
        AbstractTensor.get_tensor(alpha),
    )
    expected = (1.0 / alpha) * 44100.0 / freqs
    assert np.allclose(_np(got), expected)


def test_tensor_call_forms_are_read_from_the_class_not_invented():
    """Static-vs-method is derived, so it cannot drift from AbstractTensor."""

    from src.compiler.ssa_python_materializer import TENSOR_CALL_FORMS

    assert TENSOR_CALL_FORMS["arange"] is True        # AbstractTensor.arange(...)
    assert TENSOR_CALL_FORMS["where"] is True
    assert TENSOR_CALL_FORMS["clip"] is False         # x.clip(...)
    assert TENSOR_CALL_FORMS["index_select"] is False


def test_array_indexing_is_still_refused_loudly_not_guessed():
    """The honest current edge, recorded so it cannot regress into a guess.

    ``_relative_bandwidth``'s real body indexes and slices (``logf[1]``,
    ``logf[2:]``).  That lowers to a ``GetElementPtr`` addressing by computed
    path, which this materializer deliberately does not spell -- it refuses by
    name instead of emitting something that reads like Python and means
    something else.
    """

    source = """
import numpy as np
def relative_bandwidth(freqs):
    logf = np.log(freqs)
    return logf[1] - logf[0]
"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module, _outputs, _exports = lower_ast_source_to_ssa(
            source, "relative_bandwidth", name="rb"
        )
    _emitted, skipped = materialize_ir_module(module)
    assert skipped, "indexing should be refused, not silently emitted"
    assert any("computed path" in reason for reason in skipped.values())
