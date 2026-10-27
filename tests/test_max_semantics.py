"""Every authored meaning of ``max``/``min`` keeps Python's answer.

The planner stamps ``max_semantics`` from the call form (see
``receiver_resolution_prepasses.max_semantics_of_call``); each backend then
spells that meaning, NaN and signed zero included:

* ``select`` -- builtin ``max(a, b)``: keep ``a`` unless ``b`` is strictly
  greater (NaN never wins a comparison, so a leading NaN stays).
* ``scan`` -- builtin ``max(xs)``: the same rule along the elements.
* ``reduce`` -- ``xs.max()``: numpy's fold, NaN propagates, a tie takes the
  later element.
* ``elementwise`` -- ``x.maximum(y)``: numpy's ``maximum`` per element.
"""

import math

import numpy as np
import pytest

from tests.test_runtime_sized_spans import BACKENDS, _lower, _run

NAN = float("nan")


def _same(got, want):
    got = np.asarray(got, dtype=float).reshape(-1)
    want = np.asarray(want, dtype=float).reshape(-1)
    assert got.shape == want.shape, (got, want)
    for g, w in zip(got, want):
        if math.isnan(w):
            assert math.isnan(g), (got, want)
        else:
            assert g == w and math.copysign(1.0, g) == math.copysign(1.0, w), (
                got, want,
            )


def _compile_and_run(tmp_path, backend, source, name, rank, shape, feeds):
    module, entry, parameters, result = _lower(tmp_path, source, name, rank, shape)
    _artifact, execution = _run(
        backend, module, entry,
        {parameters[key]: np.asarray(value, dtype=float)
         for key, value in feeds.items() if key in parameters},
        tmp_path,
    )
    return execution.buffers[int(result.id)]


_PAIRS = [(NAN, 1.0), (1.0, NAN), (-0.0, 0.0), (0.0, -0.0), (2.0, 3.0)]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("builtin", [max, min], ids=["max", "min"])
def test_builtin_select_of_two_scalars(tmp_path, backend, builtin):
    source = f"def f(x, y):\n    return {builtin.__name__}(x, y)\n"
    for index, (x, y) in enumerate(_PAIRS):
        got = _compile_and_run(
            tmp_path, backend, source, f"sel_{builtin.__name__}_{backend}_{index}",
            0, None, {"x": x, "y": y},
        )
        _same(got, [builtin(x, y)])


_ROWS = [
    [NAN, 1.0, 5.0, 2.0], [1.0, NAN, 5.0, 2.0], [1.0, 5.0, 2.0, NAN],
    [-0.0, 0.0, -1.0, -2.0], [0.0, -0.0, 1.0, 2.0],
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("builtin", [max, min], ids=["max", "min"])
def test_builtin_scan_of_one_tensor(tmp_path, backend, builtin):
    source = (
        f"def f(x, y):\n    return {builtin.__name__}(x) + 0.0 * y[0]\n"
    )
    for index, row in enumerate(_ROWS):
        got = _compile_and_run(
            tmp_path, backend, source, f"scan_{builtin.__name__}_{backend}_{index}",
            1, (4,), {"x": row, "y": [0.0] * 4},
        )
        _same(got, [builtin(row) + 0.0])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("method", ["max", "min"])
def test_method_reduce_of_one_tensor(tmp_path, backend, method):
    """numpy's own fold order for a tie of signed zeros is its SIMD
    schedule's, so only the NaN rows are compared here."""

    source = f"def f(x, y):\n    return x.{method}() + 0.0 * y[0]\n"
    for index, row in enumerate(_ROWS[:3]):
        got = _compile_and_run(
            tmp_path, backend, source, f"red_{method}_{backend}_{index}",
            1, (4,), {"x": row, "y": [0.0] * 4},
        )
        _same(got, [getattr(np, method)(np.asarray(row)) + 0.0])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("method", ["emax", "emin"])
def test_emax_is_builtin_max_per_element(tmp_path, backend, method):
    source = f"def f(x, y):\n    return x.{method}(y)\n"
    x = [NAN, 1.0, -0.0, 0.0, 2.0]
    y = [1.0, NAN, 0.0, -0.0, 3.0]
    got = _compile_and_run(
        tmp_path, backend, source, f"em_{method}_{backend}", 1, (5,),
        {"x": x, "y": y},
    )
    builtin = max if method == "emax" else min
    _same(got, [builtin(a, b) for a, b in zip(x, y)])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("method", ["maximum", "minimum"])
def test_elementwise_maximum_of_two_tensors(tmp_path, backend, method):
    source = f"def f(x, y):\n    return x.{method}(y)\n"
    x = [NAN, 1.0, -0.0, 0.0, 2.0]
    y = [1.0, NAN, 0.0, -0.0, 3.0]
    got = _compile_and_run(
        tmp_path, backend, source, f"ew_{method}_{backend}", 1, (5,),
        {"x": x, "y": y},
    )
    _same(got, getattr(np, method)(np.asarray(x), np.asarray(y)))


@pytest.mark.parametrize("shape", [(3,), None], ids=["static", "runtime"])
def test_builtin_max_of_many_element_tensors_is_refused_like_python(tmp_path, shape):
    """Python raises for ``max(x, y)`` of multi-element tensors (a tensor's
    truth value is ambiguous); the planner refuses it and names ``emax``."""

    with pytest.raises(ValueError, match=r"x\.emax\(y\)"):
        _lower(
            tmp_path, "def f(x, y):\n    return max(x, y)\n",
            f"many_{shape is None}", 1, shape,
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_builtin_max_of_one_element_tensors_keeps_pythons_pick(tmp_path, backend):
    source = "def f(x, y):\n    return max(x, y)\n"
    for index, (x, y) in enumerate(_PAIRS):
        got = _compile_and_run(
            tmp_path, backend, source, f"one_{backend}_{index}", 1, (1,),
            {"x": [x], "y": [y]},
        )
        _same(got, [max(x, y)])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("method", ["emax", "maximum"])
def test_elementwise_methods_over_runtime_sized_spans(tmp_path, backend, method):
    """A method's argument is an operand of the runtime-sized law: the result
    is a span of the host's extents, never one element."""

    x = [NAN, 1.0, -0.0, 0.0, 2.0, 7.0]
    y = [1.0, NAN, 0.0, -0.0, 3.0, 6.0]
    got = _compile_and_run(
        tmp_path, backend, f"def f(x, y):\n    return x.{method}(y)\n",
        f"{method}_rt_{backend}", 1, None, {"x": x, "y": y},
    )
    if method == "emax":
        want = [max(a, b) for a, b in zip(x, y)]
    else:
        want = np.maximum(np.asarray(x), np.asarray(y))
    _same(got, want)
