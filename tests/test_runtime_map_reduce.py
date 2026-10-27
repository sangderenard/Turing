"""A reduction of an elementwise result over runtime-sized spans.

Program: ``def f(x, y): return (x * y).sum()`` with x and y spans whose
length arrives with the call.  The product is one planned region's result
and the sum another's.  Every backend used to refuse it (the sum had no
element count) or, in C, to size the product as one scalar.  Rule: the
product's storage and the reduction's trip count both come from the extents
its elementwise producer is sized by.
"""

import math

import numpy as np
import pytest

from tests.test_max_semantics import _compile_and_run, _same
from tests.test_runtime_sized_spans import BACKENDS

_NAN = float("nan")
_PROGRAMS = {
    "(x * y).sum()": lambda x, y: (x * y).sum(),
    "(x - y).max()": lambda x, y: (x - y).max(),
    "(x + y).min() * 2.0 + 1.0": lambda x, y: (x + y).min() * 2.0 + 1.0,
    "(x * y).sum() / (x * x).sum()": lambda x, y: (x * y).sum() / (x * x).sum(),
}
_FEEDS = [
    (np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.5, -1.0, 2.0, 3.0])),
    (np.array([1.0, _NAN, 3.0]), np.array([2.0, 1.0, -4.0])),
    (np.array([2.5]), np.array([-1.0])),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("source", list(_PROGRAMS))
def test_reduction_of_a_runtime_sized_product(tmp_path, backend, source):
    for index, (x, y) in enumerate(_FEEDS):
        got = _compile_and_run(
            tmp_path, backend, f"def f(x, y):\n    return {source}\n",
            f"mr_{backend}_{abs(hash(source)) % 9999}_{index}", 1, None,
            {"x": x, "y": y},
        )
        want = _PROGRAMS[source](x, y)
        _same(np.asarray(got).reshape(-1)[:1], [want])


@pytest.mark.parametrize("backend", BACKENDS)
def test_rank_two_runtime_sized_product_reduces_over_every_element(tmp_path, backend):
    x = np.arange(6.0).reshape(2, 3)
    y = np.array([[1.0, -2.0, 0.5], [3.0, 1.0, -1.0]])
    for op in ("sum", "max"):
        got = _compile_and_run(
            tmp_path, backend, f"def f(x, y):\n    return (x * y).{op}()\n",
            f"mr2_{backend}_{op}", 2, None, {"x": x, "y": y},
        )
        _same(np.asarray(got).reshape(-1)[:1], [getattr(x * y, op)()])


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b != "fortran"])
def test_empty_runtime_sized_product(tmp_path, backend):
    # Fortran's harness refuses a zero extent before the call, by name.
    empty = np.zeros((0,))
    got = _compile_and_run(
        tmp_path, backend, "def f(x, y):\n    return (x * y).sum()\n",
        f"mre_{backend}_sum", 1, None, {"x": empty, "y": empty},
    )
    _same(np.asarray(got).reshape(-1)[:1], [0.0])
    with pytest.raises(RuntimeError, match="74"):
        _compile_and_run(
            tmp_path, backend, "def f(x, y):\n    return (x * y).max()\n",
            f"mre_{backend}_max", 1, None, {"x": empty, "y": empty},
        )
