"""``y.sum(axis=k)`` and its siblings over static shapes, on every backend.

Each result is one value per lane of the kept axes, row-major, folded along
axis k in index order; ``max``/``min`` take numpy's rule, so a NaN in a lane
is that lane's result. Every backend used to refuse these by name.
"""

import numpy as np
import pytest

from tests.test_max_semantics import _compile_and_run
from tests.test_runtime_sized_spans import BACKENDS

_NAN = float("nan")
_MATRIX = np.array([[1.0, 7.0, 3.0], [4.0, _NAN, 9.0]])
_CUBE = np.arange(24, dtype=float).reshape(2, 3, 4) % 7 - 2.5


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("op", ["sum", "prod", "mean", "max", "min"])
@pytest.mark.parametrize("data,axis", [
    (_MATRIX, 0), (_MATRIX, 1), (_MATRIX, -1),
    (_CUBE, 0), (_CUBE, 1), (_CUBE, 2), (_CUBE, -2),
])
def test_axis_reduction_matches_numpy(tmp_path, backend, op, data, axis):
    got = _compile_and_run(
        tmp_path, backend, f"def f(y):\n    return y.{op}(axis={axis})\n",
        f"ax_{backend}_{op}_{data.ndim}_{axis + 3}", data.ndim, data.shape,
        {"y": data},
    )
    want = getattr(np, op)(data, axis=axis)
    np.testing.assert_allclose(
        np.asarray(got).reshape(-1), want.reshape(-1), equal_nan=True,
    )
