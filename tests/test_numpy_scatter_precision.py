"""NumPy scatter casts each accumulated result, never its source in advance.

The integer-tensor fast path must agree exactly with ``np.add.at`` and with
the existing list-index path, including when source and destination differ
in dtype.  Casting a float64 source to float32 before adding loses a
representable cancellation residual; casting it to int64 truncates too soon.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.common.tensors.numpy_backend import NumPyTensorOperations as Tensor


_DTYPES = [
    pytest.param(np.int64, -1, np.float64, 0.8, id="int64-float64"),
    pytest.param(np.float32, 16777216, np.float64, -16777215.5,
                 id="float32-float64"),
    pytest.param(np.float32, -16777216, np.int64, 16777217,
                 id="float32-int64"),
    pytest.param(np.int64, -1, np.int64, 2, id="int64-control"),
    pytest.param(np.float32, 10, np.float32, 0.5, id="float32-control"),
]
_AXES = [
    pytest.param((3,), 0, id="1d"),
    pytest.param((3, 2), 0, id="2d-axis0"),
    pytest.param((2, 3), 1, id="2d-axis1"),
    pytest.param((3, 2, 2), 0, id="3d-axis0"),
    pytest.param((2, 3, 2), 1, id="3d-axis1"),
    pytest.param((2, 2, 3), 2, id="3d-axis2"),
    pytest.param((2, 2, 3), -1, id="3d-negative-axis"),
]
_POSITIONS = [
    pytest.param([2, 0], id="unique"),
    pytest.param([2, 0, 2], id="repeated"),
]


def _assert_scatter_matches_numpy(initial, positions, values, dim):
    index = np.asarray(positions, dtype=np.int64)
    reference = initial.copy()
    indexer = [slice(None)] * initial.ndim
    indexer[dim] = index
    np.add.at(reference, tuple(indexer), values)

    # Exercise both routes through the public API, with unchanged operands.
    x = Tensor.tensor(initial.copy())
    src = Tensor.tensor(values.copy())
    tensor_index = Tensor.tensor(index.copy())
    for supplied_index in (positions, tensor_index):
        result = x.scatter(supplied_index, src, dim=dim)
        for actual, expected in ((result.data, reference), (x.data, initial),
                                 (src.data, values), (tensor_index.data, index)):
            assert actual.dtype == expected.dtype
            assert actual.shape == expected.shape
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("out_dtype,initial_value,src_dtype,source_value", _DTYPES)
@pytest.mark.parametrize("shape,dim", _AXES)
@pytest.mark.parametrize("positions", _POSITIONS)
def test_scatter_preserves_source_precision(
    out_dtype, initial_value, src_dtype, source_value, shape, dim, positions,
):
    initial = np.full(shape, initial_value, dtype=out_dtype)
    source_shape = list(shape)
    source_shape[dim] = len(positions)
    values = np.full(source_shape, source_value, dtype=src_dtype)
    _assert_scatter_matches_numpy(initial, positions, values, dim)


@pytest.mark.parametrize("out_dtype,initial_value,src_dtype,source_value", _DTYPES[:2])
@pytest.mark.parametrize("shape,dim", _AXES)
def test_empty_scatter_preserves_destination(
    out_dtype, initial_value, src_dtype, source_value, shape, dim,
):
    initial = np.full(shape, initial_value, dtype=out_dtype)
    source_shape = list(shape)
    source_shape[dim] = 0
    values = np.empty(source_shape, dtype=src_dtype)
    _assert_scatter_matches_numpy(initial, [], values, dim)


@pytest.mark.parametrize("out_dtype,initial_value,src_dtype,source_value", _DTYPES[:2])
@pytest.mark.parametrize("positions", _POSITIONS)
def test_scalar_broadcast_scatter_preserves_source_precision(
    out_dtype, initial_value, src_dtype, source_value, positions,
):
    # A scalar uses the generic path even with an integer tensor index.
    initial = np.full((2, 3), initial_value, dtype=out_dtype)
    values = np.asarray(source_value, dtype=src_dtype)
    _assert_scatter_matches_numpy(initial, positions, values, 1)


@pytest.mark.parametrize("out_dtype,initial_value,src_dtype,source_value", _DTYPES[:2])
@pytest.mark.parametrize("positions", _POSITIONS)
@pytest.mark.parametrize("shape,dim", [((3, 2), 0), ((2, 3), 1), ((2, 3, 2), 1)])
def test_singleton_broadcast_scatter_preserves_source_precision(
    out_dtype, initial_value, src_dtype, source_value, positions, shape, dim,
):
    # The native path broadcasts singleton dimensions outside the scatter axis.
    initial = np.full(shape, initial_value, dtype=out_dtype)
    source_shape = [1] * len(shape)
    source_shape[dim] = len(positions)
    values = np.full(source_shape, source_value, dtype=src_dtype)
    _assert_scatter_matches_numpy(initial, positions, values, dim)
