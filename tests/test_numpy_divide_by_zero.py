import warnings
import numpy as np
import pytest

from src.common.tensors.numpy_backend import NumPyTensorOperations


@pytest.mark.skipif(NumPyTensorOperations is None or np is None, reason="NumPy backend not available")
def test_divide_by_zero_returns_finite_no_warning():
    a = NumPyTensorOperations.tensor([1.0, 2.0])
    b = NumPyTensorOperations.tensor([0.0, 1.0])
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = a / b
    assert np.all(np.isfinite(result.tolist()))
    assert not any(issubclass(w.category, RuntimeWarning) for w in record)
    assert result.tolist() == [0.0, 2.0]
