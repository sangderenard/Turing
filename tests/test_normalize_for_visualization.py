import numpy as np
import warnings
import pytest

from src.common.tensors.abstract_convolution.riemann_convolutional_demo import (
    normalize_for_visualization,
)


def test_normalize_for_visualization_empty_array():
    arr = np.array([])
    with warnings.catch_warnings(record=True) as w:
        with pytest.raises(ValueError):
            normalize_for_visualization(arr)
    assert w == []
