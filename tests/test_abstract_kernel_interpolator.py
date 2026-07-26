import numpy as np

from src.common.tensors import AbstractTensor
from src.common.tensors.youngman.spline import AbstractKernelInterpolator


def test_kernel_interpolator_preserves_abstract_tensor_and_vector_channels():
    controls = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)))
    values = np.column_stack((
        controls,
        controls[:, 0] + controls[:, 1],
        controls[:, 0] - controls[:, 1],
    ))
    model = AbstractKernelInterpolator.fit(
        controls, values, bandwidth=0.1
    )
    result = model(AbstractTensor.tensor(((0.0, 0.0), (1.0, 1.0))))
    assert isinstance(result, AbstractTensor)
    assert result.shape == (2, 4)
    assert np.allclose(result.tolist(), values[[0, 3]], atol=2e-3)
