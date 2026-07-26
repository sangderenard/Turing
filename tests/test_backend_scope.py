import numpy as np
import pytest

from src.common.tensors import AbstractTensor


def test_explicit_numpy_backend_scope_restores_previous_default():
    previous_backend = AbstractTensor._preferred_backend
    previous_device = AbstractTensor._preferred_device
    with AbstractTensor.use_backend("numpy", "cpu"):
        value = AbstractTensor.tensor(np.asarray((1.0, 2.0)))
        assert type(value).__name__ == "NumPyTensorOperations"
    assert AbstractTensor._preferred_backend == previous_backend
    assert AbstractTensor._preferred_device == previous_device


def test_torch_backend_scope_preserves_device_and_scalar_reductions():
    pytest.importorskip("torch")
    with AbstractTensor.use_backend("torch", "cpu"):
        value = AbstractTensor.get_tensor(
            np.asarray(((1.0, 2.0), (3.0, 4.0)))
        )
        assert type(value.data).__module__ == "torch"
        assert str(value.get_device()) == "cpu"
        assert value.mean().item() == pytest.approx(2.5)
        assert value.max().item() == pytest.approx(4.0)
