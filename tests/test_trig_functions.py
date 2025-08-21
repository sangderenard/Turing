import importlib.util
import numpy as np
import pytest

from src.common.tensors.pure_backend import PurePythonTensorOperations

try:
    from src.common.tensors.torch_backend import PyTorchTensorOperations
except Exception:
    PyTorchTensorOperations = None

try:
    from src.common.tensors.numpy_backend import NumPyTensorOperations
except Exception:
    NumPyTensorOperations = None

try:
    from src.common.tensors.jax_backend import JAXTensorOperations
except Exception:
    JAXTensorOperations = None


BACKENDS = [("PurePython", PurePythonTensorOperations)]

torch_spec = importlib.util.find_spec("torch")
if PyTorchTensorOperations is not None and torch_spec is not None:
    BACKENDS.append(("PyTorch", PyTorchTensorOperations))

if NumPyTensorOperations is not None:
    BACKENDS.append(("NumPy", NumPyTensorOperations))

jax_spec = importlib.util.find_spec("jax")
if JAXTensorOperations is not None and jax_spec is not None:
    BACKENDS.append(("JAX", JAXTensorOperations))


@pytest.mark.parametrize("backend_name,BackendCls", BACKENDS)
def test_trigonometric_suite(backend_name, BackendCls):
    vals = [0.5, 0.8]
    t = BackendCls.tensor_from_list(vals, dtype=None, device=None)
    arr = np.array(vals)

    np.testing.assert_allclose(t.sin().tolist(), np.sin(arr))
    np.testing.assert_allclose(t.cos().tolist(), np.cos(arr))
    np.testing.assert_allclose(t.tan().tolist(), np.tan(arr))
    np.testing.assert_allclose(t.asin().tolist(), np.arcsin(arr))
    np.testing.assert_allclose(t.acos().tolist(), np.arccos(arr))
    np.testing.assert_allclose(t.atan().tolist(), np.arctan(arr))

    np.testing.assert_allclose(t.sinh().tolist(), np.sinh(arr))
    np.testing.assert_allclose(t.cosh().tolist(), np.cosh(arr))
    np.testing.assert_allclose(t.tanh().tolist(), np.tanh(arr))
    np.testing.assert_allclose(t.asinh().tolist(), np.arcsinh(arr))
    np.testing.assert_allclose(t.atanh().tolist(), np.arctanh(arr))

    acosh_vals = [1.0, 2.0]
    t_acosh = BackendCls.tensor_from_list(acosh_vals, dtype=None, device=None)
    np.testing.assert_allclose(t_acosh.acosh().tolist(), np.arccosh(np.array(acosh_vals)))

    np.testing.assert_allclose(t.sec().tolist(), 1 / np.cos(arr))
    np.testing.assert_allclose(t.csc().tolist(), 1 / np.sin(arr))
    np.testing.assert_allclose(t.cot().tolist(), np.cos(arr) / np.sin(arr))
    np.testing.assert_allclose(t.sech().tolist(), 1 / np.cosh(arr))
    np.testing.assert_allclose(t.csch().tolist(), 1 / np.sinh(arr))
    np.testing.assert_allclose(t.coth().tolist(), np.cosh(arr) / np.sinh(arr))
    np.testing.assert_allclose(t.sinc().tolist(), np.sin(arr) / arr)
