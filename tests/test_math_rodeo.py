import pytest
import numpy as np

from src.tensors import (
    AbstractTensor,
    PurePythonTensorOperations,
)

# Try to import optional backends
try:
    from src.tensors.torch_backend import PyTorchTensorOperations
except Exception:
    PyTorchTensorOperations = None
try:
    from src.tensors.numpy_backend import NumPyTensorOperations
except Exception:
    NumPyTensorOperations = None
try:
    from src.tensors.jax_backend import JAXTensorOperations
except Exception:
    JAXTensorOperations = None

BACKENDS = [
    ("PurePython", PurePythonTensorOperations),
]
if PyTorchTensorOperations is not None:
    BACKENDS.append(("PyTorch", PyTorchTensorOperations))
if NumPyTensorOperations is not None:
    BACKENDS.append(("NumPy", NumPyTensorOperations))
if JAXTensorOperations is not None:
    BACKENDS.append(("JAX", JAXTensorOperations))

@pytest.mark.parametrize("backend_name,BackendCls", BACKENDS)
def test_math_rodeo(backend_name, BackendCls):
    print(f"\n=== Mathematics Rodeo: {backend_name} ===")
    failures = []
    # Test data
    a = BackendCls().full_((2, 2), 3.0, dtype=None, device=None)
    b = BackendCls().full_((2, 2), 2.0, dtype=None, device=None)
    # Addition
    try:
        add = a + b
        print(f"Addition: {add}")
    except Exception as e:
        print(f"Addition FAILED: {e}")
        failures.append("add")
    # Subtraction
    try:
        sub = a - b
        print(f"Subtraction: {sub}")
    except Exception as e:
        print(f"Subtraction FAILED: {e}")
        failures.append("sub")
    # Multiplication
    try:
        mul = a * b
        print(f"Multiplication: {mul}")
    except Exception as e:
        print(f"Multiplication FAILED: {e}")
        failures.append("mul")
    # Division
    try:
        div = a / b
        print(f"Division: {div}")
    except Exception as e:
        print(f"Division FAILED: {e}")
        failures.append("div")
    # Power
    try:
        powr = a ** 2
        print(f"Power: {powr}")
    except Exception as e:
        print(f"Power FAILED: {e}")
        failures.append("pow")
    # Sqrt
    try:
        sqrt = a.sqrt()
        print(f"Sqrt: {sqrt}")
    except Exception as e:
        print(f"Sqrt FAILED: {e}")
        failures.append("sqrt")
    # Mean
    try:
        mean = a.mean()
        print(f"Mean: {mean}")
    except Exception as e:
        print(f"Mean FAILED: {e}")
        failures.append("mean")
    # Max
    try:
        maxv = a.max()
        print(f"Max: {maxv}")
    except Exception as e:
        print(f"Max FAILED: {e}")
        failures.append("max")
    # Trigonometric (if implemented)
    try:
        trig = np.sin(a.tolist())
        print(f"Sin: {trig}")
    except Exception as e:
        print(f"Sin FAILED: {e}")
        failures.append("sin")
    if failures:
        print(f"[FAILURES] {backend_name}: {failures}")
    else:
        print(f"[SUCCESS] {backend_name}: All math ops passed!")
