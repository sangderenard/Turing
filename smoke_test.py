from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.numpy_backend import NumPyTensorOperations
try:  # optional torch backend
    from src.common.tensors.torch_backend import PyTorchTensorOperations
except Exception:  # pragma: no cover - torch is optional
    PyTorchTensorOperations = None
import numpy as np


def main() -> None:
    # 1) Pure list → target backends
    L = [[0, 1, 1], [1, 0, 1]]
    x_list = AbstractTensor.get_tensor(L)
    x_np = x_list.to_backend(AbstractTensor.get_tensor(cls=NumPyTensorOperations))
    try:
        if PyTorchTensorOperations is not None:
            x_th = x_list.to_backend(
                AbstractTensor.get_tensor(cls=PyTorchTensorOperations)
            )
        else:
            x_th = None
    except Exception:
        x_th = None

    # 2) Native arrays to backends (ensure_tensor fast-paths)
    arr = np.array(L)
    np_ops = AbstractTensor.get_tensor(cls=NumPyTensorOperations)
    wrapped_np = np_ops.ensure_tensor(arr)
    try:
        if PyTorchTensorOperations is not None:
            torch_ops = AbstractTensor.get_tensor(cls=PyTorchTensorOperations)
            wrapped_np_to_torch = wrapped_np.to_backend(torch_ops)
    except Exception:
        pass

    # 3) The ascii path uses interpolate; exercise it on a tiny mask
    bm = [[0, 1], [1, 1]]
    t = AbstractTensor.get_tensor(bm)
    resized = AbstractTensor.F.interpolate(t, size=(8, 8))

    print("Smoke test completed successfully!")


if __name__ == "__main__":
    main()

