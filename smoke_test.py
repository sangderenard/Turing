from src.common.tensors.abstraction import AbstractTensor, Faculty

# 1) Pure list → target backends
L = [[0,1,1],[1,0,1]]
x_list = AbstractTensor.get_tensor(L)                 # PurePythonListTensor
x_np   = x_list.to_backend(AbstractTensor.get_tensor(faculty=Faculty.NUMPY))
try:  # Optional torch backend
    x_th   = x_list.to_backend(AbstractTensor.get_tensor(faculty=Faculty.TORCH))
except Exception:
    x_th = None

# 2) Native arrays to backends (ensure_tensor fast-paths)
import numpy as np
arr = np.array(L)
np_ops = AbstractTensor.get_tensor(faculty=Faculty.NUMPY)
wrapped_np = np_ops.ensure_tensor(arr)
try:
    torch_ops = AbstractTensor.get_tensor(faculty=Faculty.TORCH)
    wrapped_np_to_torch = wrapped_np.to_backend(torch_ops)
except Exception:
    pass

# 3) The ascii path uses interpolate; exercise it on a tiny mask
bm = [[0,1],[1,1]]
t = AbstractTensor.get_tensor(bm)
resized = AbstractTensor.F.interpolate(t, size=(8,8))  # should pick a real backend and return

print("Smoke test completed successfully!")
