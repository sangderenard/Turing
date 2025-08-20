"""NumPy implementation of :class:`AbstractTensor`."""

# TENSOR BACKEND IMPLEMENTATION GUIDELINES:
# ----------------------------------------
# 1. OPERATOR IMPLEMENTATION:
#    - DO NOT implement magic methods (__add__, __mul__, etc.)
#    - These are handled by AbstractTensor
#    - Only implement the single designated operator method from the abstract class
#
# 2. TEST COMPLIANCE:
#    - DO NOT create dummy/mock classes to pass tests
#    - DO NOT implement functions just to satisfy test requirements
#    - Either implement full functionality or leave as documented stub
#    - Failed tests are preferable to false implementations
#
# 3. BACKEND RESPONSIBILITIES:
#    - Implement only the core tensor operations defined in AbstractTensor
#    - All operator routing happens through the abstract class
#    - Let test failures expose missing functionality naturally
#
# 4. DEPENDENCIES:
#    - Import only the strictly required packages
#    - Handle import failures gracefully for optional backends
#    - Do not add dummy fallbacks for missing dependencies
#
# Remember: Magic methods and operator overloading are EXCLUSIVELY handled by
# AbstractTensor. Backend implementations provide only the raw
# tensor operations.

from typing import Any, Tuple, List, Optional


from .abstraction import AbstractTensor, register_backend



try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    np = None  # type: ignore

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
except Exception:
    import sys
    print("NumPy backend failed to import")
    sys.exit(1)

class NumPyTensorOperations(AbstractTensor):
    def max_(self, dim=None, keepdim=False):
        return np.max(self.data, axis=dim, keepdims=keepdim)

    def argmax_(self, dim=None, keepdim=False):
        arr = self.data
        if dim is None:
            return int(np.argmax(arr))
        idx = np.argmax(arr, axis=dim)
        if keepdim:
            return np.expand_dims(idx, axis=dim)
        return idx
    def where_(self, x, y):
        import numpy as np
        x = x.data if isinstance(x, AbstractTensor) else x
        y = y.data if isinstance(y, AbstractTensor) else y
        return np.where(self.data, x, y)

    def maximum_(self, other):
        import numpy as np
        other = other.data if isinstance(other, AbstractTensor) else other
        return np.maximum(self.data, other)

    def minimum_(self, other):
        import numpy as np
        other = other.data if isinstance(other, AbstractTensor) else other
        return np.minimum(self.data, other)

    def clamp_(self, min_val=None, max_val=None):
        import numpy as np
        return np.clip(self.data, a_min=min_val, a_max=max_val)

    def clamp_min_(self, min_val):
        import numpy as np
        return np.maximum(self.data, min_val)

    def clamp_max_(self, max_val):
        import numpy as np
        return np.minimum(self.data, max_val)

    def greater_(self, value):
        value = value.data if isinstance(value, AbstractTensor) else value
        return self.data > value

    def greater_equal_(self, value):
        value = value.data if isinstance(value, AbstractTensor) else value
        return self.data >= value

    def less_equal_(self, value):
        value = value.data if isinstance(value, AbstractTensor) else value
        return self.data <= value

    def equal_(self, value):
        value = value.data if isinstance(value, AbstractTensor) else value
        return self.data == value

    def logical_not_(self):
        import numpy as np
        return np.logical_not(self.data)

    def sqrt_(self):
        import numpy as np
        return np.sqrt(self.data)

    def exp_(self):
        import numpy as np
        return np.exp(self.data)

    def log_(self):
        import numpy as np
        return np.log(self.data)

    def neg_(self):
        return -self.data

    def abs_(self):
        import numpy as np
        return np.abs(self.data)

    def invert_(self):
        import numpy as np
        return np.invert(self.data)

    def round_(self, n=None):
        import numpy as np
        return np.round(self.data, n)

    def trunc_(self):
        import numpy as np
        return np.trunc(self.data)

    def floor_(self):
        import numpy as np
        return np.floor(self.data)

    def ceil_(self):
        import numpy as np
        return np.ceil(self.data)

    def softmax_(self, dim):
        import numpy as np
        x = self.data
        x_max = np.max(x, axis=dim, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x, axis=dim, keepdims=True)

    def log_softmax_(self, dim):
        import numpy as np
        x = self.data
        x_max = np.max(x, axis=dim, keepdims=True)
        e_x = np.exp(x - x_max)
        softmax = e_x / np.sum(e_x, axis=dim, keepdims=True)
        return np.log(softmax)

    def transpose_(self, dim0, dim1):
        import numpy as np
        axes = list(range(self.data.ndim))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        return np.transpose(self.data, axes)
    def __init__(self, track_time: bool = False):
        super().__init__(track_time=track_time)

    def _apply_operator__(self, op: str, left: Any, right: Any):
        """Apply arithmetic operators on NumPy arrays."""
        from .abstraction import AbstractTensor
        a = left._AbstractTensor__unwrap() if isinstance(left, AbstractTensor) else left
        b = right._AbstractTensor__unwrap() if isinstance(right, AbstractTensor) else right
        if op == "neg":
            return -a
        if op == "abs":
            return np.abs(a)
        if op == "invert":
            return np.invert(a)
        if op in ("add", "iadd"):
            return a + b
        if op == "radd":
            return b + a
        if op in ("sub", "isub"):
            return a - b
        if op == "rsub":
            return b - a
        if op in ("mul", "imul"):
            return a * b
        if op == "rmul":
            return b * a
        if op in ("truediv", "itruediv"):
            return a / b
        if op == "rtruediv":
            return b / a
        if op in ("floordiv", "ifloordiv"):
            return np.floor_divide(a, b)
        if op == "rfloordiv":
            return np.floor_divide(b, a)
        if op in ("mod", "imod"):
            return np.mod(a, b)
        if op == "rmod":
            return np.mod(b, a)
        if op in ("pow", "ipow"):
            return np.power(a, b)
        if op == "rpow":
            return np.power(b, a)
        if op in ("matmul", "imatmul"):
            return a @ b
        if op == "rmatmul":
            return b @ a
        raise NotImplementedError(f"Operator {op} not implemented for NumPy backend.")

    def _torch_dtype_to_numpy(self, dtype):
        if torch is None:
            return dtype
        if dtype == torch.float32:
            return np.float32
        if dtype == torch.int64:
            return np.int64
        if dtype == torch.float64:
            return np.float64
        if dtype == torch.int32:
            return np.int32
        if dtype == torch.bool:
            return np.bool_
        return None

    def _numpy_dtype_to_torch(self, dtype):
        if torch is None:
            return dtype
        if dtype == np.float32:
            return torch.float32
        if dtype == np.float64:
            return torch.float64
        if dtype == np.int64:
            return torch.int64
        if dtype == np.int32:
            return torch.int32
        if dtype == np.bool_:
            return torch.bool
        return None

    def full_(self, size, fill_value, dtype, device):
        return np.full(size, fill_value, dtype=self._torch_dtype_to_numpy(dtype))

    def zeros_(self, size, dtype, device):
        return np.zeros(size, dtype=self._torch_dtype_to_numpy(dtype))

    def clone_(self, tensor=None):
        tensor = self._AbstractTensor__unwrap(tensor if tensor is not None else self.data)
        return np.array(tensor, copy=True)

    def to_device_(self, tensor, device):
        return self._AbstractTensor__unwrap(tensor)

    def get_device_(self, tensor):
        return 'cpu'

    def get_dtype_(self, tensor):
        tensor = self._AbstractTensor__unwrap(tensor)
        if isinstance(tensor, np.ndarray):
            return self._numpy_dtype_to_torch(tensor.dtype)
        return tensor.dtype

    def item_(self, tensor):
        return self._AbstractTensor__unwrap(tensor).item()

    def max_(self, tensor):
        return np.max(self._AbstractTensor__unwrap(tensor))

    def long_cast_(self, tensor):
        return self._AbstractTensor__unwrap(tensor).astype(np.int64)

    def float_(self, tensor):
        return self.to_dtype_(tensor, "float")

    def double_(self, tensor):
        return self.to_dtype_(tensor, "double")

    def int_(self, tensor):
        return self.to_dtype_(tensor, "int")

    def long_(self, tensor):
        return self.to_dtype_(tensor, "long")

    def bool_(self, tensor):
        return self.to_dtype_(tensor, "bool")

    def not_equal_(self, tensor1, tensor2):
        return self._AbstractTensor__unwrap(tensor1) != self._AbstractTensor__unwrap(tensor2)

    def arange_(self, start, end=None, step=1, device=None, dtype=None):
        np_dtype = self._torch_dtype_to_numpy(dtype) if dtype is not None else None
        if end is None:
            return np.arange(start, dtype=np_dtype)
        return np.arange(start, end, step, dtype=np_dtype)

    def select_by_indices_(self, tensor, indices_dim0, indices_dim1):
        tensor = self._AbstractTensor__unwrap(tensor)
        i0 = self._AbstractTensor__unwrap(indices_dim0)
        i1 = self._AbstractTensor__unwrap(indices_dim1)
        return tensor[i0, i1]

    def log_softmax_(self, tensor, dim):
        tensor = self._AbstractTensor__unwrap(tensor)
        e_x = np.exp(tensor - np.max(tensor, axis=dim, keepdims=True))
        softmax = e_x / np.sum(e_x, axis=dim, keepdims=True)
        return np.log(softmax)

    def topk_(self, tensor, k, dim):
        tensor = self._AbstractTensor__unwrap(tensor)
        if dim < 0:
            dim = tensor.ndim + dim
        sorted_indices = np.argsort(tensor, axis=dim)
        idx_slice = [slice(None)] * tensor.ndim
        idx_slice[dim] = slice(tensor.shape[dim] - k, tensor.shape[dim])
        top_k_indices_ascending = sorted_indices[tuple(idx_slice)]
        top_k_indices = np.flip(top_k_indices_ascending, axis=dim)
        values = np.take_along_axis(tensor, top_k_indices, axis=dim)
        return values, top_k_indices

    def stack_(self, tensors, dim=0):
        tensors = [self._AbstractTensor__unwrap(t) for t in tensors]
        return np.stack(tensors, axis=dim)

    def pad_(self, pad, value=0.0):
        if len(pad) % 2 != 0:
            raise ValueError("Padding length must be even.")
        num_dims_to_pad = len(pad) // 2
        tensor = self.data
        if num_dims_to_pad > tensor.ndim:
            raise ValueError("Padding tuple length implies padding more dimensions than tensor has.")
        np_pad_width = []
        for _ in range(tensor.ndim - num_dims_to_pad):
            np_pad_width.append((0, 0))
        for i in range(num_dims_to_pad):
            left = pad[-2 * (i + 1)]
            right = pad[-2 * (i + 1) + 1]
            np_pad_width.append((left, right))
        return np.pad(tensor, pad_width=np_pad_width, constant_values=value)

    def cat_(self, tensors, dim=0):
        tensors = [self._AbstractTensor__unwrap(t) for t in tensors]
        return np.concatenate(tensors, axis=dim)

    def expand_(self, shape):
        return np.broadcast_to(self.data, shape)
    def repeat_interleave_(self, repeats=1, dim=None):
        if dim is None:
            dim = 0
        return np.repeat(self.data, repeats, axis=dim)

    def repeat_(self, repeats=None, dim: int = 0):
        """Repeat tensor along ``dim`` ``repeats`` times (stub)."""
        raise NotImplementedError("repeat not implemented for NumPy backend")

    def view_flat_(self):
        return self.data.reshape(-1)

    def assign_at_indices_(self, indices_dim0, indices_dim1, values_to_assign):
        t = self.data
        v = self._AbstractTensor__unwrap(values_to_assign)
        i0 = self._AbstractTensor__unwrap(indices_dim0)
        i1 = self._AbstractTensor__unwrap(indices_dim1)
        t[i0, i1] = v
        return t

    def increment_at_indices_(self, mask):
        t = self.data
        m = self._AbstractTensor__unwrap(mask)
        t[m] += 1
        return t

    def clamp_(self, min_val=None, max_val=None):
        return np.clip(self.data, a_min=min_val, a_max=max_val)

    def shape_(self):
        return tuple(self.data.shape)

    def numel_(self):
        return self.data.size

    def mean_(self, dim=None, keepdim=False):
        return np.mean(self.data, axis=dim, keepdims=keepdim)

    def sum_(self, dim=None, keepdim=False):
        return np.sum(self.data, axis=dim, keepdims=keepdim)

    def min_(self, dim=None, keepdim=False):
        return np.min(self.data, axis=dim, keepdims=keepdim)

    def pow_(self, exponent: float):
        return np.power(self.data, exponent)

    def sqrt_(self):
        return np.sqrt(self.data)

    def tensor_from_list_(self, data, dtype, device):
        if not isinstance(data, (list, tuple)):
            try:
                data = data.tolist()
                auto_converted = True
            except Exception:
                auto_converted = False
        else:
            auto_converted = False
        if auto_converted:
            print("[TensorBackend:numpy] Auto-converted input to list for tensor_from_list_()")
        return np.array(data, dtype=self._torch_dtype_to_numpy(dtype))

    def boolean_mask_select_(self, mask):
        tensor = self.data
        m = self._AbstractTensor__unwrap(mask)
        return tensor[m]

    def tolist_(self):
        return self._AbstractTensor__unwrap(self.data).tolist()

    def less_(self, value):
        return self.data < value

    def index_select_(self, dim, indices):
        tensor = self.data
        idx = self._AbstractTensor__unwrap(indices)
        return np.take(tensor, idx, axis=dim)

    def argmin_(self, dim=None, keepdim=False):
        return np.argmin(self.data, axis=dim, keepdims=keepdim)

    def interpolate_(self, size):
        arr = np.array(self.data)
        if len(size) != arr.ndim:
            raise ValueError("size must match tensor dimensions")
        def interp_axis(a, new_len, axis):
            old_len = a.shape[axis]
            if old_len == new_len:
                return a
            old_idx = np.arange(old_len)
            new_idx = np.linspace(0, old_len - 1, new_len)
            a = np.swapaxes(a, 0, axis)
            out_shape = (new_len,) + a.shape[1:]
            out = np.empty(out_shape, dtype=a.dtype)
            for idx in np.ndindex(a.shape[1:]):
                out[(slice(None),) + idx] = np.interp(new_idx, old_idx, a[(slice(None),) + idx])
            return np.swapaxes(out, 0, axis)
        result = arr
        for d in range(arr.ndim):
            result = interp_axis(result, size[d], d)
        return result

    def save_(self, filepath: str) -> None:
        np.save(filepath, self.data)

    def load_(self, filepath: str, dtype, device):
        arr = np.load(f"{filepath}.npy") if not filepath.endswith('.npy') else np.load(filepath)
        if dtype is not None:
            arr = arr.astype(self._torch_dtype_to_numpy(dtype))
        return arr

    def to_dtype_(self, dtype: str = "float"):
        import numpy as np
        tensor = self.data
        if dtype in ("float", "float32", "f32"):
            return tensor.astype(np.float32)
        elif dtype in ("float64", "double", "f64"):
            return tensor.astype(np.float64)
        elif dtype in ("int", "int32", "i32"):
            return tensor.astype(np.int32)
        elif dtype in ("int64", "long", "i64"):
            return tensor.astype(np.int64)
        elif dtype in ("uint8", "byte"):
            return tensor.astype(np.uint8)
        elif dtype in ("bool",):
            return tensor.astype(np.bool_)
        else:
            # Default to float32
            return tensor.astype(np.float32)

    @property
    def long_dtype_(self):
        return np.int64

    @property
    def bool_dtype_(self):
        return np.bool_

    @property
    def float_dtype_(self):
        return np.float32

    @property
    def tensor_type_(self) -> type:
        return np.ndarray

    @staticmethod
    def from_numpy(source_ops, tensor, target_ops):
        # If already numpy, just return the data (clone if needed)
        if isinstance(source_ops, NumPyTensorOperations):
            arr = source_ops.data
        else:
            import numpy as np
            arr = np.array(tensor.data, copy=False)
        result = type(target_ops)(track_time=target_ops.track_time)
        result.data = arr
        return result

    @staticmethod
    def from_torch(source_ops, tensor, target_ops):
        import numpy as np
        t = tensor.data if hasattr(tensor, "data") else tensor
        arr = t.detach().cpu().numpy()
        result = type(target_ops)(track_time=target_ops.track_time)
        result.data = arr
        return result

    @staticmethod
    def from_pure(source_ops, tensor, target_ops):
        import numpy as np
        data = tensor.data if hasattr(tensor, "data") else tensor
        result = type(target_ops)(track_time=target_ops.track_time)
        result.data = np.array(data)
        return result

    @staticmethod
    def from_jax(source_ops, tensor, target_ops):
        import numpy as np
        data = tensor.data if hasattr(tensor, "data") else tensor
        result = type(target_ops)(track_time=target_ops.track_time)
        result.data = np.array(data)
        return result

    def get_shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    def get_ndims(self) -> int:
        return self.data.ndim

    @classmethod
    def tensor_from_list(cls, data, dtype=None, device=None):
        inst = cls(track_time=False)
        inst.data = inst.tensor_from_list_(data, dtype, device)
        return inst

register_backend("numpy", NumPyTensorOperations)
