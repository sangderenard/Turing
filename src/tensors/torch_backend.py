"""PyTorch implementation of :class:`AbstractTensor`."""

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


try:
    from typing import Any, Tuple, List, Optional, Union
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None  # type: ignore
    F = None  # type: ignore
except Exception:
    import sys
    print("PyTorch backend failed to import")
    sys.exit(1)

from .abstraction import AbstractTensor

class PyTorchTensorOperations(AbstractTensor):
    def __init__(self, default_device = "cpu", track_time: bool = False):
        super().__init__(track_time=track_time)
        try:
            import torch
        except ImportError:
            raise RuntimeError("PyTorch is required for this backend")
        self.default_device = torch.device(default_device)

    def _apply_operator__(self, op, left, right):
        """Delegate arithmetic ops to PyTorch tensors. Always unwrap to raw tensors."""
        # torch is imported in __init__ and assigned to self._torch
        a = left._AbstractTensor__unwrap() if isinstance(left, AbstractTensor) else left
        b = right._AbstractTensor__unwrap() if isinstance(right, AbstractTensor) else right
        
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
            return torch.floor_divide(a, b)
        if op == "rfloordiv":
            return torch.floor_divide(b, a)
        if op in ("mod", "imod"):
            return a % b
        if op == "rmod":
            return b % a
        if op in ("pow", "ipow"):
            return a ** b
        if op == "rpow":
            return b ** a
        if op in ("matmul", "imatmul"):
            return a @ b
        if op == "rmatmul":
            return b @ a
        raise NotImplementedError(f"Operator {op} not implemented for PyTorch backend.")

    def full_(self, size, fill_value, dtype, device):
        return torch.full(size, fill_value, dtype=dtype, device=device or self.default_device)

    def zeros_(self, size, dtype, device):
        return torch.zeros(size, dtype=dtype, device=device or self.default_device)

    def clone_(self):
        return self.data.clone()

    def to_device_(self, device):
        return self.data.to(device or self.default_device)

    def get_device_(self):
        return self.data.device

    def get_dtype_(self):
        return self.data.dtype

    def item_(self):
        return self.data.item()

    def max_(self):
        return torch.max(self.data)

    def long_cast_(self):
        return self.data.long()

    def float_(self):
        return self.data.float()

    def double_(self):
        return self.data.double()

    def int_(self):
        return self.data.int()

    def long_(self):
        return self.data.long()

    def bool_(self):
        return self.data.bool()

    def not_equal_(self, other):
        from .abstraction import AbstractTensor
        return self.data != (other.data if isinstance(other, AbstractTensor) else other)

    def arange_(self, start, end=None, step=1, device=None, dtype=None):
        if end is None:
            return torch.arange(start, device=device or self.default_device, dtype=dtype)
        return torch.arange(start, end, step, device=device or self.default_device, dtype=dtype)

    def select_by_indices_(self, indices_dim0, indices_dim1):
        return self.data[indices_dim0, indices_dim1]

    def log_softmax_(self, dim):
        return F.log_softmax(self.data, dim=dim)

    def pad_(self, pad, value=0.0):
        return F.pad(self.data, pad, value=value)

    def cat_(self, tensors, dim=0):
        from .abstraction import AbstractTensor
        # torch is imported in __init__ and assigned to self._torch
        tensors = [t.data if isinstance(t, AbstractTensor) else t for t in tensors]
        return torch.cat(tensors, dim=dim)

    def topk_(self, k, dim):
        return torch.topk(self.data, k=k, dim=dim)

    def stack_(self, tensors, dim=0):
        from .abstraction import AbstractTensor
        # torch is imported in __init__ and assigned to self._torch
        tensors = [t.data if isinstance(t, AbstractTensor) else t for t in tensors]
        return torch.stack(tensors, dim=dim)

    def repeat_interleave_(self, repeats=1, dim=None):
        if dim is None:
            dim = 0
        return self.data.repeat_interleave(repeats, dim=dim)
        

    def view_flat_(self):
        return self.data.view(-1)

    def assign_at_indices_(self, indices_dim0, indices_dim1, values_to_assign):
        self.data[indices_dim0, indices_dim1] = values_to_assign
        return self.data

    def increment_at_indices_(self, mask):
        self.data[mask] += 1
        return self.data

    def clamp_(self, min_val=None, max_val=None):
        return torch.clamp(self.data, min=min_val, max=max_val)

    def numel_(self):
        return self.data.numel()

    def mean_(self, dim=None):
        return torch.mean(self.data, dim=dim)

    def pow_(self, exponent: float):
        return torch.pow(self.data, exponent)

    def sqrt_(self):
        return torch.sqrt(self.data)

    def tensor_from_list_(self, data, dtype, device):
        if not isinstance(data, (list, tuple)):
            # If not a list/tuple, try to convert to list (e.g., numpy array)
            try:
                data = data.tolist()
                auto_converted = True
            except Exception:
                auto_converted = False
        else:
            auto_converted = False
        if auto_converted:
            print("[TensorBackend:torch] Auto-converted input to list for tensor_from_list_()")
        return torch.tensor(data, dtype=dtype, device=device or self.default_device)

    def boolean_mask_select_(self, mask):
        return self.data[mask]

    def tolist_(self):
        return self.data.tolist()

    def less_(self, value):
        return self.data < value

    def index_select_(self, dim, indices):
        return torch.index_select(self.data, dim, indices)

    def argmin_(self, dim=None):
        return torch.argmin(self.data) if dim is None else torch.argmin(self.data, dim=dim)

    def get_shape(self):
        return tuple(self.data.shape)

    def get_ndims(self):
        return self.data.dim()

    def interpolate_(self, size):
        t = self.data
        if isinstance(size, int):
            size = (size,)
        if len(size) != t.dim():
            raise ValueError("size must match tensor dimensions")
        def interp_dim(t, new_len, axis):
            old_len = t.shape[axis]
            if old_len == new_len:
                return t
            pos = torch.linspace(0, old_len - 1, steps=new_len, device=t.device, dtype=torch.float32)
            left = pos.floor().long()
            right = torch.clamp(left + 1, max=old_len - 1)
            weight = (pos - left.float()).view([-1 if i == axis else 1 for i in range(t.dim())])
            left_vals = torch.gather(t, axis, left.view([-1 if i == axis else 1 for i in range(t.dim())]).expand([new_len if i == axis else s for i, s in enumerate(t.shape)]))
            right_vals = torch.gather(t, axis, right.view([-1 if i == axis else 1 for i in range(t.dim())]).expand([new_len if i == axis else s for i, s in enumerate(t.shape)]))
            return left_vals * (1 - weight) + right_vals * weight
        result = t
        for d in range(t.dim()):
            result = interp_dim(result, size[d], d)
        return result

    def save_(self, filepath: str) -> None:
        torch.save(self.data, filepath)

    def load_(self, filepath: str, dtype, device):
        t = torch.load(filepath, map_location=device or self.default_device)
        if dtype is not None:
            t = t.to(dtype)
        return t

    @property
    def long_dtype_(self):
        return torch.long

    @property
    def bool_dtype_(self):
        return torch.bool

    @property
    def float_dtype_(self):
        return torch.float32

    @property
    def tensor_type_(self) -> type:
        return torch.Tensor

    @staticmethod
    def from_numpy(source_ops, tensor, target_ops):
        if tensor is None:
            raise ValueError("from_numpy called with tensor=None")
        # torch is imported in __init__ and assigned to self._torch
        arr = tensor.data if hasattr(tensor, "data") else tensor
        result = type(target_ops)(default_device=target_ops.default_device)
        result.data = torch.from_numpy(arr).to(target_ops.default_device)
        return result

    @staticmethod
    def from_torch(source_ops, tensor, target_ops):
        # Already a torch tensor, just move to correct device if needed
        t = tensor.data if hasattr(tensor, "data") else tensor
        if isinstance(source_ops, PyTorchTensorOperations):
            t = source_ops.data
        result = type(target_ops)(default_device=target_ops.default_device)
        result.data = t.to(target_ops.default_device)
        return result

    @staticmethod
    def from_pure(source_ops, tensor, target_ops):
        # torch is imported in __init__ and assigned to self._torch
        data = tensor.data if hasattr(tensor, "data") else tensor
        result = type(target_ops)(default_device=target_ops.default_device)
        result.data = torch.tensor(data, device=target_ops.default_device)
        return result

    @staticmethod
    def from_jax(source_ops, tensor, target_ops):
        # torch is imported in __init__ and assigned to self._torch
        import numpy as np
        arr = tensor.data if hasattr(tensor, "data") else tensor
        np_array = np.array(arr)
        result = type(target_ops)(default_device=target_ops.default_device)
        result.data = torch.from_numpy(np_array).to(target_ops.default_device)
        return result

    def to_dtype_(self, dtype: str = "float"):
        if dtype in ("float", "float32", "f32"):
            return self.data.float()
        elif dtype in ("float64", "double", "f64"):
            return self.data.double()
        elif dtype in ("int", "int32", "i32"):
            return self.data.int()
        elif dtype in ("int64", "long", "i64"):
            return self.data.long()
        elif dtype in ("uint8", "byte"):
            return self.data.byte()
        elif dtype in ("bool",):
            return self.data.bool()
        else:
            return self.data.float()

    def repeat_(self, repeats = None, dim: int = 0):
        t = self.data
        repeated = t.repeat(*repeats) if isinstance(repeats, (list, tuple)) else t.repeat(repeats)
        result = type(self)(default_device=self.default_device)
        result.data = repeated
        return result

    @classmethod
    def tensor_from_list(cls, data, dtype=None, device=None):
        inst = cls(track_time=False)
        inst.data = inst.tensor_from_list_(data, dtype, device)
        return inst

from .abstraction import register_backend
register_backend("torch", PyTorchTensorOperations)

