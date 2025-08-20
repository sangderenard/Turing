"""JAX implementation of :class:`AbstractTensor`."""

from __future__ import annotations

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



try:
    import jax
    import jax.numpy as jnp
    from jax import lax
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    jax = None  # type: ignore
    jnp = None  # type: ignore
    lax = None  # type: ignore
except Exception:
    import sys
    print("JAX backend failed to import")
    sys.exit(1)

from .abstraction import AbstractTensor

class JAXTensorOperations(AbstractTensor):
    def where_(self, x, y):
        import jax.numpy as jnp
        x = x.data if hasattr(x, 'data') else x
        y = y.data if hasattr(y, 'data') else y
        return jnp.where(self.data, x, y)

    def maximum_(self, other):
        import jax.numpy as jnp
        other = other.data if hasattr(other, 'data') else other
        return jnp.maximum(self.data, other)

    def minimum_(self, other):
        import jax.numpy as jnp
        other = other.data if hasattr(other, 'data') else other
        return jnp.minimum(self.data, other)

    def clamp_(self, min_val=None, max_val=None):
        import jax.numpy as jnp
        return jnp.clip(self.data, a_min=min_val, a_max=max_val)

    def clamp_min_(self, min_val):
        import jax.numpy as jnp
        return jnp.maximum(self.data, min_val)

    def clamp_max_(self, max_val):
        import jax.numpy as jnp
        return jnp.minimum(self.data, max_val)

    def greater_(self, value):
        value = value.data if hasattr(value, 'data') else value
        return self.data > value

    def greater_equal_(self, value):
        value = value.data if hasattr(value, 'data') else value
        return self.data >= value

    def less_equal_(self, value):
        value = value.data if hasattr(value, 'data') else value
        return self.data <= value

    def equal_(self, value):
        value = value.data if hasattr(value, 'data') else value
        return self.data == value

    def logical_not_(self):
        import jax.numpy as jnp
        return jnp.logical_not(self.data)

    def sqrt_(self):
        import jax.numpy as jnp
        return jnp.sqrt(self.data)

    def exp_(self):
        import jax.numpy as jnp
        return jnp.exp(self.data)

    def log_(self):
        import jax.numpy as jnp
        return jnp.log(self.data)

    def softmax_(self, dim):
        import jax.numpy as jnp
        x = self.data
        x_max = jnp.max(x, axis=dim, keepdims=True)
        e_x = jnp.exp(x - x_max)
        return e_x / jnp.sum(e_x, axis=dim, keepdims=True)

    def log_softmax_(self, dim):
        import jax.numpy as jnp
        x = self.data
        x_max = jnp.max(x, axis=dim, keepdims=True)
        e_x = jnp.exp(x - x_max)
        softmax = e_x / jnp.sum(e_x, axis=dim, keepdims=True)
        return jnp.log(softmax)

    def transpose_(self, dim0, dim1):
        import jax.numpy as jnp
        axes = list(range(self.data.ndim))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        return jnp.transpose(self.data, axes)
    """Tensor operations powered by `jax.numpy`."""

    def __init__(self, default_device: Optional[Any] = None, track_time: bool = False) -> None:
        super().__init__(track_time=track_time)
        self.default_device = default_device
        self._validate_jax_setup()

    def _validate_jax_setup(self) -> None:
        """Validates JAX installation and available devices."""
        try:
            devices = jax.devices()
            self.available_devices = {str(d): d for d in devices}
            self.has_gpu = any('gpu' in str(d).lower() for d in devices)
            self.has_tpu = any('tpu' in str(d).lower() for d in devices)
        except Exception as e:
            raise RuntimeError(f"JAX initialization failed: {str(e)}")

    def _to_jnp(self, tensor: Any) -> jnp.ndarray:
        """Safely convert input to JAX array."""
        tensor = self._AbstractTensor__unwrap(tensor)
        if isinstance(tensor, jnp.ndarray):
            return tensor
        return jnp.array(tensor)

    def to_device_(self, tensor: Any, device: Any) -> Any:
        """Move tensor to specified device with validation."""
        target_device = device or self.default_device
        if target_device is not None:
            device_str = str(target_device).lower()
            if ('gpu' in device_str and not self.has_gpu) or \
               ('tpu' in device_str and not self.has_tpu):
                print(f"Warning: Requested device {device_str} not available. Using CPU.")
                target_device = jax.devices('cpu')[0]
        
        return jax.device_put(self._to_jnp(tensor), target_device)

    def _apply_operator__(self, op: str, left: Any, right: Any):
        """Apply arithmetic ops using JAX arrays."""
        from .abstraction import AbstractTensor
        a = self._to_jnp(left._AbstractTensor__unwrap() if isinstance(left, AbstractTensor) else left)
        b = self._to_jnp(right._AbstractTensor__unwrap() if isinstance(right, AbstractTensor) else right)
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
            return jnp.floor(a / b)
        if op == "rfloordiv":
            return jnp.floor(b / a)
        if op in ("mod", "imod"):
            return jnp.mod(a, b)
        if op == "rmod":
            return jnp.mod(b, a)
        if op in ("pow", "ipow"):
            return jnp.power(a, b)
        if op == "rpow":
            return jnp.power(b, a)
        if op in ("matmul", "imatmul"):
            return a @ b
        if op == "rmatmul":
            return b @ a
        raise NotImplementedError(f"Operator {op} not implemented for JAX backend.")

    # ------------------------------------------------------------------
    # Creation ops
    def full_(self, size: Tuple[int, ...], fill_value: Any, dtype: Any, device: Any):
        return jax.device_put(jnp.full(size, fill_value, dtype=dtype), device or self.default_device)

    def zeros_(self, size: Tuple[int, ...], dtype: Any, device: Any):
        return jax.device_put(jnp.zeros(size, dtype=dtype), device or self.default_device)

    def clone_(self, tensor: Any) -> Any:
        return jnp.array(self._AbstractTensor__unwrap(tensor), copy=True)

    # ------------------------------------------------------------------
    # Basic info
    def get_device_(self, tensor: Any) -> Any:
        return self._AbstractTensor__unwrap(tensor).device

    def get_dtype_(self, tensor: Any) -> Any:
        return self._AbstractTensor__unwrap(tensor).dtype

    def item_(self, tensor: Any) -> Any:
        return self._AbstractTensor__unwrap(tensor).item()

    def max_(self, tensor: Any) -> Any:
        return jnp.max(self._to_jnp(tensor))

    def long_cast_(self, tensor: Any) -> Any:
        return self._to_jnp(tensor).astype(jnp.int64)

    def float_(self, tensor: Any) -> Any:
        return self.to_dtype_(tensor, "float")

    def double_(self, tensor: Any) -> Any:
        return self.to_dtype_(tensor, "double")

    def int_(self, tensor: Any) -> Any:
        return self.to_dtype_(tensor, "int")

    def long_(self, tensor: Any) -> Any:
        return self.to_dtype_(tensor, "long")

    def bool_(self, tensor: Any) -> Any:
        return self.to_dtype_(tensor, "bool")

    def not_equal_(self, tensor1: Any, tensor2: Any) -> Any:
        return jnp.not_equal(self._to_jnp(tensor1), self._to_jnp(tensor2))

    def arange_(self, start: int, end: Optional[int] = None, step: int = 1, device: Any = None, dtype: Any = None) -> Any:
        arr = jnp.arange(start, end, step, dtype=dtype) if end is not None else jnp.arange(start, dtype=dtype)
        return jax.device_put(arr, device or self.default_device)

    def select_by_indices_(self, tensor: Any, indices_dim0: Any, indices_dim1: Any) -> Any:
        return self._to_jnp(tensor)[indices_dim0, indices_dim1]

    def log_softmax_(self, tensor: Any, dim: int) -> Any:
        from jax.nn import log_softmax
        return log_softmax(self._to_jnp(tensor), axis=dim)

    def pad_(self, tensor: Any, pad: Tuple[int, ...], value: float = 0) -> Any:
        if len(pad) % 2 != 0:
            raise ValueError("Padding length must be even.")
        num_dims_to_pad = len(pad) // 2
        pad_width: List[Tuple[int, int]] = []
        tensor = self._to_jnp(tensor)
        for _ in range(tensor.ndim - num_dims_to_pad):
            pad_width.append((0, 0))
        for i in range(num_dims_to_pad):
            left = pad[-2 * (i + 1)]
            right = pad[-2 * (i + 1) + 1]
            pad_width.append((left, right))
        return jnp.pad(tensor, pad_width=tuple(pad_width), constant_values=value).tolist()

    def cat_(self, tensors: List[Any], dim: int = 0) -> Any:
        tensors = [self._to_jnp(t) for t in tensors]
        return jnp.concatenate(tensors, axis=dim).tolist()

    def topk_(self, tensor: Any, k: int, dim: int) -> Tuple[Any, Any]:
        """Return the top ``k`` values and indices along ``dim``."""
        tensor = self._to_jnp(tensor)
        if dim < 0:
            dim = tensor.ndim + dim
        if dim < 0 or dim >= tensor.ndim:
            raise ValueError("dim out of range")

        if dim == tensor.ndim - 1:
            values, idxs = lax.top_k(tensor, k)
        else:
            moved = jnp.moveaxis(tensor, dim, -1)
            values, idxs = lax.top_k(moved, k)
            values = jnp.moveaxis(values, -1, dim)
            idxs = jnp.moveaxis(idxs, -1, dim)
        return values.tolist(), idxs.tolist()

    def stack_(self, tensors: List[Any], dim: int = 0) -> Any:
        tensors = [self._to_jnp(t) for t in tensors]
        return jnp.stack(tensors, axis=dim).tolist()

    def repeat_interleave_(self, tensor: Any, repeats: int, dim: Optional[int] = None) -> Any:
        return jnp.repeat(self._to_jnp(tensor), repeats, axis=dim).tolist()

    def repeat_(self, repeats: Any = None, dim: int = 0) -> Any:
        """Repeat tensor along ``dim`` ``repeats`` times (stub)."""
        raise NotImplementedError("repeat not implemented for JAX backend")

    def view_flat_(self, tensor: Any) -> Any:
        return jnp.ravel(self._to_jnp(tensor)).tolist()

    def assign_at_indices_(self, tensor_to_modify: Any, indices_dim0: Any, indices_dim1: Any, values_to_assign: Any):
        tensor_to_modify = self._to_jnp(tensor_to_modify)
        updated = tensor_to_modify.at[indices_dim0, indices_dim1].set(self._to_jnp(values_to_assign))
        return updated

    def increment_at_indices_(self, tensor_to_modify: Any, mask: Any):
        tensor_to_modify = self._to_jnp(tensor_to_modify)
        updated = tensor_to_modify.at[mask].add(1)
        return updated

    def clamp_(self, tensor: Any, min_val: Optional[float] = None, max_val: Optional[float] = None) -> Any:
        return jnp.clip(self._to_jnp(tensor), a_min=min_val, a_max=max_val)

    def shape_(self, tensor: Any) -> Tuple[int, ...]:
        return tuple(self._to_jnp(tensor).shape)

    def numel_(self, tensor: Any) -> int:
        return self._to_jnp(tensor).size

    def mean_(self, tensor: Any, dim: Optional[int] = None) -> Any:
        return jnp.mean(self._to_jnp(tensor), axis=dim)

    def pow_(self, tensor: Any, exponent: float) -> Any:
        return jnp.power(self._to_jnp(tensor), exponent)

    def sqrt_(self, tensor: Any) -> Any:
        return jnp.sqrt(self._to_jnp(tensor))

    def tensor_from_list_(self, data: list, dtype: Any, device: Any) -> Any:
        if not isinstance(data, (list, tuple)):
            try:
                data = data.tolist()
                auto_converted = True
            except Exception:
                auto_converted = False
        else:
            auto_converted = False
        if auto_converted:
            print("[TensorBackend:jax] Auto-converted input to list for tensor_from_list_()")
        arr = jnp.array(data, dtype=dtype)
        return jax.device_put(arr, device or self.default_device)

    def boolean_mask_select_(self, tensor: Any, mask: Any) -> Any:
        return self._to_jnp(tensor)[mask]

    def tolist_(self) -> list:
        return list(self._to_jnp(self.data).tolist())

    def less_(self, tensor: Any, value: Any) -> Any:
        return jnp.less(self._to_jnp(tensor), value)

    def index_select_(self, tensor: Any, dim: int, indices: Any) -> Any:
        return jnp.take(self._to_jnp(tensor), indices, axis=dim)

    def argmin_(self, tensor: Any, dim: Optional[int] = None) -> Any:
        return jnp.argmin(self._to_jnp(tensor)) if dim is None else jnp.argmin(self._to_jnp(tensor), axis=dim)

    def interpolate_(self, tensor: Any, size: Tuple[int, ...]) -> Any:
        arr = self._to_jnp(tensor)
        if len(size) != arr.ndim:
            raise ValueError("size must match tensor dimensions")

        def interp_axis(a, new_len, axis):
            old_len = a.shape[axis]
            if old_len == new_len:
                return a
            pos = jnp.linspace(0, old_len - 1, new_len)
            left = jnp.floor(pos).astype(jnp.int32)
            right = jnp.clip(left + 1, 0, old_len - 1)
            weight = pos - left
            left_vals = jnp.take(a, left, axis=axis)
            right_vals = jnp.take(a, right, axis=axis)
            shape = [1] * a.ndim
            shape[axis] = new_len
            weight = weight.reshape(shape)
            return left_vals * (1 - weight) + right_vals * weight

        result = arr
        for d in range(arr.ndim):
            result = interp_axis(result, size[d], d)
        return result


    # --- Persistence helpers ---
    def save_(self, tensor: Any, filepath: str) -> None:
        import numpy as np
        np.save(filepath, np.array(tensor))

    def load_(self, filepath: str, dtype: Any, device: Any) -> Any:
        import numpy as np
        arr = np.load(f"{filepath}.npy") if not filepath.endswith('.npy') else np.load(filepath)
        arr = jnp.array(arr, dtype=dtype) if dtype is not None else jnp.array(arr)
        return jax.device_put(arr, device or self.default_device)

    # --- Dtype helpers ---
    @property
    def long_dtype_(self) -> Any:
        return int

    @property
    def bool_dtype_(self) -> Any:
        return bool

    @property
    def float_dtype_(self) -> Any:
        return float

    @property
    def tensor_type_(self) -> type:
        return jnp.ndarray

    @staticmethod
    def test() -> None:
        """Quick smoke test for the JAX backend."""
        if jax is None:
            print("JAX not available")
            return
        ops = JAXTensorOperations()
        stacked = ops.stack([jnp.array([1, 2]), jnp.array([3, 4])], dim=0)
        assert ops.tolist(stacked) == [[1, 2], [3, 4]]
        values, idxs = ops.topk(jnp.array([1, 3, 2, 4]), k=2, dim=-1)
        assert ops.tolist(values) == [4, 3] and ops.tolist(idxs) == [3, 1]

    @staticmethod
    def from_numpy(source_ops, tensor, target_ops):
        import jax.numpy as jnp
        arr = tensor.data if hasattr(tensor, "data") else tensor
        result = type(target_ops)(track_time=target_ops.track_time)
        result.data = jnp.array(arr)
        return result

    @staticmethod
    def from_torch(source_ops, tensor, target_ops):
        import jax.numpy as jnp
        import numpy as np
        t = tensor.data if hasattr(tensor, "data") else tensor
        np_array = t.detach().cpu().numpy()
        result = type(target_ops)(track_time=target_ops.track_time)
        result.data = jnp.array(np_array)
        return result

    @staticmethod
    def from_pure(source_ops, tensor, target_ops):
        import jax.numpy as jnp
        data = tensor.data if hasattr(tensor, "data") else tensor
        result = type(target_ops)(track_time=target_ops.track_time)
        result.data = jnp.array(data)
        return result

    @staticmethod
    def from_jax(source_ops, tensor, target_ops):
        # Already a jax array, just return the data
        if isinstance(source_ops, JAXTensorOperations):
            data = source_ops.data
        else:
            data = tensor.data if hasattr(tensor, "data") else tensor
        import jax.numpy as jnp
        result = type(target_ops)(track_time=target_ops.track_time)
        result.data = jnp.array(data)
        return result

    def to_dtype_(self, tensor, dtype: str = "float"):
        import jax.numpy as jnp
        if dtype in ("float", "float32", "f32"):
            return jnp.asarray(tensor, dtype=jnp.float32)
        elif dtype in ("float64", "double", "f64"):
            return jnp.asarray(tensor, dtype=jnp.float64)
        elif dtype in ("int", "int32", "i32"):
            return jnp.asarray(tensor, dtype=jnp.int32)
        elif dtype in ("int64", "long", "i64"):
            return jnp.asarray(tensor, dtype=jnp.int64)
        elif dtype in ("uint8", "byte"):
            return jnp.asarray(tensor, dtype=jnp.uint8)
        elif dtype in ("bool",):
            return jnp.asarray(tensor, dtype=jnp.bool_)
        else:
            # Default to float32
            return jnp.asarray(tensor, dtype=jnp.float32)

    def get_shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    def get_ndims(self) -> int:
        return self.data.ndim

    @classmethod
    def tensor_from_list(cls, data, dtype=None, device=None):
        inst = cls(track_time=False)
        inst.data = inst.tensor_from_list_(data, dtype, device)
        return inst

from .abstraction import register_backend
register_backend("jax", JAXTensorOperations)
