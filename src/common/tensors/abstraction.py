"""Abstraction layer for tensor operations."""
from __future__ import annotations



from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional, List, Union, Callable, Dict, Deque
import math
import time
from collections import deque

# Wire in new abstraction_methods/properties
from .abstraction_methods import properties as _properties

# TYPE: Faculty, DEFAULT_FACULTY should be imported from .faculty
try:
    from .faculty import Faculty, DEFAULT_FACULTY
except ImportError:
    Faculty = None  # TYPE: ignore
    DEFAULT_FACULTY = None  # TYPE: ignore

DEFAULT_DEVICE = "cpu"

# Optional dependencies
try:
    import torch
except ImportError:
    torch = None  # TYPE: ignore
try:
    import numpy as np
except ImportError:
    np = None  # TYPE: ignore
try:
    from .accelerator_backends.c_backend import CTensor
except ImportError:
    CTensor = None  # TYPE: ignore

# TYPE: register_conversion, CONVERSION_REGISTRY, DEBUG, _get_ops_for_class
from . import DEBUG
def register_conversion(*args, **kwargs):
    pass
CONVERSION_REGISTRY = dict()



# ---- diagnostics ------------------------------------------------------------
import os
from dataclasses import dataclass

DIAG_LEVEL = os.environ.get("ABSTRACT_TENSOR_DIAG", "auto")  # 'concise' | 'auto' | 'verbose'

@dataclass
class _Diag:
    op: str | None = None          # e.g., "broadcast_rows"
    tensor: str | None = None      # e.g., "Linear.forward(bias)"
    expected: str | None = None    # e.g., "(1, 16) or (32, 16)"
    actual: str | None = None      # e.g., "(8, 16)"
    batch_size: int | None = None
    hint: str | None = None

class TensorShapeError(ValueError):
    def __init__(self, message: str, diag: _Diag | None = None):
        self._message = message
        self._diag = diag
        super().__init__(str(self))

    def __str__(self) -> str:
        d = self._diag
        if d is None:
            return self._message
        head = self._message
        # concise core: one line with labels + shapes
        parts = []
        if d.op:     parts.append(d.op)
        if d.tensor: parts.append(d.tensor)
        prefix = f"{' in '.join(parts)}: " if parts else ""
        line1 = f"{prefix}expected {d.expected}"
        if d.batch_size is not None:
            line1 += f" for batch_size={d.batch_size}"
        line1 += f", got {d.actual}."
        # hint policy
        want_hint = (DIAG_LEVEL == "verbose") or (DIAG_LEVEL == "auto" and d.hint)
        if want_hint:
            return head + "\n" + line1 + (f"\nHint: {d.hint}" if d.hint else "")
        return head + "\n" + line1







# --- Backend Registry Pattern ---
# This registry allows dynamic discovery and decoupling of tensor backends.
# Each backend module registers itself here at import time, avoiding all circular imports.
BACKEND_REGISTRY: dict[str, type] = {}

def register_backend(name: str, backend_cls: type) -> None:
    """
    Register a tensor backend class under a given name.
    Backends should call this after their class definition.
    """
    BACKEND_REGISTRY[name] = backend_cls

# Delayed import to avoid circular dependency
class MeshGrid:
    """Tuple-like wrapper for meshgrid results."""
    def __init__(self, tensors):
        self.tensors = tuple(tensors)

    def __iter__(self):
        return iter(self.tensors)

        return len(self.tensors)

    def __getitem__(self, idx):
        return self.tensors[idx]

    def __repr__(self):
        return f"MeshGrid(len={len(self.tensors)}, shape={self.shape})"

    @property
    def shape(self):
        return self.tensors[0].shape if self.tensors else ()




def _register_all_conversions():
    NumPyTensorOperations = BACKEND_REGISTRY.get("numpy")
    PyTorchTensorOperations = BACKEND_REGISTRY.get("torch")
    JAXTensorOperations = BACKEND_REGISTRY.get("jax")
    PurePythonTensorOperations = BACKEND_REGISTRY.get("pure_python")
class AbstractTensor:
    # --- Sentinel dtypes for use before backend is set ---
    float_dtype_ = 'float32'  # Default sentinel, can be replaced by backend
    long_dtype_ = 'int64'     # Default sentinel, can be replaced by backend
    bool_dtype_ = 'bool'      # Default sentinel, can be replaced by backend
    # --- Unary operators ---


    def __neg__(self):
        return self._apply_operator("neg", self, None)

    def __pos__(self):
        # +tensor is a no-op, return self
        return self


    def __abs__(self):
        return self._apply_operator("abs", self, None)


    def __invert__(self):
        return self._apply_operator("invert", self, None)


    def __round__(self, n=None):
        raise NotImplementedError(f"{self.__class__.__name__} must implement __round__()")


    def __trunc__(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement __trunc__()")


    def __floor__(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement __floor__()")


    def __ceil__(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement __ceil__()")

    # --- Backend hooks for unary ops (must be implemented by backends) ---
    def neg_(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement neg_()")

    def abs_(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement abs_()")

    def invert_(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement invert_()")

    def round_(self, n=None):
        raise NotImplementedError(f"{self.__class__.__name__} must implement round_()")

    def trunc_(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement trunc_()")

    def floor_(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement floor_()")

    def ceil_(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement ceil_()")
    def max_(self, dim=None, keepdim: bool = False):
        raise NotImplementedError(f"{self.__class__.__name__} must implement max_() with keepdim.")

    def argmax_(self, dim=None, keepdim: bool = False):
        raise NotImplementedError(f"{self.__class__.__name__} must implement argmax_() with keepdim.")

    def clamp_max_(self, max):
        raise NotImplementedError(f"{self.__class__.__name__} must implement clamp_max_()")

    def greater_(self, value):
        raise NotImplementedError(f"{self.__class__.__name__} must implement greater_()")
    def greater_equal_(self, value):
        raise NotImplementedError(f"{self.__class__.__name__} must implement greater_equal_()")

    def less_(self, value):
        raise NotImplementedError(f"{self.__class__.__name__} must implement less_()")

    def less_equal_(self, value):
        raise NotImplementedError(f"{self.__class__.__name__} must implement less_equal_()")

    def equal_(self, value):
        raise NotImplementedError(f"{self.__class__.__name__} must implement equal_()")
    def not_equal_(self, value):
        raise NotImplementedError(f"{self.__class__.__name__} must implement not_equal_()")

    # --- Logical ---
    def logical_not(self) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time)
        result.data = self.logical_not_()
        return result

    def logical_not_(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement logical_not_()")

    # --- Unary math ---
    def sqrt(self) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time)
        result.data = self.sqrt_()
        return result

    def sqrt_(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement sqrt_()")

    def exp(self) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time)
        result.data = self.exp_()
        return result

    def exp_(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement exp_()")

    def log(self) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time)
        result.data = self.log_()
        return result

    def log_(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement log_()")

    # --- Softmax utilities ---
    def softmax(self, dim: int = -1) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time)
        result.data = self.softmax_(dim)
        return result

    def softmax_(self, dim):
        raise NotImplementedError(f"{self.__class__.__name__} must implement softmax_()")

    def log_softmax(self, dim: int = -1) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time)
        result.data = self.log_softmax_(dim)
        return result

    def log_softmax_(self, dim):
        raise NotImplementedError(f"{self.__class__.__name__} must implement log_softmax_()")

    # --- Basic layout ---
    def reshape(self, *shape: int) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time)
        result.data = self.reshape_(shape)
        return result

    def reshape_(self, shape):
        raise NotImplementedError(f"{self.__class__.__name__} must implement reshape_()")

    def transpose(self, dim0: int = 0, dim1: int = 1) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time)
        result.data = self.transpose_(dim0, dim1)
        return result

    def transpose_(self, dim0, dim1):
        raise NotImplementedError(f"{self.__class__.__name__} must implement transpose_()")

    def squeeze(self, dim: int | None = None) -> "AbstractTensor":
        """Return a tensor with all (or one) dimensions of size 1 removed."""
        result = type(self)(track_time=self.track_time)
        result.data = self.squeeze_(dim)
        return result

    def squeeze_(self, dim: int | None = None):
        raise NotImplementedError(f"{self.__class__.__name__} must implement squeeze_()")
    def mean(self, dim=None, keepdim: bool = False):
        """Return the mean of the tensor along the specified dimension(s)."""
        return self.mean_(dim=dim, keepdim=keepdim)

    def sum(self, dim=None, keepdim: bool = False):
        """Return the sum of the tensor along the specified dimension(s)."""
        return self.sum_(dim=dim, keepdim=keepdim)

    def cumsum(self, dim: int = 0) -> "AbstractTensor":
        """Return the cumulative sum of the tensor along a dimension."""
        result = type(self)(track_time=self.track_time)
        result.data = self.cumsum_(dim)
        return result

    def min(self, dim=None, keepdim: bool = False):
        """Return the minimum of the tensor along the specified dimension(s)."""
        return self.min_(dim=dim, keepdim=keepdim)

    def argmin(self, dim: Optional[int] = None, keepdim: bool = False):
        """Return the indices of the minimum values along an axis."""
        return self.argmin_(dim, keepdim)
    # --- Backend hooks for reductions (must be implemented by backends) ---
    def mean_(self, dim=None, keepdim: bool = False):
        raise NotImplementedError(f"{self.__class__.__name__} must implement mean_() with keepdim.")

    def sum_(self, dim=None, keepdim: bool = False):
        raise NotImplementedError(f"{self.__class__.__name__} must implement sum_() with keepdim.")

    def cumsum_(self, dim: int = 0):
        raise NotImplementedError(f"{self.__class__.__name__} must implement cumsum_()")

    def min_(self, dim=None, keepdim: bool = False):
        raise NotImplementedError(f"{self.__class__.__name__} must implement min_() with keepdim.")

    def argmin_(self, dim=None, keepdim: bool = False):
        raise NotImplementedError(f"{self.__class__.__name__} must implement argmin_() with keepdim.")

    def tolist_(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement tolist_() for conversion to list.")
    def _AbstractTensor__unwrap(self, obj=None):
        """Return the underlying tensor data for this AbstractTensor or for another AbstractTensor instance."""
        if obj is None:
            return self.data
        if isinstance(obj, AbstractTensor):
            return obj.data
        return obj
    def __init__(self, track_time: bool = False):
        self.track_time = track_time

    @classmethod
    def tensor_from_list(cls, data, dtype=None, device=None):
        inst = cls(track_time=False)
        inst.data = inst.tensor_from_list_(data, dtype, device)
        return inst
    # --- Tensor creation and manipulation methods ---

    def full_(
        self,
        size: Tuple[int, ...],
        fill_value: Any,
        dtype: Any = None,
        device: Any = None,
    ):
        raise NotImplementedError(f"{self.__class__.__name__} must implement full_()")

    def zeros_(self, size: Tuple[int, ...], dtype: Any = None, device: Any = None):
        raise NotImplementedError(f"{self.__class__.__name__} must implement zeros_()")

    def ones_(self, size: Tuple[int, ...], dtype: Any = None, device: Any = None):
        raise NotImplementedError(f"{self.__class__.__name__} must implement ones_()")

    def zeros_like_(self, dtype: Any = None, device: Any = None):
        raise NotImplementedError(f"{self.__class__.__name__} must implement zeros_like_()")

    def ones_like_(self, dtype: Any = None, device: Any = None):
        raise NotImplementedError(f"{self.__class__.__name__} must implement ones_like_()")

    def full_like_(
        self,
        fill_value: Any,
        dtype: Any = None,
        device: Any = None,
    ):
        raise NotImplementedError(f"{self.__class__.__name__} must implement full_like_()")

    def clone(self) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time)
        result.data = self.clone_()
        return result

    def to_device(self, device: Any = None) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time)
        result.data = self.to_device_(device)
        return result

    def get_device(self) -> Any:
        return self.get_device_()

    def get_dtype(self) -> Any:
        return self.get_dtype_()


    # --- Properties and methods from abstraction_methods/properties.py ---
    numel = _properties.numel
    item = _properties.item
    shape = _properties.shape
    shape_ = _properties.shape_
    ndim = _properties.ndim
    dim = _properties.dim
    ndims = _properties.ndims
    datastring = _properties.datastring
    __str__ = _properties.__str__
    __repr__ = _properties.__repr__
    __len__ = _properties.__len__


    
    @classmethod
    def tensor(
        cls,
        data=None,
        *,
        dtype=None,
        device=None,
        track_time: bool = False,
        faculty: "Faculty" = None,
    ) -> "AbstractTensor":
        """
        Create an AbstractTensor from `data`.

        - If called on AbstractTensor: auto-select backend via get_tensor(...).
        - If called on a backend subclass: use that backend directly.
        - dtype/device are applied best-effort after wrapping.
        """
        if cls is AbstractTensor:
            # Auto-select backend
            return AbstractTensor.get_tensor(
                data,
                faculty=faculty,
                track_time=track_time,
                dtype=dtype,
                device=device,
            )

        # Use the specific backend class
        inst = cls(track_time=track_time)
        if data is None:
            return inst  # handle-as-backend-handle case, like get_tensor(None)

        out = inst.ensure_tensor(data)
        if dtype is not None:
            try:
                out = out.to_dtype(dtype)
            except Exception:
                pass
        if device is not None:
            try:
                out = out.to_device(device)
            except Exception:
                pass
        return out


    @staticmethod
    def range(start, end=None, step=1, *, dtype=None, device=None, cls=None):
        return AbstractTensor.arange(start, end, step, dtype=dtype, device=device, cls=cls)

    @staticmethod
    def arange(
        start,
        end=None,
        step=1,
        *,
        dtype=None,
        device=None,
        cls=None,
    ) -> "AbstractTensor":
        """
        Create an arange tensor using the best available backend if cls is None.

        Forms:
        arange(end)                -> [0, 1, ..., end-1]
        arange(start, end)         -> [start, ..., end-step]
        arange(start, end, step)   -> general form

        Notes:
        - step must be nonzero.
        - dtype/device are forwarded to the backend.
        """
        # Normalize one-argument form
        if end is None:
            start, end = 0, start

        if step == 0:
            raise ValueError("arange step must be nonzero")

        # Backend selection (attempt to register common backends)
        if cls is None:
            try:
                from . import torch_backend  # noqa: F401
            except Exception:
                pass
            try:
                from . import numpy_backend  # noqa: F401
            except Exception:
                pass
            try:
                from . import pure_backend  # noqa: F401
            except Exception:
                pass

            for backend_name in ("torch", "numpy", "pure_python"):
                backend_cls = BACKEND_REGISTRY.get(backend_name)
                if backend_cls is not None:
                    cls = backend_cls
                    break
            if cls is None:
                raise RuntimeError("No tensor backend available for arange.")

        inst = cls(track_time=False)  # Assuming default track_time
        inst.data = inst.arange_(start, end, step, dtype=dtype, device=device)
        return inst

    def select_by_indices(
        self, indices_dim0: Any = None, indices_dim1: Any = None
    ) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time)
        result.data = self.select_by_indices_(indices_dim0, indices_dim1)
        return result

    def log_softmax(self, dim: int = -1) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time)
        result.data = self.log_softmax_(dim)
        return result

    def pad(self, pad: Tuple[int, ...] = (0, 0), value: float = 0) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time)
        result.data = self.pad_(pad, value)
        return result

    # --- 2D spatial helpers -------------------------------------------------
    def pad2d(self, pad: Tuple[int, int, int, int], value: float = 0.0) -> "AbstractTensor":
        """Pad a 4D NCHW tensor with constant values.

        Parameters
        ----------
        pad:
            Tuple ``(pad_left, pad_right, pad_top, pad_bottom)``.
        value:
            Constant fill value for padded regions.
        """
        result = type(self)(track_time=self.track_time)
        result.data = self.pad2d_(pad, value)
        return result

    def pad2d_(self, pad: Tuple[int, int, int, int], value: float = 0.0):
        """Backend hook for :meth:`pad2d`."""
        # Default implementation delegates to generic ``pad``.
        return self.pad_(pad, value)

    def unfold2d(
        self,
        kernel_size: Tuple[int, int],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
    ) -> "AbstractTensor":
        """Extract sliding local blocks as columns (im2col)."""
        result = type(self)(track_time=self.track_time)
        result.data = self.unfold2d_(kernel_size, stride, padding, dilation)
        return result

    def unfold2d_(
        self,
        kernel_size: Tuple[int, int],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
    ):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement unfold2d_()"
        )

    @staticmethod
    def fold2d(
        cols: "AbstractTensor",
        output_size: Tuple[int, int, int, int],
        kernel_size: Tuple[int, int],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
    ) -> "AbstractTensor":
        """Inverse of :meth:`unfold2d`, accumulating patches back to images."""
        result = type(cols)(track_time=cols.track_time)
        result.data = cols.fold2d_(
            output_size, kernel_size, stride, padding, dilation
        )
        return result

    def fold2d_(
        self,
        output_size: Tuple[int, int, int, int],
        kernel_size: Tuple[int, int],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
    ):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement fold2d_()"
        )

    def cat(self, tensors: List[Any], dim: int = 0) -> "AbstractTensor":
        tensors = [self.ensure_tensor(t) for t in tensors]
        result = type(self)(track_time=self.track_time)
        result.data = self.cat_(tensors, dim)
        return result

    def topk(self, k: int = 1, dim: int = -1) -> Tuple["AbstractTensor", Any]:
        values, idxs = self.topk_(k, dim)
        result = type(self)(track_time=self.track_time)
        result.data = values
        return result, idxs

    def stack(self, tensors: List[Any], dim: int = 0) -> "AbstractTensor":
        tensors = [self.ensure_tensor(t) for t in tensors]
        result = type(self)(track_time=self.track_time)
        result.data = self.stack_(tensors, dim)
        return result

    # --- Broadcasting helpers (abstract, backend-agnostic) ---
    def expand(self, shape: tuple) -> "AbstractTensor":
        """Backend-agnostic expand/broadcast_to (view when possible)."""
        result = type(self)(track_time=self.track_time)
        result.data = self.expand_(shape)
        return result
    # in AbstractTensor (abstraction.py)
    def broadcast_rows(self, n: int, *, label: str | None = None) -> "AbstractTensor":
        """
        Make a (1, D) bias behave as (n, D) via view-style expand when possible.
        If already (n, D) return self.
        If neither 1 nor n rows, raise a labeled TensorShapeError.
        """
        rows, cols = self.shape[0], self.shape[1]
        if rows == n:
            return self
        if rows == 1:
            try:
                return self.expand((n, cols))
            except Exception:
                # backend lacks true view-expand; fall back to repeat
                return self.repeat_interleave(repeats=n, dim=0)

        # Build an iff-applicable hint (only when it's the classic "multi-row bias" mistake)
        hint = None
        if rows > 1 and cols >= 1:
            # This really looks like an accidental (N, D) bias; propose a safe collapse.
            hint = (
                "Bias must be a single row broadcast across the batch, or match the batch exactly. "
                "If you accidentally created an (N, D) bias, collapse it, e.g.: "
                "bias = bias.sum(dim=0, keepdim=True)  # -> (1, D)"
            )

        raise TensorShapeError(
            "Broadcast error",
            _Diag(
                op="broadcast_rows",
                tensor=label,
                expected=f"(1, {cols}) or ({n}, {cols})",
                actual=f"({rows}, {cols})",
                batch_size=n,
                hint=hint,  # only shown in 'auto' if we actually built one
            ),
        )


    def repeat_interleave(
        self, repeats: int = 1, dim: Optional[int] = None
    ) -> "AbstractTensor":
        result = AbstractTensor.get_tensor(self.repeat_interleave_(repeats, dim))
        return result

    # Backend hooks -----------------------------------------------------------
    def expand_(self, shape):  # pragma: no cover - backend required
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement expand_()"
        )

    def repeat_interleave_(
        self, repeats: int = 1, dim: Optional[int] = None
    ):  # pragma: no cover - backend required
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement repeat_interleave_()"
        )

    def view_flat(self) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time)
        result.data = self.view_flat_()
        return result

    def assign_at_indices(
        self,
        indices_dim0: Any = None,
        indices_dim1: Any = None,
        values_to_assign: Any = None,
    ) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time)
        result.data = self.assign_at_indices_(
            indices_dim0, indices_dim1, values_to_assign
        )
        return result

    def increment_at_indices(self, mask: Any = None) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time)
        result.data = self.increment_at_indices_(mask)
        return result

    # Only keep the upper, guarded to_backend (already present above)

    

    def boolean_mask_select(self, mask: Any = None) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time)
        result.data = self.boolean_mask_select_(mask)
        return result

    def tolist(self) -> List[Any]:
        return self.tolist_()


    def index_select(self, dim: int = 0, indices: Any = None) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time)
        result.data = self.index_select_(dim, indices)
        return result

    def argmin(self, dim: Optional[int] = None) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time)
        result.data = self.argmin_(dim)
        return result

    def interpolate(self, size: Tuple[int, ...] = None) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time)
        result.data = self.interpolate_(size)
        return result

    def save(self, filepath: str = None) -> None:
        self.save_(filepath)

    def load(
        self, filepath: str, dtype: Any = None, device: Any = None
    ) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time)
        result.data = self.load_(filepath, dtype, device)
        return result

    def to_dtype(self, dtype: str = "float") -> "AbstractTensor":
        result = type(self)(track_time=self.track_time)
        result.data = self.to_dtype_(dtype)
        return result

    # --- Dtype helpers ---
    @property
    def long_dtype(self) -> Any:
        return self.long_dtype_

    @property
    def bool_dtype(self) -> Any:
        return self.bool_dtype_

    @property
    def float_dtype(self) -> Any:
        return self.float_dtype_

    @property
    def tensor_type(self) -> type:
        return self.tensor_type_

    # Lightweight helper to coerce arbitrary input to this backend's tensor type


    def to_backend(self, target_ops):
        conv_func = CONVERSION_REGISTRY.get((type(self), type(target_ops)))
        if conv_func is None:
            converted = default_to_backend(self, self, target_ops)
        else:
            converted = conv_func(self, self, target_ops)

        if isinstance(converted, AbstractTensor):
            if type(converted) is type(target_ops):
                return converted
            # No progress? Inject raw data into the target and bail.
            if type(converted) is type(self):
                out = type(target_ops)(track_time=self.track_time)
                out.data = converted.data
                return out
            return converted.to_backend(target_ops)

        new_tensor = type(target_ops)(track_time=self.track_time)
        new_tensor.data = converted
        return new_tensor

    def ensure_tensor(self, tensor: Any) -> "AbstractTensor":
        """Return ``tensor`` wrapped as an ``AbstractTensor`` instance."""
        if not isinstance(self, AbstractTensor):
            raise TypeError(f"ensure_tensor called on non-AbstractTensor instance: {type(self)}")
        if tensor is None:
            raise ValueError("ensure_tensor called with tensor=None")
        backend_cls = self.__class__
        if isinstance(tensor, AbstractTensor):
            return tensor.to_backend(self)
        if isinstance(tensor, self.tensor_type):
            result = backend_cls(track_time=self.track_time)
            result.data = tensor
            return result
        if torch is not None and isinstance(tensor, torch.Tensor):
            torch_ops = AbstractTensor.get_tensor(faculty=Faculty.TORCH)
            tmp = torch_ops.__class__()
            tmp.data = tensor
            return tmp.to_backend(self)
        if np is not None and isinstance(tensor, np.ndarray):
            numpy_ops = AbstractTensor.get_tensor(faculty=Faculty.NUMPY)
            numpy_tensor = numpy_ops.__class__()
            numpy_tensor.data = tensor
            return numpy_tensor.to_backend(self)
        if isinstance(tensor, (list, tuple)):
            return self.tensor_from_list(tensor, dtype=None, device=None)
        if hasattr(tensor, "tolist"):
            return self.tensor_from_list(tensor.tolist(), dtype=None, device=None)
        return self.tensor_from_list([tensor], dtype=None, device=None)

    # --- Operator routing ---
    def _apply_operator(self, op: str, left: Any, right: Any):
        """
        Arithmetic with bool tensors:
        - if mixing with floats/complex -> cast bool to float
        - if mixing with ints          -> cast bool to int (0/1)
        - for true division            -> cast bool to float
        Promotion happens BEFORE unwrap; backends never see bool arithmetic.
        """
        # Coerce list-like operands into tensors so that operator logic remains
        # backend-agnostic. This ensures raw Python lists interoperate with
        # tensors without requiring callers to explicitly convert them.
        # In AbstractTensor._apply_operator(...)
        if isinstance(left, AbstractTensor) and isinstance(right, (list, tuple)):
            right = left.ensure_tensor(right)      # instead of get_tensor(... Faculty.NUMPY)
        elif isinstance(right, AbstractTensor) and isinstance(left, (list, tuple)):
            left = right.ensure_tensor(left)       # instead of get_tensor(... Faculty.NUMPY)

        # Optional belt-and-suspenders: align mixed backends
        if isinstance(left, AbstractTensor) and isinstance(right, AbstractTensor) and (type(left) is not type(right)):
            right = right.to_backend(left)

        arithmetic_ops = {
            "add","sub","mul","truediv","floordiv","mod","pow",
            "iadd","isub","imul","itruediv","ifloordiv","imod","ipow",
            "radd","rsub","rmul","rtruediv","rfloordiv","rmod","rpow",
            "neg","abs","invert",
        }
        div_ops = {"truediv","rtruediv","itruediv"}

        def kind(x):
            if isinstance(x, AbstractTensor):
                try:
                    dt = x.get_dtype()
                    if dt == x.bool_dtype: return "bool"
                    s = str(dt).lower()
                    if "complex" in s: return "complex"
                    if "float" in s or "half" in s or "bfloat" in s: return "float"
                    if "int" in s or "long" in s: return "int"
                except Exception:
                    return "unknown"
                return "unknown"
            if isinstance(x, bool):  return "bool"
            if isinstance(x, float): return "float"
            if isinstance(x, int):   return "int"
            return "unknown"

        def cast_bool_like(x, target_kind):
            if target_kind in ("float","complex"):
                return x.to_dtype("float")
            else:
                return x.long()

        if op in arithmetic_ops:
            lk, rk = kind(left), kind(right)
            if op in div_ops:
                if isinstance(left, AbstractTensor) and lk == "bool":
                    left = left.to_dtype("float")
                if isinstance(right, AbstractTensor) and rk == "bool":
                    right = right.to_dtype("float")
            else:
                target = "float" if ("float" in (lk, rk) or "complex" in (lk, rk)) else "int"
                if isinstance(left, AbstractTensor) and lk == "bool":
                    left = cast_bool_like(left, target)
                if isinstance(right, AbstractTensor) and rk == "bool":
                    right = cast_bool_like(right, target)

            # Handle raw Python bools symmetrically (e.g., 1 - True)
            if isinstance(left, bool):
                left = 1.0 if (op in div_ops or "float" in (rk,)) else 1
            if isinstance(right, bool):
                right = 1.0 if (op in div_ops or "float" in (lk,)) else 1

        # unwrap AFTER promotion
        l = left._AbstractTensor__unwrap() if isinstance(left, AbstractTensor) else left
        r = right._AbstractTensor__unwrap() if isinstance(right, AbstractTensor) else right

        result = type(self)(track_time=self.track_time)
        result.data = self._apply_operator__(op, l, r)

        # Record operation on the lightweight autograd tape for backends
        # lacking native autograd (pure Python and NumPy).  Torch/JAX rely on
        # their own automatic differentiation systems.
        from . import autograd as _autograd

        backend_name = type(self).__name__
        if backend_name in ("PurePythonTensorOperations", "NumPyTensorOperations"):
            _autograd.autograd.record(
                op, [x for x in (left, right) if x is not None], result
            )

        return result


    def __add__(self, other):
        return self._apply_operator("add", self, other)

    def __sub__(self, other):
        return self._apply_operator("sub", self, other)

    def __mul__(self, other):
        return self._apply_operator("mul", self, other)

    def __truediv__(self, other):
        return self._apply_operator("truediv", self, other)

    def __floordiv__(self, other):
        return self._apply_operator("floordiv", self, other)

    def __mod__(self, other):
        return self._apply_operator("mod", self, other)

    def __pow__(self, other):
        return self._apply_operator("pow", self, other)

    def __matmul__(self, other):
        return self._apply_operator("matmul", self, other)

    def __eq__(self, other):
        return self.equal(other)

    def __ne__(self, other):
        return self.not_equal(other)

    def __lt__(self, other):
        return self.less(other)

    def __le__(self, other):
        return self.less_equal(other)

    def __gt__(self, other):
        return self.greater(other)

    def __ge__(self, other):
        return self.greater_equal(other)

    # Reverse operators
    def __radd__(self, other):
        return self._apply_operator("radd", other, self)

    def __rsub__(self, other):
        return self._apply_operator("rsub", other, self)

    def __rmul__(self, other):
        return self._apply_operator("rmul", other, self)

    def __rtruediv__(self, other):
        return self._apply_operator("rtruediv", other, self)

    def __rfloordiv__(self, other):
        return self._apply_operator("rfloordiv", other, self)

    def __rmod__(self, other):
        return self._apply_operator("rmod", other, self)

    def __rpow__(self, other):
        return self._apply_operator("rpow", other, self)

    def __rmatmul__(self, other):
        return self._apply_operator("rmatmul", other, self)

    # In-place operators
    def __iadd__(self, other):
        return self._apply_operator("iadd", self, other)

    def __isub__(self, other):
        return self._apply_operator("isub", self, other)

    def __imul__(self, other):
        return self._apply_operator("imul", self, other)

    def __itruediv__(self, other):
        return self._apply_operator("itruediv", self, other)

    def __ifloordiv__(self, other):
        return self._apply_operator("ifloordiv", self, other)

    def __imod__(self, other):
        return self._apply_operator("imod", self, other)

    def __ipow__(self, other):
        return self._apply_operator("ipow", self, other)

    def __imatmul__(self, other):
        return self._apply_operator("imatmul", self, other)

    # --- Indexing helpers ---
    def __getitem__(self, idx):
        """Return an indexed view wrapped as an AbstractTensor when possible.

        Accepts either standard Python index types or another ``AbstractTensor``
        instance as ``idx``. When given an ``AbstractTensor`` the underlying
        value is extracted via ``__unwrap`` so all backends behave consistently
        for tensor-based indexing.
        """
        if DEBUG:
            print(f"__getitem__ called with idx={idx} on {self.__class__.__name__}")
        data = self.data
        if data is None:
            raise ValueError("__getitem__ called on empty tensor")

        # Ensure backend-native tensor type for indexing
        if CTensor is not None and isinstance(data, CTensor):
            # CTensor might require special handling or might not support all Python slicing.
            # For now, assume it needs unwrapped indices if idx contains AbstractTensors.
            # This placeholder allows future CTensor-specific indexing logic.
            pass  # Fall through to generic index processing for now

        if isinstance(idx, tuple):
            index = tuple(
                (
                    item._AbstractTensor__unwrap()
                    if isinstance(item, AbstractTensor)
                    else item
                )
                for item in idx
            )
        elif isinstance(idx, AbstractTensor):
            index = idx._AbstractTensor__unwrap()
        else:
            index = idx

        result = data[index]
        if isinstance(result, self.tensor_type):
            wrapped = type(self)(track_time=self.track_time)
            wrapped.data = result
            return wrapped
        return result


    def __setitem__(self, idx, value):
        """Assign to the underlying tensor using Python indexing.

        Like ``__getitem__``, ``idx`` may itself be an ``AbstractTensor``.  The
        raw value is extracted before performing the assignment so tensor-based
        indices work across all backends.
        """
        if DEBUG:
            print(
                f"__setitem__ called with idx={idx}, value={value} on {self.__class__.__name__}"
            )
        data = self.data
        if data is None:
            raise ValueError("__setitem__ called on empty tensor")
        if CTensor is not None and isinstance(data, CTensor):
            raise NotImplementedError("__setitem__ not implemented for CTensor backend")
        if isinstance(value, AbstractTensor):
            value = value.data
        index = self._AbstractTensor__unwrap(idx)
        data[index] = value

    def __bool__(self):
        try:
            n = int(self.numel())
        except Exception:
            return bool(self.item())
        if n != 1:
            raise ValueError("The truth value of a tensor with more than one element is ambiguous.")
        return bool(self.item())

    def data_or(self, obj: Any = None) -> Any:
        """Return self.data if no argument is passed, otherwise return the argument unchanged."""
        if obj is None:
            return self.data
        return obj

    @abstractmethod
    def get_shape(self) -> Tuple[int, ...]:
        """Return the shape of ``self`` as a tuple."""
        pass

    @abstractmethod
    def get_ndims(self) -> int:
        """Return the number of dimensions of ``self``."""
        pass

    def repeat(self, repeats: Any = None, dim: int = 0) -> "AbstractTensor":
        """Repeat ``self`` along ``dim`` ``repeats`` times."""
        return self.repeat_(repeats, dim)

    @staticmethod
    def get_tensor(
        data=None,
        faculty: "Faculty" = None,
        *,
        track_time: bool = False,
        dtype=None,
        device=None,
        cls=None
    ) -> "AbstractTensor":
        """
        Create and return an AbstractTensor instance from any data, auto-selecting the best backend if faculty is None.
        If faculty is provided, use the corresponding backend.
        If cls is provided, it is used directly as the backend class.
        """
        if cls is not None:
            backend_cls = cls
            # Try to pass default_device if the backend supports it
            try:
                tensor = backend_cls(default_device=DEFAULT_DEVICE, track_time=track_time)
            except TypeError:
                tensor = backend_cls(track_time=track_time)
        else:
            faculty = faculty or DEFAULT_FACULTY
            if faculty in (Faculty.TORCH, Faculty.PYGEO):
                backend_cls = BACKEND_REGISTRY.get("torch")
                if backend_cls is None:
                    from . import torch_backend  # noqa: F401
                    backend_cls = BACKEND_REGISTRY.get("torch")
                tensor = backend_cls(default_device=DEFAULT_DEVICE, track_time=track_time)
            elif faculty is Faculty.NUMPY and np is not None:
                backend_cls = BACKEND_REGISTRY.get("numpy")
                if backend_cls is None:
                    from . import numpy_backend  # noqa: F401
                    backend_cls = BACKEND_REGISTRY.get("numpy")
                tensor = backend_cls(track_time=track_time)
            elif faculty is Faculty.CTENSOR:
                from .accelerator_backends.c_backend import CTensorOperations
                tensor = CTensorOperations(track_time=track_time)
            else:
                backend_cls = BACKEND_REGISTRY.get("pure_python")
                tensor = backend_cls(track_time=track_time)

        if data is not None:
            out = tensor.ensure_tensor(data)
            # apply dtype/device if requested (best-effort)
            if dtype is not None:
                try:
                    out = out.to_dtype(dtype)
                except Exception:
                    pass
            if device is not None:
                try:
                    out = out.to_device(device)
                except Exception:
                    pass
            return out

        # no data: return a backend handle (dtype/device can be applied later when data exists)
        return tensor

class AbstractF:
    """
    Functional API for advanced tensor operations (e.g., interpolation).
    Decides the best backend and dispatches accordingly.
    """

    @staticmethod
    def interpolate(
        tensor,
        size=None,
        scale_factor=None,
        mode="bilinear",
        batch_dim=0,
        channel_dim=1,
        backend: str = None,
        align_corners=False,
    postprocess: str = None,  # 'round', 'floor', 'ceil', or None
    **kwargs,
    ):
        """
        Interpolate a tensor to a new size using the best available backend.
        - tensor: AbstractTensor or raw data
        - size: tuple of ints (new spatial size)
        - scale_factor: float or tuple (optional)
        - mode: interpolation mode (e.g., 'bilinear', 'nearest')
        - batch_dim: which dim is batch (default 0)
        - channel_dim: which dim is channel (default 1)
        - backend: 'torch', 'numpy', etc. (optional, auto if None)
        - align_corners: passed to torch if used
        """
        # Convert to AbstractTensor if needed
        tensor = AbstractTensor.get_tensor(tensor)
        orig_dtype = None
        try:
            import torch  # type: ignore
        except Exception:  # torch may be unavailable
            torch = None  # type: ignore
        if hasattr(tensor, 'data'):
            arr_data = tensor.data
        else:
            arr_data = tensor
        if torch is not None and hasattr(arr_data, 'dtype'):
            orig_dtype = arr_data.dtype
        # Backend selection
        chosen = None
        if backend == "torch":
            chosen = "torch"
        elif backend == "numpy":
            chosen = "numpy"
        else:
            try:
                import torch
                import torch.nn.functional as F

                chosen = "torch"
            except ImportError:
                chosen = "numpy"
        if chosen == "torch":
            import torch
            import torch.nn.functional as F

            arr = tensor.to_backend(AbstractTensor.get_tensor(faculty=Faculty.TORCH))
            data = arr.data if hasattr(arr, "data") else arr
            # Ensure shape is (N, C, H, W) or (N, 1, H, W)
            nd = data.dim() if hasattr(data, "dim") else len(data.shape)
            if nd == 2:
                data = data.unsqueeze(0).unsqueeze(0)
            elif nd == 3:
                # Assume (C, H, W) or (N, H, W)
                if batch_dim == 0:
                    data = data.unsqueeze(0)
                else:
                    data = data.unsqueeze(1)
            # Convert to float if needed (required by F.interpolate for most modes)
            was_int = not torch.is_floating_point(data)
            if was_int:
                data = data.float()
            out = F.interpolate(
                data,
                size=size,
                scale_factor=scale_factor,
                mode=mode,
                align_corners=(
                    align_corners
                    if mode in ("linear", "bilinear", "bicubic", "trilinear")
                    else None
                ),
            )
            # Postprocess if requested
            if postprocess is not None:
                if postprocess == 'round':
                    out = torch.round(out)
                elif postprocess == 'floor':
                    out = torch.floor(out)
                elif postprocess == 'ceil':
                    out = torch.ceil(out)
                else:
                    raise ValueError(f"Unknown postprocess: {postprocess}")
            # Cast back to original dtype if it was int
            if was_int and orig_dtype is not None:
                out = out.to(orig_dtype)
            # Remove added batch/channel dims if needed
            if nd == 2:
                out = out[0, 0]
            elif nd == 3:
                out = out[0] if batch_dim == 0 else out[:, 0]
            return AbstractTensor.get_tensor(out)
        else:
            # Numpy fallback: use PIL for images, or scipy.ndimage.zoom if available
            arr = tensor.to_backend(AbstractTensor.get_tensor(faculty=Faculty.NUMPY))
            data = arr.data if hasattr(arr, "data") else arr
            import numpy as np

            if data.ndim == 2:
                from PIL import Image

                img = Image.fromarray((data * 255).astype(np.uint8))
                img = img.resize(
                    size[::-1], Image.BILINEAR if mode == "bilinear" else Image.NEAREST
                )
                out = np.array(img) / 255.0
            else:
                # Use scipy.ndimage.zoom for nD
                try:
                    from scipy.ndimage import zoom

                    zoom_factors = [size[i] / data.shape[i] for i in range(len(size))]
                    out = zoom(data, zoom_factors, order=1 if mode == "bilinear" else 0)
                except ImportError:
                    raise RuntimeError(
                        "No suitable interpolation backend available (need torch or scipy)"
                    )
            return AbstractTensor.get_tensor(out)


# Attach to AbstractTensor
AbstractTensor.F = AbstractF

# --- Abstraction method assignments ---------------------------------------
from .abstraction_methods.creation import (
    linspace,
    meshgrid,
    zeros as create_zeros,
    ones as create_ones,
    full as create_full,
    zeros_like as create_zeros_like,
    ones_like as create_ones_like,
    full_like as create_full_like,
    random_tensor,
    randoms,
    rand_like,
    randint,
    randint_like,
)
from .abstraction_methods.reduction import (
    max as reduction_max,
    argmax as reduction_argmax,
)
from .abstraction_methods.type_ops import (
    to as type_to,
    astype as type_astype,
    where as type_where,
    long_cast as type_long_cast,
    float as type_float,
    double as type_double,
    int as type_int,
    long as type_long,
    bool as type_bool,
)
from .abstraction_methods.comparison import (
    greater as comp_greater,
    greater_equal as comp_greater_equal,
    less as comp_less,
    less_equal as comp_less_equal,
    equal as comp_equal,
    not_equal as comp_not_equal,
)
from .abstraction_methods.trigonometry import (
    sin as trig_sin,
    cos as trig_cos,
    tan as trig_tan,
    asin as trig_asin,
    acos as trig_acos,
    atan as trig_atan,
    sinh as trig_sinh,
    cosh as trig_cosh,
    tanh as trig_tanh,
    asinh as trig_asinh,
    acosh as trig_acosh,
    atanh as trig_atanh,
    sec as trig_sec,
    csc as trig_csc,
    cot as trig_cot,
    sech as trig_sech,
    csch as trig_csch,
    coth as trig_coth,
    sinc as trig_sinc,
)
from .abstraction_methods.properties import (
    numel as prop_numel,
    item as prop_item,
    shape as prop_shape,
    shape_ as prop_shape_,
    ndim as prop_ndim,
    dim as prop_dim,
    ndims as prop_ndims,
    datastring as prop_datastring,
    __str__ as prop_str,
    __format__ as prop_format,
    __repr__ as prop_repr,
    __len__ as prop_len,
)

# --- Autograd method assignments ------------------------------------------
from . import autograd as _autograd_methods

AbstractTensor.requires_grad_ = _autograd_methods.requires_grad_
AbstractTensor.requires_grad  = _autograd_methods.requires_grad
AbstractTensor.backward       = _autograd_methods.backward
AbstractTensor.grad           = _autograd_methods.grad
AbstractTensor.detach         = _autograd_methods.detach
AbstractTensor.is_leaf        = _autograd_methods.is_leaf
AbstractTensor.retain_grad    = _autograd_methods.retain_grad
AbstractTensor.grad_fn        = _autograd_methods.grad_fn
AbstractTensor.zero_grad      = _autograd_methods.zero_grad
AbstractTensor.register_hook  = _autograd_methods.register_hook

# --- Bindings to AbstractTensor -------------------------------------------

AbstractTensor.linspace     = staticmethod(linspace)
AbstractTensor.meshgrid     = staticmethod(meshgrid)
AbstractTensor.zeros        = staticmethod(create_zeros)
AbstractTensor.ones         = staticmethod(create_ones)
AbstractTensor.full         = staticmethod(create_full)
AbstractTensor.zeros_like   = create_zeros_like
AbstractTensor.ones_like    = create_ones_like
AbstractTensor.full_like    = create_full_like
AbstractTensor.random       = staticmethod(random_tensor)
AbstractTensor.randoms      = staticmethod(randoms)
AbstractTensor.rand_like    = rand_like
AbstractTensor.randint      = staticmethod(randint)
AbstractTensor.randint_like = randint_like

AbstractTensor.max    = reduction_max
AbstractTensor.argmax = reduction_argmax

AbstractTensor.to        = type_to
AbstractTensor.astype    = type_astype
AbstractTensor.where     = type_where
AbstractTensor.long_cast = type_long_cast
AbstractTensor.float     = type_float
AbstractTensor.double    = type_double
AbstractTensor.int       = type_int
AbstractTensor.long      = type_long
AbstractTensor.bool      = type_bool

AbstractTensor.greater        = comp_greater
AbstractTensor.greater_equal  = comp_greater_equal
AbstractTensor.less           = comp_less
AbstractTensor.less_equal     = comp_less_equal
AbstractTensor.equal          = comp_equal
AbstractTensor.not_equal      = comp_not_equal

AbstractTensor.sin   = trig_sin
AbstractTensor.cos   = trig_cos
AbstractTensor.tan   = trig_tan
AbstractTensor.asin  = trig_asin
AbstractTensor.acos  = trig_acos
AbstractTensor.atan  = trig_atan
AbstractTensor.sinh  = trig_sinh
AbstractTensor.cosh  = trig_cosh
AbstractTensor.tanh  = trig_tanh
AbstractTensor.asinh = trig_asinh
AbstractTensor.acosh = trig_acosh
AbstractTensor.atanh = trig_atanh
AbstractTensor.sec   = trig_sec
AbstractTensor.csc   = trig_csc
AbstractTensor.cot   = trig_cot
AbstractTensor.sech  = trig_sech
AbstractTensor.csch  = trig_csch
AbstractTensor.coth  = trig_coth
AbstractTensor.sinc  = trig_sinc

AbstractTensor.numel   = prop_numel
AbstractTensor.item    = prop_item
AbstractTensor.shape   = property(prop_shape)
AbstractTensor.shape_  = prop_shape_
AbstractTensor.ndim    = property(prop_ndim)
AbstractTensor.dim     = prop_dim
AbstractTensor.ndims   = prop_ndims
AbstractTensor.datastring = prop_datastring
AbstractTensor.__str__    = prop_str
AbstractTensor.__format__ = prop_format
AbstractTensor.__repr__   = prop_repr
AbstractTensor.__len__    = prop_len

def _get_shape(data):
    if not isinstance(data, list):
        return ()
    if not data:
        return (0,)
    return (len(data),) + _get_shape(data[0])


def _flatten(data):
    if not isinstance(data, list):
        return [data]
    return [item for sublist in data for item in _flatten(sublist)]


def default_to_backend(source_ops, tensor, target_ops):
    if type(source_ops) is type(target_ops):
        return source_ops.clone()

    data = tensor.tolist()
    dtype = None
    device = None
    try:
        dtype = source_ops.get_dtype(tensor)
    except Exception:
        dtype = None
    if isinstance(dtype, str):
        dtype = None
    try: device = source_ops.get_device(tensor)
    except: pass
    # Works if backend exposes either a classmethod, staticmethod, or instance method:
    import inspect
    cls = type(target_ops)
    raw = inspect.getattr_static(cls, "tensor_from_list", None)
    if isinstance(raw, classmethod):
        return raw.__func__(cls, data, dtype, device)
    if isinstance(raw, staticmethod):
        return raw.__func__(data, dtype, device)
    return target_ops.tensor_from_list(data, dtype=dtype, device=device)


def get_tensor_operations(
    faculty: Faculty | None = None, *, track_time: bool = False
) -> AbstractTensor:
    """[REMOVED] Use AbstractTensor.get_tensor instead."""
    raise RuntimeError(
        "get_tensor_operations is obsolete. Use AbstractTensor.get_tensor instead."
    )



# --- Delayed backend registration to avoid circular imports ---

