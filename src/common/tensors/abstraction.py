"""Abstraction layer for tensor operations."""
from __future__ import annotations



from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional, List, Union, Callable, Dict, Deque, NamedTuple, Iterable, TYPE_CHECKING
import math
import time
from collections import deque

# Wire in new abstraction_methods/properties
from .abstraction_methods import properties as _properties
from .abstraction_methods import reshape as _reshape_methods

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

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from .autograd import GradTape
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
    @staticmethod
    def _normalize_shape_args(*shape):
        """
        Accepts: reshape(-1), reshape(2,3), reshape([2,3]), reshape((2,3))
        Returns a tuple shape (e.g., (-1,), (2,3))
        """
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            return tuple(int(s) for s in shape[0])
        if len(shape) == 1 and isinstance(shape[0], int):
            return (int(shape[0]),)
        return tuple(int(s) for s in shape)

    def empty_(self, size: Tuple[int, ...], dtype: Any = None, device: Any = None):
        """Create an uninitialized tensor of the given shape (backend hook)."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement empty_()")
    @staticmethod
    def backend_class_from_backend_data(data):
        """
        Given a backend-native data object (e.g., torch.Tensor, np.ndarray),
        return the registered backend class that wraps this data type.
        """
        # Ensure registry is populated

        for backend_cls in BACKEND_REGISTRY.values():
            if type(data) == backend_cls:
                return backend_cls
            
            tensor_type = backend_cls.tensor_type_
            print(tensor_type)
            if tensor_type is not None and isinstance(data, tensor_type):
                return backend_cls
        return None

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

    # --- Clamping ---
    def clamp(self, min: float | None = None, max: float | None = None) -> "AbstractTensor":
        """Return ``self`` clamped between ``min`` and ``max``."""
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.clamp_(min_val=min, max_val=max)
        return result

    def clamp_(self, min_val: float | None = None, max_val: float | None = None):
        raise NotImplementedError(f"{self.__class__.__name__} must implement clamp_()")

    def clamp_min(self, min_val: float) -> "AbstractTensor":
        """Clamp values below ``min_val`` up to ``min_val``."""
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.clamp_min_(min_val)
        return result

    def clamp_min_(self, min_val: float):
        raise NotImplementedError(f"{self.__class__.__name__} must implement clamp_min_()")

    def clamp_max(self, max_val: float) -> "AbstractTensor":
        """Clamp values above ``max_val`` down to ``max_val``."""
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.clamp_max_(max_val)
        return result

    def clamp_max_(self, max_val: float):
        raise NotImplementedError(f"{self.__class__.__name__} must implement clamp_max_()")

    # --- API compatibility ---
    def clip(
        self,
        min: float | None = None,
        max: float | None = None,
        *,
        a_min: float | None = None,
        a_max: float | None = None,
    ) -> "AbstractTensor":
        """Alias for :meth:`clamp` accepting NumPy-style parameters."""
        if (min is not None or max is not None) and (a_min is not None or a_max is not None):
            raise TypeError("Specify either min/max or a_min/a_max, not both")
        if min is None and max is None:
            min, max = a_min, a_max
        return self.clamp(min=min, max=max)

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
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.logical_not_()
        return result

    def logical_not_(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement logical_not_()")

    # --- Unary math ---
    def sqrt(self) -> "AbstractTensor":
        if isinstance(self, AbstractTensor):
            result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        else:
            result = AbstractTensor.get_tensor(self.data, track_time=False)
        result.data = self.sqrt_()
        return result

    def sqrt_(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement sqrt_()")

    def exp(self) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.exp_()
        return result

    def exp_(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement exp_()")

    def log(self) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.log_()
        return result

    def log_(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement log_()")

    # --- Softmax utilities ---
    def softmax(self, dim: int = -1) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.softmax_(dim)
        return result

    def softmax_(self, dim):
        raise NotImplementedError(f"{self.__class__.__name__} must implement softmax_()")

    def log_softmax(self, dim: int = -1) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.log_softmax_(dim)
        return result

    def log_softmax_(self, dim):
        raise NotImplementedError(f"{self.__class__.__name__} must implement log_softmax_()")

    # --- Basic layout ---
    def mean(self, dim=None, keepdim: bool = False):
        """Return the mean of the tensor along the specified dimension(s)."""
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.mean_(dim=dim, keepdim=keepdim)
        return result

    def sum(self, dim=None, keepdim: bool = False):
        """Return the sum of the tensor along the specified dimension(s)."""
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.sum_(dim=dim, keepdim=keepdim)
        return result

    def cumsum(self, dim: int = 0) -> "AbstractTensor":
        """Return the cumulative sum of the tensor along a dimension."""
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.cumsum_(dim)
        return result

    def min(self, dim=None, keepdim: bool = False):
        """Return the minimum of the tensor along the specified dimension(s)."""
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.min_(dim=dim, keepdim=keepdim)
        return result

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
    def __init__(self, track_time: bool = False, tape: "GradTape" | None = None):
        """Create a new tensor wrapper.

        Parameters
        ----------
        track_time:
            Unused placeholder retained for API compatibility.
        tape:
            Optional :class:`GradTape` the tensor should attach to. If omitted,
            the global ``autograd`` tape is used and the tensor registers itself
            as a root node on that graph.
        """

        self.track_time = track_time
        if tape is None:
            from . import autograd as _autograd  # local import to avoid cycle
            tape = _autograd.autograd.tape
        self._tape = tape
        self._tape.create_tensor_node(self)


    def tensor_from_list_(self, data, dtype=None, device=None):
        """
        Create a tensor from a Python list using the first suitable backend in BACKEND_REGISTRY (including torch),
        or try to import numpy/pure_python as fallback. Use the first available backend found.
        """
        if isinstance(data, AbstractTensor):
            return data.data

        # 1. Try any already-registered backend
        for name, backend_cls in BACKEND_REGISTRY.items():
            if backend_cls is not None:
                inst = backend_cls(track_time=False)
                return inst.tensor_from_list_(data, dtype, device)

        # 2. Try to import and register numpy backend
        backend_cls = BACKEND_REGISTRY.get("numpy")
        if backend_cls is None:
            try:
                from . import numpy_backend  # noqa: F401
                backend_cls = BACKEND_REGISTRY.get("numpy")
            except Exception:
                backend_cls = None
        if backend_cls is not None:
            inst = backend_cls(track_time=False)
            return inst.tensor_from_list_(data, dtype, device)

        # 3. Try to import and register pure_python backend
        backend_cls = BACKEND_REGISTRY.get("pure_python")
        if backend_cls is None:
            try:
                from . import pure_backend  # noqa: F401
                backend_cls = BACKEND_REGISTRY.get("pure_python")
            except Exception:
                backend_cls = None
        if backend_cls is not None:
            inst = backend_cls(track_time=False)
            return inst.tensor_from_list_(data, dtype, device)

        # 4. If all else fails, raise an error
        raise RuntimeError("No suitable tensor backend is available to create a tensor from list.")
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
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.clone_()
        return result

    # copy is an alias of clone for API compatibility
    def copy(self) -> "AbstractTensor":
        return self.clone()

    def to_device(self, device: Any = None) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
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

    @staticmethod
    def check_or_build_registry():
        if not BACKEND_REGISTRY:
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
            raise RuntimeError("No tensor backend available for tensor creation.")
        return cls

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
            cls = cls.check_or_build_registry()

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

    @classmethod
    def from_nested(cls, data, *, dtype=None, device=None):
        """
        Recursively pack arbitrarily nested sequences of leaves (scalars, numpy/torch tensors,
        AbstractTensor instances) into a single AbstractTensor by stacking bottom-up.

        This is the safe replacement for passing a nested list to .tensor(...).
        """
        from .nested_pack import pack_nested_to_tensor

        return pack_nested_to_tensor(data, dtype=dtype, device=device, cls=cls)

    @staticmethod
    def get_tensor(data=None, *, dtype=None, device=None, cls=None, track_time=False) -> "AbstractTensor":
        """
        Get the tensor data from this AbstractTensor or create a new one if data is provided.
        If data is None, return self.
        """
        if cls is None:
            cls = AbstractTensor.check_or_build_registry()
        return cls.tensor(data, dtype=dtype, device=device, track_time=track_time)

    def tensor_like(self, data=None, *, dtype=None, device=None, cls=None) -> "AbstractTensor":
        """
        Get the tensor data from this AbstractTensor or create a new one if data is provided.
        If data is None, return self.
        """
        if cls is not None:
            return cls.tensor(data, dtype=dtype, device=device, track_time=self.track_time)
        return self.tensor(data, dtype=dtype, device=device, track_time=self.track_time)

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
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.select_by_indices_(indices_dim0, indices_dim1)
        return result

    def log_softmax(self, dim: int = -1) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.log_softmax_(dim)
        return result

    def pad(self, pad: Tuple[int, ...] = (0, 0), value: float = 0) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
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
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
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
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
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

    @staticmethod
    def cat(tensors: List[Any], dim: int = 0) -> "AbstractTensor":
        """Concatenate ``tensors`` along dimension ``dim``.

        Accepts raw backend tensors or ``AbstractTensor`` instances and
        dispatches to the underlying backend implementation via ``cat_``.
        """
        if not tensors:
            raise ValueError("cat requires at least one tensor")
        first = AbstractTensor.get_tensor(tensors[0])
        tensors = [first.ensure_tensor(t) for t in tensors]
        result = first.__class__(track_time=first.track_time)
        result.data = first.cat_(tensors, dim)
        return result

    class _TopKResult(NamedTuple):
        values: "AbstractTensor"
        indices: Any

    @staticmethod
    def topk(tensor: Any, k: int = 1, dim: int = -1) -> "AbstractTensor._TopKResult":
        """Return the top ``k`` elements and their indices along ``dim``.

        Mirrors ``torch.topk`` by returning a named tuple with ``values`` and
        ``indices`` fields.
        """
        tensor = AbstractTensor.get_tensor(tensor)
        values, idxs = tensor.topk_(k, dim)
        result = tensor.__class__(track_time=tensor.track_time)
        result.data = values
        return AbstractTensor._TopKResult(result, idxs)

    @staticmethod
    def stack(tensors: List[Any], dim: int = 0) -> "AbstractTensor":
        """Stack ``tensors`` along a new dimension ``dim``.

        This is analogous to ``torch.stack`` and can be invoked as a class
        method: ``AbstractTensor.stack([...])``.
        """
        if not tensors:
            raise ValueError("stack requires at least one tensor")
        first = AbstractTensor.get_tensor(tensors[0])
        tensors = [first.ensure_tensor(t) for t in tensors]
        result = first.__class__(track_time=first.track_time)
        result.data = first.stack_(tensors, dim)
        return result

    @staticmethod
    def diag(tensor: Any, offset: int = 0) -> "AbstractTensor":
        """Extract or construct a diagonal.

        Mirrors ``numpy.diag`` behaviour. If ``tensor`` is 1‑D, a square
        matrix is returned with ``tensor`` on the ``offset`` diagonal. If
        ``tensor`` is 2‑D, the specified diagonal is extracted.

        Args:
            tensor: Input array or ``AbstractTensor``.
            offset: Which diagonal to consider. ``0`` selects the main
                diagonal, positive values move up, negative move down.

        Returns:
            ``AbstractTensor`` holding the resulting diagonal data.
        """
        t = AbstractTensor.get_tensor(tensor)
        result = t.__class__(track_time=t.track_time)
        result.data = t.diag_(offset)
        return result


    # --- Broadcasting helpers (abstract, backend-agnostic) ---
    def expand(self, shape: tuple) -> "AbstractTensor":
        """Backend-agnostic expand/broadcast_to (view when possible)."""
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
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
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.view_flat_()
        return result

    def assign_at_indices(
        self,
        indices_dim0: Any = None,
        indices_dim1: Any = None,
        values_to_assign: Any = None,
    ) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.assign_at_indices_(
            indices_dim0, indices_dim1, values_to_assign
        )
        return result

    def increment_at_indices(self, mask: Any = None) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.increment_at_indices_(mask)
        return result

    # Only keep the upper, guarded to_backend (already present above)

    

    def boolean_mask_select(self, mask: Any = None) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.boolean_mask_select_(mask)
        return result

    def tolist(self) -> List[Any]:
        return self.tolist_()


    def index_select(self, dim: int = 0, indices: Any = None) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.index_select_(dim, indices)
        return result

    def argmin(self, dim: Optional[int] = None) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.argmin_(dim)
        return result

    def interpolate(self, size: Tuple[int, ...] = None) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.interpolate_(size)
        return result

    def save(self, filepath: str = None) -> None:
        self.save_(filepath)

    def load(
        self, filepath: str, dtype: Any = None, device: Any = None
    ) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.load_(filepath, dtype, device)
        return result

    def to_dtype(self, dtype: str = "float") -> "AbstractTensor":
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
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
            try:
                from .torch_backend import PyTorchTensorOperations
                torch_ops = AbstractTensor.get_tensor(cls=PyTorchTensorOperations)
                tmp = torch_ops.__class__()
                tmp.data = tensor
                return tmp.to_backend(self)
            except Exception:
                pass
        if np is not None and isinstance(tensor, np.ndarray):
            from .numpy_backend import NumPyTensorOperations
            numpy_ops = AbstractTensor.get_tensor(cls=NumPyTensorOperations)
            numpy_tensor = numpy_ops.__class__()
            numpy_tensor.data = tensor
            return numpy_tensor.to_backend(self)
        if isinstance(tensor, (list, tuple)):
            # Mixed or nested sequences are routed through the nested packer
            if any(isinstance(elem, (list, tuple, AbstractTensor)) for elem in tensor):
                return self.__class__.from_nested(tensor)
            try:
                return self.tensor_from_list(tensor, dtype=None, device=None)
            except Exception:
                # numpy/pure backends may choke on ragged lists; fall back to nested pack
                return self.__class__.from_nested(tensor)
        if hasattr(tensor, "tolist"):
            return self.ensure_tensor(tensor.tolist())
        return self.tensor_from_list([tensor], dtype=None, device=None)

    # --- Operator routing ---
    @staticmethod
    def _pre_autograd(op: str, inputs: Iterable[Any]):
        """Return a callback that records ``op`` on the autograd tape.

        Parameters
        ----------
        op:
            Name of the operator about to be executed.
        inputs:
            Sequence of operands participating in the operation.
        """
        from . import autograd as _autograd

        inputs = list(inputs)
        # Promote non-tensor operands to tensors so backward rules always see wrappers.
        first = next((t for t in inputs if isinstance(t, AbstractTensor)), None)
        if first is not None:
            inputs = [x if isinstance(x, AbstractTensor) else first.ensure_tensor(x) for x in inputs]

        tape = None
        for t in inputs:
            if isinstance(t, AbstractTensor):
                tape = getattr(t, "_tape", None)
                if tape is not None:
                    break
        if tape is None:
            tape = _autograd.autograd.tape

        requires = any(
            getattr(x, "requires_grad", False) for x in inputs if isinstance(x, AbstractTensor)
        )
        track = any(
            getattr(x, "track_time", False) for x in inputs if isinstance(x, AbstractTensor)
        )

        if requires or track:
            start = time.perf_counter() if track else None

            def finalize(result: Any):
                end = time.perf_counter() if track else None
                if requires:
                    try:
                        result._requires_grad = True  # type: ignore[attr-defined]
                    except Exception:
                        pass
                if getattr(AbstractTensor.autograd, "_no_grad_depth", 0) == 0:
                    AbstractTensor.autograd.record(op, inputs, result, start=start, end=end)
                return result

            return finalize

        return lambda result: result

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

        finalize = AbstractTensor._pre_autograd(op, [x for x in (left, right) if x is not None])

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

        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self._apply_operator__(op, l, r)
        return finalize(result)


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
            wrapped = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
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
                
        if isinstance(idx, tuple):
            index = tuple(
                item._AbstractTensor__unwrap() if isinstance(item, AbstractTensor) else item
                for item in idx
            )
        elif isinstance(idx, AbstractTensor):
            index = idx._AbstractTensor__unwrap()
        else:
            index = idx
        data[index] = value

    def __bool__(self):
        try:
            n = int(self.numel())
        except Exception:
            return bool(self.item())
        if n != 1:
            raise ValueError("The truth value of a tensor with more than one element is ambiguous.")
        return bool(self.item())

    def numpy(self):
        """Convert the tensor to a NumPy array."""
        return np.array(self.data)

    @staticmethod
    def benchmark(
        fn: Callable, *args, repeat: int = 1, warmup: int = 0, **kwargs
    ) -> "TapeProfiler":
        """Run ``fn`` repeatedly and profile recorded operations.

        The callable is executed ``warmup`` times without recording to allow the
        runtime to stabilise.  A fresh :class:`GradTape` is then installed on the
        global autograd engine and ``fn`` is executed ``repeat`` additional
        times.  The returned :class:`TapeProfiler` exposes statistics for all
        operations that were recorded on the tape.
        """

        from . import autograd as _autograd

        for _ in range(max(warmup, 0)):
            fn(*args, **kwargs)

        tape = _autograd.GradTape()
        _autograd.autograd.tape = tape
        for _ in range(max(repeat, 0)):
            fn(*args, **kwargs)
        return _autograd.TapeProfiler(tape)

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




# --- Added backend hook methods/properties ---
    def reshape_(self, shape):
        raise NotImplementedError(f"{self.__class__.__name__} must implement reshape_()")

    def transpose_(self, dim0, dim1):
        raise NotImplementedError(f"{self.__class__.__name__} must implement transpose_()")

    def squeeze_(self, dim: int | None = None):
        raise NotImplementedError(f"{self.__class__.__name__} must implement squeeze_()")

    def unravel_index_(self, shape):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement unravel_index_()"
        )


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

            from .torch_backend import PyTorchTensorOperations
            arr = tensor.to_backend(
                AbstractTensor.get_tensor(cls=PyTorchTensorOperations)
            )
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
            from .numpy_backend import NumPyTensorOperations
            arr = tensor.to_backend(
                AbstractTensor.get_tensor(cls=NumPyTensorOperations)
            )
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
    randn,
    randint,
    randint_like,
    empty as create_empty,
)
from .abstraction_methods.reduction import (
    max as reduction_max,
    argmax as reduction_argmax,
    prod as reduction_prod,
)
from .abstraction_methods.indexing import (
    unravel_index as indexing_unravel_index,
)
from .abstraction_methods.type_ops import (
    to as type_to,
    astype as type_astype,
    long_cast as type_long_cast,
    float as type_float,
    double as type_double,
    int as type_int,
    long as type_long,
    bool as type_bool,
    cpu as type_cpu,
    cuda as type_cuda,
)
from .abstraction_methods.comparison import (
    greater as comp_greater,
    greater_equal as comp_greater_equal,
    less as comp_less,
    less_equal as comp_less_equal,
    equal as comp_equal,
    not_equal as comp_not_equal,
    any as comp_any,
    where as comp_where,
    nonzero as comp_nonzero,
    isnan as comp_isnan,
    isinf as comp_isinf,
    all as comp_all,
    isfinite as comp_isfinite,
    allclose as comp_allclose,
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
from .abstraction_methods.fourier import (
    fft as fourier_fft,
    ifft as fourier_ifft,
)
from .abstraction_methods.properties import (
    numel as prop_numel,
    item as prop_item,
    shape as prop_shape,
    shape_ as prop_shape_,
    ndim as prop_ndim,
    dim as prop_dim,
    ndims as prop_ndims,
    device as prop_device,
    dtype as prop_dtype,
    datastring as prop_datastring,
    __str__ as prop_str,
    __format__ as prop_format,
    __repr__ as prop_repr,
    __len__ as prop_len,
)

# --- Autograd method assignments ------------------------------------------
from . import autograd as _autograd_methods
from .backward import BACKWARD_REGISTRY

AbstractTensor.requires_grad_ = _autograd_methods.requires_grad_
AbstractTensor.requires_grad  = _autograd_methods.requires_grad
AbstractTensor.backward       = _autograd_methods.backward
AbstractTensor.grad           = _autograd_methods.grad
AbstractTensor.detach         = _autograd_methods.detach
AbstractTensor.is_leaf        = _autograd_methods.is_leaf
AbstractTensor.retain_grad    = _autograd_methods.retain_grad
AbstractTensor.grad_fn        = _autograd_methods.grad_fn
AbstractTensor.zero_grad      = _autograd_methods.zero_grad

# Register backward algorithms and expose a helper for building pipelines
_BACKWARD = BACKWARD_REGISTRY.register_from_module(_autograd_methods)


def get_backward_tool(ops: Iterable[str]):
    """Return a callable executing backward rules for ``ops`` in order."""
    return _BACKWARD.build(ops)


AbstractTensor.get_backward_tool = staticmethod(get_backward_tool)
AbstractTensor.register_hook  = _autograd_methods.register_hook

# --- Bindings to AbstractTensor -------------------------------------------

AbstractTensor.linspace = staticmethod(linspace)
AbstractTensor.meshgrid = staticmethod(meshgrid)
AbstractTensor.zeros = staticmethod(create_zeros)
AbstractTensor.ones = staticmethod(create_ones)
AbstractTensor.full = staticmethod(create_full)
AbstractTensor.empty = staticmethod(create_empty)
from .abstraction_methods.random import Random as _RandomClass
AbstractTensor.random = _RandomClass()
AbstractTensor.randoms = staticmethod(randoms)
AbstractTensor.randint = staticmethod(randint)
AbstractTensor.unravel_index = staticmethod(indexing_unravel_index)


def _wrap_with_autograd(name: str, func: Callable) -> Callable:
    def wrapped(self, *args, **kwargs):
        tensor_args = [a for a in args if isinstance(a, AbstractTensor)]
        tensor_args += [v for v in kwargs.values() if isinstance(v, AbstractTensor)]
        finalize = AbstractTensor._pre_autograd(name, [self] + tensor_args)
        result = func(self, *args, **kwargs)
        return finalize(result)
    return wrapped


def _bind_and_wrap(mapping: Dict[str, Callable]) -> None:
    for _name, _func in mapping.items():
        setattr(AbstractTensor, _name, _wrap_with_autograd(_name, _func))


_bind_and_wrap({
    "reshape": _reshape_methods.reshape,
    "flatten": _reshape_methods.flatten,
    "transpose": _reshape_methods.transpose,
    "unsqueeze": _reshape_methods.unsqueeze,
    "squeeze": _reshape_methods.squeeze,
    "repeat": _reshape_methods.repeat,
    "repeat_interleave": _reshape_methods.repeat_interleave,
    "zeros_like": create_zeros_like,
    "ones_like": create_ones_like,
    "full_like": create_full_like,
    "rand_like": rand_like,
    "randn": randn,
    "randint_like": randint_like,
    "max": reduction_max,
    "argmax": reduction_argmax,
    "prod": reduction_prod,
    "to": type_to,
    "astype": type_astype,
    "long_cast": type_long_cast,
    "float": type_float,
    "double": type_double,
    "int": type_int,
    "long": type_long,
    "bool": type_bool,
    "cpu": type_cpu,
    "cuda": type_cuda,
    "greater": comp_greater,
    "greater_equal": comp_greater_equal,
    "less": comp_less,
    "less_equal": comp_less_equal,
    "equal": comp_equal,
    "not_equal": comp_not_equal,
    #"where": comp_where, this is now handled by the elementwise set
    "all": comp_all,
    "any": comp_any,
    "nonzero": comp_nonzero,
    "isfinite": comp_isfinite,
    "isnan": comp_isnan,
    "isinf": comp_isinf,
    "allclose": comp_allclose,
    "sin": trig_sin,
    "cos": trig_cos,
    "tan": trig_tan,
    "asin": trig_asin,
    "acos": trig_acos,
    "atan": trig_atan,
    "sinh": trig_sinh,
    "cosh": trig_cosh,
    "tanh": trig_tanh,
    "asinh": trig_asinh,
    "acosh": trig_acosh,
    "atanh": trig_atanh,
    "sec": trig_sec,
    "csc": trig_csc,
    "cot": trig_cot,
    "sech": trig_sech,
    "csch": trig_csch,
    "coth": trig_coth,
    "sinc": trig_sinc,
    "fft": fourier_fft,
    "ifft": fourier_ifft,
})

AbstractTensor.numel   = prop_numel
AbstractTensor.item    = prop_item
AbstractTensor.shape   = property(prop_shape)
AbstractTensor.shape_  = prop_shape_
AbstractTensor.ndim    = property(prop_ndim)
AbstractTensor.dim     = prop_dim
AbstractTensor.ndims   = prop_ndims
AbstractTensor.device  = property(prop_device)
AbstractTensor.dtype   = property(prop_dtype)
AbstractTensor.datastring = prop_datastring
AbstractTensor.__str__    = prop_str
AbstractTensor.__format__ = prop_format
AbstractTensor.__repr__   = prop_repr
AbstractTensor.__len__    = prop_len

from .linalg import (
    dot as linalg_dot,
    norm as linalg_norm,
    cross as linalg_cross,
    trace as linalg_trace,
    det as linalg_det,
    solve as linalg_solve,
    inv as linalg_inv,
    eye as linalg_eye,
)

class _LinalgNS:
    pass

if not hasattr(AbstractTensor, "linalg"):
    AbstractTensor.linalg = _LinalgNS()

# namespace (numpy-like)
AbstractTensor.linalg.dot   = staticmethod(linalg_dot)
AbstractTensor.linalg.norm  = staticmethod(linalg_norm)
AbstractTensor.linalg.cross = staticmethod(linalg_cross)
AbstractTensor.linalg.trace = staticmethod(linalg_trace)
AbstractTensor.linalg.det   = staticmethod(linalg_det)
AbstractTensor.linalg.solve = staticmethod(linalg_solve)
AbstractTensor.linalg.inv   = staticmethod(linalg_inv)
AbstractTensor.linalg.eye   = staticmethod(linalg_eye)

# optional top-level shorthands used by your code already
AbstractTensor.dot   = staticmethod(linalg_dot)
AbstractTensor.norm  = staticmethod(linalg_norm)
AbstractTensor.cross = staticmethod(linalg_cross)
AbstractTensor.trace = staticmethod(linalg_trace)
AbstractTensor.det   = staticmethod(linalg_det)
AbstractTensor.solve = staticmethod(linalg_solve)
AbstractTensor.inv   = staticmethod(linalg_inv)
AbstractTensor.eye   = staticmethod(linalg_eye)
AbstractTensor.inverse = staticmethod(linalg_inv)

def _cbrt(x):
    import numpy as np
    if isinstance(x, AbstractTensor):
        result = type(x)(track_time=x.track_time)
        result.data = np.cbrt(x.data)
        return result
    return np.cbrt(x)

AbstractTensor.cbrt = staticmethod(_cbrt)

def _einsum(equation: str, *tensors: "AbstractTensor") -> "AbstractTensor":
    cls = type(tensors[0])
    data = [t.data if isinstance(t, AbstractTensor) else t for t in tensors]
    result = cls(track_time=False)
    result.data = cls.einsum_(equation, *data)
    return result

AbstractTensor.einsum = staticmethod(_einsum)

def _sparse_coo_tensor(indices, values, size):
    from .coo_matrix import COOMatrix
    return COOMatrix(indices, values, size)

AbstractTensor.sparse_coo_tensor = staticmethod(_sparse_coo_tensor)


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
from .abstraction_methods.elementwise import (
    __eq__ as elementwise_eq,
    __ne__ as elementwise_ne,
    __lt__ as elementwise_lt,
    __le__ as elementwise_le,
    __gt__ as elementwise_gt,
    __ge__ as elementwise_ge,
    __and__ as elementwise_and,
    __or__ as elementwise_or,
    __xor__ as elementwise_xor,
    __invert__ as elementwise_invert,
    where as elementwise_where,
    _as_scalar, _scalar_kernel, 
    _v1_valuewise, _v2_valuewise, _v3_valuewise
)

# --- Elementwise operator assignments (from abstraction_methods/elementwise.py) ---
AbstractTensor.__eq__    = elementwise_eq
AbstractTensor.__ne__    = elementwise_ne
AbstractTensor.__lt__    = elementwise_lt
AbstractTensor.__le__    = elementwise_le
AbstractTensor.__gt__    = elementwise_gt
AbstractTensor.__ge__    = elementwise_ge
AbstractTensor.__and__   = elementwise_and
AbstractTensor.__or__    = elementwise_or
AbstractTensor.__xor__   = elementwise_xor
AbstractTensor.__invert__= elementwise_invert
AbstractTensor.where     = staticmethod(elementwise_where)

AbstractTensor._as_scalar   = staticmethod(_as_scalar)
AbstractTensor._scalar_kernel = staticmethod(_scalar_kernel)
AbstractTensor._v1_valuewise  = _v1_valuewise
AbstractTensor._v2_valuewise  = _v2_valuewise
AbstractTensor._v3_valuewise  = _v3_valuewise
