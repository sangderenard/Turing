"""Abstraction layer for tensor operations."""
from __future__ import annotations

try:
    from abc import ABC, abstractmethod
    from typing import Any, Tuple, Optional, List, Union, Callable, Dict, Deque
    import math
    import time
    from collections import deque

    from .faculty import Faculty, DEFAULT_FACULTY
    DEFAULT_DEVICE = "cpu"
    import torch
    import numpy as np
    from .accelerator_backends.c_backend import CTensor
except ModuleNotFoundError:
    torch = None  # type: ignore
    np = None  # type: ignore
    CTensor = None  # type: ignore
except Exception:
    print("Failed to import required modules for tensor operations.")
    import sys

    sys.exit(1)

CONVERSION_REGISTRY: Dict[
    Tuple[type, type], Callable[["AbstractTensor", Any, "AbstractTensor"], Any]
] = {}

OPS_CACHE: Dict[type, "AbstractTensor"] = {}

DEBUG = False  # True


class ShapeAccessor:
    """Proxy object allowing both ``tensor.shape`` and ``tensor.shape()``."""

    def __init__(self, owner: "AbstractTensor") -> None:
        self.owner = owner

    def __call__(self) -> Tuple[int, ...]:
        """Return the shape of ``owner`` as a tuple."""
        return self.owner.shape_()

    def __iter__(self):  # type: ignore[override]
        return iter(self())

    def __len__(self) -> int:  # type: ignore[override]
        return len(self())

    def __getitem__(self, idx: int) -> int:  # type: ignore[override]
        return self()[idx]

    def __repr__(self) -> str:  # type: ignore[override]
        return repr(self())


def register_conversion(
    src_cls: type,
    tgt_cls: type,
    func: Callable[["AbstractTensor", Any, "AbstractTensor"], Any],
) -> None:
    """Register a direct tensor conversion function."""
    CONVERSION_REGISTRY[(src_cls, tgt_cls)] = func


def _get_ops_for_class(cls: type) -> "AbstractTensor":
    if cls in OPS_CACHE:
        return OPS_CACHE[cls]
    if cls.__name__.startswith("PyTorch"):
        ops = AbstractTensor.get_tensor(faculty=Faculty.TORCH)
    elif cls.__name__.startswith("NumPy"):
        ops = AbstractTensor.get_tensor(faculty=Faculty.NUMPY)
    elif cls.__name__.startswith("PurePython"):
        ops = AbstractTensor.get_tensor(faculty=Faculty.PURE_PYTHON)
    elif cls.__name__.startswith("JAX"):
        ops = AbstractTensor.get_tensor(faculty=Faculty.NUMPY)  # approximate
    else:
        ops = AbstractTensor.get_tensor()
    OPS_CACHE[cls] = ops
    return ops


def _find_conversion_path(src_cls: type, tgt_cls: type) -> List[Tuple[type, type]]:
    if src_cls == tgt_cls:
        return []
    q: Deque[Tuple[type, List[Tuple[type, type]]]] = deque([(src_cls, [])])
    seen = {src_cls}
    while q:
        cur, path = q.popleft()
        if cur == tgt_cls:
            return path
        for (a, b), _ in CONVERSION_REGISTRY.items():
            if a == cur and b not in seen:
                q.append((b, path + [(a, b)]))
                seen.add(b)
    return []


class AbstractTensor(ABC):
    def __init__(self, track_time: bool = False) -> None:
        """Optional benchmark support for tensor operations."""
        self.track_time = track_time
        self.last_op_time: float | None = None
        self.data = None  # Holds the tensor's data

    def _AbstractTensor__wrap(self, tensor):
        """Default wrap: should be overridden by backend. Wraps a backend-native tensor."""
        obj = type(self)()
        obj.data = tensor
        return obj

    # --------------------------------------------------------------
    # Internal helpers
    def __unwrap(self) -> Any:
        """Internal helper to access the raw tensor value."""
        return self.data

    def benchmark(self, call: "Callable[[], Any]") -> Any:
        """Run ``call`` and store elapsed time if benchmarking is enabled."""
        if self.track_time:
            start = time.process_time()
            result = call()
            self.last_op_time = time.process_time() - start
            return result
        return call()

    # --- Tensor creation and manipulation methods ---
    @classmethod
    def full(
        cls,
        size: Tuple[int, ...],
        fill_value: Any,
        dtype: Any = None,
        device: Any = None,
    ) -> "AbstractTensor":
        ops = _get_ops_for_class(cls)
        result = cls(track_time=getattr(ops, "track_time", False))
        result.data = ops.full_(size, fill_value, dtype, device)
        return result

    def full(self, size: Tuple[int, ...], fill_value: Any, dtype: Any = None, device: Any = None):
        return type(self).full(size, fill_value, dtype, device)

    @classmethod
    def zeros(
        cls, size: Tuple[int, ...], dtype: Any = None, device: Any = None
    ) -> "AbstractTensor":
        ops = _get_ops_for_class(cls)
        result = cls(track_time=getattr(ops, "track_time", False))
        result.data = ops.zeros_(size, dtype, device)
        return result

    def zeros(self, size: Tuple[int, ...], dtype: Any = None, device: Any = None):
        return type(self).zeros(size, dtype, device)

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

    def item(self) -> Union[int, float, bool]:
        return self.item_()

    def max(self) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time)
        result.data = self.max_()
        return result

    def long_cast(self) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time)
        result.data = self.long_cast_()
        return result

    def float(self) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time)
        result.data = self.float_()
        return result

    def double(self) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time)
        result.data = self.double_()
        return result

    def int(self) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time)
        result.data = self.int_()
        return result

    def long(self) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time)
        result.data = self.long_()
        return result

    def bool(self) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time)
        result.data = self.bool_()
        return result

    def not_equal(self, other: Any) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time)
        result.data = self.not_equal_(other)
        return result

    @classmethod
    def arange(
        cls,
        start: int,
        end: Optional[int] = None,
        step: int = 1,
        device: Any = None,
        dtype: Any = None,
    ) -> "AbstractTensor":
        ops = _get_ops_for_class(cls)
        result = cls(track_time=getattr(ops, "track_time", False))
        result.data = ops.arange_(start, end, step, device, dtype)
        return result

    def arange(
        self,
        start: int,
        end: Optional[int] = None,
        step: int = 1,
        device: Any = None,
        dtype: Any = None,
    ) -> "AbstractTensor":
        return type(self).arange(start, end, step, device, dtype)

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

    def repeat_interleave(
        self, repeats: int = 1, dim: Optional[int] = None
    ) -> "AbstractTensor":
        result = AbstractTensor.get_tensor(self.repeat_interleave_(repeats, dim))
        return result

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

    def clamp(
        self, min_val: Optional[float] = None, max_val: Optional[float] = None
    ) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time)
        result.data = self.clamp_(min_val, max_val)
        return result

    def numel(self) -> int:
        return self.numel_()

    def mean(self, dim: Optional[int] = None) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time)
        result.data = self.mean_(dim)
        return result

    def pow(self, exponent: float = 1.0) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time)
        result.data = self.pow_(exponent)
        return result

    def sqrt(self) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time)
        result.data = self.sqrt_()
        return result

    @classmethod
    def tensor_from_list(
        cls, data: List[Any], dtype: Any = None, device: Any = None
    ) -> "AbstractTensor":
        ops = _get_ops_for_class(cls)
        result = cls(track_time=getattr(ops, "track_time", False))
        result.data = ops.tensor_from_list_(data, dtype, device)
        return result

    def tensor_from_list(
        self, data: List[Any], dtype: Any = None, device: Any = None
    ) -> "AbstractTensor":
        return type(self).tensor_from_list(data, dtype, device)

    def boolean_mask_select(self, mask: Any = None) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time)
        result.data = self.boolean_mask_select_(mask)
        return result

    def tolist(self) -> List[Any]:
        return self.tolist_()

    def less(self, value: Any = None) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time)
        result.data = self.less_(value)
        return result

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

    # --- Shape and dimension accessors (overloads) ---
    @property
    def shape(self) -> ShapeAccessor:
        """Return a callable/iterable shape accessor."""
        return ShapeAccessor(self)

    def shape_(self) -> Tuple[int, ...]:
        """Return the shape of the tensor as a tuple (backend hook)."""
        return self.get_shape()

    @property
    def ndim(self):
        """Return the number of dimensions (property, numpy style)."""
        return self.get_ndims(self.data)

    def dim(self) -> int:
        """Return the number of dimensions (method, torch style)."""
        return self.get_ndims(self.data)

    def ndims(self) -> int:
        """Return the number of dimensions (method, project style)."""
        return self.get_ndims(self.data)

    # Lightweight helper to coerce arbitrary input to this backend's tensor type
    def to_backend(
        self,
        target_ops: "AbstractTensor",
    ) -> "AbstractTensor":
        """Return ``self`` converted to ``target_ops`` backend."""

        if not isinstance(target_ops, AbstractTensor):
            raise TypeError("target_ops must be an AbstractTensor instance")

        if type(self) is type(target_ops):
            result = type(target_ops)(track_time=self.track_time)
            result.data = self.clone_()
            return result

        conv_func = CONVERSION_REGISTRY.get((type(self), type(target_ops)))
        if conv_func is None:
            converted = default_to_backend(self, self, target_ops)
        else:
            converted = conv_func(self, self, target_ops)

        if isinstance(converted, AbstractTensor):
            return converted.to_backend(target_ops)

        new_tensor = type(target_ops)(track_time=self.track_time)
        new_tensor.data = converted
        return new_tensor

    def ensure_tensor(self, tensor: Any) -> "AbstractTensor":
        """Return ``tensor`` wrapped as an ``AbstractTensor`` instance."""
        if tensor is None:
            raise ValueError("ensure_tensor called with tensor=None")
        if isinstance(tensor, AbstractTensor):
            return tensor.to_backend(self)
        if isinstance(tensor, self.tensor_type):
            result = type(self)(track_time=self.track_time)
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
        if isinstance(tensor, list):
            return self.tensor_from_list(tensor, dtype=None, device=None)
        if hasattr(tensor, "tolist"):
            return self.tensor_from_list(tensor.tolist(), dtype=None, device=None)
        return self.tensor_from_list([tensor], dtype=None, device=None)

    # --- Operator routing ---
    def _apply_operator(self, op: str, left: Any, right: Any):
        """Apply ``op`` to ``left`` and ``right`` returning a new tensor."""
        l = left._AbstractTensor__unwrap() if isinstance(left, AbstractTensor) else left
        r = (
            right._AbstractTensor__unwrap()
            if isinstance(right, AbstractTensor)
            else right
        )

        result = type(self)(track_time=self.track_time)

        result.data = self._apply_operator__(op, l, r)
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

    def __str__(self):
        # Unified print: show the underlying tensor's string representation
        return self.datastring(self.data)

    def datastring(self, data: Any) -> str:
        """Return a pretty string representation of ``data`` for console output."""

        if data is None:
            return "AbstractTensor (None)"

        try:
            shape = self.get_shape()
        except Exception:
            shape = ()

        try:
            dtype = self.get_dtype(data)
        except Exception:
            dtype = getattr(data, "dtype", None)

        try:
            device = self.get_device(data)
        except Exception:
            device = getattr(data, "device", None)

        header = f"shape={shape} dtype={dtype} device={device}"

        # Attempt color support via colorama
        try:
            from colorama import Fore, Style
        except Exception:  # pragma: no cover - optional dependency

            class _NoColor:
                RED = BLUE = CYAN = YELLOW = GREEN = MAGENTA = WHITE = RESET_ALL = ""

            Fore = Style = _NoColor()  # type: ignore

        if hasattr(data, "tolist"):
            values = data.tolist()
        else:
            values = data

        if not isinstance(values, list):
            values = [values]

        if shape and len(shape) == 1:
            values = [values]

        rows = len(values)
        cols = len(values[0]) if rows and isinstance(values[0], list) else 1

        flat_vals = [
            float(x)
            for row in values
            for x in (row if isinstance(row, list) else [row])
            if isinstance(x, (int, float))
        ]
        if flat_vals:
            min_val, max_val = min(flat_vals), max(flat_vals)
            spread = max_val - min_val or 1.0
        else:
            min_val, max_val, spread = 0.0, 0.0, 1.0

        def colorize(v: Any) -> str:
            if not isinstance(v, (int, float)):
                return str(v)
            norm = (float(v) - min_val) / spread
            palette = [Fore.BLUE, Fore.CYAN, Fore.GREEN, Fore.YELLOW, Fore.RED]
            idx = int(norm * (len(palette) - 1))
            return f"{palette[idx]}{v:.4e}{Style.RESET_ALL}"

        cell_w = 10
        col_cap = 6
        row_cap = 10
        lines = []
        border = "+" + "+".join(["-" * cell_w] * min(cols, col_cap)) + "+"
        lines.append(border)
        for r in range(min(rows, row_cap)):
            row = values[r] if isinstance(values[r], list) else [values[r]]
            cells = []
            for c in range(min(cols, col_cap)):
                if c < len(row):
                    cell = colorize(row[c]).ljust(cell_w)
                else:
                    cell = "".ljust(cell_w)
                cells.append(cell)
            lines.append("|" + "|".join(cells) + "|")
        if rows > row_cap or cols > col_cap:
            ell = "...".center(cell_w)
            lines.append("|" + "|".join([ell] * min(cols, col_cap)) + "|")
        lines.append(border)

        table = "\n".join(lines)
        return f"\n\n{header}\n{table}\n\n"

    def __repr__(self):
        # Unified repr: AbstractTensor (BackendClass (backend data repr))
        backend_class = (
            type(self.data).__name__ if self.data is not None else "NoneType"
        )
        backend_data_repr = repr(self.data)
        return f"AbstractTensor ({backend_class} ({backend_data_repr}))"

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

    def __len__(self):
        """Return the length of the underlying tensor along the first dimension."""
        if DEBUG:
            print(f"__len__ called on {self.__class__.__name__}")
        data = self.data
        # Removed unconditional print: print(f"{type(data)}, {data.shape if hasattr(data, 'shape') else 'no shape'}")

        if data is None:
            raise ValueError("__len__ called on empty tensor")
        return len(data)

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
        data=None, faculty: "Faculty" = None, *, track_time: bool = False
    ) -> "AbstractTensor":
        """
        Create and return an AbstractTensor instance from any data, auto-selecting the best backend if faculty is None.
        If faculty is provided, use the corresponding backend.
        """
        faculty = faculty or DEFAULT_FACULTY
        if faculty in (Faculty.TORCH, Faculty.PYGEO):
            from .torch_backend import PyTorchTensorOperations

            tensor = PyTorchTensorOperations(
                default_device=DEFAULT_DEVICE, track_time=track_time
            )
        elif faculty is Faculty.NUMPY and np is not None:
            from .numpy_backend import NumPyTensorOperations

            tensor = NumPyTensorOperations(track_time=track_time)
        elif faculty is Faculty.CTENSOR:
            from .c_backend import CTensorOperations

            tensor = CTensorOperations(track_time=track_time)
        else:
            from .pure_backend import PurePythonTensorOperations

            tensor = PurePythonTensorOperations(track_time=track_time)
        if data is not None:
            return tensor.ensure_tensor(data)
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
            # Now data is (N, C, H, W)
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


def default_to_backend(
    source_ops: "AbstractTensor", tensor: Any, target_ops: "AbstractTensor"
) -> Any:
    """Fallback conversion using :meth:`tolist` and :meth:`tensor_from_list`."""
    if type(source_ops) is type(target_ops):
        return source_ops.clone(tensor)
    data = source_ops.tolist(tensor)
    dtype = None
    device = None
    try:
        dtype = source_ops.get_dtype(tensor)
    except Exception:
        pass
    try:
        device = source_ops.get_device(tensor)
    except Exception:
        pass
    return target_ops.tensor_from_list(data, dtype=dtype, device=device)


def get_tensor_operations(
    faculty: Faculty | None = None, *, track_time: bool = False
) -> AbstractTensor:
    """[REMOVED] Use AbstractTensor.get_tensor instead."""
    raise RuntimeError(
        "get_tensor_operations is obsolete. Use AbstractTensor.get_tensor instead."
    )


# --- Conversion registration ---
try:
    from .torch_backend import PyTorchTensorOperations
except Exception:  # pragma: no cover - optional backend
    PyTorchTensorOperations = None  # type: ignore
try:
    from .numpy_backend import NumPyTensorOperations
except Exception:  # pragma: no cover - optional backend
    NumPyTensorOperations = None  # type: ignore
from .pure_backend import PurePythonTensorOperations

if np is not None and NumPyTensorOperations is not None:
    register_conversion(
        NumPyTensorOperations,
        PurePythonTensorOperations,
        PurePythonTensorOperations.from_numpy,
    )
    register_conversion(
        PurePythonTensorOperations,
        NumPyTensorOperations,
        NumPyTensorOperations.from_pure,
    )

if (
    torch is not None
    and PyTorchTensorOperations is not None
    and NumPyTensorOperations is not None
):
    register_conversion(
        PyTorchTensorOperations, NumPyTensorOperations, NumPyTensorOperations.from_torch
    )
    register_conversion(
        NumPyTensorOperations,
        PyTorchTensorOperations,
        PyTorchTensorOperations.from_numpy,
    )
    register_conversion(
        PyTorchTensorOperations,
        PurePythonTensorOperations,
        PurePythonTensorOperations.from_torch,
    )
    register_conversion(
        PurePythonTensorOperations,
        PyTorchTensorOperations,
        PyTorchTensorOperations.from_pure,
    )

try:
    from .jax_backend import JAXTensorOperations
except Exception:  # pragma: no cover - optional backend
    JAXTensorOperations = None  # type: ignore

if (
    np is not None
    and NumPyTensorOperations is not None
    and JAXTensorOperations is not None
):
    register_conversion(
        JAXTensorOperations, NumPyTensorOperations, NumPyTensorOperations.from_jax
    )
    register_conversion(
        NumPyTensorOperations, JAXTensorOperations, JAXTensorOperations.from_numpy
    )

if (
    torch is not None
    and PyTorchTensorOperations is not None
    and JAXTensorOperations is not None
):
    register_conversion(
        PyTorchTensorOperations, JAXTensorOperations, JAXTensorOperations.from_torch
    )
    register_conversion(
        JAXTensorOperations, PyTorchTensorOperations, PyTorchTensorOperations.from_jax
    )

if JAXTensorOperations is not None:
    register_conversion(
        JAXTensorOperations,
        PurePythonTensorOperations,
        PurePythonTensorOperations.from_jax,
    )
    register_conversion(
        PurePythonTensorOperations, JAXTensorOperations, JAXTensorOperations.from_pure
    )
