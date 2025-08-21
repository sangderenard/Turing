from __future__ import annotations

from typing import Any


def to(self, *args, **kwargs):
    """Mimic ``torch.Tensor.to`` by accepting device and dtype arguments.

    The first positional argument is interpreted as either a device or a dtype.
    Unknown targets are ignored so non-device backends continue to operate.
    """
    device = kwargs.get("device")
    dtype = kwargs.get("dtype")

    if args:
        first = args[0]
        if isinstance(first, str) or hasattr(first, "type"):
            device = first
        else:
            dtype = first
        if len(args) > 1:
            second = args[1]
            if device is None:
                device = second
            else:
                dtype = second

    result = self
    if device is not None:
        try:
            result = result.to_device(device)
        except Exception:
            pass
    if dtype is not None:
        try:
            result = result.to_dtype(dtype)
        except Exception:
            pass
    return result


def astype(self, dtype):
    """Redirect to to_dtype for compatibility with backend-style dtype conversion."""
    return self.to_dtype(dtype)


def where(self, x: Any, y: Any) -> "AbstractTensor":
    """Elementwise select: self as bool mask, x if True else y."""
    result = type(self)(track_time=self.track_time)
    result.data = self.where_(x, y)
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


def cpu(self) -> "AbstractTensor":
    try:
        return self.to_device("cpu")
    except Exception:
        return self


def cuda(self) -> "AbstractTensor":
    try:
        return self.to_device("cuda")
    except Exception:
        return self
