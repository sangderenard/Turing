from __future__ import annotations

from typing import Any


def to(self, dtype):
    """Redirect to to_dtype for compatibility with backend-style dtype conversion."""
    return self.to_dtype(dtype)


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
