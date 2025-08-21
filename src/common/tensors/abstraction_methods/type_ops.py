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
