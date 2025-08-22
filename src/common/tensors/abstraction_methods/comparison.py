from __future__ import annotations

from typing import Any

def where(self, x: Any, y: Any) -> "AbstractTensor":
    """Elementwise select: self as bool mask, x if True else y."""
    result = type(self)(track_time=self.track_time)
    result.data = self.where_(x, y)
    return result

def any(self) -> Any:
    """Return True if any element of the tensor is True."""
    return self.any_()

def greater(self, value: Any) -> "AbstractTensor":
    result = type(self)(track_time=self.track_time)
    result.data = self.greater_(value)
    return result


def greater_equal(self, value: Any) -> "AbstractTensor":
    result = type(self)(track_time=self.track_time)
    result.data = self.greater_equal_(value)
    return result


def less(self, value: Any) -> "AbstractTensor":
    result = type(self)(track_time=self.track_time)
    result.data = self.less_(value)
    return result


def less_equal(self, value: Any) -> "AbstractTensor":
    result = type(self)(track_time=self.track_time)
    result.data = self.less_equal_(value)
    return result


def equal(self, value: Any) -> "AbstractTensor":
    result = type(self)(track_time=self.track_time)
    result.data = self.equal_(value)
    return result


def not_equal(self, value: Any) -> "AbstractTensor":
    result = type(self)(track_time=self.track_time)
    result.data = self.not_equal_(value)
    return result

def nonzero(self, as_tuple: bool = False):
    """Return indices of nonzero elements. as_tuple matches numpy/torch API."""
    return self.nonzero_(as_tuple=as_tuple)
