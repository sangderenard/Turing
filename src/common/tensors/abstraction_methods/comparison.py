
from __future__ import annotations

from typing import Any

def where(self, x: Any, y: Any) -> "AbstractTensor":
    """Elementwise select: self as bool mask, x if True else y."""
    from ..abstraction import AbstractTensor
    if not isinstance(self, AbstractTensor):
        self = AbstractTensor.tensor(self)
    result = type(self)(track_time=self.track_time)
    result.data = self.where_(x, y)
    return result
def argwhere(self) -> "AbstractTensor":
    """Return the indices where condition is True. Like np.argwhere, always returns a 2D array of indices."""
    from ..abstraction import AbstractTensor
    if not isinstance(self, AbstractTensor):
        self = AbstractTensor.tensor(self)
    result = type(self)(track_time=self.track_time)
    result.data = self.argwhere_()
    return result

def all(self, dim=None) -> Any:
    """Return True if all elements of the tensor are True."""
    from ..abstraction import AbstractTensor
    if not isinstance(self, AbstractTensor):
        self = AbstractTensor.tensor(self)
    result = type(self)(track_time=self.track_time)
    result.data = self.all_(dim)
    return result

def any(self, dim=None) -> Any:
    """Return True if any element of the tensor is True."""
    from ..abstraction import AbstractTensor
    if not isinstance(self, AbstractTensor):
        self = AbstractTensor.tensor(self)
    result = type(self)(track_time=self.track_time)
    result.data = self.any_(dim)
    return result

def isnan(self) -> "AbstractTensor":
    from ..abstraction import AbstractTensor
    if not isinstance(self, AbstractTensor):
        self = AbstractTensor.tensor(self)
    result = type(self)(track_time=self.track_time)
    result.data = self.isnan_()
    return result

def isfinite(self) -> "AbstractTensor":
    from ..abstraction import AbstractTensor
    if not isinstance(self, AbstractTensor):
        self = AbstractTensor.tensor(self)
    result = type(self)(track_time=self.track_time)
    result.data = self.isfinite_()
    return result

def isinf(self) -> "AbstractTensor":
    from ..abstraction import AbstractTensor
    if not isinstance(self, AbstractTensor):
        self = AbstractTensor.tensor(self)
    result = type(self)(track_time=self.track_time)
    result.data = self.isinf_()
    return result

def greater(self, value: Any) -> "AbstractTensor":
    from ..abstraction import AbstractTensor
    if not isinstance(self, AbstractTensor):
        self = AbstractTensor.tensor(self)
    result = type(self)(track_time=self.track_time)
    result.data = self.greater_(value)
    return result


def greater_equal(self, value: Any) -> "AbstractTensor":
    from ..abstraction import AbstractTensor
    if not isinstance(self, AbstractTensor):
        self = AbstractTensor.tensor(self)
    result = type(self)(track_time=self.track_time)
    result.data = self.greater_equal_(value)
    return result


def less(self, value: Any) -> "AbstractTensor":
    from ..abstraction import AbstractTensor
    if not isinstance(self, AbstractTensor):
        self = AbstractTensor.tensor(self)
    result = type(self)(track_time=self.track_time)
    result.data = self.less_(value)
    return result


def less_equal(self, value: Any) -> "AbstractTensor":
    from ..abstraction import AbstractTensor
    if not isinstance(self, AbstractTensor):
        self = AbstractTensor.tensor(self)
    result = type(self)(track_time=self.track_time)
    result.data = self.less_equal_(value)
    return result


def equal(self, value: Any) -> "AbstractTensor":
    from ..abstraction import AbstractTensor
    if not isinstance(self, AbstractTensor):
        self = AbstractTensor.tensor(self)
    result = type(self)(track_time=self.track_time)
    result.data = self.equal_(value)
    return result


def not_equal(self, value: Any) -> "AbstractTensor":
    from ..abstraction import AbstractTensor
    if not isinstance(self, AbstractTensor):
        self = AbstractTensor.tensor(self)
    result = type(self)(track_time=self.track_time)
    result.data = self.not_equal_(value)
    return result
def allclose(self, other, rtol=1e-5, atol=1e-8, equal_nan=False):
    """Return True if all elements of self and other are close within tolerance."""
    from ..abstraction import AbstractTensor
    if not isinstance(self, AbstractTensor):
        self = AbstractTensor.tensor(self)
    if not isinstance(other, AbstractTensor):
        other = AbstractTensor.tensor(other)
    return self.allclose_(other, rtol=rtol, atol=atol, equal_nan=equal_nan)

def nonzero(self, as_tuple: bool = False):
    """Return indices of nonzero elements. as_tuple matches numpy/torch API."""
    from ..abstraction import AbstractTensor
    if not isinstance(self, AbstractTensor):
        self = AbstractTensor.tensor(self)
    return self.nonzero_(as_tuple=as_tuple)
