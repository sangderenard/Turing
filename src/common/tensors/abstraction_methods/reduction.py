from __future__ import annotations

from typing import Optional

from ..abstraction import AbstractTensor


def max(self, dim=None, keepdim: bool = False):
    """Return the maximum of the tensor along the specified dimension(s)."""
    finalize = AbstractTensor._pre_autograd(
        "max", [self], params={"axis": dim, "keepdim": keepdim}
    )
    result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
    result.data = self.max_(dim=dim, keepdim=keepdim)
    return finalize(result)


def argmax(self, dim: Optional[int] = None, keepdim: bool = False):
    """Return the indices of the maximum values along an axis."""
    finalize = AbstractTensor._pre_autograd(
        "argmax", [self], params={"axis": dim, "keepdim": keepdim}
    )
    result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
    result.data = self.argmax_(dim, keepdim)
    return finalize(result)


def prod(self, dim=None, keepdim: bool = False):
    """Return the product of tensor elements along a dimension."""
    finalize = AbstractTensor._pre_autograd(
        "prod", [self], params={"axis": dim, "keepdim": keepdim}
    )
    result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
    result.data = self.prod_(dim=dim, keepdim=keepdim)
    return finalize(result)
