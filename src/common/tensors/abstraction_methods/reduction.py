from __future__ import annotations

from typing import Optional

from ..abstraction import AbstractTensor, AbstractScalar


def _wrap_scalar(result):
    if getattr(result.data, "shape", ()) == ():
        return AbstractScalar.__new__(AbstractScalar, result)
    return result



def max(
    self,
    dim=None,
    keepdim: bool = False,
    *,
    axis=None,
    dtype=None,
    out=None,
    keepdims=None,
):
    """Return the maximum of the tensor along the specified dimension(s)."""
    source, dim, keepdim = self._normalize_reduction_call(
        "max", dim, keepdim, axis=axis, dtype=dtype, out=out, keepdims=keepdims
    )
    finalize = AbstractTensor._pre_autograd(
        "max", [source], params={"axis": dim, "keepdim": keepdim}
    )
    result = type(source)(track_time=source.track_time, tape=getattr(source, "_tape", None))
    result.data = source.max_(dim=dim, keepdim=keepdim)
    result = finalize(result)
    return _wrap_scalar(result)


def argmax(self, dim: Optional[int] = None, keepdim: bool = False):
    """Return the indices of the maximum values along an axis."""
    finalize = AbstractTensor._pre_autograd(
        "argmax", [self], params={"axis": dim, "keepdim": keepdim}
    )
    result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
    result.data = self.argmax_(dim, keepdim)
    result = finalize(result)
    return _wrap_scalar(result)


def argmin(self, dim: Optional[int] = None, keepdim: bool = False):
    """Return the indices of the minimum values along an axis."""
    finalize = AbstractTensor._pre_autograd(
        "argmin", [self], params={"axis": dim, "keepdim": keepdim}
    )
    result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
    result.data = self.argmin_(dim, keepdim)
    result = finalize(result)
    return _wrap_scalar(result)


def prod(
    self,
    dim=None,
    keepdim: bool = False,
    *,
    axis=None,
    dtype=None,
    out=None,
    keepdims=None,
):
    """Return the product of tensor elements along a dimension."""
    source, dim, keepdim = self._normalize_reduction_call(
        "prod", dim, keepdim, axis=axis, dtype=dtype, out=out, keepdims=keepdims
    )
    finalize = AbstractTensor._pre_autograd(
        "prod", [source], params={"axis": dim, "keepdim": keepdim}
    )
    result = type(source)(track_time=source.track_time, tape=getattr(source, "_tape", None))
    result.data = source.prod_(dim=dim, keepdim=keepdim)
    result = finalize(result)
    return _wrap_scalar(result)
