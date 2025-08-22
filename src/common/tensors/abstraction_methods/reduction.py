from __future__ import annotations

from typing import Optional


def max(self, dim=None, keepdim: bool = False):
    """Return the maximum of the tensor along the specified dimension(s)."""
    return self.max_(dim=dim, keepdim=keepdim)


def argmax(self, dim: Optional[int] = None, keepdim: bool = False):
    """Return the indices of the maximum values along an axis."""
    return self.argmax_(dim, keepdim)


def prod(self, dim=None, keepdim: bool = False):
    """Return the product of tensor elements along a dimension."""
    return self.prod_(dim=dim, keepdim=keepdim)
