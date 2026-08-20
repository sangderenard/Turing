"""AbstractTensor entry points for compiler-owned BLAS semantics.

This module is the explicit seam between the public tensor operation and the
plain compiler source in :mod:`src.common.tensors.blas`.  It deliberately does
not implement a second matrix multiply: eager execution still travels through
the selected AbstractTensor backend, while extraction can identify the exact
``gemm`` semantic material from which compiled backend variants are derived.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..blas import GEMM_SOURCE

if TYPE_CHECKING:
    from ..abstraction import AbstractTensor


MATMUL_BLAS_SEMANTICS = {
    "library": "src.common.tensors.blas",
    "kernel": "gemm",
    "source_symbol": "GEMM_SOURCE",
    "intrinsic_family": "blas.gemm",
    "alpha": 1.0,
    "beta": 0.0,
}


def matmul(tensor: "AbstractTensor", other: Any) -> "AbstractTensor":
    """Dispatch ``tensor @ other`` with canonical ``gemm(alpha=1,beta=0)`` semantics."""

    return tensor._apply_operator("matmul", tensor, other)


def rmatmul(tensor: "AbstractTensor", other: Any) -> "AbstractTensor":
    """Dispatch ``other @ tensor`` with canonical ``gemm(alpha=1,beta=0)`` semantics."""

    return tensor._apply_operator("matmul", other, tensor)


def imatmul(tensor: "AbstractTensor", other: Any) -> "AbstractTensor":
    """Dispatch ``tensor @= other`` through the same mathematical operation."""

    return tensor._apply_operator("imatmul", tensor, other)


__all__ = [
    "GEMM_SOURCE",
    "MATMUL_BLAS_SEMANTICS",
    "imatmul",
    "matmul",
    "rmatmul",
]
