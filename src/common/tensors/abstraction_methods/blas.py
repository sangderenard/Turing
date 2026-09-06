"""AbstractTensor entry points for compiler-owned BLAS semantics.

This module is the explicit seam between the public tensor operation and the
plain compiler source in :mod:`src.common.tensors.blas`.  It deliberately does
not implement a second matrix multiply: eager execution still travels through
the selected AbstractTensor backend, while extraction can identify the exact
``gemm`` semantic material from which compiled backend variants are derived.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..blas import GEMM_SOURCE, blas_role

if TYPE_CHECKING:
    from ..abstraction import AbstractTensor


_GEMM_ROLE = blas_role("gemm")

MATMUL_BLAS_SEMANTICS = {
    "library": "src.common.tensors.blas",
    "kernel": _GEMM_ROLE.name,
    "source_symbol": "GEMM_SOURCE",
    "intrinsic_family": _GEMM_ROLE.identity,
    "alpha": 1.0,
    "beta": 0.0,
}


def call_blas_role(
    tensor: "AbstractTensor",
    role_name: str,
    *operands: Any,
    reverse: bool = False,
    inplace: bool = False,
) -> "AbstractTensor":
    """Resolve one finite BLAS role through the selected tensor backend."""

    role = blas_role(role_name)
    if role.abstract_operator == "matmul" and len(operands) == 1:
        left, right = (
            (operands[0], tensor) if reverse else (tensor, operands[0])
        )
        return tensor._apply_operator(
            "imatmul" if inplace else "matmul", left, right,
        )
    raise NotImplementedError(
        f"AbstractTensor has no eager adapter for BLAS role {role.identity!r}"
    )


def matmul(tensor: "AbstractTensor", other: Any) -> "AbstractTensor":
    """Dispatch ``tensor @ other`` with canonical ``gemm(alpha=1,beta=0)`` semantics."""

    return call_blas_role(tensor, "gemm", other)


def rmatmul(tensor: "AbstractTensor", other: Any) -> "AbstractTensor":
    """Dispatch ``other @ tensor`` with canonical ``gemm(alpha=1,beta=0)`` semantics."""

    return call_blas_role(tensor, "gemm", other, reverse=True)


def imatmul(tensor: "AbstractTensor", other: Any) -> "AbstractTensor":
    """Dispatch ``tensor @= other`` through the same mathematical operation."""

    return call_blas_role(tensor, "gemm", other, inplace=True)


__all__ = [
    "GEMM_SOURCE",
    "MATMUL_BLAS_SEMANTICS",
    "call_blas_role",
    "imatmul",
    "matmul",
    "rmatmul",
]
