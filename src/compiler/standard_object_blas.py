"""BLAS as the first generic compiled standard-object catalog."""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

from ..common.tensors.accelerator_backends.ssa_backend import (
    SSATensorOperations,
    SSATensorProgram,
)
from ..common.tensors.mathematical_library import AbstractTensorBLAS, BLAS_LIBRARY
from .kernel_bank import blas_kernel_specs
from .standard_object_product import (
    MethodGraphCapture,
    StandardObject,
    mathematical_sublibrary_object,
)


def _capture(method: str) -> MethodGraphCapture:
    program = SSATensorProgram(f"standard_blas_{method}")

    def inputs(**shapes):
        return {
            name: SSATensorOperations.input(program, tuple(shape))
            for name, shape in shapes.items()
        }

    blas = AbstractTensorBLAS(SSATensorOperations)
    if method == "scal":
        values = inputs(x=(4,), alpha=())
        output = blas.scal(values["x"], values["alpha"])
    elif method == "axpy":
        values = inputs(x=(4,), y=(4,), alpha=())
        output = blas.axpy(values["x"], values["y"], values["alpha"])
    elif method == "dot":
        values = inputs(x=(4,), y=(4,))
        output = blas.dot(values["x"], values["y"])
    elif method == "gemv":
        values = inputs(a=(2, 3), x=(3,), y=(2,), alpha=(), beta=())
        output = blas.gemv(
            values["a"], values["x"], y=values["y"],
            alpha=values["alpha"], beta=values["beta"],
        )
    elif method == "gemm":
        values = inputs(a=(2, 3), b=(3, 2), c=(2, 2), alpha=(), beta=())
        output = blas.gemm(
            values["a"], values["b"], c=values["c"],
            alpha=values["alpha"], beta=values["beta"],
        )
    elif method == "rot":
        values = inputs(x=(4,), y=(4,), c=(), s=())
        output = blas.rot(
            values["x"], values["y"], values["c"], values["s"],
        )
    else:
        raise KeyError(f"unknown standard BLAS method {method!r}")
    return MethodGraphCapture(
        output=output,
        bindings=values,
        wrt_value_ids=tuple(
            int(value.data.value.id) for value in values.values()
        ),
    )


def blas_graph_captures() -> Mapping[str, Callable[[], MethodGraphCapture]]:
    """Fresh semantic captures for every method in the canonical catalog."""

    return {
        method.name: (lambda name=method.name: _capture(name))
        for method in BLAS_LIBRARY.methods
    }


def blas_standard_object(
    *,
    specializations: Mapping[str, Sequence[Mapping[str, int]]] | None = None,
) -> StandardObject:
    """Return BLAS through the same adapter later used by ``linalg``."""

    return mathematical_sublibrary_object(
        BLAS_LIBRARY,
        kernels=blas_kernel_specs(),
        graph_captures=blas_graph_captures(),
        specializations=specializations,
    )


__all__ = ["blas_graph_captures", "blas_standard_object"]
