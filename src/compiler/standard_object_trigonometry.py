"""Compiler adapter for the existing AbstractTensor trigonometry surface."""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

from ..common.tensors.accelerator_backends.ssa_backend import (
    SSATensorOperations,
    SSATensorProgram,
)
from ..common.tensors.mathematical_library import TRIGONOMETRY_LIBRARY
from .standard_object_product import (
    MethodGraphCapture,
    StandardObject,
    mathematical_sublibrary_object,
)


def _capture_existing_method(method: str) -> MethodGraphCapture:
    """Run the existing method on SSA tensors; do not restate its algorithm."""

    program = SSATensorProgram(f"standard_trigonometry_{method}")
    value = SSATensorOperations.input(program, (4,))
    output = getattr(value, method)()
    return MethodGraphCapture(
        output=output,
        bindings={"value": value},
        wrt_value_ids=(int(value.data.value.id),),
    )


def trigonometry_graph_captures() -> Mapping[
    str, Callable[[], MethodGraphCapture]
]:
    """Discover every method from the existing canonical surface."""

    return {
        method.name: (
            lambda name=method.name: _capture_existing_method(name)
        )
        for method in TRIGONOMETRY_LIBRARY.methods
    }


def trigonometry_standard_object(
    *,
    specializations: Mapping[str, Sequence[Mapping[str, int]]] | None = None,
) -> StandardObject:
    """Give the existing trigonometry catalog to the generic object maker."""

    return mathematical_sublibrary_object(
        TRIGONOMETRY_LIBRARY,
        kernels=None,
        graph_captures=trigonometry_graph_captures(),
        specializations=specializations,
    )


__all__ = ["trigonometry_graph_captures", "trigonometry_standard_object"]
