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


#: Width the core bakes at when a row does not name one.  Every trigonometry
#: method is elementwise, so the width is a pure DEPLOYMENT choice: the
#: repository kernel behind these captures (``unary_double``) already takes its
#: element count as a runtime argument, and was measured serving n = 4, 5 and
#: 4096 correctly out of a pack baked at 4.  Only the generated wrapper's call
#: site freezes the count, so a row here selects a bake, not a capability.
DEFAULT_BAKE_LENGTH = 4


def _bake_length(parameters: Mapping[str, object] | None) -> int:
    selected = dict(parameters or {})
    length = int(selected.get("length", DEFAULT_BAKE_LENGTH))
    if length < 1:
        raise ValueError(f"trigonometry bake length must be positive, got {length!r}")
    return length


def _capture_existing_method(
    method: str, parameters: Mapping[str, object] | None = None,
) -> MethodGraphCapture:
    """Run the existing method on SSA tensors; do not restate its algorithm."""

    length = _bake_length(parameters)
    program = SSATensorProgram(f"standard_trigonometry_{method}_{length}")
    value = SSATensorOperations.input(program, (length,))
    output = getattr(value, method)()
    return MethodGraphCapture(
        output=output,
        bindings={"value": value},
        wrt_value_ids=(int(value.data.value.id),),
    )


def trigonometry_graph_captures() -> Mapping[
    str, Callable[[Mapping[str, object]], MethodGraphCapture]
]:
    """Discover every method from the existing canonical surface."""

    return {
        method.name: (
            lambda parameters, name=method.name: _capture_existing_method(
                name, parameters,
            )
        )
        for method in TRIGONOMETRY_LIBRARY.methods
    }


def trigonometry_bake_domains(
    lengths: Sequence[int],
) -> Mapping[str, Mapping[str, Sequence[int]]]:
    """One ``length`` axis per method, so every method bakes the same widths."""

    widths = tuple(dict.fromkeys(int(length) for length in lengths))
    if not widths or any(width < 1 for width in widths):
        raise ValueError(f"trigonometry bake lengths must be positive: {lengths!r}")
    return {
        method.name: {"length": widths}
        for method in TRIGONOMETRY_LIBRARY.methods
    }


def trigonometry_bake_baselines(
    length: int = DEFAULT_BAKE_LENGTH,
) -> Mapping[str, Mapping[str, int]]:
    """Which baked width answers a call that names no deployment row."""

    return {
        method.name: {"length": int(length)}
        for method in TRIGONOMETRY_LIBRARY.methods
    }


def trigonometry_standard_object(
    *,
    specializations: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
    parameter_domains: Mapping[
        str, Mapping[str, Sequence[object]]
    ] | None = None,
    baseline_parameters: Mapping[str, Mapping[str, object]] | None = None,
) -> StandardObject:
    """Give the existing trigonometry catalog to the generic object maker.

    ``parameter_domains={'sin': {'length': (4, 64)}}`` bakes one core per width
    and states them as deployment rows; the baseline row is the width a call
    that names nothing gets.
    """

    return mathematical_sublibrary_object(
        TRIGONOMETRY_LIBRARY,
        kernels=None,
        graph_captures=trigonometry_graph_captures(),
        specializations=specializations,
        parameter_domains=parameter_domains,
        baseline_parameters=baseline_parameters,
    )


__all__ = [
    "DEFAULT_BAKE_LENGTH",
    "trigonometry_bake_baselines",
    "trigonometry_bake_domains",
    "trigonometry_graph_captures",
    "trigonometry_standard_object",
]
