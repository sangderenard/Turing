"""Generic source capture for AbstractTensor-backed standard-object methods."""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import Any, Callable, Mapping

from ..common.tensors.abstraction import AbstractTensor
from ..common.tensors.accelerator_backends.ssa_backend import (
    SSATensorOperations,
    SSATensorProgram,
)
from .standard_object_product import MethodGraphCapture


ShapeResolver = tuple[int, ...] | Callable[[Mapping[str, Any]], tuple[int, ...]]


@dataclass(frozen=True)
class TensorMethodCaptureContract:
    """The tensor feeds and compile-time defaults needed to realize source."""

    input_shapes: Mapping[str, ShapeResolver]
    compile_defaults: Mapping[str, Any] = field(default_factory=dict)


def capture_tensor_method(
    function: Callable[..., Any],
    contract: TensorMethodCaptureContract,
    parameters: Mapping[str, Any] | None = None,
    *,
    name: str | None = None,
) -> MethodGraphCapture:
    """Execute an authored tensor function once into repository SSA.

    ``parameters`` may contain ordinary source arguments (for example
    ``sweeps``) or capture-only layout axes used by shape resolvers (for
    example ``n`` for an input matrix).  The object maker owns Cartesian row
    generation; this function only realizes whichever row it is handed.
    """

    selected = {**dict(contract.compile_defaults), **dict(parameters or {})}
    program = SSATensorProgram(name or f"standard_{function.__name__}")
    bindings = {}
    for argument, resolver in contract.input_shapes.items():
        shape = resolver(selected) if callable(resolver) else resolver
        bindings[argument] = SSATensorOperations.input(
            program, tuple(int(extent) for extent in shape),
        )

    signature = inspect.signature(function)
    call = {}
    for argument, parameter in signature.parameters.items():
        if argument in bindings:
            call[argument] = bindings[argument]
        elif argument in selected:
            call[argument] = selected[argument]
        elif parameter.default is inspect.Parameter.empty:
            raise ValueError(
                f"capture of {function.__module__}.{function.__name__} "
                f"requires compile parameter {argument!r}"
            )

    with program.activate(), AbstractTensor.use_backend("ssa"):
        output = function(**call)
    return MethodGraphCapture(
        output=output,
        bindings=bindings,
        wrt_value_ids=tuple(
            int(value.data.value.id) for value in bindings.values()
        ),
    )


__all__ = ["TensorMethodCaptureContract", "capture_tensor_method"]
