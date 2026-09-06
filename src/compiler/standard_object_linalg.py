"""The existing linalg namespace supplied to the generic object maker."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from ..common.tensors import linalg
from ..common.tensors.mathematical_library import LINALG_LIBRARY
from .standard_object_product import (
    MethodGraphCapture,
    StandardObject,
    mathematical_sublibrary_object,
)
from .standard_object_tensor_capture import (
    TensorMethodCaptureContract,
    capture_tensor_method,
)


def _n(parameters: Mapping[str, Any]) -> int:
    return int(parameters.get("n", 3))


def _vector(parameters: Mapping[str, Any]) -> tuple[int, ...]:
    return (int(parameters.get("length", parameters.get("n", 4))),)


def _matrix(parameters: Mapping[str, Any]) -> tuple[int, ...]:
    n = _n(parameters)
    return (n, n)


def _rhs(parameters: Mapping[str, Any]) -> tuple[int, ...]:
    return (_n(parameters),)


LINALG_CAPTURE_CONTRACTS = {
    "eye": TensorMethodCaptureContract({}, {"n": 3}),
    "dot": TensorMethodCaptureContract({"a": _vector, "b": _vector}),
    "norm": TensorMethodCaptureContract({"x": _vector}),
    "cross": TensorMethodCaptureContract({"a": (3,), "b": (3,)}),
    "trace": TensorMethodCaptureContract({"A": _matrix}),
    "det": TensorMethodCaptureContract({"A": _matrix}),
    "solve": TensorMethodCaptureContract({"A": _matrix, "b": _rhs}),
    "inv": TensorMethodCaptureContract({"A": _matrix}),
    "eigh": TensorMethodCaptureContract({"A": _matrix}),
    "cholesky": TensorMethodCaptureContract({"A": _matrix}),
}


def linalg_graph_captures() -> Mapping[
    str, Callable[[Mapping[str, Any]], MethodGraphCapture]
]:
    """Capture every method declared by the existing linalg source surface."""

    return {
        method.name: (
            lambda parameters, method_name=method.name: capture_tensor_method(
                getattr(linalg, method_name),
                LINALG_CAPTURE_CONTRACTS[method_name],
                parameters,
                name=f"standard_linalg_{method_name}",
            )
        )
        for method in LINALG_LIBRARY.methods
    }


def linalg_standard_object(
    *,
    parameter_domains: Mapping[
        str, Mapping[str, Sequence[Any]]
    ] | None = None,
    baseline_parameters: Mapping[str, Mapping[str, Any]] | None = None,
) -> StandardObject:
    """Supply linalg unchanged to the generic standard-object compiler."""

    return mathematical_sublibrary_object(
        LINALG_LIBRARY,
        kernels=None,
        graph_captures=linalg_graph_captures(),
        parameter_domains=parameter_domains,
        baseline_parameters=baseline_parameters,
    )


__all__ = [
    "LINALG_CAPTURE_CONTRACTS",
    "linalg_graph_captures",
    "linalg_standard_object",
]
