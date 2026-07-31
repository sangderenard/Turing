"""Shared numerical corpus for eager and compiled AbstractTensor backends.

Every case contains two independently authored programs:

``numpy_program``
    Operates directly on ``numpy.ndarray`` objects and is the behavioral
    oracle.  It never imports or constructs :class:`AbstractTensor`.

``tensor_program``
    Expresses the same program through the public AbstractTensor surface.
    Eager backends execute it directly; accelerated backends capture and
    compile this exact program.

The corpus is deliberately deterministic.  Its semantic payload is suitable
for repository-local precompile cache identities and contains no Python object
addresses or live tensor identities.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Callable, Mapping

import numpy as np


ArrayMap = Mapping[str, np.ndarray]
OutputMap = Mapping[str, Any]
Program = Callable[[Mapping[str, Any]], OutputMap]


class TortureTier(str, Enum):
    ISOLATED = "isolated"
    GRAB_BAG = "grab_bag"
    ADVANCED = "advanced"
    LARGE = "large"


@dataclass(frozen=True)
class TensorTortureCase:
    name: str
    tier: TortureTier
    inputs: ArrayMap
    numpy_program: Program
    tensor_program: Program
    operations: tuple[str, ...]
    rtol: float = 1.0e-5
    atol: float = 1.0e-6
    # Real `def` source for the AOT precompiler path
    # (accelerator_backends.aot_compile.compile_ast_aot), which needs actual
    # parseable Python source -- not a lambda closure -- and, per the one
    # verified working shape, an entrypoint that calls another function
    # rather than computing its result inline. None for cases not expressed
    # this way; the AOT backends report those as unsupported rather than
    # guess at a translation.
    ast_source: str | None = None
    ast_entrypoint: str | None = None

    def numpy_reference(self) -> dict[str, np.ndarray]:
        outputs = self.numpy_program(
            {name: np.asarray(value).copy() for name, value in self.inputs.items()}
        )
        return {
            name: np.asarray(value)
            for name, value in outputs.items()
        }

    def semantic_record(self) -> dict[str, Any]:
        return {
            # v2 adds the reference output signature.  v1 hashed only the
            # inputs, so editing a case body left the digest unchanged and a
            # stale compiled artifact was served for the new program.
            "schema": "turing-abstract-tensor-torture-v2",
            "name": self.name,
            "tier": self.tier.value,
            "operations": list(self.operations),
            "inputs": {
                name: {
                    "shape": list(value.shape),
                    "dtype": value.dtype.str,
                    "sha256": hashlib.sha256(
                        np.ascontiguousarray(value).tobytes()
                    ).hexdigest(),
                }
                for name, value in sorted(self.inputs.items())
            },
            "outputs": {
                name: {
                    "shape": list(np.asarray(value).shape),
                    "dtype": np.asarray(value).dtype.str,
                }
                for name, value in sorted(self.numpy_reference().items())
            },
            "rtol": self.rtol,
            "atol": self.atol,
        }

    @property
    def semantic_digest(self) -> str:
        payload = json.dumps(
            self.semantic_record(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CapturedTortureCase:
    case: TensorTortureCase
    tape: Any
    inputs: Mapping[str, Any]
    outputs: Mapping[str, Any]


def _isolated_cases(
    *,
    tier: TortureTier = TortureTier.ISOLATED,
    shape: tuple[int, int] = (4, 6),
    matmul_shape: tuple[int, int, int] = (3, 5, 4),
    reshape_target: tuple[int, int, int] = (2, 3, 4),
    name_suffix: str = "",
) -> tuple[TensorTortureCase, ...]:
    """The per-operator cases, at whichever size the tier asks for.

    The operator set is the same at every size on purpose: which backend wins
    is a function of size, so comparing backends only at one shape answers the
    wrong question.  The large tier reuses these definitions rather than
    inventing separate ones, so a difference between tiers is a difference in
    size and nothing else.
    """

    count = int(np.prod(shape))
    signed = np.linspace(-1.75, 2.25, count, dtype=np.float64).reshape(shape)
    positive = np.linspace(0.125, 3.0, count, dtype=np.float64).reshape(shape)
    other = np.linspace(2.5, -1.0, count, dtype=np.float64).reshape(shape)
    rows, inner, columns = matmul_shape
    left = np.linspace(
        -1.0, 1.0, rows * inner, dtype=np.float64
    ).reshape(rows, inner)
    right = np.linspace(
        0.25, 1.75, inner * columns, dtype=np.float64
    ).reshape(inner, columns)

    # Stated rather than derived: the reshape target is part of what the case
    # means, and quietly recomputing it would change the operation being
    # measured whenever the shape moved.
    if int(np.prod(reshape_target)) != count:
        raise ValueError(
            f"reshape target {reshape_target} does not hold {count} elements"
        )

    def _binary_ast_source(op_name: str, expression: str) -> str:
        # Two functions, not one: the AOT precompiler
        # (accelerator_backends.aot_compile.compile_ast_aot) only has a
        # verified-working shape when the entrypoint calls another function
        # rather than computing its result inline.
        return (
            f"def {op_name}(left, right):\n"
            f"    return {expression}\n"
            "\n"
            "def compute(left, right):\n"
            f"    return {op_name}(left=left, right=right)\n"
        )

    def _unary_ast_source(op_name: str, expression: str) -> str:
        return (
            f"def {op_name}(value):\n"
            f"    return {expression}\n"
            "\n"
            "def compute(value):\n"
            f"    return {op_name}(value=value)\n"
        )

    def elementwise(
        name: str,
        operation: str,
        numpy_operation: Callable[[np.ndarray, np.ndarray], Any],
        tensor_operation: Callable[[Any, Any], Any],
        *,
        ast_expression: str | None = None,
    ) -> TensorTortureCase:
        return TensorTortureCase(
            name=name + name_suffix,
            tier=tier,
            inputs={"left": signed, "right": other},
            numpy_program=lambda values: {
                "result": numpy_operation(values["left"], values["right"])
            },
            tensor_program=lambda values: {
                "result": tensor_operation(values["left"], values["right"])
            },
            operations=(operation,),
            ast_source=(
                _binary_ast_source(name, ast_expression)
                if ast_expression is not None
                else None
            ),
            ast_entrypoint="compute" if ast_expression is not None else None,
        )

    def unary(
        name: str,
        source: np.ndarray,
        numpy_operation: Callable[[np.ndarray], Any],
        tensor_operation: Callable[[Any], Any],
        *,
        ast_expression: str | None = None,
    ) -> TensorTortureCase:
        return TensorTortureCase(
            name=name + name_suffix,
            tier=tier,
            inputs={"value": source},
            numpy_program=lambda values: {
                "result": numpy_operation(values["value"])
            },
            tensor_program=lambda values: {
                "result": tensor_operation(values["value"])
            },
            operations=(name,),
            ast_source=(
                _unary_ast_source(name, ast_expression)
                if ast_expression is not None
                else None
            ),
            ast_entrypoint="compute" if ast_expression is not None else None,
        )

    cases = [
        elementwise(
            "add", "add", np.add, lambda a, b: a + b,
            ast_expression="left + right",
        ),
        elementwise(
            "subtract", "sub", np.subtract, lambda a, b: a - b,
            ast_expression="left - right",
        ),
        elementwise(
            "multiply", "mul", np.multiply, lambda a, b: a * b,
            ast_expression="left * right",
        ),
        elementwise(
            "divide",
            "truediv",
            lambda a, b: a / (np.abs(b) + 1.0),
            lambda a, b: a / (b.abs() + 1.0),
            ast_expression="left / (right.abs() + 1.0)",
        ),
        elementwise(
            "maximum", "maximum", np.maximum, lambda a, b: a.maximum(b),
            ast_expression="left.maximum(right)",
        ),
        elementwise(
            "minimum", "minimum", np.minimum, lambda a, b: a.minimum(b),
            ast_expression="left.minimum(right)",
        ),
        elementwise(
            "less", "less", np.less, lambda a, b: a < b,
            ast_expression="left < right",
        ),
        elementwise(
            "greater_equal", "greater_equal", np.greater_equal, lambda a, b: a >= b,
            ast_expression="left >= right",
        ),
        unary(
            "neg", signed, np.negative, lambda value: -value,
            ast_expression="-value",
        ),
        unary(
            "abs", signed, np.abs, lambda value: value.abs(),
            ast_expression="value.abs()",
        ),
        unary(
            "sqrt", positive, np.sqrt, lambda value: value.sqrt(),
            ast_expression="value.sqrt()",
        ),
        unary(
            "exp", signed, np.exp, lambda value: value.exp(),
            ast_expression="value.exp()",
        ),
        unary(
            "log", positive, np.log, lambda value: value.log(),
            ast_expression="value.log()",
        ),
        unary(
            "sin", signed, np.sin, lambda value: value.sin(),
            ast_expression="value.sin()",
        ),
        unary(
            "cos", signed, np.cos, lambda value: value.cos(),
            ast_expression="value.cos()",
        ),
        unary(
            "tan", signed * 0.4, np.tan, lambda value: value.tan(),
            ast_expression="value.tan()",
        ),
        TensorTortureCase(
            name="scalar_broadcast" + name_suffix,
            tier=tier,
            inputs={"value": signed},
            numpy_program=lambda values: {"result": values["value"] * 1.75 + 0.25},
            tensor_program=lambda values: {"result": values["value"] * 1.75 + 0.25},
            ast_source=_unary_ast_source(
                "scalar_broadcast", "value * 1.75 + 0.25"
            ),
            ast_entrypoint="compute",
            operations=("mul", "add"),
        ),
        TensorTortureCase(
            name="where" + name_suffix,
            tier=tier,
            inputs={"left": signed, "right": other},
            numpy_program=lambda values: {
                "result": np.where(
                    values["left"] > values["right"],
                    values["left"],
                    values["right"],
                )
            },
            tensor_program=lambda values: {
                "result": type(values["left"]).where(
                    values["left"] > values["right"],
                    values["left"],
                    values["right"],
                )
            },
            operations=("greater", "where"),
        ),
        TensorTortureCase(
            name="flat_sum" + name_suffix,
            tier=tier,
            inputs={"value": signed},
            numpy_program=lambda values: {"result": values["value"].sum()},
            tensor_program=lambda values: {"result": values["value"].sum()},
            operations=("sum",),
            ast_source=_unary_ast_source("flat_sum", "value.sum()"),
            ast_entrypoint="compute",
        ),
        TensorTortureCase(
            name="dim_sum" + name_suffix,
            tier=tier,
            inputs={"value": signed},
            numpy_program=lambda values: {
                "result": values["value"].sum(axis=1, keepdims=True)
            },
            tensor_program=lambda values: {
                "result": values["value"].sum(dim=1, keepdim=True)
            },
            operations=("sum",),
            ast_source=_unary_ast_source(
                "dim_sum", "value.sum(dim=1, keepdim=True)"
            ),
            ast_entrypoint="compute",
        ),
        TensorTortureCase(
            name="cumsum" + name_suffix,
            tier=tier,
            inputs={"value": signed},
            numpy_program=lambda values: {
                "result": values["value"].cumsum(axis=1)
            },
            tensor_program=lambda values: {"result": values["value"].cumsum(1)},
            operations=("cumsum",),
            ast_source=_unary_ast_source("cumsum_op", "value.cumsum(1)"),
            ast_entrypoint="compute",
        ),
        TensorTortureCase(
            name="matmul" + name_suffix,
            tier=tier,
            inputs={"left": left, "right": right},
            numpy_program=lambda values: {
                "result": values["left"] @ values["right"]
            },
            tensor_program=lambda values: {
                "result": values["left"] @ values["right"]
            },
            operations=("matmul",),
            ast_source=_binary_ast_source("matmul_op", "left.matmul(right)"),
            ast_entrypoint="compute",
        ),
        TensorTortureCase(
            name="reshape_transpose" + name_suffix,
            tier=tier,
            inputs={"value": signed},
            numpy_program=lambda values: {
                "result": values["value"]
                .reshape(*reshape_target)
                .transpose(2, 0, 1)
            },
            tensor_program=lambda values: {
                "result": values["value"]
                .reshape(*reshape_target)
                .permute(2, 0, 1)
            },
            operations=("reshape", "permute"),
            ast_source=_unary_ast_source(
                "reshape_transpose_op",
                f"value.reshape({', '.join(map(str, reshape_target))})"
                ".permute(2, 0, 1)",
            ),
            ast_entrypoint="compute",
        ),
        TensorTortureCase(
            name="stack" + name_suffix,
            tier=tier,
            inputs={"left": signed, "right": other},
            numpy_program=lambda values: {
                "result": np.stack((values["left"], values["right"]), axis=1)
            },
            tensor_program=lambda values: {
                "result": type(values["left"]).stack(
                    (values["left"], values["right"]), dim=1
                )
            },
            operations=("stack",),
        ),
        TensorTortureCase(
            name="cat" + name_suffix,
            tier=tier,
            inputs={"left": signed, "right": other},
            numpy_program=lambda values: {
                "result": np.concatenate((values["left"], values["right"]), axis=0)
            },
            tensor_program=lambda values: {
                "result": type(values["left"]).cat(
                    (values["left"], values["right"]), dim=0
                )
            },
            operations=("cat",),
        ),
    ]
    return tuple(cases)



def _fused_chain_case(
    *,
    tier: TortureTier = TortureTier.ISOLATED,
    shape: tuple[int, int] = (4, 6),
    name_suffix: str = "",
) -> TensorTortureCase:
    """One long elementwise chain that lowers to a single fused program.

    The per-operator cases measure one operation surrounded by launch
    overhead, which is the regime where an eager backend looks best: NumPy
    pays one pass and one temporary, and so does everything else.

    A long chain measures the opposite thing.  Eager evaluation walks memory
    once per operation and materialises a temporary each time; a fused program
    walks it once for the whole chain.  That difference is the entire argument
    for compiling, and it is invisible until the chain is long.
    """

    count = int(np.prod(shape))
    left = np.linspace(0.35, 2.4, count, dtype=np.float64).reshape(shape)
    right = np.linspace(2.1, 0.4, count, dtype=np.float64).reshape(shape)

    def numpy_program(values):
        a = values["left"]
        b = values["right"]
        x = a * 1.25 + b * 0.5
        x = np.sin(x) + np.cos(b * 0.125)
        x = x * x + a * 0.75
        x = np.sqrt(np.abs(x) + 0.5)
        x = x - b * 0.25
        x = np.exp(-np.abs(x) * 0.5)
        x = x + np.log(np.abs(a) + 1.5)
        x = np.maximum(x, b * 0.1)
        x = np.minimum(x, a * 3.0)
        x = x * 0.5 + np.sin(a * 0.25)
        x = x / (np.abs(b) + 1.25)
        x = np.where(x > 0.0, x, -x)
        return {"result": x * 2.0 - 0.125}

    def tensor_program(values):
        a = values["left"]
        b = values["right"]
        x = a * 1.25 + b * 0.5
        x = x.sin() + (b * 0.125).cos()
        x = x * x + a * 0.75
        x = ((x.abs() + 0.5)).sqrt()
        x = x - b * 0.25
        x = (-(x.abs() * 0.5)).exp()
        x = x + (a.abs() + 1.5).log()
        x = type(a).maximum(x, b * 0.1)
        x = type(a).minimum(x, a * 3.0)
        x = x * 0.5 + (a * 0.25).sin()
        x = x / (b.abs() + 1.25)
        x = type(a).where(x > 0.0, x, -x)
        return {"result": x * 2.0 - 0.125}

    return TensorTortureCase(
        name="fused_chain" + name_suffix,
        tier=tier,
        inputs={"left": left, "right": right},
        numpy_program=numpy_program,
        tensor_program=tensor_program,
        operations=(
            "mul", "add", "sin", "cos", "sqrt", "abs", "sub", "neg",
            "exp", "log", "maximum", "minimum", "truediv", "greater", "where",
        ),
        rtol=1.0e-5,
        atol=1.0e-6,
    )


def _grab_bag_case() -> TensorTortureCase:
    left = np.linspace(-1.5, 2.0, 48, dtype=np.float64).reshape(6, 8)
    right = np.linspace(2.25, -0.75, 48, dtype=np.float64).reshape(6, 8)

    def numpy_program(values):
        left_value = values["left"]
        right_value = values["right"]
        mixed = (left_value * 1.25 + right_value * 0.5) / (
            np.abs(right_value) + 1.75
        )
        wave = np.sin(mixed) + np.cos(left_value * 0.3)
        selected = np.where(left_value >= right_value, wave, -wave)
        return {"result": np.maximum(selected, -0.8) * 0.625 + 0.125}

    def tensor_program(values):
        left_value = values["left"]
        right_value = values["right"]
        mixed = (left_value * 1.25 + right_value * 0.5) / (
            right_value.abs() + 1.75
        )
        wave = mixed.sin() + (left_value * 0.3).cos()
        selected = type(left_value).where(
            left_value >= right_value, wave, -wave
        )
        return {"result": selected.maximum(-0.8) * 0.625 + 0.125}

    return TensorTortureCase(
        name="operator_grab_bag",
        tier=TortureTier.GRAB_BAG,
        inputs={"left": left, "right": right},
        numpy_program=numpy_program,
        tensor_program=tensor_program,
        operations=(
            "mul",
            "add",
            "abs",
            "truediv",
            "sin",
            "cos",
            "greater_equal",
            "where",
            "neg",
            "maximum",
        ),
    )


def _advanced_case() -> TensorTortureCase:
    cube = np.linspace(-1.0, 2.0, 24, dtype=np.float64).reshape(2, 3, 4)
    weights = np.linspace(0.25, 1.25, 20, dtype=np.float64).reshape(4, 5)

    def numpy_program(values):
        cube_value = values["cube"]
        weights_value = values["weights"]
        projected = cube_value.reshape(6, 4) @ weights_value
        restored = projected.reshape(2, 3, 5).transpose(1, 0, 2)
        paired = np.stack((restored, restored * -0.5), axis=1)
        prefix = paired.cumsum(axis=-1)
        joined = np.concatenate((prefix, np.abs(prefix)), axis=0)
        selected = np.where(joined >= 0.0, joined, joined * 0.25)
        return {
            "tensor": selected,
            "summary": selected.sum(axis=-1, keepdims=True),
        }

    def tensor_program(values):
        cube_value = values["cube"]
        weights_value = values["weights"]
        projected = cube_value.reshape(6, 4) @ weights_value
        restored = projected.reshape(2, 3, 5).permute(1, 0, 2)
        paired = type(restored).stack((restored, restored * -0.5), dim=1)
        prefix = paired.cumsum(-1)
        joined = type(prefix).cat((prefix, prefix.abs()), dim=0)
        selected = type(joined).where(
            joined >= 0.0, joined, joined * 0.25
        )
        return {
            "tensor": selected,
            "summary": selected.sum(dim=-1, keepdim=True),
        }

    return TensorTortureCase(
        name="advanced_tensor_topology",
        tier=TortureTier.ADVANCED,
        inputs={"cube": cube, "weights": weights},
        numpy_program=numpy_program,
        tensor_program=tensor_program,
        operations=(
            "reshape",
            "matmul",
            "permute",
            "mul",
            "stack",
            "cumsum",
            "abs",
            "cat",
            "greater_equal",
            "where",
            "sum",
        ),
        rtol=2.0e-5,
        atol=2.0e-6,
    )


# One megaelement per operand: past every cache level, so the large tier
# measures memory behaviour rather than launch overhead.  Matmul is sized
# separately because its cost is cubic and it would otherwise dominate the run.
LARGE_SHAPE = (1024, 1024)
LARGE_MATMUL_SHAPE = (256, 256, 256)
LARGE_RESHAPE_TARGET = (2, 512, 1024)


def _large_cases() -> tuple[TensorTortureCase, ...]:
    frame_shape = (8, 512, 512)
    count = int(np.prod(frame_shape))
    frame_values = np.linspace(
        -1.25, 1.75, count, dtype=np.float64
    ).reshape(frame_shape)
    companion = np.linspace(
        1.5, -0.75, count, dtype=np.float64
    ).reshape(frame_shape)
    matrix_left = np.linspace(
        -0.75, 1.25, 256 * 256, dtype=np.float64
    ).reshape(256, 256)
    matrix_right = np.linspace(
        1.0, -0.5, 256 * 256, dtype=np.float64
    ).reshape(256, 256)

    def frame_numpy(values):
        left = values["left"]
        right = values["right"]
        mixed = left * 0.75 + right * 0.25
        wave = np.sin(mixed) + np.cos(right * 0.125)
        return {
            "frames": np.where(left >= right, wave, wave * -0.5)
        }

    def frame_tensor(values):
        left = values["left"]
        right = values["right"]
        mixed = left * 0.75 + right * 0.25
        wave = mixed.sin() + (right * 0.125).cos()
        return {
            "frames": type(left).where(
                left >= right, wave, wave * -0.5
            )
        }

    def reduction_numpy(values):
        prefix = values["frames"].cumsum(axis=-1)
        return {
            "row_totals": prefix.sum(axis=1, keepdims=True),
        }

    def reduction_tensor(values):
        prefix = values["frames"].cumsum(-1)
        return {
            "row_totals": prefix.sum(dim=1, keepdim=True),
        }

    return (
        TensorTortureCase(
            name="large_frame_batch",
            tier=TortureTier.LARGE,
            inputs={"left": frame_values, "right": companion},
            numpy_program=frame_numpy,
            tensor_program=frame_tensor,
            operations=(
                "mul",
                "add",
                "sin",
                "cos",
                "greater_equal",
                "where",
            ),
            rtol=1.0e-4,
            atol=1.0e-4,
        ),
        TensorTortureCase(
            name="large_reduction",
            tier=TortureTier.LARGE,
            inputs={"frames": frame_values},
            numpy_program=reduction_numpy,
            tensor_program=reduction_tensor,
            operations=("cumsum", "sum"),
            rtol=2.0e-4,
            atol=2.0e-3,
        ),
        TensorTortureCase(
            name="large_matmul",
            tier=TortureTier.LARGE,
            inputs={"left": matrix_left, "right": matrix_right},
            numpy_program=lambda values: {
                "result": values["left"] @ values["right"]
            },
            tensor_program=lambda values: {
                "result": values["left"] @ values["right"]
            },
            operations=("matmul",),
            rtol=2.0e-4,
            atol=2.0e-3,
        ),
    )


def tensor_torture_cases(
    *,
    include_large: bool = False,
) -> tuple[TensorTortureCase, ...]:
    cases = (
        *_isolated_cases(),
        _fused_chain_case(),
        _grab_bag_case(),
        _advanced_case(),
    )
    if not include_large:
        return cases
    # The same operators again at size, plus the bespoke frame/reduction cases.
    return (
        *cases,
        *_isolated_cases(
            tier=TortureTier.LARGE,
            shape=LARGE_SHAPE,
            matmul_shape=LARGE_MATMUL_SHAPE,
            reshape_target=LARGE_RESHAPE_TARGET,
            name_suffix="_large",
        ),
        _fused_chain_case(
            tier=TortureTier.LARGE,
            shape=LARGE_SHAPE,
            name_suffix="_large",
        ),
        *_large_cases(),
    )


def eager_backend_result(
    case: TensorTortureCase,
    backend_type: type,
    *,
    device: str | None = None,
) -> dict[str, np.ndarray]:
    values = {
        name: backend_type.tensor(np.asarray(value).copy(), device=device)
        for name, value in case.inputs.items()
    }
    outputs = case.tensor_program(values)
    return {
        name: np.asarray(
            value.data.detach().cpu().numpy() if hasattr(getattr(value, "data", None), "detach") else (value.numpy() if hasattr(value, "numpy") else value.tolist())
        )
        for name, value in outputs.items()
    }


def capture_torture_case(case: TensorTortureCase) -> CapturedTortureCase:
    """Record the exact AbstractTensor program once using NumPy storage."""

    from ..autograd import autograd
    from ..numpy_backend import NumPyTensorOperations

    inputs = {
        name: NumPyTensorOperations.tensor(np.asarray(value).copy())
        for name, value in case.inputs.items()
    }
    with autograd.forward_capture() as tape:
        outputs = dict(case.tensor_program(inputs))
    return CapturedTortureCase(
        case=case,
        tape=tape,
        inputs=inputs,
        outputs=outputs,
    )


__all__ = [
    "CapturedTortureCase",
    "TensorTortureCase",
    "TortureTier",
    "capture_torture_case",
    "eager_backend_result",
    "tensor_torture_cases",
]
