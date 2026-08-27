"""Canonical scalar numeric spellings for repository-SSA JavaScript.

The table is keyed by the shared ``FusedProgram``/AbstractTensor operation
names. Repository handlers and ``Call[tensor_operation=...]`` instructions are
only two representations of that same vocabulary, so both resolve through
one table. This keeps JavaScript parity measurable against the Python numeric
oracle without making either backend import the other backend's internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from ..common.tensors.fused_ir import ELEMENTWISE_BINARY, ELEMENTWISE_UNARY
from .ssa_numeric_operators import TENSOR_SSA_OPERATORS


@dataclass(frozen=True, slots=True)
class JavaScriptNumericOperator:
    arity: int
    expression: str

    def render(self, arguments: Sequence[str]) -> str | None:
        if len(arguments) != self.arity:
            return None
        return self.expression.format(*arguments)


_EXPRESSIONS = {
    "add": (2, "({0} + {1})"),
    "sub": (2, "({0} - {1})"),
    "mul": (2, "({0} * {1})"),
    "truediv": (2, "({0} / {1})"),
    "pow": (2, "(({0}) ** ({1}))"),
    "mod": (2, "turingMod({0}, {1})"),
    "floordiv": (2, "Math.floor(({0}) / ({1}))"),
    "less": (2, "({0} < {1})"),
    "less_equal": (2, "({0} <= {1})"),
    "greater": (2, "({0} > {1})"),
    "greater_equal": (2, "({0} >= {1})"),
    "equal": (2, "({0} === {1})"),
    "not_equal": (2, "({0} !== {1})"),
    "maximum": (2, "Math.max({0}, {1})"),
    "minimum": (2, "Math.min({0}, {1})"),
    "bitand": (2, "(({0}) & ({1}))"),
    "bitor": (2, "(({0}) | ({1}))"),
    "bitxor": (2, "(({0}) ^ ({1}))"),
    "shl": (2, "(({0}) << ({1}))"),
    "shr": (2, "(({0}) >> ({1}))"),
    "logical_and": (2, "(Boolean({0}) && Boolean({1}))"),
    "logical_or": (2, "(Boolean({0}) || Boolean({1}))"),
    "neg": (1, "(-({0}))"),
    "abs": (1, "Math.abs({0})"),
    "sqrt": (1, "Math.sqrt({0})"),
    "exp": (1, "Math.exp({0})"),
    "log": (1, "Math.log({0})"),
    "tanh": (1, "Math.tanh({0})"),
    "sigmoid": (1, "(1 / (1 + Math.exp(-({0}))))"),
    "sin": (1, "Math.sin({0})"),
    "cos": (1, "Math.cos({0})"),
    "tan": (1, "Math.tan({0})"),
    "asin": (1, "Math.asin({0})"),
    "acos": (1, "Math.acos({0})"),
    "atan": (1, "Math.atan({0})"),
    "sinh": (1, "Math.sinh({0})"),
    "cosh": (1, "Math.cosh({0})"),
    "asinh": (1, "Math.asinh({0})"),
    "acosh": (1, "Math.acosh({0})"),
    "atanh": (1, "Math.atanh({0})"),
    "sign": (1, "Math.sign({0})"),
    "round": (1, "turingRoundEven({0})"),
    "trunc": (1, "Math.trunc({0})"),
    "floor": (1, "Math.floor({0})"),
    "ceil": (1, "Math.ceil({0})"),
    "isfinite": (1, "Number.isFinite({0})"),
    "isnan": (1, "Number.isNaN({0})"),
    "isinf": (1, "(typeof ({0}) === 'number' && !Number.isFinite({0}) && !Number.isNaN({0}))"),
    "logical_not": (1, "(!Boolean({0}))"),
    "invert": (1, "(~({0}))"),
    "int_trunc": (1, "Math.trunc({0})"),
    "zext": (1, "(({0}) >>> 0)"),
    "sext": (1, "(({0}) | 0)"),
    "fptosi": (1, "(Math.trunc({0}) | 0)"),
    "fptoui": (1, "(Math.trunc({0}) >>> 0)"),
    "sitofp": (1, "Number({0})"),
    "uitofp": (1, "Number(({0}) >>> 0)"),
}

JAVASCRIPT_NUMERIC_OPERATORS: Mapping[str, JavaScriptNumericOperator] = (
    MappingProxyType({
        name: JavaScriptNumericOperator(arity, expression)
        for name, (arity, expression) in _EXPRESSIONS.items()
    })
)

_missing = (ELEMENTWISE_UNARY | ELEMENTWISE_BINARY) - JAVASCRIPT_NUMERIC_OPERATORS.keys()
_extra = JAVASCRIPT_NUMERIC_OPERATORS.keys() - (ELEMENTWISE_UNARY | ELEMENTWISE_BINARY)
if _missing or _extra:
    raise RuntimeError(
        "JavaScript numeric table drifted from the portable elementwise "
        f"catalogue: missing={sorted(_missing)!r}, extra={sorted(_extra)!r}"
    )

_repository_aliases = {
    row.handler.value: row.name
    for row in TENSOR_SSA_OPERATORS
    if row.is_direct and row.name in JAVASCRIPT_NUMERIC_OPERATORS
}
_repository_aliases.update({
    # Accepted repository spellings which are not direct tensor rows.
    "Max": "maximum", "Min": "minimum",
    "Exp": "exp", "Log": "log",
    "BitAnd": "bitand", "BitOr": "bitor", "BitXor": "bitxor",
    "Invert": "invert",
    "Round": "round", "Floor": "floor", "Ceil": "ceil", "Trunc": "trunc",
    "SIToFP": "sitofp", "UiToFp": "uitofp", "FPToSI": "fptosi",
    "FpToSi": "fptosi", "FpToUi": "fptoui",
})
for _canonical in JAVASCRIPT_NUMERIC_OPERATORS:
    _repository_aliases.setdefault(_canonical, _canonical)

REPOSITORY_SSA_TO_JAVASCRIPT_NUMERIC: Mapping[str, str] = MappingProxyType(
    _repository_aliases
)

JAVASCRIPT_BITWISE_OPERATIONS = frozenset({
    "bitand", "bitor", "bitxor", "shl", "shr", "invert", "zext", "sext",
})


def canonical_javascript_numeric_operation(
    repository_operation: str,
    *,
    tensor_operation: str | None = None,
) -> str | None:
    """Resolve either repository spelling to one canonical numeric name."""

    if tensor_operation is not None:
        candidate = str(tensor_operation)
        return candidate if candidate in JAVASCRIPT_NUMERIC_OPERATORS else None
    return REPOSITORY_SSA_TO_JAVASCRIPT_NUMERIC.get(str(repository_operation))


def render_javascript_numeric_operation(
    repository_operation: str,
    arguments: Sequence[str],
    *,
    tensor_operation: str | None = None,
) -> tuple[str | None, str | None]:
    """Return ``(canonical_name, expression)``; expression is absent on mismatch."""

    canonical = canonical_javascript_numeric_operation(
        repository_operation, tensor_operation=tensor_operation,
    )
    if canonical is None:
        return None, None
    return canonical, JAVASCRIPT_NUMERIC_OPERATORS[canonical].render(arguments)


def supported_javascript_numeric_operations() -> frozenset[str]:
    return frozenset(JAVASCRIPT_NUMERIC_OPERATORS)


__all__ = [
    "JAVASCRIPT_BITWISE_OPERATIONS",
    "JAVASCRIPT_NUMERIC_OPERATORS",
    "JavaScriptNumericOperator",
    "REPOSITORY_SSA_TO_JAVASCRIPT_NUMERIC",
    "canonical_javascript_numeric_operation",
    "render_javascript_numeric_operation",
    "supported_javascript_numeric_operations",
]
