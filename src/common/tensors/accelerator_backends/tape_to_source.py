"""Print a recorded AbstractTensor tape as ordinary NumPy or PyTorch source.

An AbstractTensor program is recorded on a ``GradTape`` as a flat sequence of
canonical operations.  NumPy and PyTorch expose very nearly the same set of
operations over the same array model, so one traversal serves both: the
dialects differ only where the two libraries genuinely disagree -- the name of
concatenation, whether an axis argument is called ``axis`` or ``dim``, and how
a permutation is spelled.  Everything else is the same call.

The output is real Python source for a real ``def``, so it can be read,
diffed, checked into a repository, imported, or handed to ``ast.parse`` and
compiled further.  That is the point: it turns a traced AbstractTensor
program into an ordinary function with no AbstractTensor dependency at all.

Node ordering and dead-operation removal come from
``c_jit_backend._required_nodes``, the same traversal the C and Fortran JIT
backends already use, so a printed program contains exactly the operations
those backends compile -- no more, and in the same order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..fused_ir import canonical_elementwise_op
from .c_jit_backend import _flatten, _required_nodes


class TapePrintShortfall(RuntimeError):
    """An operation with no NumPy/PyTorch spelling registered."""


@dataclass(frozen=True)
class Dialect:
    """Where NumPy and PyTorch actually differ."""

    name: str
    module: str
    imports: tuple[str, ...]
    axis: str
    concatenate: str
    permute: str
    array: str

    def call(self, function: str, *arguments: str) -> str:
        return f"{self.module}.{function}({', '.join(arguments)})"


NUMPY = Dialect(
    name="numpy",
    module="np",
    imports=("import numpy as np",),
    axis="axis",
    concatenate="concatenate",
    permute="transpose",
    array="asarray",
)

TORCH = Dialect(
    name="torch",
    module="torch",
    imports=("import torch",),
    axis="dim",
    concatenate="cat",
    permute="permute",
    array="as_tensor",
)

DIALECTS = {"numpy": NUMPY, "torch": TORCH}

# Operations that are the same call in both libraries, taking only operands.
_ELEMENTWISE = {
    "add": "{0} + {1}",
    "sub": "{0} - {1}",
    "mul": "{0} * {1}",
    "truediv": "{0} / {1}",
    "div": "{0} / {1}",
    "pow": "{0} ** {1}",
    "mod": "{0} % {1}",
    "floordiv": "{0} // {1}",
    "neg": "-{0}",
    "less": "{0} < {1}",
    "less_equal": "{0} <= {1}",
    "greater": "{0} > {1}",
    "greater_equal": "{0} >= {1}",
    "equal": "{0} == {1}",
    "not_equal": "{0} != {1}",
    "logical_and": "{0} & {1}",
    "logical_or": "{0} | {1}",
    "matmul": "{0} @ {1}",
}

# Same function name on both libraries.
_SHARED_FUNCTIONS = {
    "abs", "sqrt", "exp", "log", "sin", "cos", "tan",
    "asin", "acos", "atan", "sinh", "cosh", "tanh",
    "floor", "ceil", "round", "sign", "maximum", "minimum",
    "where", "clip",
}

# Reductions and scans, which take the axis argument under different names.
_AXIS_FUNCTIONS = {"sum", "prod", "mean", "max", "min", "cumsum", "cumprod", "all", "any"}

# PyTorch's two-tensor elementwise functions reject a Python scalar where the
# NumPy equivalents accept one.
_TENSOR_ONLY = {"maximum", "minimum", "where"}


def _is_scalar_literal(expression: str) -> bool:
    """Whether an emitted expression is a bare Python number.

    Checked on the emitted text rather than recorded at each site, because a
    constant reaches the output through more than one path (an untraced
    operand, or a recorded ``tensor_from_list``) and only one property
    matters afterwards: that the name holds a scalar, not an array.
    """

    try:
        float(expression)
    except (TypeError, ValueError):
        return expression in ("True", "False")
    return True


def _literal(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return repr(value)
    return repr(value)


class _Printer:
    def __init__(self, dialect: Dialect):
        self.dialect = dialect
        self.names: dict[int, str] = {}
        self.lines: list[str] = []
        self._counter = 0
        # Operand expressions that are bare Python literals rather than
        # array-valued names, so a dialect that refuses scalars can wrap them.
        self._literal_expressions: set[str] = set()

    def fresh(self) -> str:
        name = f"v{self._counter}"
        self._counter += 1
        return name

    def operand(self, value: Any) -> str:
        name = self.names.get(id(value))
        if name is not None:
            return name
        # A value the tape never produced is a literal constant folded into
        # the expression rather than a variable.
        try:
            flat = tuple(_flatten(value.tolist()))
        except Exception:
            expression = _literal(value)
            self._literal_expressions.add(expression)
            return expression
        if len(flat) == 1:
            expression = _literal(flat[0])
            self._literal_expressions.add(expression)
            return expression
        return self.dialect.call(self.dialect.array, repr(list(flat)))

    def emit(self, node: Any) -> None:
        dialect = self.dialect
        raw = str(node.op)
        result = node.ctx["result"]
        inputs = tuple(node.ctx.get("inputs", ()))
        params = dict(node.ctx.get("params") or {})
        args = [self.operand(value) for value in inputs]
        target = self.fresh()

        expression = self._expression(raw, args, params, node)
        if expression is None:
            raise TapePrintShortfall(
                f"{raw} has no {dialect.name} spelling registered"
            )
        self.lines.append(f"    {target} = {expression}")
        self.names[id(result)] = target
        if _is_scalar_literal(expression):
            # A recorded constant becomes its own assignment, so the name now
            # holds a bare Python scalar and carries that property forward.
            self._literal_expressions.add(target)

    def _expression(
        self, raw: str, args: list[str], params: dict, node: Any
    ) -> str | None:
        dialect = self.dialect

        if raw == "tensor_from_list":
            data = list(_flatten(params.get("data")))
            if len(data) == 1:
                return _literal(data[0])
            return dialect.call(dialect.array, repr(data))

        if raw in {"reshape", "view"}:
            shape = tuple(params.get("new_shape") or params.get("shape") or ())
            return f"{args[0]}.reshape({repr(shape)})"

        if raw == "permute":
            perm = list(params.get("perm") or params.get("dims") or ())
            return dialect.call(dialect.permute, args[0], repr(tuple(perm)))

        if raw in {"concat", "cat"}:
            dim = int(params.get("dim", params.get("axis", 0)))
            return dialect.call(
                dialect.concatenate,
                "[" + ", ".join(args) + "]",
                f"{dialect.axis}={dim}",
            )

        if raw == "stack":
            dim = int(params.get("dim", params.get("axis", 0)))
            return dialect.call(
                "stack",
                "[" + ", ".join(args) + "]",
                f"{dialect.axis}={dim}",
            )

        if raw in _AXIS_FUNCTIONS:
            axis = params.get("axis", params.get("dim"))
            if axis is None:
                return dialect.call(raw, args[0])
            call = dialect.call(raw, args[0], f"{dialect.axis}={int(axis)}")
            if params.get("keepdim") or params.get("keepdims"):
                # NumPy spells this keepdims and takes it inline; torch spells
                # it keepdim. Both accept it alongside the axis argument.
                keep = "keepdims" if dialect is NUMPY else "keepdim"
                call = dialect.call(
                    raw, args[0], f"{dialect.axis}={int(axis)}", f"{keep}=True"
                )
            return call

        if raw in _SHARED_FUNCTIONS:
            return dialect.call(raw, *self._tensorize(raw, args))

        operation, reverse = self._canonical(raw)
        if reverse and len(args) == 2:
            args = [args[1], args[0]]
        if operation in _ELEMENTWISE:
            template = _ELEMENTWISE[operation]
            if len(args) == 1 and "{1}" in template:
                return None
            return "(" + template.format(*args) + ")"
        if operation in _SHARED_FUNCTIONS:
            return dialect.call(operation, *args)
        if operation in _AXIS_FUNCTIONS:
            return dialect.call(operation, args[0])
        return None

    def _tensorize(self, function: str, args: Sequence[str]) -> list[str]:
        """Give PyTorch a tensor where it will not accept a scalar.

        ``np.maximum(x, 0.5)`` is fine, ``torch.maximum(x, 0.5)`` is a
        TypeError -- the two-tensor elementwise functions are the one place
        the libraries genuinely disagree about scalars.
        """

        if self.dialect is not TORCH or function not in _TENSOR_ONLY:
            return list(args)
        return [
            (
                self.dialect.call(self.dialect.array, argument)
                if argument in self._literal_expressions
                else argument
            )
            for argument in args
        ]

    @staticmethod
    def _canonical(raw: str) -> tuple[str, bool]:
        try:
            return canonical_elementwise_op(raw)
        except KeyError:
            return raw, False


def emit_tape_source(
    tape: Any,
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Any],
    *,
    backend: str = "numpy",
    function_name: str = "compiled_program",
) -> str:
    """Print the recorded program as a NumPy or PyTorch function.

    ``inputs`` names the tensors that become parameters and ``outputs`` the
    tensors the function returns; both are the same tensors that were traced,
    identified the way the tape identifies everything, by object identity.
    """

    dialect = DIALECTS.get(backend)
    if dialect is None:
        raise ValueError(
            f"unknown backend {backend!r}; one of {sorted(DIALECTS)}"
        )

    printer = _Printer(dialect)
    for name, value in inputs.items():
        printer.names[id(value)] = name

    # The same traversal the C and Fortran JIT backends use: execution order,
    # with operations that no requested output depends on removed.
    class _Captured:
        pass

    captured = _Captured()
    captured.tape = tape
    captured.outputs = dict(outputs)
    for node in _required_nodes(captured):
        if id(node.ctx["result"]) in printer.names:
            continue
        printer.emit(node)

    returned = ", ".join(
        printer.operand(value) for value in outputs.values()
    )
    if len(outputs) == 1:
        returned_expression = returned
    else:
        returned_expression = f"({returned})"

    signature = ", ".join(inputs)
    lines = [
        *dialect.imports,
        "",
        "",
        f"def {function_name}({signature}):",
        *printer.lines,
        f"    return {returned_expression}",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "DIALECTS",
    "Dialect",
    "NUMPY",
    "TORCH",
    "TapePrintShortfall",
    "emit_tape_source",
]
