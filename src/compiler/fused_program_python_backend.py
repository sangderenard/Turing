"""Lower a ``FusedProgram`` numeric region straight to Python -- no SSA, no
C, no GLSL -- as real ``ast`` statements in a selectable dialect (plain
NumPy, plain PyTorch, or AbstractTensor, which routes through whichever
backend is current, nodus included).

This module lowers the *numeric* IR only (``FusedProgram``: a flat list of
``OpStep``, no control flow -- see ``fused_ir.py``).  ``FusedProgram`` is the
same class regardless of which front end produced it: the tape-walking JIT
front end (``c_primitive_program.compile_elementwise_tape``) and the
AST-ingestion AOT front end (``ProcessGraph.build_from_ast`` ->
``process_graph_fusion.plan_process_graph_dispatches`` ->
``dispatch_region_to_fused_program``) both hand a ``FusedProgram`` to their
backends -- see ``program_order.py``'s module docstring: "these are two
different products, not one pipeline wearing two hats".  This module's job
starts *after* a ``FusedProgram`` already exists; it does not care, and must
not be made to care, which front end built it.  ``fused_program_python_aot``
below is the half of this module that does pick a front end, and it always
picks the AST one -- a program meant to be lowered ahead-of-time should be
discovered by ingesting its source, not by tracing a tape.

Each numeric statement is produced by parsing a small, individually-valid
source fragment (``ast.parse(f"{target} = {expr}").body[0]``) rather than by
hand-building ``ast.BinOp``/``ast.Call`` nodes for every operator a dialect
can spell.  The result is still genuine ``ast``: real ``ast.Assign`` nodes
assembled into one ``ast.Module``, inspectable, ``ast.unparse``-able, and
diffable -- not a string-templated whole-program print like
``tape_to_source.py``'s (which also, deliberately, is not touched or reused
here: it walks a tape, and dialect tables that shape belong to a JIT
consumer, not this one).

The composed region source is finalized through the existing control-shell
runtime (``control_source.compile_python_shell``), the same "instantiate a
real callable from finalized region bodies" contract
``ssa_fortran_backend.compile_module`` uses for Fortran -- so a program
compiled here is a real, loadable Python function, not a second, parallel
execution mechanism.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Mapping

from ..common.tensors.fused_ir import (
    ELEMENTWISE_BINARY,
    ELEMENTWISE_UNARY,
    FusedProgram,
    OpStep,
    ordered_feed_ids,
)
from .control_source import (
    ControlProgram,
    ControlTarget,
    RegionCode,
    StatementBlock,
    compile_python_shell,
)


class PythonLoweringShortfall(ValueError):
    """A ``FusedProgram`` step has no spelling registered for a dialect."""


@dataclass(frozen=True)
class Dialect:
    """Where NumPy, PyTorch, and AbstractTensor genuinely differ in how an
    elementwise call is spelled: NumPy/PyTorch call a module-level function
    (``np.sqrt(x)``); AbstractTensor calls an instance method (``x.sqrt()``).
    Operator syntax (``x + y``) is the same across all three and is not
    dialect-dependent."""

    name: str
    imports: tuple[str, ...]
    style: str  # "module" or "method"
    module: str | None = None


NUMPY = Dialect(name="numpy", imports=("import numpy as np",), style="module", module="np")
TORCH = Dialect(name="torch", imports=("import torch",), style="module", module="torch")
ABSTRACT_TENSOR = Dialect(
    name="abstract_tensor",
    imports=("from src.common.tensors.abstraction import AbstractTensor",),
    style="method",
)

DIALECTS: dict[str, Dialect] = {d.name: d for d in (NUMPY, TORCH, ABSTRACT_TENSOR)}

# Span-memory initialisation constructors and their implicit fill scalar.
# ``None`` requires an explicit ``fill_value`` (``full``/``fill``).
_SPAN_INIT_DEFAULTS: dict[str, float | None] = {
    "fill": None,
    "zeros": 0.0,
    "zeros_like": 0.0,
    "empty": 0.0,
    "empty_like": 0.0,
    "ones": 1.0,
    "ones_like": 1.0,
    "full": None,
    "full_like": None,
}


# Operator syntax: identical across every dialect, so these are not part of
# the Dialect record.
_ELEMENTWISE_TEMPLATES: dict[str, str] = {
    "add": "{0} + {1}",
    "sub": "{0} - {1}",
    "mul": "{0} * {1}",
    "truediv": "{0} / {1}",
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
    "bitand": "{0} & {1}",
    "bitor": "{0} | {1}",
    "bitxor": "{0} ^ {1}",
    "shl": "{0} << {1}",
    "shr": "{0} >> {1}",
    "invert": "~{0}",
    "logical_and": "{0} & {1}",
    "logical_or": "{0} | {1}",
}

# Named functions: same name on every dialect, only the calling convention
# (module function vs. instance method, handled by ``_call``) differs.
_NAMED_FUNCTIONS = frozenset(
    {
        "abs", "sqrt", "exp", "log", "sin", "cos", "tan",
        "asin", "acos", "atan", "sinh", "cosh", "tanh",
        "asinh", "acosh", "atanh", "sign", "floor",
        "isfinite", "isnan", "isinf", "logical_not",
        "maximum", "minimum",
    }
)


def _dialect_namespace(dialect: Dialect) -> dict[str, Any]:
    """Bind whatever bare name a dialect's emitted calls reference.

    ``compile_python_shell`` execs the rendered source in a namespace it is
    given; a call like ``np.abs(x)`` needs ``np`` bound in that namespace the
    same way the rendered ``import numpy as np`` line would bind it at
    module scope.
    """

    if dialect is NUMPY:
        import numpy as np

        return {"np": np}
    if dialect is TORCH:
        import torch

        return {"torch": torch}
    return {}


def _call(dialect: Dialect, function: str, args: list[str]) -> str:
    if dialect.style == "method":
        head, *rest = args
        return f"{head}.{function}({', '.join(rest)})"
    return f"{dialect.module}.{function}({', '.join(args)})"


# Axis reductions the numeric backends fold away. Kept aligned with the
# WebAssembly backend's ``_REDUCE_FOLD`` so a program that lowers to one also
# lowers to the other.
_REDUCE_NUMPY: dict[str, str] = {
    "sum": "sum", "mean": "mean", "prod": "prod",
    "min": "min", "amin": "min", "max": "max", "amax": "max",
}
_REDUCE_TORCH: dict[str, str] = {
    "sum": "sum", "mean": "mean", "prod": "prod",
    "min": "amin", "amin": "amin", "max": "amax", "amax": "amax",
}
_REDUCE_ABSTRACT: dict[str, str] = {
    "sum": "sum", "mean": "mean", "prod": "prod",
    "min": "amin", "amin": "amin", "max": "amax", "amax": "amax",
}


def _step_expression(dialect: Dialect, step: OpStep, names: dict[int, str]) -> str:
    op = step.op_name
    if op == "tensor_from_list":
        if "values" not in step.attrs:
            raise PythonLoweringShortfall("tensor_from_list has no values")
        values = repr(step.attrs["values"])
        if dialect is NUMPY:
            return f"np.asarray({values})"
        if dialect is TORCH:
            return f"torch.tensor({values})"
        if dialect is ABSTRACT_TENSOR:
            return f"AbstractTensor.tensor({values})"
        raise PythonLoweringShortfall(
            f"tensor_from_list has no {dialect.name} spelling registered"
        )
    if op in _SPAN_INIT_DEFAULTS:
        shape = tuple(step.attrs.get("shape", ()))
        value = step.attrs.get(
            "fill_value", step.attrs.get("value", _SPAN_INIT_DEFAULTS[op])
        )
        if value is None:
            raise PythonLoweringShortfall(f"{op} requires an explicit fill_value")
        value = float(value)
        shape_src = repr(shape)
        if value == 0.0:
            # Zero-fill is the calloc case; the constructors already zero-page.
            if dialect is NUMPY:
                return f"np.zeros({shape_src})"
            if dialect is TORCH:
                return f"torch.zeros({shape_src})"
            if dialect is ABSTRACT_TENSOR:
                return f"AbstractTensor.zeros({shape_src})"
        else:
            if dialect is NUMPY:
                return f"np.full({shape_src}, {value!r})"
            if dialect is TORCH:
                return f"torch.full({shape_src}, {value!r})"
            if dialect is ABSTRACT_TENSOR:
                return f"AbstractTensor.full({shape_src}, {value!r})"
        raise PythonLoweringShortfall(
            f"{op} has no {dialect.name} spelling registered"
        )
    a = names[step.input_ids[0]]
    reduce_numpy = _REDUCE_NUMPY.get(op)
    if reduce_numpy is not None:
        axis = step.attrs.get("axis")
        keepdim = bool(step.attrs.get("keepdim", False))
        if dialect is NUMPY:
            return f"np.{reduce_numpy}({a}, axis={axis!r}, keepdims={keepdim!r})"
        if dialect is TORCH:
            fn = _REDUCE_TORCH[op]
            if axis is None:
                return f"torch.{fn}({a})"
            return f"torch.{fn}({a}, dim={axis!r}, keepdim={keepdim!r})"
        if dialect is ABSTRACT_TENSOR:
            return f"{a}.{_REDUCE_ABSTRACT[op]}(dim={axis!r}, keepdim={keepdim!r})"
        raise PythonLoweringShortfall(
            f"{op} has no {dialect.name} spelling registered"
        )
    if op in ELEMENTWISE_UNARY:
        args = [a]
    elif op in ELEMENTWISE_BINARY:
        if len(step.input_ids) == 2:
            args = [a, names[step.input_ids[1]]]
        elif "right_scalar" in step.attrs:
            args = [a, repr(step.attrs["right_scalar"])]
        else:
            raise PythonLoweringShortfall(f"{op} has no right operand: {step!r}")
        if step.attrs.get("reverse", False):
            args = [args[1], args[0]]
    else:
        raise PythonLoweringShortfall(f"{op} is not an elementwise op")

    template = _ELEMENTWISE_TEMPLATES.get(op)
    if template is not None:
        if len(args) == 1 and "{1}" in template:
            raise PythonLoweringShortfall(f"{op} needs two operands, got one")
        return "(" + template.format(*args) + ")"
    if op in _NAMED_FUNCTIONS:
        return _call(dialect, op, args)
    raise PythonLoweringShortfall(f"{op} has no {dialect.name} spelling registered")


@dataclass(frozen=True)
class LoweredRegion:
    """One region's numeric IR, lowered to real ``ast`` statements."""

    statements: tuple[ast.stmt, ...]
    output_names: dict[str, str]  # FusedProgram output name -> local variable
    value_names: dict[int, str]  # value id -> local variable (feeds + results)


def lower_fused_program_region(
    program: FusedProgram,
    feed_names: Mapping[int, str],
    *,
    dialect: Dialect,
    name_prefix: str = "v",
) -> LoweredRegion:
    """Build one ``ast.Assign`` per ``OpStep``, in the program's own order.

    ``feed_names`` maps each of ``program.feeds`` to the local name already
    bound to it (a function parameter, or a prior region's output) -- this
    function does not decide what a feed is called, only how each step
    computes from names it is given.
    """

    names: dict[int, str] = dict(feed_names)
    statements: list[ast.stmt] = []
    for index, step in enumerate(program.steps):
        target = f"{name_prefix}{index}"
        expression = _step_expression(dialect, step, names)
        statement = ast.parse(f"{target} = {expression}\n").body[0]
        statements.append(statement)
        names[step.result_id] = target
    output_names = {
        name: names[value_id] for name, value_id in program.outputs.items()
    }
    return LoweredRegion(
        statements=tuple(statements),
        output_names=output_names,
        value_names=names,
    )


def region_source_lines(lowered: LoweredRegion) -> tuple[str, ...]:
    """The lowered region's assignments as source lines, for splicing into
    a ``control_source.StatementBlock`` region body."""

    return tuple(ast.unparse(statement) for statement in lowered.statements)


@dataclass(frozen=True)
class CompiledPythonProgram:
    """A ``FusedProgram`` region, instantiated as a real Python callable --
    the same "finalized shell" shape ``ssa_fortran_backend.compile_module``
    returns for Fortran, minus the subprocess compiler."""

    callable: Any
    source: str
    dialect: str


def compile_single_region_python(
    program: FusedProgram,
    feed_names: Mapping[int, str],
    *,
    dialect: str = "numpy",
    function_name: str = "compiled_program",
    abstract_tensor_backend: str | None = None,
) -> CompiledPythonProgram:
    """Finalize one region's ``FusedProgram`` as a real Python callable.

    Composes through ``control_source.compile_python_shell`` with the
    trivial single-region control shell (call the one region, then return),
    the same region-marker/``RegionCode`` contract every other target
    (C, GLSL, Fortran) already uses.
    """

    chosen = DIALECTS.get(dialect)
    if chosen is None:
        raise ValueError(f"unknown dialect {dialect!r}; one of {sorted(DIALECTS)}")

    lowered = lower_fused_program_region(program, feed_names, dialect=chosen)
    output_items = list(program.outputs.items())
    if not output_items:
        raise ValueError("a compiled program needs at least one output")
    if len(output_items) == 1:
        return_expression = lowered.output_names[output_items[0][0]]
    else:
        return_expression = "(" + ", ".join(
            lowered.output_names[name] for name, _ in output_items
        ) + ")"

    region_lines = region_source_lines(lowered) + (f"return {return_expression}",)
    control = ControlProgram(
        root=StatementBlock(("__scheduled_region_0__",)),
        region_indices=(0,),
    )
    region = RegionCode(
        region_index=0,
        target=ControlTarget.PYTHON,
        body=StatementBlock(region_lines),
    )
    parameters = tuple(feed_names[feed_id] for feed_id in ordered_feed_ids(program))
    function = compile_python_shell(
        control,
        (region,),
        function_name=function_name,
        parameters=parameters,
        namespace=_dialect_namespace(chosen),
        abstract_tensor_backend=abstract_tensor_backend,
    )
    # compile_python_shell attaches the exact source it compiled (including
    # its own AbstractTensor.use_backend wrapping, when requested) to the
    # function it returns.
    source = function.__compiled_shell_source__
    return CompiledPythonProgram(callable=function, source=source, dialect=dialect)


__all__ = [
    "ABSTRACT_TENSOR",
    "CompiledPythonProgram",
    "DIALECTS",
    "Dialect",
    "LoweredRegion",
    "NUMPY",
    "PythonLoweringShortfall",
    "TORCH",
    "compile_single_region_python",
    "lower_fused_program_region",
    "region_source_lines",
]
