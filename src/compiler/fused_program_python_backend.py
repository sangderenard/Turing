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

Whole functions (``lower_functions_to_numpy``, at the end of this file) take the
same route one level up: a function written against AbstractTensor is parsed
with ``ast``, every AbstractTensor spelling in it is rewritten to the NumPy
dialect's spelling from the tables here, and every ``if``/``while``/``for``/
``return``/``try`` stays exactly as authored -- control flow is not
rediscovered, it is never taken apart.  The product is Python source, cached
by the digest of the source it was lowered from plus this file.

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


def supported_elementwise_operations() -> frozenset[str]:
    """Elementwise operations the Python fidelity oracle can really lower."""

    return frozenset(
        (set(_ELEMENTWISE_TEMPLATES) | set(_NAMED_FUNCTIONS))
        & (ELEMENTWISE_UNARY | ELEMENTWISE_BINARY)
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
    if op == "matmul":
        if len(step.input_ids) != 2:
            raise PythonLoweringShortfall("matmul requires two operands")
        b = names[step.input_ids[1]]
        if dialect is NUMPY:
            return f"np.matmul({a}, {b})"
        if dialect is TORCH:
            return f"torch.matmul({a}, {b})"
        if dialect is ABSTRACT_TENSOR:
            return f"{a}.matmul({b})"
        raise PythonLoweringShortfall(
            f"matmul has no {dialect.name} spelling registered"
        )
    if op in {"reshape", "view"}:
        # The extents are the step's remaining operands (``x.reshape(2, 3)``
        # captures 2 and 3 as values), else a recorded ``shape``.
        if len(step.input_ids) > 1:
            extents = [f"int({names[i]})" for i in step.input_ids[1:]]
        elif "shape" in step.attrs:
            extents = [repr(int(e)) for e in step.attrs["shape"]]
        else:
            raise PythonLoweringShortfall(f"{op} has no target shape: {step!r}")
        return f"{a}.reshape({', '.join(extents)})"
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


# ----------------------------------------------------------------------------
# Whole-function lowering: AbstractTensor source -> NumPy dialect source.
#
# The value domain.  An AbstractTensor on the NumPy backend IS a NumPy array
# (``.data``); every operation in the tables below is that backend's own call
# (``numpy_backend.maximum_`` is ``np.maximum``, ``clamp_`` is ``np.clip`` ...),
# so the lowered function computes the same IEEE results, in the same kernels,
# on the same shapes and dtypes.  The one spelling that is not a single NumPy
# call, ``AbstractTensor.tensor``, is a named helper in ``_NUMPY_PRELUDE`` that
# reproduces the library's shape/dtype rule exactly; ``isinstance(x,
# AbstractTensor)`` is ``isinstance(x, np.ndarray)``.
# ----------------------------------------------------------------------------

#: ``AbstractTensor.<name>(...)`` -> NumPy dialect call.  Each row is the
#: backend call the AbstractTensor function makes (``numpy_backend``).
NUMPY_ABSTRACT_TENSOR_FUNCTIONS: dict[str, str] = {
    "tensor": "_at_tensor",          # shape/dtype rule: see _NUMPY_PRELUDE
    "maximum": "np.maximum",         # numpy_backend.maximum_ (NaN-propagating)
    "minimum": "np.minimum",         # numpy_backend.minimum_ (NaN-propagating)
    "where": "np.where",             # numpy_backend.where_
    "zeros_like": "np.zeros_like",
    "ones_like": "np.ones_like",
    "zeros": "np.zeros",
}

#: ``x.<name>(...)`` tensor methods that NumPy arrays do not carry under that
#: name -> the backend's module-function call.
NUMPY_TENSOR_METHODS: dict[str, str] = {
    "log": "np.log", "exp": "np.exp", "sqrt": "np.sqrt", "floor": "np.floor",
    "isfinite": "np.isfinite",
}

#: Reductions AbstractTensor spells ``dim=``/``keepdim=`` and NumPy ``axis=``/``keepdims=``.
NUMPY_AXIS_METHODS = frozenset({"sum", "max", "min", "any", "all", "mean", "prod"})

_NUMPY_PRELUDE = '''
def _at_tensor(data):
    """``AbstractTensor.tensor(data)`` on the NumPy backend, rule for rule: an
    array of any rank is copied as it is; a list or tuple is ``np.array``; a
    Python or NumPy scalar becomes a one-element array whose dtype is its kind
    (bool, int64, or float64 for every float width)."""
    if isinstance(data, np.ndarray):
        return np.array(data)
    if isinstance(data, (list, tuple)):
        return np.array(data)
    if isinstance(data, (bool, np.bool_)):
        return np.array([data], dtype=np.bool_)
    if isinstance(data, (int, np.integer)):
        return np.array([data], dtype=np.int64)
    return np.array([data], dtype=np.float64)


def _at_truediv(a, b):
    """``a / b`` where either side is an AbstractTensor: the backend's
    ``_safe_divide`` in its ``zero`` mode (a zero divisor gives 0, not inf or
    nan; the result is at least float32).  Neither side a tensor is Python's
    own division, ZeroDivisionError included."""
    if not (isinstance(a, np.ndarray) or isinstance(b, np.ndarray)):
        return a / b
    dtype = np.result_type(a, b, np.float32)
    a = np.asarray(a, dtype=dtype)
    b = np.asarray(b, dtype=dtype)
    out = np.empty(np.broadcast(a, b).shape, dtype=dtype)
    mask = np.broadcast_to(b != 0, out.shape)
    np.divide(a, b, out=out, where=mask)
    out[~mask] = 0
    return out
'''

#: The division mode ``_at_truediv`` spells.  Lowering refuses a backend whose
#: ``DIVIDE_MODE`` is anything else rather than approximate it.
NUMPY_DIVIDE_MODE = "zero"

_BUILTIN_NAMES = frozenset(dir(__import__("builtins")))


class _NumpyDialectRewriter(ast.NodeTransformer):
    """Rewrite one function's AbstractTensor spellings; touch nothing else."""

    #: Names bound to modules in the function's globals: ``math.sqrt(x)`` is a
    #: module function, not the tensor method ``x.sqrt()``.
    modules: frozenset = frozenset({"np", "numpy", "math"})

    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.op, ast.Div):
            return ast.copy_location(
                ast.Call(ast.Name("_at_truediv", ast.Load()), [node.left, node.right], []), node
            )
        return node

    def visit_AugAssign(self, node):
        if isinstance(node.op, ast.Div):
            raise PythonLoweringShortfall(
                "an in-place true division has no NumPy dialect spelling here "
                f"(line {getattr(node, 'lineno', '?')})"
            )
        return self.generic_visit(node)

    def visit_Name(self, node):
        if node.id == "AbstractTensor":
            return ast.copy_location(
                ast.Attribute(ast.Name("np", ast.Load()), "ndarray", node.ctx), node
            )
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        function = node.func
        if not isinstance(function, ast.Attribute):
            return node
        owner = function.value
        if isinstance(owner, ast.Name) and owner.id in self.modules:
            return node
        # ``AbstractTensor.<fn>`` arrives here as ``np.ndarray.<fn>``.
        if (
            isinstance(owner, ast.Attribute) and owner.attr == "ndarray"
            and isinstance(owner.value, ast.Name) and owner.value.id == "np"
            and function.attr in NUMPY_ABSTRACT_TENSOR_FUNCTIONS
        ):
            head = ast.parse(NUMPY_ABSTRACT_TENSOR_FUNCTIONS[function.attr], mode="eval").body
            node.func = ast.copy_location(head, function)
            return node
        if function.attr in NUMPY_TENSOR_METHODS and not node.keywords:
            head = ast.parse(NUMPY_TENSOR_METHODS[function.attr], mode="eval").body
            return ast.copy_location(
                ast.Call(head, [function.value, *node.args], []), node
            )
        if function.attr == "clamp":
            keywords = {k.arg: k.value for k in node.keywords}
            positional = list(node.args)
            low = keywords.get("min", positional[0] if positional else ast.Constant(None))
            high = keywords.get("max", positional[1] if len(positional) > 1 else ast.Constant(None))
            head = ast.parse("np.clip", mode="eval").body
            return ast.copy_location(ast.Call(head, [function.value, low, high], []), node)
        if function.attr in NUMPY_AXIS_METHODS:
            for keyword in node.keywords:
                if keyword.arg == "dim":
                    keyword.arg = "axis"
                elif keyword.arg == "keepdim":
                    keyword.arg = "keepdims"
        return node


def _bound_names(tree) -> set[str]:
    """Every name a lowered function binds itself (parameters, assignments,
    loop and comprehension targets, nested defs, imports)."""
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, ast.arguments):
            for argument in (*node.posonlyargs, *node.args, *node.kwonlyargs, node.vararg, node.kwarg):
                if argument is not None:
                    bound.add(argument.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node is not tree:
            bound.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
    return bound


def _importable(value):
    module = getattr(value, "__module__", None)
    name = getattr(value, "__name__", None)
    if module is None or name is None or getattr(value, "__qualname__", name) != name:
        return None
    import importlib

    try:
        if getattr(importlib.import_module(module), name) is value:
            return module, name
    except Exception:
        return None
    return None


@dataclass(frozen=True)
class LoweredPythonModule:
    """Whole functions lowered to the NumPy dialect: the source, the functions
    it defines, the names the runtime must still bind (the pieces a generated
    program calls by name), and the digest it is cached under."""

    source: str
    functions: tuple[str, ...]
    injected: tuple[str, ...]
    digest: str


def lower_functions_to_numpy(
    roots,
    *,
    namespace: Mapping[str, Any],
    owned,
    subclass_methods: Mapping[Any, tuple[str, ...]] | None = None,
) -> LoweredPythonModule:
    """Lower ``roots`` and everything they reach that ``owned`` admits.

    ``roots`` are function objects or source texts (generated source, whose
    free names resolve in ``namespace``).  A function ``owned(value)`` admits
    is lowered with its callers.  Any other name a lowered function reads is
    bound by an ``import`` or literal at the top of the product, or listed in
    ``injected`` when only the runtime has it.  ``subclass_methods`` maps an
    owned class to the methods to lower; the product defines a subclass of the
    original carrying them, so ``isinstance`` and the fields stay the original's.
    """
    import hashlib
    import inspect
    import sys
    import textwrap

    subclass_methods = dict(subclass_methods or {})
    queue: list[tuple[str, ast.AST, Mapping[str, Any]]] = []
    seen: set[int] = set()
    lowered: dict[str, ast.AST] = {}
    bases: dict[str, Any] = {}
    imports: dict[str, str] = {}
    literals: dict[str, str] = {}
    injected: set[str] = set()
    sources: list[str] = []
    defined: set[str] = {"_at_tensor", "_at_truediv"}  # the prelude's helpers
    from ..common.tensors.numpy_backend import NumPyTensorOperations

    if NumPyTensorOperations.DIVIDE_MODE != NUMPY_DIVIDE_MODE:
        raise PythonLoweringShortfall(
            f"the NumPy backend divides in mode {NumPyTensorOperations.DIVIDE_MODE!r}; "
            f"the dialect spells {NUMPY_DIVIDE_MODE!r}"
        )

    def enqueue_function(function):
        if id(function) in seen:
            return
        seen.add(id(function))
        text = textwrap.dedent(inspect.getsource(function))
        sources.append(text)
        queue.append((function.__name__, ast.parse(text).body[0], function.__globals__))
        defined.add(function.__name__)

    def enqueue_class(cls, methods):
        if id(cls) in seen:
            return
        seen.add(id(cls))
        text = textwrap.dedent(inspect.getsource(cls))
        sources.append(text)
        tree = ast.parse(text).body[0]
        tree.body = [
            member for member in tree.body
            if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
            and member.name in methods
        ]
        tree.decorator_list = []
        tree.bases = [ast.Name(f"_{cls.__name__}_base", ast.Load())]
        bases[cls.__name__] = cls
        defined.add(f"_{cls.__name__}_base")
        queue.append((cls.__name__, tree, sys.modules[cls.__module__].__dict__))
        defined.add(cls.__name__)

    for root in roots:
        if isinstance(root, str):
            sources.append(root)
            for node in ast.parse(root).body:
                if isinstance(node, ast.FunctionDef):
                    queue.append((node.name, node, namespace))
                    defined.add(node.name)
        else:
            enqueue_function(root)
    for cls, methods in subclass_methods.items():
        enqueue_class(cls, methods)

    def strip_imports(tree, globals_):
        """Drop local imports of AbstractTensor and of lowered functions;
        make every other relative import absolute."""
        import importlib

        class Strip(ast.NodeTransformer):
            def visit_ImportFrom(self, node):
                module = node.module or ""
                if node.level:
                    parts = (globals_.get("__package__") or "").split(".")
                    base = ".".join(parts[: len(parts) - node.level + 1])
                    module = f"{base}.{module}" if module else base
                kept = []
                for alias in node.names:
                    if alias.name == "AbstractTensor":
                        continue
                    target = getattr(importlib.import_module(module), alias.name, None)
                    if inspect.isfunction(target) and owned(target):
                        enqueue_function(target)
                        continue
                    kept.append(alias)
                if not kept:
                    return None
                return ast.copy_location(ast.ImportFrom(module=module, names=kept, level=0), node)

        return Strip().visit(tree)

    rewriter = _NumpyDialectRewriter()
    processed = 0
    while processed < len(queue):
        name, tree, globals_ = queue[processed]
        processed += 1
        rewriter.modules = frozenset({"np", "numpy"}) | frozenset(
            key for key, value in globals_.items() if inspect.ismodule(value)
        ) | frozenset(
            (alias.asname or alias.name).split(".")[0]
            for node in ast.walk(tree) if isinstance(node, ast.Import)
            for alias in node.names
        )
        tree = strip_imports(rewriter.visit(tree), globals_)
        ast.fix_missing_locations(tree)
        lowered[name] = tree
        local = _bound_names(tree)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)):
                continue
            ident = node.id
            if ident in local or ident in defined or ident == "np" or ident in _BUILTIN_NAMES:
                continue
            if ident not in globals_:
                injected.add(ident)
                continue
            value = globals_[ident]
            if inspect.isfunction(value) and owned(value):
                enqueue_function(value)
            elif inspect.isclass(value) and value in subclass_methods:
                enqueue_class(value, subclass_methods[value])
            elif inspect.ismodule(value):
                imports[ident] = f"import {value.__name__}" + (
                    "" if value.__name__ == ident else f" as {ident}"
                )
            elif isinstance(value, (bool, int, float, str, type(None))):
                literals[ident] = repr(value)
            else:
                found = _importable(value)
                if found is None:
                    injected.add(ident)
                else:
                    module, attribute = found
                    imports[ident] = f"from {module} import {attribute}" + (
                        "" if attribute == ident else f" as {ident}"
                    )

    for cls in bases.values():
        imports[f"_{cls.__name__}_base"] = (
            f"from {cls.__module__} import {cls.__name__} as _{cls.__name__}_base"
        )
    digest = hashlib.sha256(
        ("\0".join(sources) + "\0" + inspect.getsource(sys.modules[__name__])).encode("utf-8")
    ).hexdigest()[:16]
    banner = (
        '"""Whole functions lowered from AbstractTensor source to the NumPy dialect '
        f"(source digest {digest}).\n\nGenerated by "
        "src.compiler.fused_program_python_backend.lower_functions_to_numpy; "
        'change the source and it is lowered again.  Do not edit."""\n'
    )
    header = "\n".join((
        "from __future__ import annotations",
        "import numpy as np",
        *sorted(set(imports.values())),
        *(f"{key} = {value}" for key, value in sorted(literals.items())),
    ))
    body = "\n\n\n".join(ast.unparse(tree) for tree in lowered.values())
    return LoweredPythonModule(
        source=banner + header + "\n" + _NUMPY_PRELUDE + "\n\n" + body + "\n",
        functions=tuple(lowered),
        injected=tuple(sorted(injected)),
        digest=digest,
    )


__all__ = [
    "ABSTRACT_TENSOR",
    "CompiledPythonProgram",
    "DIALECTS",
    "Dialect",
    "LoweredPythonModule",
    "LoweredRegion",
    "NUMPY",
    "NUMPY_ABSTRACT_TENSOR_FUNCTIONS",
    "NUMPY_TENSOR_METHODS",
    "PythonLoweringShortfall",
    "TORCH",
    "compile_single_region_python",
    "lower_functions_to_numpy",
    "lower_fused_program_region",
    "region_source_lines",
    "supported_elementwise_operations",
]
