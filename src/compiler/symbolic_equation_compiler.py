"""Compile named SymPy equations into one repository-SSA function.

The equations are the numerical authority.  This module only coordinates the
existing SymPy -> ProcessGraph translator and ProcessGraph -> SSA scheduler;
it does not evaluate, rewrite, or reimplement their right-hand sides.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import copy
import inspect
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

import sympy

from .hierarchical_plan import is_predicate_operation
from .ir_identities import reduce_constant_exponent_pow
from .ssa_builder import process_graph_to_ssa_instrs
from .symbolic_process_graph import (
    ingest_sympy_expression,
    ingest_sympy_expressions,
    matrix_component_name,
)
from .external_functions import declared_external_functions
from .sympy_dual_ir_cache import SympyDualIRCache
from ..common.tensors.accelerator_backends.aot_checkpoint import callable_digest
from ..common.tensors.accelerator_backends.artifact_cache import implementation_digest
from ..transmogrifier.graph.graph_express2 import ProcessGraph
from ..transmogrifier.ssa import BasicBlock, Function, IRModule, Instr, SSAValue


@dataclass(frozen=True, slots=True)
class SymbolicPublication:
    """Backend-neutral meaning assigned to one named symbolic result."""

    output: str
    semantic: str
    presentation: str = "field"
    unit: str | None = None


@dataclass(frozen=True, slots=True)
class SymbolicEquationCompilation:
    """Inspectable checkpoints from authored equations to repository SSA."""

    equations: tuple[sympy.Equality, ...]
    process_graph: ProcessGraph
    instructions: tuple[Instr, ...]
    function: Function
    module: IRModule
    input_ids: Mapping[str, int]
    output_ids: Mapping[str, int]
    publications: tuple[SymbolicPublication, ...]
    cache_identity: str | None = None
    cache_hit: bool = False


def _numeric_constant(value: Any) -> Any:
    # ProcessGraph spells SymPy's singletons (One, Zero, NegativeOne) as
    # Python ints.  Left as ints, a node this module declares float64 kept
    # an integer payload, and a consumer that stores the constant into a
    # value slot (a Piecewise Select arm) stored ``i64 1`` and read it back
    # as the double 5e-324.  A predicate's bool is not a number here.
    if isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, sympy.Integer):
        return float(value)
    if isinstance(value, (sympy.Rational, sympy.Float)):
        return float(value)
    return value


@dataclass(frozen=True, slots=True)
class SymbolicOutputDeclaration:
    """One declared output of an authored equation set.

    ``component`` is the element index of a matrix-valued equation (``()``
    for a scalar one).  ``form`` is ``"assignment"`` when the lhs is a
    declared name (a Symbol, or an element of a MatrixSymbol) and the output
    is the rhs, or ``"residual"`` when the lhs is any other expression and
    the output is ``lhs - rhs`` (the lane's linear-system convention; zero
    where the equation holds).
    """

    name: str
    equation_index: int
    component: tuple[int, ...]
    form: str
    expression: sympy.Basic


def _is_matrix(value: Any) -> bool:
    return isinstance(value, (sympy.MatrixBase, sympy.MatrixExpr))


def _scalar_output(
    lhs: sympy.Basic, rhs: sympy.Basic, unnamed: str,
    equation_index: int, component: tuple[int, ...],
) -> SymbolicOutputDeclaration:
    from sympy.matrices.expressions.matexpr import MatrixElement


    if isinstance(lhs, sympy.Symbol):
        return SymbolicOutputDeclaration(
            str(lhs), equation_index, component, "assignment", rhs)
    if (
        isinstance(lhs, MatrixElement)
        and isinstance(lhs.args[0], sympy.MatrixSymbol)
        and all(getattr(item, "is_Integer", False) for item in lhs.args[1:])
    ):
        return SymbolicOutputDeclaration(
            matrix_component_name(
                lhs.args[0].name, tuple(int(item) for item in lhs.args[1:])),
            equation_index, component, "assignment", rhs)
    # As written: the residual is the two authored sides, unevaluated.
    residual = sympy.Add(
        lhs, sympy.Mul(sympy.Integer(-1), rhs, evaluate=False), evaluate=False)
    return SymbolicOutputDeclaration(
        unnamed, equation_index, component, "residual", residual)


def declared_symbolic_outputs(
    equations: Sequence[sympy.Equality], name: str,
) -> tuple[SymbolicOutputDeclaration, ...]:
    """The outputs an authored equation set declares, in authored order.

    A scalar equation declares one output; a matrix-valued equation (lhs a
    Matrix, MatrixSymbol or matrix expression, rhs of the same shape)
    declares one output per component, row-major.  An output whose lhs is
    not a declared name is named from the equation: a MatrixSymbol lhs names
    its components ``matrix_component_name(M, (i, j))``; any other equation
    is named ``<name>_<equation index>`` and its components
    ``<name>_<equation index>_<i>_<j>``.
    """


    declarations: list[SymbolicOutputDeclaration] = []
    for index, equation in enumerate(equations):
        if not isinstance(equation, sympy.Equality):
            raise TypeError(f"expected a SymPy Equality, got {equation!r}")
        lhs, rhs = equation.lhs, equation.rhs
        unnamed = f"{name}_{index}"
        if _is_matrix(lhs) or _is_matrix(rhs):
            if not (_is_matrix(lhs) and _is_matrix(rhs)):
                raise TypeError(
                    "a matrix-valued equation needs a matrix on both sides: "
                    f"{equation!r}")
            if tuple(lhs.shape) != tuple(rhs.shape):
                raise TypeError(
                    f"equation sides differ in shape {tuple(lhs.shape)} vs "
                    f"{tuple(rhs.shape)}: {equation!r}")
            rows, columns = (int(extent) for extent in lhs.shape)
            base = lhs.name if isinstance(lhs, sympy.MatrixSymbol) else unnamed
            for row in range(rows):
                for column in range(columns):
                    component = (row, column)
                    declarations.append(_scalar_output(
                        lhs[row, column], rhs[row, column],
                        matrix_component_name(base, component),
                        index, component,
                    ))
            continue
        declarations.append(_scalar_output(lhs, rhs, unnamed, index, ()))
    return tuple(declarations)


def _compile_sympy_equations_uncached(
    equations: Sequence[sympy.Equality],
    *,
    name: str = "symbolic_equation_step",
    schedule: str = "asap",
    publications: Sequence[SymbolicPublication] = (),
    dtype: str = "float64",
) -> SymbolicEquationCompilation:
    """Lower simultaneous named SymPy equations into repository SSA.

    Each equation declares its outputs (``declared_symbolic_outputs``): a
    Symbol lhs names one result, a matrix-valued equation one result per
    component, and an lhs that is not a declared name contributes its
    residual.  Every output expression is compiled verbatim through the
    canonical SymPy ProcessGraph importer.  Output names are not permitted
    as right-hand-side inputs: one invocation is a simultaneous state
    transition, and recurrence belongs to the caller that feeds the returned
    state into the next invocation.
    """

    authored = tuple(equations)
    if not authored:
        raise ValueError("symbolic equation program requires equations")
    declarations = declared_symbolic_outputs(authored, name)
    output_names = tuple(row.name for row in declarations)
    if len(output_names) != len(set(output_names)):
        raise ValueError("symbolic equation output names must be unique")
    output_symbols = frozenset(
        equation.lhs for equation in authored
        if isinstance(equation.lhs, sympy.Symbol)
    )
    recursive = {
        str(symbol)
        for row in declarations if row.form == "assignment"
        for symbol in row.expression.free_symbols & output_symbols
    }
    if recursive:
        raise ValueError(
            "simultaneous next-state outputs cannot be RHS inputs: "
            + ", ".join(sorted(recursive))
        )

    publication_rows = tuple(publications)
    unknown_publications = {
        row.output for row in publication_rows
    } - set(output_names)
    if unknown_publications:
        raise ValueError(
            "publications name unknown symbolic outputs: "
            + ", ".join(sorted(unknown_publications))
        )

    if dtype not in {"float32", "float64"}:
        raise ValueError("symbolic equation dtype must be float32 or float64")
    graph = ProcessGraph(materialize_memory=False, source_language="sympy")
    # The law's equations and declared outputs are on the book before the
    # graph exists, so every ingested node derives from its equation's output
    # cell (the same rows ``_post_symbolic_outputs`` posts per call).
    output_cells = _post_symbolic_program(
        name,
        tuple(sympy.srepr(equation) for equation in authored),
        authored,
        tuple(
            (row.name, row.equation_index, row.component, row.form)
            for row in declarations
        ),
    )
    roots = ingest_sympy_expressions(
        graph,
        tuple(row.expression for row in declarations),
        output_names=output_names,
        strict=True,
        expression_sources=tuple(output_cells[row.name] for row in declarations),
    )
    # Input columns are declared by the importer (a Symbol by its name, an
    # element of a MatrixSymbol by ``matrix_component_name``).  Two inputs
    # spelled alike, or an input spelled like an output, would be one column
    # standing for two values: refuse rather than let the spelling decide.
    input_columns: dict[str, sympy.Basic] = {}
    for _node_id, data in graph.G.nodes(data=True):
        if data.get("op") not in {"input", "Input", "Symbol"}:
            continue
        column = str((data.get("attributes") or {}).get("binding_name"))
        value = data.get("expr_obj")
        incumbent = input_columns.setdefault(column, value)
        if incumbent != value:
            raise ValueError(
                f"symbolic input column {column!r} is declared by both "
                f"{incumbent!r} and {value!r}")
    recursive_columns = set(input_columns) & set(output_names)
    if recursive_columns:
        raise ValueError(
            "simultaneous next-state outputs cannot be RHS inputs: "
            + ", ".join(sorted(recursive_columns))
        )
    matrix_element_inputs = tuple(sorted(
        (
            str(attributes["binding_name"]),
            str(attributes["matrix_symbol"]),
            tuple(attributes["matrix_shape"]),
            tuple(attributes["matrix_index"]),
        )
        for _node_id, data in graph.G.nodes(data=True)
        if data.get("op") in {"input", "Input", "Symbol"}
        for attributes in ((data.get("attributes") or {}),)
        if "matrix_symbol" in attributes
    ))
    # These equations are a floating physical model.  SymPy retains exact
    # integer/rational literals in the authored form, while the compiled ABI
    # consistently carries scalar f64 values across all native targets.
    for _node_id, data in graph.G.nodes(data=True):
        # A relation's result is not a value of the model, it is a
        # predicate, and blanket float64 erased that. The backend cannot
        # recover it either: `Lt` emits `fcmp`, which yields i1 whatever
        # the SSA declared, so the value disagreed with its own rendering
        # and the first consumer of it -- a Piecewise select -- failed
        # verification. Declaring it here fixes every target at once.
        spelling = str(data.get("op") or data.get("type") or "")
        data["tensor"] = {
            "dtype": "bool" if is_predicate_operation(spelling) else dtype,
            "shape": (),
        }
        if str(data.get("type") or data.get("op") or "").casefold() in {
            "const", "constant",
        }:
            attributes = data.setdefault("attributes", {})
            if "value" in attributes:
                attributes["value"] = _numeric_constant(attributes["value"])
            if "constant" in attributes:
                attributes["constant"] = _numeric_constant(
                    attributes["constant"]
                )
            if "constant" in data:
                data["constant"] = _numeric_constant(data["constant"])
        if data.get("op") in {"input", "Input", "Symbol"}:
            data["type"] = "Input"
            data["op"] = "input"
            data.setdefault("attributes", {})["binding_kind"] = "parameter"

    symbolic_inputs = sorted(
        (
            str(data.get("attributes", {}).get("binding_name")),
            int(node_id),
        )
        for node_id, data in graph.G.nodes(data=True)
        if data.get("op") == "input"
    )
    graph.G.graph.update(
        function_name=name,
        function_parameters=tuple(row[0] for row in symbolic_inputs),
        positional_parameters=tuple(row[0] for row in symbolic_inputs),
        keyword_only_parameters=(),
        parameter_defaults={},
        canonical_value_ids=True,
        identity_table={
            **{input_name: (node_id,) for input_name, node_id in symbolic_inputs},
            **{output_name: (int(root),) for output_name, root in zip(output_names, roots)},
        },
        symbolic_equations=tuple(sympy.srepr(eq) for eq in authored),
    )

    # Scheduling may install storage nodes.  Retain the pre-schedule graph as
    # the authored function body that other front ends link through the shared
    # FunctionTable, and schedule an independent copy into repository SSA.
    authored_graph = graph
    scheduled = tuple(
        process_graph_to_ssa_instrs(copy.deepcopy(graph), schedule=schedule)
    )
    if dtype == "float32":
        # The scheduler historically spells exact numeric constants as f64
        # even when their graph node carries an explicit f32 contract.  A
        # symbolic WebGPU program needs one consistent storage dtype, so
        # preserve predicates and narrow only those residual numeric values.
        for instruction in scheduled:
            for value in (*instruction.args, instruction.res):
                if value is not None and value.dtype in {None, "float64", "double", "f64"}:
                    value.dtype = "float32"
    input_instructions = {
        int(instruction.res.id): instruction
        for instruction in scheduled
        if instruction.op in {"input", "Input", "Symbol"}
    }
    input_rows = sorted(
        (
            str(instruction.attributes.get("binding_name")),
            value_id,
            instruction,
        )
        for value_id, instruction in input_instructions.items()
    )
    function_args = [instruction.res for _name, _id, instruction in input_rows]
    body: list[Instr] = []
    for instruction in scheduled:
        if instruction.op in {"input", "Input", "Symbol"}:
            continue
        if str(instruction.op).startswith("Store["):
            continue
        if instruction.op in {"const", "Constant"}:
            attributes = dict(instruction.attributes)
            payload = attributes.get("constant", attributes.get("value"))
            attributes["constant"] = _numeric_constant(payload)
            instruction = Instr(
                "Const", list(instruction.args), instruction.res,
                arg_roles=list(instruction.arg_roles),
                attributes=attributes,
                source_span=instruction.source_span,
            )
        # SymPy function nodes arrive through ProcessGraph as direct calls
        # (for example ``callee='acos'``).  Late native backends intentionally
        # consume the canonical tensor primitive ABI rather than linking an
        # arbitrary source-language function name.  Preserve the operation in
        # metadata and route it through that ABI so symbolic arc/contact math
        # can lower to C, LLVM, and the other tensor targets uniformly.
        if instruction.op in {"Call", "call"}:
            callee = str(instruction.attributes.get("callee") or "")
            if tuple(getattr(instruction.res, "shape", ()) or ()) and callee in {
                "acos", "acosh", "asin", "asinh", "atan", "atanh",
                "cos", "cosh", "exp", "log", "sin", "sinh", "sqrt",
                "tan", "tanh",
            }:
                attributes = dict(instruction.attributes)
                attributes["callee"] = "unary_double"
                attributes["tensor_operation"] = callee
                instruction = Instr(
                    "Call", list(instruction.args), instruction.res,
                    arg_roles=list(instruction.arg_roles),
                    attributes=attributes,
                    source_span=instruction.source_span,
                )
        body.append(instruction)
    if dtype == "float32":
        for instruction in body:
            for value in (*instruction.args, instruction.res):
                if value is not None and value.dtype in {None, "float64", "double", "f64"}:
                    value.dtype = "float32"
    output_values = [SSAValue(int(root), dtype) for root in roots]
    body.append(Instr("Ret", output_values, None))
    function = Function(
        name,
        function_args,
        {"entry": BasicBlock("entry", body)},
        metadata={
            "argument_names": tuple(row[0] for row in input_rows),
            "output_names": output_names,
            "parameter_names": tuple(
                (row[0], int(row[2].res.id)) for row in input_rows
            ),
            "named_outputs": tuple(
                (output_name, int(root))
                for output_name, root in zip(output_names, roots)
            ),
            "symbolic_equations": tuple(sympy.srepr(eq) for eq in authored),
            # (output, equation index, component index, form) per output,
            # in output order; posted on the book by compile_sympy_equations.
            "symbolic_outputs": tuple(
                (row.name, row.equation_index, row.component, row.form)
                for row in declarations
            ),
            # (column, MatrixSymbol, shape, element index) per input column
            # that is an element of a MatrixSymbol.
            "matrix_element_inputs": matrix_element_inputs,
            # (name, arity) per undefined Function the outputs apply: each is
            # a declared external the host supplies at runtime
            # (``external_functions``).
            "external_functions": tuple(
                (row.name, row.arity)
                for row in declared_external_functions(
                    tuple(row.expression for row in declarations))),
            "symbolic_source": "sympy",
            "symbolic_dtype": dtype,
            "publications": tuple(
                {
                    "output": row.output,
                    "semantic": row.semantic,
                    "presentation": row.presentation,
                    "unit": row.unit,
                }
                for row in publication_rows
            ),
        },
    )
    module = IRModule({name: function})
    # The same contract-governed identity pass the whole-program path runs at
    # finalization; without it the direct scalar lanes would receive raw Pow
    # and each backend's private spelling table would become a second,
    # unaudited policy.
    reduce_constant_exponent_pow(module.functions)
    if dtype == "float32":
        for current in module.functions.values():
            for block in current.blocks.values():
                for instruction in block.instrs:
                    for value in (*instruction.args, instruction.res):
                        if value is not None and value.dtype in {None, "float64", "double", "f64"}:
                            value.dtype = "float32"
    return SymbolicEquationCompilation(
        equations=authored,
        process_graph=authored_graph,
        instructions=tuple(function.blocks["entry"].instrs),
        function=function,
        module=module,
        input_ids={row[0]: row[1] for row in input_rows},
        output_ids=dict(zip(output_names, roots)),
        publications=publication_rows,
    )


def _publication_record(publication: SymbolicPublication) -> Mapping[str, Any]:
    return {
        "output": publication.output,
        "semantic": publication.semantic,
        "presentation": publication.presentation,
        "unit": publication.unit,
    }


def compile_sympy_equations(
    equations: Sequence[sympy.Equality],
    *,
    name: str = "symbolic_equation_step",
    schedule: str = "asap",
    publications: Sequence[SymbolicPublication] = (),
    dtype: str = "float64",
) -> SymbolicEquationCompilation:
    """Lower equations once, then reuse their persistent repository dual IR.

    The cache identity contains the canonical symbolic structure, ordered live
    parameter ABI (inherent in the equations), publications, dtype, scheduling
    policy, interpreter/SymPy serialization versions, and the lowering
    implementation digest.  Runtime parameter values remain outside the key
    unless a caller intentionally specializes them into the equations.
    """

    authored = tuple(equations)
    publication_rows = tuple(publications)
    implementation = _pipeline_implementation()
    record = {
        "name": str(name),
        "schedule": str(schedule),
        "dtype": str(dtype),
        "equations": tuple(sympy.srepr(equation) for equation in authored),
        "publications": tuple(
            _publication_record(publication) for publication in publication_rows
        ),
        "python_cache_tag": sys.implementation.cache_tag,
        "sympy_version": sympy.__version__,
    }
    cached = SympyDualIRCache(implementation).dual_ir(
        record,
        lambda: _compile_sympy_equations_uncached(
            authored,
            name=name,
            schedule=schedule,
            publications=publication_rows,
            dtype=dtype,
        ),
    )
    if not isinstance(cached.value, SymbolicEquationCompilation):
        # A locally corrupted or obsolete payload must never cross the public
        # compiler boundary. Recompute with caching disabled for this call.
        value = _compile_sympy_equations_uncached(
            authored,
            name=name,
            schedule=schedule,
            publications=publication_rows,
            dtype=dtype,
        )
        value = replace(value, cache_identity=cached.identity, cache_hit=False)
    else:
        value = replace(
            cached.value,
            cache_identity=cached.identity,
            cache_hit=cached.hit,
        )
    _post_symbolic_outputs(value, name)
    return value


def symbolic_program_scope(
    compilation: SymbolicEquationCompilation, name: str,
) -> tuple[str, str]:
    """The book scope of one compiled set: (law name, digest of its
    authored equations), the ``program`` field of ``symbolic_equation`` and
    ``symbolic_equation_output`` rows."""

    reprs = tuple(compilation.function.metadata.get("symbolic_equations") or ())
    return _symbolic_program_key(name, reprs)


def _symbolic_program_key(name: str, reprs: Sequence[str]) -> tuple[str, str]:
    """(law name, digest of the authored equations' sreprs): the one key
    ``symbolic_program_scope`` reads back from a compilation's metadata and
    ``_post_symbolic_program`` posts under before ingestion."""

    import hashlib

    return (
        str(name),
        hashlib.sha256("\n".join(reprs).encode("utf-8")).hexdigest(),
    )


def _post_symbolic_program(
    name: str,
    reprs: Sequence[str],
    equations: Sequence[sympy.Equality],
    outputs: Sequence[tuple],
) -> dict[str, Any]:
    """Post each authored equation (NOVEL root) and each declared output
    (DERIVED from its equation's cell) on the active book; return the
    output cells by output name.  ``CONCORD``: posting the same set again
    writes no new cell, so the pre-ingestion post and the per-call post of
    ``_post_symbolic_outputs`` agree."""

    import hashlib

    from .concordance_declarations import (
        INGEST_SOURCE, INGESTION, SYMBOLIC_EQUATION, SYMBOLIC_EQUATION_OUTPUT,
        SymbolicEquationFact, SymbolicOutputFact, SymbolicOutputForm,
    )
    from .identity_concordance import (
        Derived, Mode, Novel, current_identity_book,
    )

    program = _symbolic_program_key(name, reprs)
    book = current_identity_book()
    cells = {}
    for index, (text, equation) in enumerate(zip(reprs, equations)):
        lhs = equation.lhs
        cells[index] = book.post(
            SYMBOLIC_EQUATION, (program, index),
            SymbolicEquationFact(
                hashlib.sha256(text.encode("utf-8")).hexdigest(),
                tuple(int(extent) for extent in lhs.shape) if _is_matrix(lhs) else (),
            ),
            stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()),
            mode=Mode.CONCORD,
        )
    output_cells = {}
    for output, equation_index, component, form in outputs:
        output_cells[str(output)] = book.post(
            SYMBOLIC_EQUATION_OUTPUT, (program, str(output)),
            SymbolicOutputFact(
                int(equation_index), tuple(component), SymbolicOutputForm(form)),
            stage=INGESTION, provenance=Derived((cells[int(equation_index)],)),
            mode=Mode.CONCORD,
        )
    # Undefined Functions the outputs apply: declared externals (NOVEL,
    # minted) and their callsites (DERIVED), ``external_functions``.
    from .external_functions import post_external_functions

    post_external_functions(
        program, tuple(equations), cells,
        tuple((row.name, row.equation_index, row.expression)
              for row in declared_symbolic_outputs(tuple(equations), name)),
        output_cells,
    )
    return output_cells


def _post_symbolic_outputs(
    compilation: SymbolicEquationCompilation, name: str,
) -> None:
    """Record each authored equation and the outputs it declares on the book.

    Posted on every call, cache hit or not, into the active book (the one the
    compilation's own module tables were posted to).  The equation is a NOVEL
    root; each output is DERIVED from its equation's cell, with the component
    index and the form.  ``CONCORD``: the same set compiled again writes no
    new cell.
    """

    metadata = compilation.function.metadata
    reprs = tuple(metadata.get("symbolic_equations") or ())
    outputs = tuple(metadata.get("symbolic_outputs") or ())
    if not reprs or not outputs:
        return
    _post_symbolic_program(name, reprs, compilation.equations, outputs)


def _pipeline_implementation() -> str:
    """Digest of the lowering implementation every cached layer depends on."""

    return callable_digest(
        _compile_sympy_equations_uncached,
        declared_symbolic_outputs,
        _scalar_output,
        SymbolicOutputDeclaration,
        declared_external_functions,
        matrix_component_name,
        ingest_sympy_expression,
        ingest_sympy_expressions,
        process_graph_to_ssa_instrs,
        reduce_constant_exponent_pow,
        ProcessGraph,
        Function,
        IRModule,
    )


def _source_files(*values: Any) -> tuple[Path, ...]:
    paths: list[Path] = []
    for value in values:
        try:
            path = inspect.getsourcefile(value)
        except TypeError:
            path = None
        if not path:
            raise TypeError(
                f"{value!r} has no source file; a symbolic producer must be "
                "authored in a module so its revision can be digested"
            )
        paths.append(Path(path))
    return tuple(paths)


def _producer_record(
    producer: Callable[[], Any], key_sources: Sequence[Any],
) -> tuple[Mapping[str, Any], str]:
    """Cheap, construction-free identity of an authored symbolic program.

    The producer builds sympy expressions whose automatic evaluation can take
    minutes for a large model; that cost must not be paid to discover whether
    the result is already on disk.  A symbolic program with no runtime
    parameters is a pure function of its source, so the key is the digest of
    the source FILE of the producer (and of any ``key_sources`` it draws
    helpers or constants from), plus the interpreter and SymPy versions.
    Any edit to those files changes the key, so a stale program can never
    survive an edit silently.
    """

    files = _source_files(producer, *key_sources)
    source_digest = implementation_digest(files)
    record = {
        "producer": f"{producer.__module__}.{producer.__qualname__}",
        "producer_sources": source_digest,
        "python_cache_tag": sys.implementation.cache_tag,
        "sympy_version": sympy.__version__,
    }
    return record, source_digest


def symbolic_equations_cached(
    producer: Callable[[], Any], *, key_sources: Sequence[Any] = (),
) -> Any:
    """Run a zero-argument authored equation producer once per source revision.

    Returns exactly what ``producer`` returns (typically
    ``(equations, symbols)``), loaded from the persistent ``solved-equations``
    layer when the producer's source files are unchanged.
    """

    record, source_digest = _producer_record(producer, key_sources)
    cached = SympyDualIRCache(source_digest).solved_equations(record, producer)
    return cached.value


def compile_symbolic_program(
    producer: Callable[[], Any],
    *,
    name: str,
    schedule: str = "asap",
    publications: Sequence[SymbolicPublication] = (),
    dtype: str = "float64",
    key_sources: Sequence[Any] = (),
) -> SymbolicEquationCompilation:
    """Compile an authored symbolic program once per source revision.

    This is the entry point every ``compile_*_ssa`` should use.  On a hit the
    finished :class:`SymbolicEquationCompilation` is loaded without
    constructing a single sympy expression; on a miss the producer runs
    (through :func:`symbolic_equations_cached`, so a sibling that needs only
    the equations shares the work) and :func:`compile_sympy_equations`
    lowers it, populating the ``dual-ir`` layer as before.
    """

    publication_rows = tuple(publications)
    producer_record, source_digest = _producer_record(producer, key_sources)
    record = {
        **producer_record,
        "name": str(name),
        "schedule": str(schedule),
        "dtype": str(dtype),
        "publications": tuple(
            _publication_record(publication) for publication in publication_rows
        ),
    }
    implementation = f"{source_digest}:{_pipeline_implementation()}"

    def lower() -> SymbolicEquationCompilation:
        equations, _symbols = symbolic_equations_cached(
            producer, key_sources=key_sources,
        )
        return compile_sympy_equations(
            equations, name=name, schedule=schedule,
            publications=publication_rows, dtype=dtype,
        )

    cached = SympyDualIRCache(implementation).get_or_compute(
        "symbolic-program", record, lower,
    )
    value = cached.value
    if not isinstance(value, SymbolicEquationCompilation):
        value = lower()
        return replace(value, cache_identity=cached.identity, cache_hit=False)
    return replace(value, cache_identity=cached.identity, cache_hit=cached.hit)


__all__ = [
    "SymbolicEquationCompilation",
    "SymbolicPublication",
    "compile_sympy_equations",
    "compile_symbolic_program",
    "symbolic_equations_cached",
]
