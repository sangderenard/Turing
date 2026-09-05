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

from .hierarchical_plan import PREDICATE_OPERATIONS
from .ir_identities import reduce_constant_exponent_pow
from .ssa_builder import process_graph_to_ssa_instrs
from .symbolic_process_graph import ingest_sympy_expressions
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
    if isinstance(value, sympy.Integer):
        return float(value)
    if isinstance(value, (sympy.Rational, sympy.Float)):
        return float(value)
    return value


def _compile_sympy_equations_uncached(
    equations: Sequence[sympy.Equality],
    *,
    name: str = "symbolic_equation_step",
    schedule: str = "asap",
    publications: Sequence[SymbolicPublication] = (),
    dtype: str = "float64",
) -> SymbolicEquationCompilation:
    """Lower simultaneous named SymPy equations into repository SSA.

    Each left-hand side names one result and each right-hand side is compiled
    verbatim through the canonical SymPy ProcessGraph importer.  Output names
    are not permitted as right-hand-side inputs: one invocation is a
    simultaneous state transition, and recurrence belongs to the caller that
    feeds the returned state into the next invocation.
    """

    authored = tuple(equations)
    if not authored:
        raise ValueError("symbolic equation program requires equations")
    for equation in authored:
        if not isinstance(equation, sympy.Equality):
            raise TypeError(f"expected a SymPy Equality, got {equation!r}")
        if not isinstance(equation.lhs, sympy.Symbol):
            raise TypeError(f"equation output must be a Symbol: {equation!r}")
    output_names = tuple(str(equation.lhs) for equation in authored)
    if len(output_names) != len(set(output_names)):
        raise ValueError("symbolic equation output names must be unique")
    output_symbols = frozenset(equation.lhs for equation in authored)
    recursive = {
        str(symbol)
        for equation in authored
        for symbol in equation.rhs.free_symbols & output_symbols
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
    roots = ingest_sympy_expressions(
        graph,
        tuple(equation.rhs for equation in authored),
        output_names=output_names,
        strict=True,
    )
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
            "dtype": "bool" if spelling in PREDICATE_OPERATIONS else dtype,
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
        return replace(value, cache_identity=cached.identity, cache_hit=False)
    return replace(
        cached.value,
        cache_identity=cached.identity,
        cache_hit=cached.hit,
    )


def _pipeline_implementation() -> str:
    """Digest of the lowering implementation every cached layer depends on."""

    return callable_digest(
        _compile_sympy_equations_uncached,
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
