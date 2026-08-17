"""Compile named SymPy equations into one repository-SSA function.

The equations are the numerical authority.  This module only coordinates the
existing SymPy -> ProcessGraph translator and ProcessGraph -> SSA scheduler;
it does not evaluate, rewrite, or reimplement their right-hand sides.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Any, Mapping, Sequence

import sympy

from .hierarchical_plan import PREDICATE_OPERATIONS
from .ssa_builder import process_graph_to_ssa_instrs
from .symbolic_process_graph import ingest_sympy_expressions
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


def _numeric_constant(value: Any) -> Any:
    if isinstance(value, sympy.Integer):
        return float(value)
    if isinstance(value, (sympy.Rational, sympy.Float)):
        return float(value)
    return value


def compile_sympy_equations(
    equations: Sequence[sympy.Equality],
    *,
    name: str = "symbolic_equation_step",
    schedule: str = "asap",
    publications: Sequence[SymbolicPublication] = (),
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
            "dtype": "bool" if spelling in PREDICATE_OPERATIONS else "float64",
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
        body.append(instruction)
    output_values = [SSAValue(int(root), "float64") for root in roots]
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
    return SymbolicEquationCompilation(
        equations=authored,
        process_graph=authored_graph,
        instructions=tuple(body),
        function=function,
        module=module,
        input_ids={row[0]: row[1] for row in input_rows},
        output_ids=dict(zip(output_names, roots)),
        publications=publication_rows,
    )


__all__ = [
    "SymbolicEquationCompilation",
    "SymbolicPublication",
    "compile_sympy_equations",
]
