"""Presentation adapter for an existing SymPy ProcessGraph target.

This module does not translate operations.  It selects the established
``process_graph_to_sympy_relations`` target for a reduced ``FusedProgram``
and serializes the returned SymPy objects as native presentation MathML for
the web shell.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import sympy


@dataclass(frozen=True)
class SympyMathDocument:
    """A browser-ready view of one exact symbolic process model."""

    equations: tuple[Mapping[str, Any], ...]
    outputs: tuple[Mapping[str, Any], ...]
    input_names: tuple[str, ...]
    constraint_count: int
    uninterpreted: tuple[tuple[int, str], ...]
    node_count: int
    program_relation_head: str
    program_relation_arity: int
    program_kind: str
    program_name: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": "turing-sympy-process-model-v1",
            "target": "sympy",
            "projection": "process_graph_to_sympy_relations",
            "node_count": self.node_count,
            "equation_count": len(self.equations),
            "constraint_count": self.constraint_count,
            "input_names": list(self.input_names),
            "outputs": [dict(output) for output in self.outputs],
            "uninterpreted": [
                {"node_id": node_id, "operation": operation}
                for node_id, operation in self.uninterpreted
            ],
            "program_relation": {
                "head": self.program_relation_head,
                "arity": self.program_relation_arity,
                "arguments": "equations[*]",
            },
            "depiction": {
                "kind": self.program_kind,
                "name": self.program_name,
                "inputs": list(self.input_names),
                "outputs": [str(output["name"]) for output in self.outputs],
            },
            "equations": [dict(equation) for equation in self.equations],
        }


def _presentation_mathml(expression: sympy.Basic) -> str:
    """Wrap SymPy's presentation printer output as native MathML."""

    # SymPy emits named MathML entities such as ``&InvisibleTimes;``. They
    # work when markup is parsed as HTML, but the shell deliberately uses an
    # XML parser before importing MathML into the document, and XML only
    # predefines five entities. Resolve the standard names to their Unicode
    # code points at build time so every valid SymPy product survives that
    # safety boundary.
    body = html.unescape(
        sympy.printing.mathml(expression, printer="presentation")
    )
    return (
        '<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">'
        f"{body}</math>"
    )


def render_reduced_program_mathematics(
    program: Any,
    *,
    input_names: Sequence[str] | None = None,
    program_name: str = "program",
) -> SympyMathDocument:
    """Select the existing SymPy target for a normalized reduced program.

    Scalar constructor folding and dead-step pruning are the same normalization
    used by the language backends.  The semantic projection and every equation
    come from compiler facilities that already exist; this function only adds
    labels and presentation markup.
    """

    from .backend_sources import normalized_program
    from .process_graph_fusion import fused_program_to_process_graph
    from . import symbolic_process_graph

    target = getattr(
        symbolic_process_graph, "process_graph_to_sympy_relations", None
    )
    if target is None:
        raise RuntimeError(
            "the existing process_graph_to_sympy_relations target is required"
        )

    reduced = normalized_program(program)
    graph = fused_program_to_process_graph(reduced)
    output_ids = tuple(reduced.outputs.values())
    model = target(graph, output_ids=output_ids)
    # The equation list is useful for solvers and inspection, but its Boolean
    # conjunction is the program as one relation. Keep the aggregate as a
    # genuine SymPy object here; the JSON document references its clauses
    # rather than duplicating several megabytes of MathML in one expression.
    program_relation = sympy.And(*model.relations, evaluate=False)
    metadata = reduced.meta or {}
    output_dtypes = tuple(
        str(getattr(metadata.get(node_id), "dtype", "")).lower()
        for node_id in output_ids
    )
    if reduced.state_in:
        program_kind = "transition"
    elif output_dtypes and all("bool" in dtype for dtype in output_dtypes):
        program_kind = "predicate"
    elif output_ids:
        program_kind = "function"
    else:
        program_kind = "relation"

    equation_by_symbol = {
        equation.lhs: equation
        for equation in model.equations
        if isinstance(equation, sympy.Equality)
    }
    node_by_symbol = {
        expression: int(node_id)
        for node_id, expression in model.expressions.items()
    }
    equations = []
    for equation in (*model.equations, *model.constraints):
        lhs = equation.lhs if isinstance(equation, sympy.Equality) else None
        node_id = node_by_symbol.get(lhs) if lhs is not None else None
        spec = model.node_specs.get(node_id) if node_id is not None else None
        equations.append({
            "node_id": node_id,
            "operation": spec.operation if spec is not None else "constraint",
            "text": sympy.sstr(equation),
            "mathml": _presentation_mathml(equation),
        })

    outputs = []
    for (name, node_id), symbol in zip(reduced.outputs.items(), model.outputs):
        relation = equation_by_symbol.get(symbol, sympy.Eq(symbol, symbol))
        outputs.append({
            "name": str(name),
            "node_id": int(node_id),
            "symbol": sympy.sstr(symbol),
            "text": sympy.sstr(relation),
            "mathml": _presentation_mathml(relation),
        })

    return SympyMathDocument(
        equations=tuple(equations),
        outputs=tuple(outputs),
        input_names=tuple(map(str, input_names or model.inputs)),
        constraint_count=len(model.constraints),
        uninterpreted=tuple(model.uninterpreted),
        node_count=len(model.node_specs),
        program_relation_head=type(program_relation).__name__,
        program_relation_arity=len(program_relation.args),
        program_kind=program_kind,
        program_name=str(program_name),
    )


__all__ = ["SympyMathDocument", "render_reduced_program_mathematics"]
