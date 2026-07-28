"""Explicit reference tables carried by a compiled ProcessGraph shell.

The ProcessGraph remains the authority for topology.  These tables give a
backend-facing shell compact, monotonically indexed views of the references
that survive at its boundary:

* functions visible through the shared function tables;
* literal constants owned by this graph;
* input, output, and field-addressable memory references;
* correlations back to ProcessGraph nodes and source references.

They deliberately contain no allocation policy and perform no lowering.
Backends may later replace the local IDs with addresses, bindings, offsets, or
inlined definitions without losing the source correlation.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ShellFunctionReference:
    """One shell-local function-table slot."""

    index: int
    namespace: str
    source_address: int | None
    name: str
    qualified_name: str
    graph_backed: bool
    external: bool


@dataclass(frozen=True)
class ShellConstantReference:
    """One literal constant owned by the shell graph."""

    index: int
    value: Any
    value_type: str


@dataclass(frozen=True)
class ShellMemoryReference:
    """One boundary or derived storage reference."""

    index: int
    graph_node_id: Any
    name: str
    roles: tuple[str, ...]
    base_node_id: Any | None = None
    field_name: str | None = None


@dataclass(frozen=True)
class ShellReferenceCorrelation:
    """Trace one local table slot back to its ProcessGraph origin."""

    table: str
    index: int
    graph_node_id: Any | None
    source_kind: str
    source_reference: Any | None = None
    source_name: str | None = None


@dataclass
class ShellReferenceTables:
    """Ordinary indexed lists installed on one deployment shell."""

    functions: list[ShellFunctionReference]
    constants: list[ShellConstantReference]
    memory: list[ShellMemoryReference]
    correlations: list[ShellReferenceCorrelation]

    def copy(self) -> "ShellReferenceTables":
        return ShellReferenceTables(
            functions=list(self.functions),
            constants=list(self.constants),
            memory=list(self.memory),
            correlations=list(self.correlations),
        )


def _ordered_nodes(graph: Any) -> list[Any]:
    try:
        import networkx as nx

        return list(nx.lexicographical_topological_sort(graph.G, key=str))
    except (ImportError, TypeError, ValueError):
        return list(graph.G)


def _constant_value(data: dict[str, Any]) -> tuple[bool, Any]:
    expression = data.get("expr_obj")
    if isinstance(expression, ast.Constant):
        return True, expression.value
    if "constant" in data:
        return True, data["constant"]
    attributes = data.get("attributes") or {}
    if str(data.get("type")) in {"Const", "const", "Constant"}:
        return True, attributes.get("value")
    return False, None


def _function_usage(
    graph: Any,
    node_id: Any,
    data: dict[str, Any],
) -> tuple[str, int | str] | None:
    attributes = data.get("attributes") or {}
    if attributes.get("callee_ref") is not None:
        return "graph", int(attributes["callee_ref"])
    if attributes.get("external_callee_ref") is not None:
        return "external", int(attributes["external_callee_ref"])
    if attributes.get("static_python_reference"):
        return "static", str(attributes["static_python_reference"])

    # Attribute-call syntax has already been reduced to an operation name at
    # this point.  Correlate that operation with an existing function-table
    # entry without restoring Python attribute or bound-method semantics.
    expression = data.get("expr_obj")
    if isinstance(expression, ast.Call) and isinstance(
        expression.func,
        ast.Attribute,
    ):
        table = getattr(graph, "function_table", None)
        if table is not None:
            reference = table.reference(str(data.get("type")))
            if reference is not None:
                return "graph", int(reference.address)
    return None


def build_shell_reference_tables(graph: Any) -> ShellReferenceTables:
    """Package one graph's visible references into monotonic local lists."""

    functions: list[ShellFunctionReference] = []
    constants: list[ShellConstantReference] = []
    memory: list[ShellMemoryReference] = []
    correlations: list[ShellReferenceCorrelation] = []
    function_indices: dict[tuple[str, int | str], int] = {}

    function_table = getattr(graph, "function_table", None)
    if function_table is not None:
        for entry in sorted(
            function_table,
            key=lambda item: item.reference.address,
        ):
            index = len(functions)
            key = ("graph", int(entry.reference.address))
            function_indices[key] = index
            functions.append(
                ShellFunctionReference(
                    index=index,
                    namespace="graph",
                    source_address=int(entry.reference.address),
                    name=str(entry.name),
                    qualified_name=str(entry.qualified_name),
                    graph_backed=entry.graph is not None,
                    external=False,
                )
            )
            correlations.append(
                ShellReferenceCorrelation(
                    table="functions",
                    index=index,
                    graph_node_id=None,
                    source_kind="function_table",
                    source_reference=int(entry.reference.address),
                    source_name=str(entry.qualified_name),
                )
            )

    external_table = getattr(graph, "external_function_table", None)
    if external_table is not None:
        for entry in sorted(
            external_table,
            key=lambda item: item.reference.address,
        ):
            index = len(functions)
            key = ("external", int(entry.reference.address))
            function_indices[key] = index
            functions.append(
                ShellFunctionReference(
                    index=index,
                    namespace="external",
                    source_address=int(entry.reference.address),
                    name=str(entry.name),
                    qualified_name=str(entry.qualified_name),
                    graph_backed=False,
                    external=True,
                )
            )
            correlations.append(
                ShellReferenceCorrelation(
                    table="functions",
                    index=index,
                    graph_node_id=None,
                    source_kind="external_function_table",
                    source_reference=int(entry.reference.address),
                    source_name=str(entry.qualified_name),
                )
            )

    ordered_nodes = _ordered_nodes(graph)
    output_names = tuple(graph.G.graph.get("function_outputs", ()))
    identities = dict(graph.G.graph.get("identity_table", {}) or {})
    output_node_names = {
        node_id: name
        for name in output_names
        for node_id in identities.get(name, ())[-1:]
    }
    output_nodes = set(output_node_names)

    for node_id in ordered_nodes:
        data = graph.G.nodes[node_id]
        usage = _function_usage(graph, node_id, data)
        if usage is not None:
            index = function_indices.get(usage)
            if index is None and usage[0] == "static":
                index = len(functions)
                function_indices[usage] = index
                functions.append(
                    ShellFunctionReference(
                        index=index,
                        namespace="static",
                        source_address=None,
                        name=str(usage[1]).rsplit(".", 1)[-1],
                        qualified_name=str(usage[1]),
                        graph_backed=False,
                        external=True,
                    )
                )
            if index is not None:
                correlations.append(
                    ShellReferenceCorrelation(
                        table="functions",
                        index=index,
                        graph_node_id=node_id,
                        source_kind="call",
                        source_reference=usage[1],
                        source_name=functions[index].qualified_name,
                    )
                )

        is_constant, value = _constant_value(data)
        if is_constant:
            index = len(constants)
            constants.append(
                ShellConstantReference(
                    index=index,
                    value=value,
                    value_type=type(value).__name__,
                )
            )
            correlations.append(
                ShellReferenceCorrelation(
                    table="constants",
                    index=index,
                    graph_node_id=node_id,
                    source_kind="literal",
                    source_reference=node_id,
                    source_name=str(data.get("label", value)),
                )
            )

        expression = data.get("expr_obj")
        node_type = str(data.get("type"))
        is_input = node_type in {"Input", "input"}
        is_attribute = isinstance(expression, ast.Attribute)
        is_output = node_id in output_nodes
        if not (is_input or is_attribute or is_output):
            continue
        roles = []
        if is_input:
            roles.append("input")
        if is_attribute:
            roles.append("attribute")
        if is_output:
            roles.append("output")
        attributes = data.get("attributes") or {}
        field_name = expression.attr if is_attribute else None
        base_node_id = next(
            (
                parent
                for parent, role in data.get("parents", ())
                if str(role) in {"value", "base", "operand"}
            ),
            None,
        )
        name = str(
            output_node_names.get(
                node_id,
                attributes.get(
                    "binding_name",
                    field_name or data.get("label", node_id),
                ),
            )
        )
        index = len(memory)
        memory.append(
            ShellMemoryReference(
                index=index,
                graph_node_id=node_id,
                name=name,
                roles=tuple(roles),
                base_node_id=base_node_id,
                field_name=field_name,
            )
        )
        correlations.append(
            ShellReferenceCorrelation(
                table="memory",
                index=index,
                graph_node_id=node_id,
                source_kind="+".join(roles),
                source_reference=node_id,
                source_name=name,
            )
        )

    return ShellReferenceTables(
        functions=functions,
        constants=constants,
        memory=memory,
        correlations=correlations,
    )


__all__ = [
    "ShellConstantReference",
    "ShellFunctionReference",
    "ShellMemoryReference",
    "ShellReferenceCorrelation",
    "ShellReferenceTables",
    "build_shell_reference_tables",
]
