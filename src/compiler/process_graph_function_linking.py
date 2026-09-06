"""Link already-authored ProcessGraph functions into a source graph.

This is the source-neutral function-table operation needed when different
front ends meet at ProcessGraph.  In particular, a Python control program may
call a numerical function authored by SymPy without rendering that numerical
function back into Python or recapturing it through a tape.
"""

from __future__ import annotations

from typing import Any, Mapping


def _graphs(root: Any) -> tuple[Any, ...]:
    seen: set[int] = set()
    pending = [root]
    result = []
    while pending:
        graph = pending.pop()
        if id(graph) in seen:
            continue
        seen.add(id(graph))
        result.append(graph)
        table = getattr(graph, "function_table", None)
        if table is not None:
            pending.extend(
                entry.graph for entry in table if entry.graph is not None
            )
    return tuple(result)


def link_process_graph_functions(
    root: Any,
    functions: Mapping[str, Any],
) -> Mapping[str, int]:
    """Resolve named source calls directly to supplied ProcessGraph bodies."""

    table = root.function_table
    references: dict[str, int] = {}
    for name, callee in sorted(functions.items()):
        function_name = str(name)
        parameters = tuple(callee.G.graph.get("function_parameters") or ())
        if not parameters:
            raise ValueError(
                f"linked ProcessGraph {function_name!r} has no parameters"
            )
        outputs = tuple(callee.G.graph.get("function_outputs") or ())
        if not outputs:
            raise ValueError(
                f"linked ProcessGraph {function_name!r} has no outputs"
            )
        reference = table.declare(
            function_name,
            qualified_name=function_name,
            metadata={
                "source_language": callee.G.graph.get("symbolic_source"),
                "linked_process_graph": True,
            },
        )
        callee.function_table = table
        callee.external_function_table = root.external_function_table
        callee.G.graph.update(
            function_ref=int(reference.address),
            function_name=function_name,
        )
        table.resolve_graph(reference, callee)
        references[function_name] = int(reference.address)

    for graph in _graphs(root):
        for node_id, data in tuple(graph.G.nodes(data=True)):
            if data.get("op") not in {"Call", "call"}:
                continue
            parents = tuple(data.get("parents") or ())
            matched = None
            for parent_id, role in parents:
                if str(role) != "callee" or parent_id not in graph.G:
                    continue
                parent = graph.G.nodes[parent_id]
                binding_name = str(
                    (parent.get("attributes") or {}).get("binding_name") or ""
                )
                if binding_name in references:
                    matched = (int(parent_id), binding_name)
                    break
            if matched is None:
                continue
            callee_id, binding_name = matched
            data["parents"] = [
                (parent_id, role)
                for parent_id, role in parents
                if int(parent_id) != callee_id
            ]
            data.setdefault("attributes", {}).update(
                callee_ref=references[binding_name],
                callee_resolution="linked-process-graph",
            )
            if graph.G.has_edge(callee_id, node_id):
                graph.G.remove_edge(callee_id, node_id)
            children = graph.G.nodes[callee_id].get("children") or []
            graph.G.nodes[callee_id]["children"] = [
                child for child in children if int(child[0]) != int(node_id)
            ]
            if not graph.G.nodes[callee_id]["children"]:
                graph.G.remove_node(callee_id)
                graph.node_map.pop(callee_id, None)
            unresolved = tuple(graph.G.graph.get("unresolved_ast_calls") or ())
            graph.G.graph["unresolved_ast_calls"] = tuple(
                record for record in unresolved
                if str(record.get("name")) != binding_name
            )
    return references


__all__ = ["link_process_graph_functions"]
