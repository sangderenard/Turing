"""Link already-authored ProcessGraph functions into a source graph.

This is the source-neutral function-table operation needed when different
front ends meet at ProcessGraph.  In particular, a Python control program may
call a numerical function authored by SymPy without rendering that numerical
function back into Python or recapturing it through a tape.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class GraphFunctionSocketBinding:
    """One exact unresolved function class spliced from one graph body."""

    function: Any
    graph: Any
    call_node_ids: tuple[int, ...]
    inserted_node_ids: tuple[int, ...]


def _replace_graph_metadata_value(graph: Any, old_id: int, new_id: int) -> None:
    """Move public graph references when a socket itself is an output."""

    graph.roots = [new_id if int(value) == old_id else value for value in graph.roots]
    metadata = graph.G.graph
    for key in ("deployment_outputs",):
        if key in metadata:
            metadata[key] = tuple(
                new_id if int(value) == old_id else value
                for value in metadata[key]
            )
    identity_table = metadata.get("identity_table")
    if isinstance(identity_table, dict):
        for name, values in tuple(identity_table.items()):
            identity_table[name] = tuple(
                new_id if int(value) == old_id else value for value in values
            )


def _post_spliced_node(
    envelope: Any,
    node_id: int,
    data: Mapping[str, Any],
    body: Any,
    body_node_id: int,
    call_node_id: int,
) -> None:
    """Give a copied body value its exact two-parent identity receipt."""

    from ..common.tensors.topological_reducer import node_identity_cell
    from .concordance_declarations import (
        GRAPH_FUNCTION_SPLICE,
        GRAPH_FUNCTION_SPLICE_STAGE,
        INGESTION_VALUE,
        NodeFact,
    )
    from .identity_concordance import Mode, Novel, current_identity_book

    scope = envelope.G.graph.get("ingestion_value_scope")
    if scope is None:
        raise ValueError("graph function envelope has no ingestion identity scope")
    call_cell = node_identity_cell(envelope, call_node_id)
    body_cell = node_identity_cell(body, body_node_id)
    if call_cell is None or body_cell is None:
        raise ValueError("graph function splice requires identified envelope and body values")
    current_identity_book().post(
        INGESTION_VALUE,
        (scope, int(node_id)),
        NodeFact(
            str(data.get("type") or ""),
            str(data.get("op") or ""),
            str(data.get("label") or ""),
        ),
        stage=GRAPH_FUNCTION_SPLICE_STAGE,
        provenance=Novel(GRAPH_FUNCTION_SPLICE, (call_cell, body_cell)),
        mode=Mode.CONCORD,
    )


def _post_socket_binding(
    envelope: Any,
    call_id: int,
    body: Any,
    function_identity: str,
    parameter_ids: tuple[int, ...],
    argument_ids: tuple[int, ...],
    body_output_id: int,
    replacement_id: int,
) -> None:
    """Record the exact parameter and result wiring of one substitution."""

    from ..common.tensors.topological_reducer import node_identity_cell
    from .concordance_declarations import (
        GRAPH_FUNCTION_SOCKET,
        GRAPH_FUNCTION_SPLICE,
        GRAPH_FUNCTION_SPLICE_STAGE,
        GraphFunctionSocketFact,
    )
    from .identity_concordance import Mode, Novel, current_identity_book

    scope = envelope.G.graph.get("ingestion_value_scope")
    body_scope = body.G.graph.get("ingestion_value_scope")
    if scope is None or body_scope is None:
        raise ValueError("graph function splice requires identified envelope and body scopes")
    call_cell = node_identity_cell(envelope, call_id)
    output_cell = node_identity_cell(body, body_output_id)
    current_identity_book().post(
        GRAPH_FUNCTION_SOCKET,
        (scope, int(call_id)),
        GraphFunctionSocketFact(
            str(function_identity), tuple(body_scope), tuple(parameter_ids),
            tuple(argument_ids), int(body_output_id), int(replacement_id),
        ),
        stage=GRAPH_FUNCTION_SPLICE_STAGE,
        provenance=Novel(GRAPH_FUNCTION_SPLICE, (call_cell, output_cell)),
        mode=Mode.CONCORD,
    )


def _ordered_socket_arguments(data: Mapping[str, Any], arity: int) -> tuple[int, ...]:
    arguments: dict[int, int] = {}
    for parent_id, role in data.get("parents") or ():
        prefix, separator, index = str(role).partition(":")
        if prefix != "arg" or not separator or not index.isdigit():
            raise ValueError("graph function socket arguments must use arg:N roles")
        position = int(index)
        if position in arguments:
            raise ValueError("graph function socket has a duplicate argument position")
        arguments[position] = int(parent_id)
    if tuple(sorted(arguments)) != tuple(range(arity)):
        raise ValueError(
            f"graph function socket has argument positions {tuple(sorted(arguments))}; "
            f"expected {tuple(range(arity))}"
        )
    return tuple(arguments[index] for index in range(arity))


def _splice_call(
    envelope: Any, call_id: int, body: Any, function_identity: str,
) -> tuple[int, ...]:
    """Replace one unresolved Call with a copied body subgraph."""

    import networkx as nx

    from ..common.tensors.topological_reducer import _redirect_value, _set_operands
    from .concordance_declarations import GRAPH_FUNCTION_SPLICE
    from .process_graph_value_ids import next_process_value_id

    parameters = tuple(body.G.graph.get("function_parameters") or ())
    outputs = tuple(body.G.graph.get("function_outputs") or ())
    if not parameters or len(outputs) != 1:
        raise ValueError("a graph function socket body needs parameters and one output")
    identity_table = body.G.graph.get("identity_table") or {}
    try:
        parameter_ids = tuple(int(identity_table[name][-1]) for name in parameters)
        output_id = int(identity_table[outputs[0]][-1])
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("graph function socket body has an incomplete identity table") from error
    if output_id not in body.G:
        raise ValueError("graph function socket output is not present in its body graph")

    arguments = _ordered_socket_arguments(envelope.G.nodes[call_id], len(parameters))
    remap = dict(zip(parameter_ids, arguments))
    required = nx.ancestors(body.G, output_id) | {output_id}
    inserted: list[int] = []
    for source_id in nx.topological_sort(body.G.subgraph(required)):
        source_id = int(source_id)
        if source_id in remap:
            continue
        source = body.G.nodes[source_id]
        parents = []
        for parent_id, role in source.get("parents") or ():
            parent_id = int(parent_id)
            if parent_id not in remap:
                raise ValueError(
                    f"graph function body value {source_id} depends on unmapped value {parent_id}"
                )
            parents.append((remap[parent_id], str(role)))
        new_id = next_process_value_id(envelope)
        remap[source_id] = new_id
        data = copy.deepcopy(dict(source))
        data["parents"] = []
        data["children"] = []
        data["value_id"] = new_id
        attributes = dict(data.get("attributes") or {})
        attributes.update(
            graph_socket_call=int(call_id),
            graph_socket_body_value=source_id,
        )
        data["attributes"] = attributes
        data["extra_args"] = copy.deepcopy(attributes)
        envelope.G.add_node(new_id, **data)
        envelope.node_map[new_id] = data.get("expr_obj")
        _post_spliced_node(envelope, new_id, data, body, source_id, call_id)
        _set_operands(
            envelope, new_id, parents, cause=GRAPH_FUNCTION_SPLICE,
        )
        inserted.append(new_id)

    replacement = remap[output_id]
    _post_socket_binding(
        envelope, call_id, body, function_identity,
        parameter_ids, arguments, output_id, replacement,
    )
    _replace_graph_metadata_value(envelope, call_id, replacement)
    _redirect_value(envelope, call_id, replacement)
    envelope.node_map.pop(call_id, None)
    return tuple(inserted)


def bind_process_graph_function_sockets(
    envelope: Any,
    sockets: Mapping[Any, Any],
) -> tuple[GraphFunctionSocketBinding, ...]:
    """Splice graph bodies into unresolved function applications by identity.

    The symbolic envelope retains the applied undefined function object on
    each Call node.  That object -- not its spelling -- selects the supplied
    body.  Each occurrence receives fresh value identities whose receipts name
    both that exact application and the exact body values copied into it.
    Unmentioned unresolved functions remain external declarations.

    ``envelope`` is changed in place.  The body is not converted back through
    source or rebuilt from its expression labels; its existing nodes and edges
    are copied directly, with its parameter nodes wired to the call operands.
    """

    import sympy
    from sympy.core.function import AppliedUndef, UndefinedFunction

    requested = tuple(sockets.items())
    if not requested:
        return ()
    keys = []
    for function, _body in requested:
        if not isinstance(function, UndefinedFunction):
            raise TypeError("graph function socket must be an undefined function class")
        keys.append(sympy.srepr(function))
    if len(set(keys)) != len(keys):
        raise ValueError("graph function socket declaration identities must be unique")

    bindings = []
    for (function, body), socket_key in zip(requested, keys):
        name = getattr(function, "__name__", None)
        if not name:
            raise TypeError("graph function socket must be an undefined function class")
        parameters = tuple(body.G.graph.get("function_parameters") or ())
        outputs = tuple(body.G.graph.get("function_outputs") or ())
        if not parameters or len(outputs) != 1:
            raise ValueError(
                f"graph function socket {name!r} needs parameters and exactly one output"
            )
        matched = []
        for node_id, data in envelope.G.nodes(data=True):
            expression = data.get("expr_obj")
            attributes = data.get("attributes") or {}
            if (
                not isinstance(expression, AppliedUndef)
                or attributes.get("external_function_identity") != socket_key
            ):
                continue
            matched.append(int(node_id))
        if not matched:
            raise ValueError(f"graph function socket {name!r} has no callsite in the envelope")
        inserted = tuple(
            node_id
            for call_id in matched
            for node_id in _splice_call(envelope, call_id, body, socket_key)
        )
        bindings.append(GraphFunctionSocketBinding(
            function, body, tuple(matched), inserted,
        ))
    envelope.G.graph["graph_function_sockets"] = tuple(
        {
            "function": getattr(binding.function, "__name__", str(binding.function)),
            "calls": binding.call_node_ids,
            "inserted": binding.inserted_node_ids,
        }
        for binding in bindings
    )
    return tuple(bindings)


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
        from .symbolic_equation_compiler import (
            rebuild_symbolic_process_graph_in_current_book,
        )

        callee = rebuild_symbolic_process_graph_in_current_book(callee)
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
        from ..transmogrifier.graph.graph_express2 import (
            resolve_source_definition,
        )

        source_definitions = tuple(
            callee.G.graph.get("function_source_definitions") or ()
        )
        if not source_definitions:
            raise ValueError(
                f"linked ProcessGraph {function_name!r} has no authored "
                "function source definitions"
            )
        source_cells = tuple(
            resolve_source_definition(definition)
            for definition in source_definitions
        )
        reference = table.declare(
            function_name,
            qualified_name=function_name,
            metadata={
                "source_language": callee.G.graph.get("symbolic_source"),
                "linked_process_graph": True,
            },
            source=source_cells,
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
        # Source lowering registers linked functions before it ingests the
        # caller.  In that phase the root has no callsites to inspect.  Avoid
        # materializing its empty node view: under the disk-backed graph
        # container that view is not published until ingestion begins.
        if graph is root and not graph.node_map:
            continue
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
            # The attributes are the read view of the ``call_binding`` row
            # (plan 80, A2.8): DERIVED from the call node's identity cell and
            # the linked callee's ``function_address`` cell.
            from .glsl_deployment_strategy import _post_call_binding

            _post_call_binding(
                graph, int(node_id), references[binding_name],
                "linked_process_graph",
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


__all__ = [
    "GraphFunctionSocketBinding",
    "bind_process_graph_function_sockets",
    "link_process_graph_functions",
]
