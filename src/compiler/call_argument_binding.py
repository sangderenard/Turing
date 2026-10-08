"""Call-argument binding: positional/parameter naming, aggregate-leaf ledger lookup and expanded callsite argument edges."""

from __future__ import annotations

import ast
from typing import Any

from ..transmogrifier.graph.edge_roles import positional_argument_index


def _positional_argument_index(role: str) -> int | None:
    """Normalize the two ProcessGraph spellings used for positional edges.

    Delegates to the graph package, which owns edge roles and therefore
    owns the operator that reads them. Kept as a name here because callers
    in this module already use it; the implementation must not be a second
    copy, since the two would drift and a position is not a thing to be
    slightly wrong about.
    """

    return positional_argument_index(role)


def _method_parameter_layout(graph: Any) -> tuple[
    str | None, tuple[str, ...], tuple[str, ...]
]:
    """Return receiver, positional-call parameters, and every parameter.

    ProcessGraph parameter identities are SSA-oriented and their combined
    ``function_parameters`` tuple is not an ABI guarantee: in particular an
    instance receiver can appear after another parameter.  Call binding must
    identify the receiver as a receiver, not manufacture an offset into that
    tuple.  The source spellings ``self`` and ``cls`` are authoritative when
    retained; a source positional list is the fallback for unconventional
    receiver names.
    """

    metadata = graph.graph
    all_parameters = tuple(metadata.get("function_parameters", ()))
    positional = tuple(
        metadata.get("positional_parameters", ())
        or all_parameters
    )
    binding = metadata.get("method_binding")
    receiver = None
    if binding == "instance":
        receiver = next(
            (name for name in all_parameters if name == "self"),
            positional[0] if positional else None,
        )
    elif binding == "class":
        receiver = next(
            (name for name in all_parameters if name == "cls"),
            positional[0] if positional else None,
        )
    call_positional = tuple(
        name for name in positional if name != receiver
    )
    return receiver, call_positional, all_parameters


def _callsite_parameter_name(
    role: str,
    receiver: str | None,
    positional: tuple[str, ...],
) -> str | None:
    """Resolve one call edge through the callee's authored ABI layout.

    A bound-method receiver is an argument edge too.  It is spelled
    ``operand`` rather than ``arg:N`` in ProcessGraph, so consumers that only
    decoded positional/keyword roles silently omitted the exact record
    descriptor for ``self``.  Optional receiver fields were then lowered as
    always-present payload storage instead of payload/presence pairs.
    """

    position = _positional_argument_index(str(role))
    if position is not None and position < len(positional):
        return str(positional[position])
    if str(role) == "operand" and receiver is not None:
        return str(receiver)
    if str(role).startswith("kw:"):
        return str(role).split(":", 1)[1]
    return None


def _record_aggregate_ledger_lookup(
    graph: Any, value_id: int, data: Any, attributes: Any, leaves: Any,
) -> None:
    """Put every ledger lookup on the book, found or not.

    This is the chokepoint every aggregate-member decision passes through,
    and a miss here is silent: no ledger means no callsite aggregate
    descriptors, so no indexed member leaves, so no ``parameter_member_
    formals`` receipt, so the members read as formals no caller can name --
    and the first thing that says so is the full-native contract at the far
    end of the build, naming value ids with nothing to tie them back to.

    ``restore(self, snapshot)`` is the live case.  Its snapshot comes from
    ``saved = state.copy_shallow() if rollback else None`` -- a CONDITIONAL
    producer.  The ledger belongs to ``copy_shallow()``'s own result node,
    and what the conditional node in between carries instead is what this
    record exists to show, without guessing at it.
    """
    try:
        from .identity_concordance import (
            authored_function_name,
            concordant_shape_transformation_state,
            current_identity_book,
            descriptor_from_shape_transformation_state,
        )

        owner = (
            graph.G.graph.get("function_name")
            or graph.G.graph.get("qualified_name")
            or graph.G.graph.get("function_ref")
        )
        interesting = {
            "aggregate_leaf_value_ids", "optional_presence",
            "conditional_member_bindings", "producer_kind", "aggregate_kind",
            "aggregate_index", "aggregate_parent_binding",
        }
        # An EMPTIED ledger -- the key present, the tuple empty -- is the
        # case worth the extra detail.  It is indistinguishable from "never
        # had one" to every reader, because the lookup below tests the
        # tuple's truth and not the key's presence.  The full attribute key
        # set fingerprints which writer produced the node, which is the one
        # thing the shorter record could not say.
        emptied = "aggregate_leaf_value_ids" in attributes and not leaves
        # Holding a ledger is not enough.  The consumer additionally
        # requires every leaf to still be RESIDENT in the graph
        # (``all(int(leaf) in caller.G ...)`` where callsite aggregate
        # descriptors are collected), and a ledger whose leaves were folded
        # away fails that test as silently as a missing ledger does.
        # ``_repair_missing_aggregate_leaf_projections`` is supposed to
        # rebuild those, but it declines whenever the stored descriptor
        # count does not match the leaf count, which leaves the dangling
        # ledger in place and nobody the wiser.
        resident = sum(1 for leaf in leaves if int(leaf) in graph.G)
        current_identity_book().page("aggregate_ledger").set(
            (str(owner), int(value_id), "ledger_lookup"),
            0,
            (
                "emptied" if emptied
                else "absent" if not leaves
                else "resident" if resident == len(leaves)
                else "dangling",
                len(leaves),
                resident,
                str(data.get("type") or data.get("op") or ""),
                type(data.get("expr_obj")).__name__,
                tuple(sorted(interesting.intersection(attributes))),
                *((
                    tuple(sorted(map(str, attributes))),
                    len(tuple(attributes.get("tensor_output_descriptors") or ())),
                ) if emptied else ()),
            ),
        )
    except Exception:  # noqa: BLE001 -- diagnostics never fail a build
        pass


def _authored_aggregate_leaves(graph: Any, value_id: int) -> tuple[int, ...]:
    """Return the exact leaf ledger for an authored aggregate value.

    ``*value`` has its own AST node, while the aggregate ledger belongs to
    ``value``. Follow only that explicit Starred-to-value edge; absence of a
    ledger is absence of a compiler fact and must not trigger reconstruction
    from shapes, names, or neighboring values.
    """

    value_id = int(value_id)
    if value_id not in graph.G:
        return ()
    data = graph.G.nodes[value_id]
    attributes = data.get("attributes") or {}
    leaves = tuple(map(int, attributes.get("aggregate_leaf_value_ids", ())))
    _record_aggregate_ledger_lookup(graph, value_id, data, attributes, leaves)
    if leaves:
        return leaves
    if not isinstance(data.get("expr_obj"), ast.Starred):
        return ()
    sources = tuple(
        int(parent)
        for parent, role in data.get("parents") or ()
        if str(role) in {"value", "operand", "arg:0"}
    )
    if len(sources) != 1 or sources[0] not in graph.G:
        return ()
    source_attributes = graph.G.nodes[sources[0]].get("attributes") or {}
    return tuple(map(
        int, source_attributes.get("aggregate_leaf_value_ids", ()),
    ))


def _expanded_callsite_argument_edges(
    graph: Any, node_id: int,
) -> tuple[tuple[int, str], ...]:
    """Apply Python's positional ``*aggregate`` binding to graph edges.

    Expansion is permitted only from ProcessGraph's authored aggregate-leaf
    ledger. An unrecorded aggregate remains unexpanded so the missing source
    identity becomes an honest call-boundary error instead of an inferred ABI.
    """

    raw = tuple(
        (int(parent), str(role))
        for parent, role in graph.G.nodes[int(node_id)].get("parents") or ()
        if str(role) not in {"callee", "func", "definition"}
    )
    positional = sorted(
        (
            (_positional_argument_index(role), parent)
            for parent, role in raw
            if _positional_argument_index(role) is not None
        ),
        key=lambda item: int(item[0]),
    )
    non_positional = tuple(
        (parent, role)
        for parent, role in raw
        if _positional_argument_index(role) is None
    )
    expanded: list[tuple[int, str]] = []
    next_position = 0
    for original_position, parent in positional:
        next_position = max(next_position, int(original_position))
        parent_data = graph.G.nodes.get(int(parent), {})
        if isinstance(parent_data.get("expr_obj"), ast.Starred):
            leaves = _authored_aggregate_leaves(graph, int(parent))
            if leaves:
                expanded.extend(
                    (leaf, f"arg:{next_position + offset}")
                    for offset, leaf in enumerate(leaves)
                )
                next_position += len(leaves)
                continue
        expanded.append((int(parent), f"arg:{next_position}"))
        next_position += 1
    return (*expanded, *non_positional)


def _declared_output_terminals(
    graph: Any,
    *,
    produced_values: set[int] | None = None,
) -> dict[str, int]:
    """Expand structural return aggregates into their numerical SSA leaves."""

    identities = graph.G.graph.get("identity_table") or {}
    terminals: dict[str, int] = {}
    visiting: set[int] = set()

    def expand(name: str, node_id: int) -> None:
        node_id = int(node_id)
        if node_id in visiting or node_id not in graph.G:
            return
        visiting.add(node_id)
        data = graph.G.nodes[node_id]
        expression = data.get("expr_obj")
        attributes = data.get("attributes") or {}
        class_name = attributes.get("class_ref")
        if data.get("type") in {"LoopExit", "LoopResult"}:
            value_parent = next(
                (
                    int(parent)
                    for parent, role in (data.get("parents") or ())
                    if str(role) == "value"
                ),
                None,
            )
            if value_parent is not None:
                expand(name, value_parent)
        elif isinstance(expression, ast.Call) and class_name is not None:
            descriptor = (
                graph.G.graph.get("class_table", {}).get(class_name)
                or {}
            )
            fields = tuple(descriptor.get("fields") or ())
            parents = tuple(
                (int(parent), str(role))
                for parent, role in (data.get("parents") or ())
                if str(role) not in {"callee", "func", "definition"}
            )
            positional = {
                position: parent
                for parent, role in parents
                if (
                    position := _positional_argument_index(role)
                ) is not None
            }
            keywords = {
                role[3:]: parent
                for parent, role in parents
                if role.startswith("kw:")
            }
            for index, field in enumerate(fields):
                parent = keywords.get(field, positional.get(index))
                if parent is not None:
                    expand(f"{name}.{field}", parent)
        elif isinstance(
            expression, (ast.Tuple, ast.List)
        ):
            elements = [
                int(parent)
                for parent, role in (data.get("parents") or ())
                if str(role) in {"elts", "element", "item"}
            ]
            for index, parent in enumerate(elements):
                expand(f"{name}.{index}", parent)
        elif produced_values is None or node_id in produced_values:
            terminals[str(name)] = node_id
        visiting.remove(node_id)

    for name in graph.G.graph.get("function_outputs", ()):
        values = identities.get(name, ())
        if values:
            expand(str(name), int(values[-1]))
    return terminals


def _call_arguments(
    parents: tuple[tuple[int, str], ...],
    values: dict[int, Any],
    static_arguments: dict[str, Any] | None = None,
    graph: Any | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """Reconstruct positional and keyword arguments from ProcessGraph roles."""

    positional: dict[int, Any] = {}
    starred: set[int] = set()
    keywords: dict[str, Any] = {}
    fallback_index = 1 << 30
    for parent, role_value in parents:
        role = str(role_value)
        if role == "kwargs":
            keywords.update(dict(values[parent]))
            continue
        if role == "args":
            for value in values[parent]:
                positional[len(positional)] = value
            continue
        if role.startswith("kw:"):
            keywords[role[3:]] = values[parent]
            continue
        if role in {"operand", "func", "callee"}:
            continue
        index = fallback_index
        declared = positional_argument_index(role)
        if declared is not None:
            index = declared
        elif role == "arg":
            # A bare, unnumbered positional edge: the only case where
            # arrival order legitimately decides, because the role states
            # no index of its own.
            index = len(positional)
        else:
            continue
        positional[index] = values[parent]
        if (
            graph is not None
            and parent in graph.G
            and isinstance(
                graph.G.nodes[parent].get("expr_obj"), ast.Starred
            )
        ):
            starred.add(index)
        fallback_index += 1
    for role, value in (static_arguments or {}).items():
        if role.startswith("kw:"):
            keywords[role[3:]] = value
        elif (index := _positional_argument_index(role)) is not None:
            positional[index] = value
    arguments = []
    for index in sorted(positional):
        value = positional[index]
        if index in starred:
            arguments.extend(value)
        else:
            arguments.append(value)
    return arguments, keywords
