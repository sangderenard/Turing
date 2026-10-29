"""Source-owned exceptional transfers for native control composition.

The value graph records the expression that constructs an exception.  This
topology records where execution goes when that expression raises.  In
particular, a raise in an ``except`` suite cannot be caught by that suite's
own ``try``; it crosses its ``finally`` and seeks an outer protected body.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExceptionScope:
    try_node_id: int
    arm: str
    handler_node_id: int | None
    has_finally: bool


@dataclass(frozen=True)
class ExceptionTransfer:
    source_node_id: int
    source_kind: str
    catch_try_node_id: int | None
    scopes: tuple[ExceptionScope, ...]
    unwind_finally_node_ids: tuple[int, ...]


def resolved_handler_declarations(
    graph: Any, handler_node_id: int,
) -> tuple[tuple[int, str], ...]:
    """Return handler class values and identities from static-use edges.

    The authored ``except`` type occurrence is the key.  Its
    ``static_reference_use`` provenance names the actual class declaration;
    a coincidentally equal name is never enough to select a native handler.
    """

    from .concordance_declarations import (
        STATIC_REFERENCE_USE, STATIC_SYMBOL_DEFINITION,
    )
    from .identity_concordance import current_identity_book

    G = getattr(graph, "G", graph)
    handler = G.nodes[int(handler_node_id)].get("expr_obj")
    if not isinstance(handler, ast.ExceptHandler):
        raise ValueError(
            f"exception handler {handler_node_id} has no exact source AST"
        )
    if handler.type is None:
        return ()
    type_id = next((
        int(parent) for parent, role in G.nodes[int(handler_node_id)].get("parents", ())
        if str(role) == "type"
    ), None)
    if type_id is None or type_id not in G:
        raise ValueError(f"handler {handler_node_id} has no type value edge")
    declarations = (
        tuple(handler.type.elts)
        if isinstance(handler.type, ast.Tuple) else (handler.type,)
    )
    values = (
        tuple(next((
            int(parent)
            for parent, role in G.nodes[type_id].get("parents", ())
            if str(role) == f"elt:{index}"
        ), None) for index in range(len(declarations)))
        if isinstance(handler.type, ast.Tuple)
        else (type_id,)
    )
    if any(value is None for value in values):
        raise ValueError(
            f"handler {handler_node_id} lacks exact declaration value edges"
        )
    book = current_identity_book()
    uses = book.page(STATIC_REFERENCE_USE)
    result = []
    for declaration, value_id in zip(declarations, values):
        identities = set()
        for row in uses.rows():
            if int(row[1]) != id(declaration):
                continue
            use_cell = book.latest_ref(STATIC_REFERENCE_USE, row)
            if use_cell is None:
                continue
            identities.update(
                (str(source.row[0]), str(source.row[1]))
                for source, _stage in book.edges_into(use_cell)
                if source.page == STATIC_SYMBOL_DEFINITION
                and len(source.row) == 3 and not source.row[2]
            )
        if len(identities) != 1:
            raise ValueError(
                "handler exception declaration has no unique static "
                f"identity: handler={handler_node_id}, "
                f"occurrence={id(declaration)}, identities={identities!r}"
            )
        module, qualname = identities.pop()
        result.append((int(value_id), f"{module}.{qualname}"))
    return tuple(result)


def build_exception_topology(graph: Any) -> dict[int, ExceptionTransfer]:
    """Post exact source exceptional edges to the process graph's book.

    Scope is tracked by AST containment, with a distinct arm for the body,
    each handler, else and finally.  A body can select its own ``try``'s
    handlers; a handler/else/finally cannot.  A failed handler therefore
    leapfrogs that ``try`` and crosses its finalizer before reaching an outer
    catch or the function's fault exit.
    """

    from ..common.tensors.topological_reducer import node_identity_cell
    from .concordance_declarations import (
        EXCEPTION_TOPOLOGY, SOURCE_EXCEPTION_TRANSFER,
        ExceptionTransferFact,
    )
    from .identity_concordance import Derived, Mode, current_identity_book

    G = getattr(graph, "G", graph)
    body = tuple(G.graph.get("function_body") or ())
    if not body:
        return {}
    scope = G.graph.get("lexical_read_scope")
    if scope is None:
        raise ValueError("exception topology has no lexical read scope")
    by_expression: dict[int, list[int]] = {}
    for node_id, data in G.nodes(data=True):
        expression = data.get("expr_obj")
        if isinstance(expression, ast.AST):
            by_expression.setdefault(id(expression), []).append(int(node_id))

    def source_node(expression: ast.AST, *, required: bool = True) -> int | None:
        candidates = tuple(by_expression.get(id(expression), ()))
        if not candidates and not required:
            return None
        if len(candidates) != 1:
            raise ValueError(
                "exception topology needs one exact source graph node: "
                f"{type(expression).__name__} at "
                f"{getattr(expression, 'lineno', None)}; "
                f"candidates={candidates!r}"
            )
        return candidates[0]

    transfers: dict[int, ExceptionTransfer] = {}

    def record(expression: ast.AST, scopes: tuple[ExceptionScope, ...]) -> None:
        # Only a protected call needs an internal exception edge.  An
        # authored Raise always has one, including at function scope.
        if isinstance(expression, ast.Call) and not scopes:
            return
        source_id = source_node(expression, required=False)
        if source_id is None:
            # Structural specialization has removed this source operation
            # from the active value graph.  No live execution can transfer
            # from it; retained source syntax still supplies nesting for
            # operations that remain represented.
            return
        destination_index = next((
            index for index in range(len(scopes) - 1, -1, -1)
            if scopes[index].arm == "body"
        ), None)
        destination = (
            None if destination_index is None
            else scopes[destination_index].try_node_id
        )
        crossed = (
            scopes if destination_index is None
            else scopes[destination_index + 1:]
        )
        unwind = tuple(dict.fromkeys(
            item.try_node_id
            for item in reversed(crossed)
            if item.has_finally and item.arm != "finalbody"
        ))
        transfer = ExceptionTransfer(
            source_node_id=source_id,
            source_kind="raise" if isinstance(expression, ast.Raise) else "call",
            catch_try_node_id=destination,
            scopes=scopes,
            unwind_finally_node_ids=unwind,
        )
        if source_id in transfers and transfers[source_id] != transfer:
            raise ValueError(
                "source exception transfer has two scope contexts: "
                f"node={source_id}"
            )
        transfers[source_id] = transfer

    def walk(node: ast.AST, scopes: tuple[ExceptionScope, ...]) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # A nested definition has its own function exception boundary.
            return
        if isinstance(node, ast.Try):
            try_id = source_node(node, required=False)
            if try_id is None:
                if any(id(member) in by_expression for member in ast.walk(node)):
                    raise ValueError(
                        "live exception operation has no exact Try owner"
                    )
                return
            has_finally = bool(node.finalbody)

            def arm(name: str, handler_id: int | None = None):
                return scopes + (ExceptionScope(
                    try_id, name, handler_id, has_finally,
                ),)

            for member in node.body:
                walk(member, arm("body"))
            for handler in node.handlers:
                handler_id = source_node(handler)
                # Handler type expressions are evaluated during dispatch,
                # outside this try's protected body.
                if handler.type is not None:
                    walk(handler.type, arm("handler", handler_id))
                for member in handler.body:
                    walk(member, arm("handler", handler_id))
            for member in node.orelse:
                walk(member, arm("orelse"))
            for member in node.finalbody:
                walk(member, arm("finalbody"))
            return
        if isinstance(node, ast.TryStar):
            raise ValueError("except* requires its own exception-group topology")
        if isinstance(node, (ast.Raise, ast.Call)):
            record(node, scopes)
        for child in ast.iter_child_nodes(node):
            walk(child, scopes)

    for statement in body:
        walk(statement, ())

    book = current_identity_book()
    for source_id, transfer in sorted(transfers.items()):
        source_cell = node_identity_cell(graph, source_id)
        target_cell = (
            None if transfer.catch_try_node_id is None
            else node_identity_cell(graph, transfer.catch_try_node_id)
        )
        scope_cells = tuple(
            (
                node_identity_cell(graph, item.try_node_id),
                item.arm,
                None if item.handler_node_id is None
                else node_identity_cell(graph, item.handler_node_id),
                item.has_finally,
            ) for item in transfer.scopes
        )
        finalizer_cells = tuple(
            node_identity_cell(graph, node_id)
            for node_id in transfer.unwind_finally_node_ids
        )
        fact = ExceptionTransferFact(
            source_cell, transfer.source_kind, target_cell,
            scope_cells, finalizer_cells,
        )
        dependencies = tuple(dict.fromkeys((
            source_cell,
            *((target_cell,) if target_cell is not None else ()),
            *(item for owner, _arm, handler, _has_finally in scope_cells
              for item in (owner, handler) if item is not None),
            *finalizer_cells,
        )))
        row = (scope, source_id)
        previous = book.page(SOURCE_EXCEPTION_TRANSFER).latest(row)
        book.post(
            SOURCE_EXCEPTION_TRANSFER, row, fact,
            stage=EXCEPTION_TOPOLOGY,
            provenance=Derived(dependencies),
            mode=Mode.CONCORD if previous is None or previous == fact
            else Mode.REVISE,
        )
    G.graph["source_exception_transfers"] = transfers
    return transfers
