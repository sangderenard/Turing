"""Derive exceptional routes from reducer-owned source site cells."""

from __future__ import annotations

import ast
from typing import Any

from .concordance_declarations import (
    EXCEPTION_TOPOLOGY, INGESTION_VALUE, NAME_BINDING,
    SOURCE_EXCEPTION_HANDLER_CLASS,
    SOURCE_EXCEPTION_HANDLER_ENTRY, SOURCE_EXCEPTION_HANDLER_PHI,
    SOURCE_EXCEPTION_STATUS_ROUTE, NATIVE_STATUS_CLASS,
    SOURCE_EXCEPTION_SITE_STATE, SOURCE_EXCEPTION_TRANSFER,
    STATIC_REFERENCE_USE, STATIC_SYMBOL_DEFINITION,
    ExceptionHandlerClassFact, ExceptionHandlerEntryFact,
    ExceptionHandlerPhiFact, ExceptionStatusRouteFact,
    ExceptionSiteStateFact, ExceptionTransferFact,
    canonical_python_subtype_identities, concorded_native_status_classes,
    ensure_builtin_class_topology,
)
from .identity_concordance import Derived, Mode, Ref, current_identity_book
from .native_faults import authored_raise_status
from ..common.tensors.topological_reducer import node_identity_cell


def _raised_class_definition(
    graph: Any, scope: Any, statement: ast.Raise,
) -> Ref | None:
    """Read the authored constructor's static-use edge, when it has one."""

    raised = statement.exc
    if not isinstance(raised, ast.Call) or id(raised) not in graph.G:
        return None
    callees = tuple(
        int(parent)
        for parent, role in graph.G.nodes[id(raised)].get("parents", ())
        if str(role) == "callee"
    )
    if len(callees) != 1:
        raise ValueError(
            "raised constructor has no unique callee edge: "
            f"raise={id(statement)}, callees={callees!r}"
        )
    book = current_identity_book()
    definitions = set()
    for row in book.page(STATIC_REFERENCE_USE).rows():
        if (
            row[0] != scope or int(row[1]) != id(raised.func)
            or int(row[2]) != callees[0]
        ):
            continue
        use = book.latest_ref(STATIC_REFERENCE_USE, row)
        if use is None:
            continue
        definitions.update(
            source for source, _stage in book.edges_into(use)
            if source.page == STATIC_SYMBOL_DEFINITION
            and len(source.row) == 3 and not source.row[2]
        )
    if len(definitions) > 1:
        raise ValueError(
            "raised constructor has conflicting static classes: "
            f"raise={id(statement)}, definitions={definitions!r}"
        )
    return next(iter(definitions), None)


def post_source_exception_transfers(
    graph: Any, definition: ast.FunctionDef | ast.AsyncFunctionDef,
    ingestion_scope: Any,
) -> None:
    """Post each recorded site's catch and finalizer path before relabeling.

    The reducer posts a site only when evaluating that exact AST occurrence.
    Its scope cells and name-binding cells are the causes of this route; this
    pass reads them rather than reconstructing a lexical environment from
    the final state of the Try body.
    """

    book = current_identity_book()
    syntax = {id(node): node for node in ast.walk(definition)}
    site_page = book.page(SOURCE_EXCEPTION_SITE_STATE)
    for row in site_page.scope_rows(ingestion_scope):
        site = site_page.latest(row)
        if not isinstance(site, ExceptionSiteStateFact):
            raise ValueError(f"exception site has no exact state: {row!r}")
        source_id = int(row[1])
        expression = syntax.get(source_id)
        if isinstance(expression, (ast.Return, ast.Break, ast.Continue)):
            # These exits carry finalizer inputs but have no native
            # exception status to dispatch through except handlers.
            continue
        if not isinstance(expression, (ast.Call, ast.Raise, ast.Assert)):
            raise ValueError(
                "exception site has no authored call, raise or assert: "
                f"source={source_id}"
            )
        source = site.source
        if (
            not isinstance(source, Ref)
            or source.page != INGESTION_VALUE
            or int(source.row[1]) != source_id
            or book.page(INGESTION_VALUE).cells.get((
                source.row, source.column,
            )) is None
        ):
            raise ValueError(
                f"exception site disagrees with source cell: {row!r}"
            )
        scopes = []
        for owner, arm, handler in site.scopes:
            owner_ast = syntax.get(int(owner.row[1]))
            if not isinstance(owner_ast, ast.Try):
                raise ValueError(
                    "exception scope has no authored Try: "
                    f"source={source_id}, owner={owner!r}"
                )
            if arm == "handler":
                if handler is None or not isinstance(
                    syntax.get(int(handler.row[1])), ast.ExceptHandler,
                ):
                    raise ValueError(
                        "exception handler scope lacks an exact handler: "
                        f"source={source_id}, handler={handler!r}"
                    )
                if syntax[int(handler.row[1])] not in owner_ast.handlers:
                    raise ValueError(
                        "exception handler belongs to another Try: "
                        f"source={source_id}, handler={handler!r}"
                    )
            scopes.append((owner, arm, handler, bool(owner_ast.finalbody)))
        catch_index = next((
            index for index in range(len(scopes) - 1, -1, -1)
            if scopes[index][1] == "body"
        ), None)
        catch_try = None if catch_index is None else scopes[catch_index][0]
        crossed = scopes if catch_index is None else scopes[catch_index + 1:]
        unwind = tuple(dict.fromkeys(
            owner for owner, arm, _handler, has_finally in reversed(crossed)
            if has_finally and arm != "finalbody"
        ))
        raised_class = None
        if isinstance(expression, ast.Raise):
            raised_class = _raised_class_definition(
                graph, ingestion_scope, expression,
            )
        elif isinstance(expression, ast.Assert):
            ensure_builtin_class_topology(book)
            raised_class = book.latest_ref(
                STATIC_SYMBOL_DEFINITION, ("builtins", "AssertionError", ""),
            )
            if raised_class is None:
                raise ValueError(
                    f"authored Assert lacks AssertionError class: {source_id}"
                )
        fact = ExceptionTransferFact(
            source,
            "reraise" if isinstance(expression, ast.Raise)
            and expression.exc is None else
            "raise" if isinstance(expression, ast.Raise) else
            "assert" if isinstance(expression, ast.Assert) else "call",
            catch_try, tuple(scopes), unwind, raised_class,
            site.bindings,
        )
        site_cell = book.latest_ref(SOURCE_EXCEPTION_SITE_STATE, row)
        causes = tuple(dict.fromkeys((
            site_cell, source,
            *(owner for owner, _arm, _handler, _final in scopes),
            *(handler for _owner, _arm, handler, _final in scopes
              if handler is not None),
            *(unwind),
            *((raised_class,) if raised_class is not None else ()),
            *(binding for _name, binding in site.bindings),
        )))
        book.post(
            SOURCE_EXCEPTION_TRANSFER, row, fact,
            stage=EXCEPTION_TOPOLOGY,
            provenance=Derived(causes), mode=Mode.CONCORD,
        )

    # Handler declarations are separate from the transfer. A Call can fail
    # with several classes; the first matching authored handler is selected
    # only after its class operands and the Call's native status classes are
    # both known. Keep the occurrence's static-use edge as the class source.
    from ..common.tensors.topological_reducer import _source_read_names

    use_page = book.page(STATIC_REFERENCE_USE)
    transfer_page = book.page(SOURCE_EXCEPTION_TRANSFER)
    for owner_ast in syntax.values():
        if not isinstance(owner_ast, ast.Try) or id(owner_ast) not in graph.G:
            continue
        owner = node_identity_cell(graph, id(owner_ast))
        if owner is None:
            raise ValueError(
                f"retained Try has no ingestion cell: {id(owner_ast)}"
            )
        for ordinal, handler_ast in enumerate(owner_ast.handlers):
            handler_id = id(handler_ast)
            if handler_id not in graph.G:
                raise ValueError(
                    "retained Try has no exact handler node: "
                    f"try={id(owner_ast)}, handler={handler_id}"
                )
            handler = node_identity_cell(graph, handler_id)
            if handler is None:
                raise ValueError(
                    f"handler has no ingestion cell: {handler_id}"
                )
            declarations = () if handler_ast.type is None else (
                tuple(handler_ast.type.elts)
                if isinstance(handler_ast.type, ast.Tuple)
                else (handler_ast.type,)
            )
            type_id = next((
                int(parent)
                for parent, role in graph.G.nodes[handler_id].get("parents", ())
                if str(role) == "type"
            ), None)
            if declarations and type_id is None:
                raise ValueError(
                    f"handler has no exact type operand: {handler_id}"
                )
            values = (
                tuple(next((
                    int(parent) for parent, role in
                    graph.G.nodes[type_id].get("parents", ())
                    if str(role) == f"elt:{index}"
                ), None) for index in range(len(declarations)))
                if isinstance(handler_ast.type, ast.Tuple)
                else (type_id,) if declarations else ()
            )
            if any(value is None for value in values):
                raise ValueError(
                    f"handler has missing type member edges: {handler_id}"
                )
            resolved = []
            for declaration, value_id in zip(declarations, values):
                value = node_identity_cell(graph, int(value_id))
                if value is None:
                    raise ValueError(
                        "handler type value has no ingestion cell: "
                        f"handler={handler_id}, value={value_id}"
                    )
                definitions = set()
                for use_row in use_page.scope_rows(ingestion_scope):
                    if (
                        int(use_row[1]) != id(declaration)
                        or int(use_row[2]) != int(value_id)
                    ):
                        continue
                    use = book.latest_ref(STATIC_REFERENCE_USE, use_row)
                    if use is not None:
                        definitions.update(
                            source for source, _stage in book.edges_into(use)
                            if source.page == STATIC_SYMBOL_DEFINITION
                            and len(source.row) == 3 and not source.row[2]
                        )
                if len(definitions) > 1:
                    raise ValueError(
                        "handler type has conflicting class declarations: "
                        f"handler={handler_id}, occurrence={id(declaration)}"
                    )
                resolved.append((value, next(iter(definitions), None)))
            fact = ExceptionHandlerClassFact(
                owner, handler, ordinal, tuple(resolved),
            )
            dependencies = tuple(dict.fromkeys((
                owner, handler,
                *(cell for value, definition in resolved
                  for cell in (value, definition) if cell is not None),
            )))
            book.post(
                SOURCE_EXCEPTION_HANDLER_CLASS,
                (ingestion_scope, handler_id), fact,
                stage=EXCEPTION_TOPOLOGY,
                provenance=Derived(dependencies), mode=Mode.CONCORD,
            )
            candidates = tuple(
                (transfer_row, transfer_page.latest(transfer_row))
                for transfer_row in transfer_page.scope_rows(ingestion_scope)
                if transfer_page.latest(transfer_row).catch_try == owner
            )
            candidate_cells = tuple(
                book.latest_ref(SOURCE_EXCEPTION_TRANSFER, transfer_row)
                for transfer_row, _transfer in candidates
            )
            if any(cell is None for cell in candidate_cells):
                raise ValueError(
                    f"handler candidate has no transfer cell: {handler_id}"
                )
            handler_reads = set().union(*(
                _source_read_names(nested) for nested in handler_ast.body
            ))
            if handler_ast.name:
                handler_reads.discard(str(handler_ast.name))
            # Static class references do not need a runtime lexical slot.
            # Site demand was posted while the reducer knew which names were
            # static in that arm. Keep only names demanded on a candidate
            # transfer; a missing binding is explicit on that edge.
            required = tuple(sorted(handler_reads.intersection(
                name
                for transfer_row, _transfer in candidates
                for name in site_page.latest(transfer_row).required_names
            )))
            incoming = tuple(
                (name, tuple(
                    (cell, dict(transfer.entry_bindings).get(name))
                    for cell, (_row, transfer) in zip(candidate_cells, candidates)
                ))
                for name in required
            )
            entry = ExceptionHandlerEntryFact(
                owner, handler, candidate_cells, required, incoming,
            )
            entry_causes = tuple(dict.fromkeys((
                owner, handler, *candidate_cells,
                *(binding for _name, arms in incoming
                  for _transfer, binding in arms if binding is not None),
            )))
            book.post(
                SOURCE_EXCEPTION_HANDLER_ENTRY,
                (ingestion_scope, handler_id), entry,
                stage=EXCEPTION_TOPOLOGY,
                provenance=Derived(entry_causes), mode=Mode.CONCORD,
            )

    # A protected Call or reraise can return any catchable native status;
    # its actual code is selected at runtime. Authored Raise and Assert have
    # an exact static class and one status. All paths select handlers through
    # the same ordered declaration cells.
    concorded_native_status_classes(book)
    status_page = book.page(NATIVE_STATUS_CLASS)
    handler_page = book.page(SOURCE_EXCEPTION_HANDLER_CLASS)
    for transfer_row in transfer_page.scope_rows(ingestion_scope):
        transfer = transfer_page.latest(transfer_row)
        if transfer.source_kind in {"raise", "assert"} and (
            transfer.raised_class_definition is None
        ):
            continue
        if transfer.source_kind in {"raise", "assert"}:
            raised_identity = ".".join(
                transfer.raised_class_definition.row[:2]
            )
            status = authored_raise_status(raised_identity)
            if status is None:
                continue
            statuses = (int(status),)
        elif transfer.source_kind in {"call", "reraise"}:
            statuses = tuple(int(row[0]) for row in status_page.rows())
        else:
            raise ValueError(
                f"exception transfer has unknown kind: {transfer_row!r}"
            )
        handlers = sorted((
            (handler_page.latest(row).ordinal,
             book.latest_ref(SOURCE_EXCEPTION_HANDLER_CLASS, row),
             handler_page.latest(row))
            for row in handler_page.scope_rows(ingestion_scope)
            if handler_page.latest(row).owner == transfer.catch_try
        ), key=lambda item: item[0])
        if tuple(ordinal for ordinal, _cell, _fact in handlers) != tuple(
            range(len(handlers))
        ) or any(cell is None for _ordinal, cell, _fact in handlers):
            raise ValueError(
                f"catch owner has incomplete authored handler order: {transfer_row!r}"
            )
        for status in statuses:
            status_class = book.latest_ref(NATIVE_STATUS_CLASS, (status,))
            definition = status_page.latest((status,))
            if status_class is None or definition is None:
                raise ValueError(
                    f"exception status has no class: {transfer_row!r}, {status}"
                )
            if transfer.source_kind in {"raise", "assert"} and (
                definition != transfer.raised_class_definition
            ):
                raise ValueError(
                    "authored raise status disagrees with class: "
                    f"{transfer_row!r}"
                )
            raised_identity = ".".join(definition.row[:2])
            examined = []
            selected = matched = None
            for _ordinal, handler_cell, handler in handlers:
                examined.append(handler_cell)
                if not handler.declarations:
                    selected = handler.handler
                    break
                for _value, handler_definition in handler.declarations:
                    if handler_definition is None:
                        raise ValueError(
                            "handler has no exact class declaration: "
                            f"{handler_cell!r}"
                        )
                    handler_identity = ".".join(handler_definition.row[:2])
                    if raised_identity in canonical_python_subtype_identities(
                        handler_identity, book=book,
                    ):
                        selected, matched = handler.handler, handler_definition
                        break
                if selected is not None:
                    break
            transfer_cell = book.latest_ref(
                SOURCE_EXCEPTION_TRANSFER, transfer_row,
            )
            route = ExceptionStatusRouteFact(
                transfer.catch_try, transfer_cell, status_class,
                tuple(examined), selected, matched, transfer.unwind_finally,
            )
            causes = tuple(dict.fromkeys((
                transfer_cell, status_class,
                *((transfer.catch_try,) if transfer.catch_try is not None else ()),
                *examined,
                *((selected,) if selected is not None else ()),
                *((matched,) if matched is not None else ()),
                *transfer.unwind_finally,
            )))
            book.post(
                SOURCE_EXCEPTION_STATUS_ROUTE,
                (ingestion_scope, transfer_row[1], status), route,
                stage=EXCEPTION_TOPOLOGY,
                provenance=Derived(causes), mode=Mode.CONCORD,
            )

    binding_page = book.page(NAME_BINDING)
    for phi_id, data in graph.G.nodes(data=True):
        attributes = data.get("attributes") or {}
        if "source_exception_handler_id" not in attributes:
            continue
        handler_id = int(attributes["source_exception_handler_id"])
        entry = book.page(SOURCE_EXCEPTION_HANDLER_ENTRY).latest((
            ingestion_scope, handler_id,
        ))
        if not isinstance(entry, ExceptionHandlerEntryFact):
            raise ValueError(
                f"handler Phi has no exact entry fact: phi={phi_id}"
            )
        name = str(attributes["source_exception_handler_name"])
        site_ids = tuple(map(
            int, attributes["source_exception_handler_sites"]
        ))
        parents = tuple(data.get("parents") or ())
        if len(site_ids) != len(parents):
            raise ValueError(
                f"handler Phi has unequal sites and arms: phi={phi_id}"
            )
        arms = []
        for index, (site_id, (value_id, role)) in enumerate(zip(
            site_ids, parents,
        )):
            transfer = book.latest_ref(
                SOURCE_EXCEPTION_TRANSFER, (ingestion_scope, site_id),
            )
            binding = dict(dict(entry.incoming_bindings)[name]).get(transfer)
            value_fact = (
                None if binding is None else
                binding_page.cells.get((binding.row, binding.column))
            )
            if (
                role != f"entry:{index}"
                or binding is None
                or value_fact is None
                or int(value_fact.value_id) != int(value_id)
            ):
                raise ValueError(
                    "handler Phi arm disagrees with transfer binding: "
                    f"phi={phi_id}, site={site_id}, value={value_id}"
                )
            arms.append((transfer, binding))
        phi = node_identity_cell(graph, int(phi_id))
        if phi is None:
            raise ValueError(f"handler Phi has no ingestion cell: {phi_id}")
        fact = ExceptionHandlerPhiFact(
            entry.owner, entry.handler, phi, name, tuple(arms),
        )
        book.post(
            SOURCE_EXCEPTION_HANDLER_PHI,
            (ingestion_scope, int(phi_id)), fact,
            stage=EXCEPTION_TOPOLOGY,
            provenance=Derived(tuple(dict.fromkeys((
                entry.owner, entry.handler, phi,
                *(cell for transfer, binding in arms
                  for cell in (transfer, binding)),
            )))), mode=Mode.CONCORD,
        )
