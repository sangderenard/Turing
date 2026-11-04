"""Relay an already proved single bytes return through a reached source call."""

from .concordance_declarations import (
    CANONICAL_VALUE, FUNCTION_ADDRESS, REDUCTION, SEMANTIC_OUTPUT,
    SOURCE_ACTIVITY_SCOPE, SOURCE_CALL_RESULT_IDENTITY,
    SOURCE_FUNCTION_REACHABILITY, SOURCE_VALUE_CLASS,
)
from .identity_concordance import Derived, Mode
from .source_bytes_concat_class import _class_matches_operand


def post_reached_bytes_return_classes(function_table, book) -> int:
    """Finish an incumbent call result after eager callee class publication.

    The reducer reaches a call before the eager bytes Add class is available.
    Its SOURCE_FUNCTION_REACHABILITY and SEMANTIC_OUTPUT remain the exact
    call-to-return route; this pass only reuses the incumbent result pages.
    """

    from ..common.tensors.topological_reducer import (
        _concord_source_value_class, _source_numeric_scope,
    )
    from ..transmogrifier.function_table import function_address_identity

    entries = tuple(entry for entry in function_table
                    if getattr(entry, "graph", None) is not None)
    reach_page = book.page(SOURCE_FUNCTION_REACHABILITY)
    scope_page = book.page(SOURCE_ACTIVITY_SCOPE)
    output_page = book.page(SEMANTIC_OUTPUT)
    posted = 0
    for reach_row in reach_page.rows():
        caller_scope, call_id, callee_scope = reach_row
        reached = book.latest_ref(SOURCE_FUNCTION_REACHABILITY, reach_row)
        if reached is None or reach_page.cell(
            reached.row, reached.column,
        ) is not True:
            continue
        reach_causes = {parent for parent, _ in book.edges_into(reached)}
        calls = tuple(parent for parent in reach_causes
                      if parent.page == CANONICAL_VALUE)
        addresses = tuple(parent for parent in reach_causes
                          if parent.page == FUNCTION_ADDRESS)
        if len(calls) != 1 or len(addresses) != 1:
            continue
        call, reached_address = calls[0], addresses[0]
        if call.row[1] != call_id:
            continue
        matching_entries = tuple(
            entry for entry in entries
            if _source_numeric_scope(entry.graph) == callee_scope
            and book.page(FUNCTION_ADDRESS).cell(
                reached_address.row, reached_address.column,
            ) == int(entry.reference.address)
        )
        if len(matching_entries) != 1:
            continue
        callee_entry = matching_entries[0]
        authored_identity = tuple(callee_entry.graph.G.graph.get(
            "authored_function_identity"
        ) or ())
        equivalents = tuple(
            entry for entry in entries
            if tuple(entry.graph.G.graph.get(
                "authored_function_identity"
            ) or ()) == authored_identity
        )
        canonical_entry = min(
            equivalents, key=lambda entry: int(entry.reference.address),
        )
        canonical_address = book.latest_ref(
            FUNCTION_ADDRESS, (function_address_identity(canonical_entry),),
        )
        if canonical_address is None:
            continue
        outputs = []
        for scope_row in scope_page.rows():
            read_scope = scope_row[0]
            if scope_page.latest(scope_row) != callee_scope:
                continue
            output_rows = tuple(output_page.scope_rows(read_scope))
            if output_rows != ((read_scope, 0),):
                continue
            output = book.latest_ref(SEMANTIC_OUTPUT, output_rows[0])
            output_fact = output_page.cell(output.row, output.column)
            if not isinstance(output_fact, tuple) or len(output_fact) != 2:
                continue
            returned_id = int(output_fact[1])
            returned = book.latest_ref(
                CANONICAL_VALUE, (read_scope, returned_id),
            )
            class_cell = book.latest_ref(
                SOURCE_VALUE_CLASS, (callee_scope, returned_id),
            )
            if returned is None or class_cell is None or returned not in {
                parent for parent, _ in book.edges_into(output)
            } or not _class_matches_operand(book, class_cell, returned):
                continue
            class_fact = book.page(SOURCE_VALUE_CLASS).cell(
                class_cell.row, class_cell.column,
            )
            if class_fact[:2] != ("builtins.bytes", 1):
                continue
            outputs.append((output, returned, class_cell))
        if len(outputs) != 1:
            continue
        output, returned, class_cell = outputs[0]
        result_row = (caller_scope, int(call_id))
        result_fact = (
            int(canonical_entry.reference.address), "builtins.bytes", 1,
        )
        incumbent = book.page(SOURCE_CALL_RESULT_IDENTITY).latest(result_row)
        if incumbent is not None and tuple(incumbent) != result_fact:
            continue
        if incumbent is None:
            book.post(
                SOURCE_CALL_RESULT_IDENTITY, result_row, result_fact,
                stage=REDUCTION,
                provenance=Derived((
                    call, reached, reached_address, canonical_address,
                    output, returned, class_cell,
                )),
                mode=Mode.CONCORD,
            )
            posted += 1
        result_cell = book.latest_ref(SOURCE_CALL_RESULT_IDENTITY, result_row)
        if book.latest_ref(SOURCE_VALUE_CLASS, result_row) is None:
            _concord_source_value_class(
                caller_scope, int(call_id), "builtins.bytes",
                source="eager reached bytes return",
                provenance=Derived((result_cell,)),
            )
            posted += 1
    return posted
