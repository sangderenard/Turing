"""Forward a closed semantic class domain through an exact formal return."""

from .concordance_declarations import (
    CANONICAL_VALUE, REDUCTION, SEMANTIC_OUTPUT,
    SOURCE_ACTIVITY_SCOPE, SOURCE_CALL_ACTUAL_FORMAL,
    SOURCE_CALL_CLOSED_CLASS_ALTERNATIVES,
    SOURCE_FORMAL_CLOSED_CLASS_ALTERNATIVES,
    SOURCE_FUNCTION_REACHABILITY,
)
from .identity_concordance import Derived, Mode, Ref
from .source_evidence_resolver import (
    ClassAlternativesEvidence, require_class_alternatives,
)


def post_formal_return_class_domains(book) -> int:
    """Post a call domain when its sole output is an exactly bound formal.

    `relay(value): return value` carries the A/B source domain supplied by
    `root` only when every reached call to relay binds that formal and the
    formal resolver has a complete closed domain. No physical arm is minted.
    """

    posted = 0
    formal_page = book.page(SOURCE_FORMAL_CLOSED_CLASS_ALTERNATIVES)
    reach_page = book.page(SOURCE_FUNCTION_REACHABILITY)
    actual_page = book.page(SOURCE_CALL_ACTUAL_FORMAL)
    for row in formal_page.rows():
        formal_scope, formal_id, caller_scope, call_id = row
        # This narrow transport has one exact call to the forwarding formal.
        # Several distinct callers remain unresolved until a control-owned
        # join has a receipt for their combined domain.
        if sum(candidate[:2] == row[:2] for candidate in
               formal_page.rows()) != 1:
            continue
        answer = require_class_alternatives(
            formal_scope, int(formal_id), book=book,
        )
        if not isinstance(answer, ClassAlternativesEvidence):
            continue
        callee_scope = book.page(SOURCE_ACTIVITY_SCOPE).latest(
            (formal_scope,))
        if callee_scope is None:
            continue
        reach_row = (caller_scope, int(call_id), callee_scope)
        reached = book.latest_ref(SOURCE_FUNCTION_REACHABILITY, reach_row)
        if reached is None or reach_page.cell(
            reached.row, reached.column) is not True:
            continue
        reached_calls = {
            candidate[:2] for candidate in reach_page.rows()
            if candidate[2] == callee_scope
            and reach_page.latest(candidate) is True
        }
        bound_calls = {
            candidate[:2] for candidate in actual_page.rows()
            if candidate[2] == callee_scope
            and int(candidate[3]) == int(formal_id)
            and actual_page.latest(candidate) is not None
        }
        if reached_calls != bound_calls or reached_calls != {
            (caller_scope, int(call_id))
        }:
            continue
        output_rows = tuple(candidate for candidate in
                            book.page(SEMANTIC_OUTPUT).scope_rows(formal_scope))
        if output_rows != ((formal_scope, 0),):
            continue
        output = book.latest_ref(SEMANTIC_OUTPUT, output_rows[0])
        returned = book.latest_ref(
            CANONICAL_VALUE, (formal_scope, int(formal_id)),
        )
        if output is None or returned is None or (
            book.page(SEMANTIC_OUTPUT).cell(output.row, output.column)[1]
            != int(formal_id)
        ) or returned not in {
            source for source, _stage in book.edges_into(output)
        }:
            continue
        calls = tuple(
            source for source, _stage in book.edges_into(reached)
            if source.page == CANONICAL_VALUE
            and int(source.row[1]) == int(call_id)
        )
        if len(calls) != 1:
            continue
        formal_domain = book.latest_ref(
            SOURCE_FORMAL_CLOSED_CLASS_ALTERNATIVES, row,
        )
        if formal_domain is None:
            continue
        result_row = (
            caller_scope, int(call_id), callee_scope, int(formal_id),
        )
        if book.latest_ref(SOURCE_CALL_CLOSED_CLASS_ALTERNATIVES,
                           result_row) is not None:
            continue
        # Program: root -> choose(A/B) -> relay(value) -> root Call 2.
        # The callee output is the formal Input, whose complete call binding
        # owns the same closed arms. Preserve that exact route in the call
        # result view, without assigning one class or a native union ABI.
        book.post(
            SOURCE_CALL_CLOSED_CLASS_ALTERNATIVES, result_row,
            (output, returned, formal_domain, reached, calls[0]),
            stage=REDUCTION,
            provenance=Derived((
                output, returned, formal_domain, reached, calls[0],
            )),
            mode=Mode.CONCORD,
        )
        posted += 1
    return posted
