"""Read admitted intrinsic result fields from the source concordance."""

from __future__ import annotations

from .concordance_declarations import (
    CLASS_DECLARATION, FUNCTION_RETURN_ANNOTATION, INGESTION_VALUE,
    SOURCE_CALLABLE_SIGNATURE, SOURCE_INTRINSIC_ADMISSION,
    SOURCE_NAMED_TUPLE_LAYOUT,
)
from .concorded_literals import origin_cell
from .identity_concordance import Ref, current_identity_book


def intrinsic_result_fields(graph, node_id: int):
    """Return ``(ordered (name, field Ref)s, proof Refs)`` for one intrinsic.

    The call's exact canonical-to-ingestion lineage selects its admission.
    The callable's authored return annotation selects one declared class; its
    NamedTuple layout supplies order. This reports no member dtype or physical
    output representation.
    """

    from ..common.tensors.topological_reducer import node_identity_cell

    book = current_identity_book()
    call = node_identity_cell(graph, int(node_id))
    if call is None:
        return (), ()
    try:
        origin = origin_cell(book, call)
    except ValueError:
        return (), ()
    if origin.page != INGESTION_VALUE:
        return (), ()
    admission = book.latest_ref(SOURCE_INTRINSIC_ADMISSION, (origin,))
    if admission is None:
        return (), ()
    admission_fact = book.page(SOURCE_INTRINSIC_ADMISSION).cell(
        admission.row, admission.column,
    )
    if not (
        isinstance(admission_fact, tuple) and len(admission_fact) == 3
        and isinstance(admission_fact[1], dict)
        and admission_fact[1].get("action") == "intrinsic"
        and isinstance(admission_fact[2], Ref)
        and admission_fact[2].page == SOURCE_CALLABLE_SIGNATURE
    ):
        return (), ()
    signature = admission_fact[2]
    signature_fact = book.page(SOURCE_CALLABLE_SIGNATURE).cell(
        signature.row, signature.column,
    )
    if not (isinstance(signature_fact, tuple) and signature_fact):
        return (), ()
    definition = signature_fact[0]
    if not isinstance(definition, Ref):
        return (), ()
    annotation = book.latest_ref(
        FUNCTION_RETURN_ANNOTATION, definition.row[:2],
    )
    if annotation is None:
        return (), ()
    classes = tuple(
        source for source, _ in book.edges_into(annotation)
        if source.page == CLASS_DECLARATION
    )
    if len(classes) != 1:
        return (), ()
    layout = book.latest_ref(SOURCE_NAMED_TUPLE_LAYOUT, (classes[0],))
    if layout is None:
        return (), ()
    fact = book.page(SOURCE_NAMED_TUPLE_LAYOUT).cell(
        layout.row, layout.column,
    )
    if not (isinstance(fact, tuple) and len(fact) == 2):
        return (), ()
    members = fact[1]
    if not isinstance(members, tuple) or not members:
        return (), ()
    fields = tuple((name, field) for name, field, _span in members)
    return fields, (
        call, origin, admission, signature, definition, annotation,
        classes[0], layout,
    )
