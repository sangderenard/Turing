"""Formal shape/literal ledger: publication and proof of one formal's shape or literal across callsites."""

from __future__ import annotations

from typing import Any


_FORMAL_LITERAL_CONFLICT = object()


def _publish_formal_literal(
    function: str, parameter: str, value: Any, caller: str, cells: tuple = (),
) -> None:
    """Record that a callsite proved one literal for one formal.

    Page ``formal_literal`` row ``(authored function, parameter)``: the
    first callsite posts ``("proven", value, caller)`` DERIVED from the
    argument cells that proved it; a later callsite with another value
    revises the row to ``Unresolved(FORMAL_LITERAL_CONFLICT)`` reading both
    -- genuinely parametric, two callsites, two values.  A caller with no
    cells posts ``Unsourced(RAW_PRIMITIVE)`` under the latch.
    """

    from .concordance_declarations import (
        FORMAL_LITERAL, FORMAL_LITERAL_CONFLICT, PLANNER_SPECIALIZATION,
    )
    from .identity_concordance import (
        Derived, Mode, RAW_PRIMITIVE, Unresolved, Unsourced,
        current_identity_book,
    )

    book = current_identity_book()
    row = (str(function), str(parameter))
    previous = book.latest_ref(FORMAL_LITERAL, row)
    cells = tuple(cells)
    provenance = Derived(cells) if cells else Unsourced(RAW_PRIMITIVE)
    if previous is None:
        book.post(
            FORMAL_LITERAL, row, ("proven", value, caller),
            stage=PLANNER_SPECIALIZATION, provenance=provenance,
            mode=Mode.REVISE,
        )
        return
    incumbent = book.pages[FORMAL_LITERAL.name].latest(row)
    if isinstance(incumbent, Unresolved):
        return
    try:
        agrees = bool(incumbent[1] == value)
    except Exception:
        agrees = False
    if agrees:
        return
    read = (previous, *cells)
    book.post(
        FORMAL_LITERAL, row, Unresolved(FORMAL_LITERAL_CONFLICT, read=read),
        stage=PLANNER_SPECIALIZATION,
        provenance=Derived(read) if cells else Unsourced(RAW_PRIMITIVE),
        mode=Mode.REVISE,
    )


def _publish_formal_shape(
    function: str, parameter: str, descriptor: Any, caller: str,
    cells: tuple = (),
) -> bool:
    """Record that a callsite proved one shape for one formal.

    Mirrors ``_publish_formal_literal``: ``("proven", extents, dtype,
    caller)`` DERIVED from the argument cells; a differing extent revises to
    ``Unresolved(FORMAL_SHAPE_CONFLICT)``.  A third caller's shape is a
    revision with a changed source, no longer lost.
    """

    from .concordance_declarations import (
        FORMAL_SHAPE, FORMAL_SHAPE_CONFLICT, PLANNER_TENSOR_SPECIALIZATION,
    )
    from .identity_concordance import (
        Derived, Mode, RAW_PRIMITIVE, Unresolved, Unsourced,
        authored_function_name, current_identity_book,
    )

    extents = tuple(int(e) for e in (descriptor.get("shape") or ()))
    dtype = str(descriptor.get("dtype") or "float64")
    book = current_identity_book()
    row = (authored_function_name(function), str(parameter))
    previous = book.latest_ref(FORMAL_SHAPE, row)
    cells = tuple(cells)
    if previous is None:
        book.post(
            FORMAL_SHAPE, row, ("proven", extents, dtype, caller),
            stage=PLANNER_TENSOR_SPECIALIZATION,
            provenance=Derived(cells) if cells else Unsourced(RAW_PRIMITIVE),
            mode=Mode.REVISE,
        )
        return True
    incumbent = book.pages[FORMAL_SHAPE.name].latest(row)
    read = (previous, *cells)
    if isinstance(incumbent, Unresolved):
        # Already parametric.  A third caller is a revision only when it is
        # a new source (its cells are not the ones already read); the same
        # caller on a later fixed-point round is not.
        if not cells or {cell.key for cell in read} == {
            source.key for source, _stage in book.edges_into(previous)
        }:
            return False
        book.post(
            FORMAL_SHAPE, row, Unresolved(FORMAL_SHAPE_CONFLICT, read=read),
            stage=PLANNER_TENSOR_SPECIALIZATION, provenance=Derived(read),
            mode=Mode.REVISE,
        )
        return True
    if (
        isinstance(incumbent, tuple)
        and incumbent[0] == "proven"
        and tuple(incumbent[1]) != extents
    ):
        book.post(
            FORMAL_SHAPE, row, Unresolved(FORMAL_SHAPE_CONFLICT, read=read),
            stage=PLANNER_TENSOR_SPECIALIZATION,
            provenance=Derived(read) if cells else Unsourced(RAW_PRIMITIVE),
            mode=Mode.REVISE,
        )
        return True
    return False


def _proven_formal_shape(graph: Any, name: Any) -> Any:
    """The shape every callsite agrees this formal carries, if any."""

    if not name:
        return None
    try:
        from .identity_concordance import (
            authored_function_name,
            current_identity_book,
        )

        page = current_identity_book().page("formal_shape")
        row = (
            authored_function_name(graph.G.graph.get("function_name")),
            str(name),
        )
        proven = page.latest(row)
    except Exception:
        return None
    # ``Unresolved(FORMAL_SHAPE_CONFLICT)`` is the conflicting fact.
    if not isinstance(proven, tuple) or proven[0] != "proven":
        return None
    return {
        "shape": tuple(proven[1]),
        "dtype": str(proven[2]),
        "rank": len(tuple(proven[1])),
    }


def _proven_formal_literal(graph: Any, name: Any) -> Any:
    """The literal every callsite agrees this formal carries, if any."""

    if not name:
        return _FORMAL_LITERAL_CONFLICT
    try:
        from .identity_concordance import current_identity_book

        page = current_identity_book().page("formal_literal")
        row = (str(graph.G.graph.get("function_name")), str(name))
        proven = page.latest(row)
    except Exception:
        return _FORMAL_LITERAL_CONFLICT
    # ``Unresolved(FORMAL_LITERAL_CONFLICT)`` is the conflicting fact.
    if not isinstance(proven, tuple) or proven[0] != "proven":
        return _FORMAL_LITERAL_CONFLICT
    return proven[1]
