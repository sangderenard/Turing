"""The frame linker's identity-book access: scope, cells, posts, mints, storage leases (relocated from fortran_c_shell)."""

from __future__ import annotations

from typing import Any, Iterable


# --------------------------------------------------------------------------
# The frame linker's book access (concordance step 7, plan 90 section 3).
#
# One identity per value, one key to it: an SSA value's identity row is step
# 5's ``ssa_value`` page under the function's control scope.  Every value the
# linker mints is a NOVEL row there, with the transform that made it and the
# cell it was made from; every statement the linker writes derives from the
# cells it read, or says why it could not.
# --------------------------------------------------------------------------


def _frame_book_scope(function: Any) -> str:
    """The ``ssa_value`` row scope of ``function``: its control scope, else
    its name (the same key step 5's builder and region lowering use)."""

    metadata = getattr(function, "metadata", None) or {}
    return str(
        metadata.get("tensor_shape_concordance_scope")
        or getattr(function, "name", function)
    )


def _frame_cells(*items: Any) -> tuple[Any, ...]:
    """The distinct Refs among ``items`` (None and non-Refs dropped)."""

    from .identity_concordance import Ref

    found: list = []
    for item in items:
        if isinstance(item, Ref) and item not in found:
            found.append(item)
    return tuple(found)


def _frame_value_cell(function: Any, value_id: Any) -> Any:
    """The ``ssa_value`` cell of ``value_id`` in ``function``, or None."""

    from .concordance_declarations import SSA_VALUE
    from .identity_concordance import current_identity_book

    if function is None or value_id is None:
        return None
    return current_identity_book().latest_ref(
        SSA_VALUE, (_frame_book_scope(function), int(value_id)),
    )


def _frame_graph_cell(graph: Any, value_id: Any) -> Any:
    """The ``canonical_value`` cell of graph id ``value_id`` in the source
    graph ``graph`` (its ``lexical_read_scope``), or None."""

    from .concordance_declarations import CANONICAL_VALUE
    from .identity_concordance import current_identity_book

    if graph is None or value_id is None:
        return None
    read_scope = (getattr(graph, "graph", None) or {}).get("lexical_read_scope")
    if read_scope is None:
        return None
    return current_identity_book().latest_ref(
        CANONICAL_VALUE, (tuple(read_scope), int(value_id)),
    )


def _frame_table_cell(table: Any, value_id: Any, page: Any) -> Any:
    """The ``record_member`` / ``sequence_member`` / descriptor cell of
    ``value_id`` on a book-backed SSA table, or None."""

    from .identity_concordance import current_identity_book

    owner = getattr(table, "owner", None)
    if table is None or owner is None or value_id is None:
        return None
    return current_identity_book().latest_ref(page, (owner, int(value_id)))


def _frame_post(
    page: Any, row: tuple, fact: Any, *, stage: Any, cells: Any = (),
    mode: Any = None, reason: Any = None,
) -> Any:
    """Post ``fact`` DERIVED from ``cells``; ``Unsourced(reason)`` when no
    cell could be named (default reason ``FRAME_SOURCE_CELL_ABSENT``).

    CONCORD (the default) keeps ``concord``'s rule: a different fact for a
    recorded row raises.  REVISE with an unchanged fact posts nothing (the
    previous cell is returned); a revision the api refuses for want of a
    changed source is recorded ``Unsourced`` rather than dropped.
    """

    from .concordance_declarations import FRAME_SOURCE_CELL_ABSENT
    from .identity_concordance import (
        ConcordanceRefusal, Derived, Mode, Unsourced, current_identity_book,
    )

    book = current_identity_book()
    mode = Mode.CONCORD if mode is None else mode
    reason = FRAME_SOURCE_CELL_ABSENT if reason is None else reason
    cells = _frame_cells(*cells)
    if mode is Mode.REVISE:
        previous = book.latest_ref(page, row)
        if previous is not None:
            recorded = book.pages[page.name].latest(row)
            try:
                unchanged = bool(recorded == fact)
            except Exception:  # an array-valued literal compares elementwise
                unchanged = recorded is fact
            if unchanged:
                return previous
    if cells:
        try:
            return book.post(
                page, row, fact, stage=stage, provenance=Derived(cells),
                mode=mode,
            )
        except ConcordanceRefusal:
            if mode is not Mode.REVISE:
                raise
    return book.post(
        page, row, fact, stage=stage, provenance=Unsourced(reason), mode=mode,
    )


def _frame_mint(
    function: Any, transform: Any, operands: Any = (), *,
    dtype: Any = None, shape: Any = (), stage: Any,
) -> int:
    """Mint one SSA id for ``function`` through the book: a NOVEL
    ``ssa_value`` row under its scope with ``transform`` and the one cell in
    ``operands`` (several become a ``cell_set`` row; none names the
    function root, exactly as ``_ControlSSABuilder.fresh_value`` does)."""

    from .identity_concordance import current_identity_book
    from .precompile_to_ssa import _function_root_cell, _mint_ssa_id

    book = current_identity_book()
    scope = _frame_book_scope(function)
    cells = _frame_cells(*operands) or (_function_root_cell(book, scope),)
    return _mint_ssa_id(
        book, scope, transform, cells,
        dtype=dtype, shape=tuple(shape or ()), stage=stage,
    )


class _RecordAbiMinter:
    """The id source of one record-abi materialization, routed through
    the book.

    ``materialize_parameter_record_abi`` mints the physical parts of a
    declared record (columns, lengths, capacities, strides, pointers,
    status and presence cells, pooled scalars, token constants) at some
    thirty-five sites, every one of them ``GLOBAL_MONOTONIC_IDS.mint()``.
    Bound to that name inside the function, this object makes each of them
    a NOVEL ``ssa_value`` row under the function's scope with the
    transform NESTED_RECORD_PART and, as operand, the declaration cell the
    materialization is currently expanding (``declare``), else the function
    root -- the same rule ``_ControlSSABuilder.fresh_value`` applies to a
    mint with no more specific cell.  Nothing else in the file sees it.
    """

    def __init__(self, function: Any, transform: Any, stage: Any) -> None:
        self.function = function
        self.transform = transform
        self.stage = stage
        self.declaration: Any = None

    def declare(self, cell: Any) -> None:
        """Name the declaration cell the next mints are parts of."""

        self.declaration = cell

    def mint(self, *operands: Any, dtype: Any = None, shape: Any = ()) -> int:
        return _frame_mint(
            self.function, self.transform,
            (*operands, self.declaration),
            dtype=dtype, shape=shape, stage=self.stage,
        )


def _result_storage_lease_cell(
    caller_symbol: Any, callsite_id: Any, callee_value_id: Any,
) -> Any:
    """The latest ``result_storage_binding`` cell for one callee value at
    one call, or None.  Its fact is the leased storage's ``ssa_value``
    cell; the storage id is that cell's ``row[1]``."""

    from .concordance_declarations import RESULT_STORAGE_BINDING
    from .identity_concordance import current_identity_book

    book = current_identity_book()
    page = book.pages.get(RESULT_STORAGE_BINDING.name)
    if page is None:
        return None
    found = None
    for row in page.scope_rows(str(caller_symbol)):
        if (
            len(row) == 4
            and row[1] == callsite_id
            and int(row[2]) == int(callee_value_id)
        ):
            found = row
    return None if found is None else book.latest_ref(RESULT_STORAGE_BINDING, found)


def _post_result_storage_lease(
    caller_function: Any, caller_symbol: Any, callsite_id: Any,
    callee_value_id: int, storage_id: int, cells: Any, *, stage: Any,
) -> Any:
    """Record one leased result slot: row ``(caller, callsite, callee value,
    serial)`` whose fact is the storage's ``ssa_value`` cell, DERIVED from
    the callee value's cells.  A ``distinct_slot`` lease for the same callee
    value is the next serial, never a disagreement (plan 90 R7.3)."""

    from .concordance_declarations import RESULT_STORAGE_BINDING
    from .identity_concordance import current_identity_book

    book = current_identity_book()
    page = book.page(RESULT_STORAGE_BINDING)
    serial = sum(
        1 for row in page.scope_rows(str(caller_symbol))
        if len(row) == 4
        and row[1] == callsite_id
        and int(row[2]) == int(callee_value_id)
    )
    storage_cell = _frame_value_cell(caller_function, storage_id)
    return _frame_post(
        RESULT_STORAGE_BINDING,
        (str(caller_symbol), callsite_id, int(callee_value_id), int(serial)),
        storage_cell, stage=stage, cells=cells,
    )


def _argument_binding_fact(book: Any, callee_symbol: Any, formal_id: Any, callsite_id: Any) -> Any:
    """The ``argument_binding`` fact for one callee formal at one callsite:
    the step-7 row ``(callee, formal, callsite)`` first, else the pre-step-7
    row ``(callee, formal, "binding")`` at column ``callsite``.  None when
    neither exists; an ``Unresolved`` is returned as such."""

    from .concordance_declarations import ARGUMENT_BINDING

    page = book.pages.get(ARGUMENT_BINDING.name)
    if page is None or callsite_id is None:
        return None
    row = (str(callee_symbol), int(formal_id), callsite_id)
    fact = page.latest(row)
    if fact is not None:
        return fact
    return page.cells.get(
        ((str(callee_symbol), int(formal_id), "binding"), int(callsite_id)),
    )


def _rebind_linked_storage_alias(
    argument: SSAValue,
    placeholder: SSAValue,
    replacement: SSAValue,
) -> SSAValue:
    """Move one proven ordered view from a semantic to physical container.

    Region calls deliberately clone ``SSAValue`` objects so one storage can
    carry a different shape at each call position.  Source-call linking may
    then freshen the aggregate container behind that storage.  Object-only
    replacement misses those views, while replacement by numeric id can
    capture an unrelated value from another numbering domain.  The explicit
    ``ssa_storage_alias`` receipt is the narrow bridge between the two.
    """

    from ..transmogrifier.ssa import SSAValue

    if argument is placeholder:
        return replacement
    accounting = dict(argument.accounting or {})
    # A region feed can be a typed view of a loop-carried Phi which happens
    # to share the linked aggregate placeholder's physical id.  Its explicit
    # carried receipt outranks the generic storage-alias receipt: rebinding it
    # would replace the Phi with a repeated seed projection and reset the loop
    # on every iteration.
    if accounting.get("ssa_loop_carried_feed") is not None:
        return argument
    storage_alias = accounting.get("ssa_storage_alias")
    variant_source = accounting.get("unbound_variant_source_id")
    if (
        storage_alias != int(placeholder.id)
        and variant_source != int(placeholder.id)
    ):
        return argument
    return SSAValue(
        int(replacement.id),
        dtype=argument.dtype or replacement.dtype,
        shape=tuple(argument.shape or ()),
        device=argument.device or replacement.device,
        accounting={
            **accounting,
            "ssa_storage_alias": int(replacement.id),
            "ssa_linked_storage_from": int(placeholder.id),
            **(
                {
                    "ssa_linked_variant_row_from": int(storage_alias),
                }
                if variant_source == int(placeholder.id)
                and storage_alias != int(placeholder.id)
                else {}
            ),
        },
    )


def _frame_binding_value_ids(
    bindings: Iterable[tuple[int, str, Any]],
) -> tuple[int, ...]:
    """Return only physical SSA ids from a linked-call frame receipt.

    Integer caller/default literals and opaque function-reference tokens are
    payloads, not members of the caller's value-id namespace.  Letting them
    seed the fresh allocator creates id()-scale SSA values.
    """

    value_kinds = {"caller_value", "caller_alias", "caller_storage"}
    return tuple(
        int(source)
        for _callee_id, kind, source in bindings
        if kind in value_kinds and isinstance(source, int)
    )


def _monotonic_ssa_ids(values: Iterable[Any]) -> tuple[int, ...]:
    """Select graph ids that belong to the repository-SSA counter domain."""

    from .ssa_self_check import ID_SCALE_THRESHOLD

    selected = []
    for value in values:
        try:
            value_id = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= value_id < ID_SCALE_THRESHOLD:
            selected.append(value_id)
    return tuple(selected)
