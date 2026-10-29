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


def _frame_owner_book(function: Any) -> tuple[Any, Any]:
    """``(book, artifact_sha256)`` that owns ``function``'s rows: its accepted
    component's retained book and verified digest, else the ambient book and
    None (the same owner rule as ``_accepted_component_identity_book``)."""

    from .identity_concordance import current_identity_book

    accepted = dict(
        (getattr(function, "metadata", None) or {}).get(
            "accepted_ssa_component"
        ) or {}
    )
    owner = accepted.get("identity_book")
    if owner is None:
        return current_identity_book(), None
    return owner, accepted.get("artifact_sha256")


def _frame_cite(book: Any, cell: Any, artifact_sha256: Any) -> Any:
    """``cell`` as the ambient book may cite it: itself when ``book`` is the
    ambient book, else the IMPORTED_CELL receipt (``import_cell``) carrying
    the owner's verified digest."""

    from .identity_concordance import current_identity_book

    ambient = current_identity_book()
    if cell is None or book is ambient:
        return cell
    return ambient.import_cell(book, cell, artifact_sha256)


# Program: a result storage lease for an accepted callee.  Wrong record: the
# callee's ``ssa_value`` cell was looked up on the AMBIENT book (None,
# silently dropped from the lease's cause set, or another row of the same
# key) instead of the book that owns the callee.  Rule: a function's cells are
# read in its owner's book and cited to the caller's book by IMPORTED_CELL
# receipt; a cell of the ambient book is cited directly.
def _frame_value_cell(function: Any, value_id: Any, *, book: Any = None) -> Any:
    """The ``ssa_value`` cell of ``value_id`` in ``function``, or None.

    Read in ``book`` or ``function``'s owner book and returned as the ambient
    book may cite it (an IMPORTED_CELL receipt for an accepted callee)."""

    from .concordance_declarations import SSA_VALUE

    if function is None or value_id is None:
        return None
    owner, digest = _frame_owner_book(function)
    if book is not None:
        owner = book
    cell = owner.latest_ref(
        SSA_VALUE, (_frame_book_scope(function), int(value_id)),
    )
    return _frame_cite(owner, cell, digest)


# Program: a linked call whose result storage / output is read from the
# callee's output (``fresh`` record copy, ``history_projection``,
# ``loop_call_carry``; audit kind ``ssa-physical-unposted``).  Wrong record:
# the linker's value rows carried no physical set -- the lease mint passed no
# ``physical=``, and the output rebuild kept KEEP_PHYSICAL while copying the
# callee's ``accounting`` dict onto the object.  Rule: a value that is a
# view of a callee value posts that value's PHYSICAL fields as recorded on
# the callee's own ``ssa_value`` row, read in the owner's book (the same
# owner rule as ``_frame_value_cell``); accounting is never the source.  A
# callee row that records none posts none -- the residue then belongs to the
# callee's producer, not to this writer.
def _frame_field_recorded(name: str, value: Any) -> bool:
    """Whether a physical field is RECORDED on a row.  Program: ``fresh`` /
    ``keep`` (a scalar result, rank 0).  Wrong record: the test was
    ``value not in (None, False)`` and ``0 == False``, so a recorded rank of
    0 read as "records none" and the lease posted no rank.  Rule: ``arena``
    is a flag (recorded when True); every other field is recorded when it is
    not None, so rank 0 is a recorded rank."""

    if name == "arena":
        return value is True
    return value is not None


def _frame_physical_fields(function: Any, value_id: Any, *, book: Any = None) -> Any:
    """The physical fields (``physical_dtype``, ``storage``, ``rank``,
    ``arena``, ``device``) recorded on ``value_id``'s ``ssa_value`` row in
    ``function``'s owner book (or ``book``), as a mapping a mint or
    ``publish_ssa_value_type(physical=...)`` accepts; ``KEEP_PHYSICAL`` when
    the function has no such row or the row records none."""

    from .concordance_declarations import KEEP_PHYSICAL, SSA_VALUE, SSAValueFact

    if function is None or value_id is None:
        return KEEP_PHYSICAL
    owner, _ = _frame_owner_book(function)
    if book is not None:
        owner = book
    fact = owner.page(SSA_VALUE).latest(
        (_frame_book_scope(function), int(value_id))
    )
    if not isinstance(fact, SSAValueFact):
        return KEEP_PHYSICAL
    # Only the fields the callee row RECORDS are posted; an unrecorded field
    # is not asserted (and cannot erase one the receiving row already has).
    posted = {
        name: getattr(fact, name)
        for name in ("physical_dtype", "storage", "rank", "arena", "device")
        if _frame_field_recorded(name, getattr(fact, name))
    }
    return posted or KEEP_PHYSICAL


def _frame_accounting_from_posted(accounting: Any, physical: Any) -> dict:
    """``accounting`` with its five physical labels (``physical_dtype``,
    ``program_abi_storage``, ``program_abi_rank``, ``sequence_arena``,
    ``device``) set from the physical fields just POSTED -- the row is the
    truth, the label follows it.

    Program: ``keep`` / ``fresh`` / ``coerce`` / ``loop_call_carry``.  Wrong
    record: only storage and rank were mirrored, and only where accounting
    already carried the key, so a posted dtype, arena or device never reached
    accounting.  Rule: accounting mirrors the fact -- every field the fact
    records is written, whether or not the label was present."""

    accounting = dict(accounting or {})
    if isinstance(physical, dict):
        for field, label in (
            ("physical_dtype", "physical_dtype"),
            ("storage", "program_abi_storage"),
            ("rank", "program_abi_rank"),
            ("arena", "sequence_arena"),
            ("device", "device"),
        ):
            # A recorded rank of 0 is a fact (``0 == False`` made the old
            # ``not in (None, False)`` test drop it: a scalar formal's
            # ``program_abi_rank`` never mirrored its row).
            if _frame_field_recorded(field, physical.get(field)):
                accounting[label] = physical[field]
    return accounting


def _frame_required_value_cell(function: Any, value_id: Any) -> Any:
    """``_frame_value_cell`` for a cell the caller is about to cite: a missing
    row of an accepted callee is a lost proof and refuses with the function,
    id and book consulted; an ambient function's missing row stays None."""

    from .identity_concordance import ConcordanceRefusal

    cell = _frame_value_cell(function, value_id)
    accepted = function is not None and bool(
        (getattr(function, "metadata", None) or {}).get(
            "accepted_ssa_component"
        )
    )
    if cell is None and accepted and value_id is not None:
        owner, _ = _frame_owner_book(function)
        raise ConcordanceRefusal(
            f"lost proof: no ssa_value row for value {int(value_id)} of "
            f"accepted callee {getattr(function, 'name', function)!r} in its "
            f"accepted component's identity book {id(owner):#x}"
        )
    return cell


def _frame_graph_cell(graph: Any, value_id: Any) -> Any:
    """The identity cell of graph id ``value_id`` in the source graph
    ``graph``: its ``canonical_value`` row under the graph's lexical read
    scope, else its ``ingestion_value`` row under the graph's ingestion
    scope (a specialized or instance copy keeps its nodes there) -- the one
    reader ``node_identity_cell`` is -- or None when ``value_id`` names no
    node of ``graph``."""

    from ..common.tensors.topological_reducer import node_identity_cell

    if graph is None or value_id is None:
        return None
    node_id = int(value_id)
    if node_id not in graph:
        # Rows are keyed by node; a value minted on a node (a JoinedStr's
        # resident sequence) is named by ``value_id``.
        node_id = next((
            node for node, data in graph.nodes(data=True)
            if data.get("value_id") == node_id
        ), None)
        if node_id is None:
            return None
    return node_identity_cell(graph, node_id)


# Program: a result storage lease for an accepted callee.  Wrong record: the
# callee's table member cell looked up on the ambient book.  Rule: ``function``
# (the table's owning function, when it is an accepted callee) names the book
# the row is read in; the cell is cited by IMPORTED_CELL receipt.
def _frame_table_cell(
    table: Any, value_id: Any, page: Any, *, book: Any = None,
    function: Any = None,
) -> Any:
    """The ``record_member`` / ``sequence_member`` / descriptor cell of
    ``value_id`` on a book-backed SSA table, or None.

    Read in ``book``, else the owner book of ``function`` (the table's
    function), else the ambient book."""

    owner = getattr(table, "owner", None)
    if table is None or owner is None or value_id is None:
        return None
    source, digest = _frame_owner_book(function)
    if book is not None:
        source = book
    cell = source.latest_ref(page, (owner, int(value_id)))
    return _frame_cite(source, cell, digest)


def _frame_register_sequence(
    function: Any, table: Any, descriptor: Any, *extra: Any, stage: Any = None,
    revise: bool = False,
) -> Any:
    """Publish a sequence descriptor on the book, DERIVED from the cells it
    stands on: ``extra`` (the descriptor it is mapped from), the incumbent
    row it revises, and the ``ssa_value`` cells of the values it names in
    ``function``; the function root when none has a cell (``_frame_mint``'s
    rule).

    ``revise`` replaces the incumbent row (``table.sequences.assign``);
    otherwise ``table.register`` publishes it and refuses disagreement.
    """

    from .concordance_declarations import SEQUENCE_DESCRIPTOR
    from .identity_concordance import current_identity_book
    from .precompile_to_ssa import _function_root_cell

    pool = descriptor.child_table_pool
    named = (
        descriptor.sequence_id, *descriptor.column_value_ids,
        descriptor.length_address_id, descriptor.capacity_value_id,
        descriptor.status_address_id, descriptor.live_flags_value_id,
        descriptor.heap_owner_value_id,
        *(() if pool is None else (
            *pool.column_value_ids, pool.length_value_id,
            pool.capacity_value_id, pool.row_stride_value_id,
            pool.shape_value_id, pool.rank_value_id,
            pool.shape_stride_value_id, pool.status_value_id,
            pool.live_flags_value_id,
        )),
    )
    sources = _frame_cells(
        *extra,
        _frame_table_cell(table, descriptor.sequence_id, SEQUENCE_DESCRIPTOR),
        *(_frame_value_cell(function, value_id) for value_id in named),
    ) or (_function_root_cell(
        current_identity_book(), _frame_book_scope(function),
    ),)
    if revise:
        table.sequences.assign(
            int(descriptor.sequence_id), descriptor, sources=sources,
            stage=stage,
        )
        return descriptor
    return table.register(descriptor, sources=sources, stage=stage)


def _record_field_declaration_cell(
    record: Any, field_name: Any, identity: Any = None,
) -> Any:
    """The ``class_field_declaration`` cell of ``field_name`` of a declared
    ProgramABI ``record``: the row its ``source_record_view_schema`` names
    when it carries one, else the declaration row whose class identity is
    the record's identity (``identity`` overrides it).  None when ingestion
    posted neither."""

    from .concordance_declarations import CLASS_FIELD_DECLARATION
    from .identity_concordance import current_identity_book

    book = current_identity_book()
    record = record or {}
    schema_row = record.get("source_record_view_schema")
    if isinstance(schema_row, (tuple, list)) and len(schema_row) == 2:
        cell = book.latest_ref(
            CLASS_FIELD_DECLARATION, (*tuple(schema_row), str(field_name)),
        )
        if cell is not None:
            return cell
    class_identity = str(
        record.get("identity") if identity is None else identity
    )
    page = book.pages.get(CLASS_FIELD_DECLARATION.name)
    qualified: set = set()
    for row in () if page is None else page.rows():
        # The declaration row is keyed ``(module, class_identity, field)``.
        # A contract record names its class by the qualified identity
        # ``module.Class`` (program_extraction.yaml:
        # ``src.common.dt_system.dt_scaler.Metrics``), while ingestion posts
        # the row as ``('src.common.dt_system.dt_scaler', 'Metrics',
        # 'dt_limit')``.  Explicit contract records override the inferred
        # schemas (fortran_c_shell ``program_abi_receipt["records"]``), so a
        # contract record never carries ``source_record_view_schema`` and
        # this join is its only route to the cell.  Comparing the bare
        # ``row[1]`` to the qualified identity matched nothing, and the
        # optional-field presence lowering refused ``Metrics.dt_limit`` /
        # ``STController.dt_min`` although their cells were posted.  The
        # row's own ``(module, class)`` pair IS the qualified identity.
        if len(row) != 3 or str(row[2]) != str(field_name):
            continue
        row_identity = (
            f"{row[0]}.{row[1]}" if str(row[0]) else str(row[1])
        )
        if (
            (row_identity == class_identity or str(row[1]) == class_identity)
            and page.latest(row) is not None
        ):
            return book.latest_ref(CLASS_FIELD_DECLARATION, row)
        # Program: the octet codec's ``CanonicalHuffmanTable.symbols``.
        # Wrong record: a dependency class is declared under its qualified
        # identity (``('src...huffman', 'src...huffman.CanonicalHuffmanTable',
        # 'symbols')``) while the method's record names it bare
        # (``CanonicalHuffmanTable``), so neither spelling above matched.
        # Rule: a bare record identity names the declaration whose
        # qualified class ends in it, when exactly one class does.
        if (
            "." not in class_identity
            and str(row[1]).endswith("." + class_identity)
            and page.latest(row) is not None
        ):
            qualified.add((str(row[0]), str(row[1]), row))
    if len({(module, owner) for module, owner, _row in qualified}) == 1:
        return book.latest_ref(CLASS_FIELD_DECLARATION, next(iter(qualified))[2])
    return None


def _frame_call_cells(graph: Any, callsite_id: Any) -> tuple[Any, ...]:
    """The cells a call record of ``callsite_id`` in ``graph`` is built from:
    the call's ``call_binding`` cell (caller read scope, call node), else its
    node identity ``canonical_value`` cell."""

    from .concordance_declarations import CALL_BINDING
    from .identity_concordance import current_identity_book

    if graph is None or callsite_id is None:
        return ()
    scope = (getattr(graph, "graph", None) or {}).get("lexical_read_scope")
    if scope is None:
        return ()
    bound = current_identity_book().latest_ref(
        CALL_BINDING, (tuple(scope), int(callsite_id)),
    )
    return _frame_cells(bound, _frame_graph_cell(graph, callsite_id))


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
        Derived, Mode, current_identity_book,
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
    # No unsourced path: a writer that hands no cell is named here, with
    # the reason it passed, instead of being tagged.
    if not cells:
        raise ValueError(
            f"{page.name}: no source cell for {row!r} -- {reason}"
        )
    return book.post(
        page, row, fact, stage=stage, provenance=Derived(cells), mode=mode,
    )


def _frame_mint(
    function: Any, transform: Any, operands: Any = (), *,
    dtype: Any = None, shape: Any = (), stage: Any, physical: Any = None,
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
    # Rule: a mint declares its physical fields (``physical``); the
    # KEEP_PHYSICAL answer of ``_frame_physical_fields`` (nothing recorded to
    # derive from) declares none.
    from .concordance_declarations import KEEP_PHYSICAL

    if physical is KEEP_PHYSICAL:
        physical = None
    return _mint_ssa_id(
        book, scope, transform, cells,
        dtype=dtype, shape=tuple(shape or ()), stage=stage, physical=physical,
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

    def mint(self, *operands: Any, dtype: Any = None, shape: Any = (),
             physical: Any = None) -> int:
        return _frame_mint(
            self.function, self.transform,
            (*operands, self.declaration),
            dtype=dtype, shape=shape, stage=self.stage, physical=physical,
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


def _argument_binding_ref(
    book: Any, callee_symbol: Any, formal_id: Any, callsite_id: Any,
    *, caller_symbol: Any = None,
) -> Any:
    """The exact caller-qualified binding cell, with legacy-row support."""

    from .concordance_declarations import ARGUMENT_BINDING
    from .identity_concordance import Ref

    page = book.pages.get(ARGUMENT_BINDING.name)
    if page is None or callsite_id is None:
        return None
    callee = str(callee_symbol)
    if caller_symbol is not None:
        row = (callee, str(caller_symbol), int(formal_id), callsite_id)
        ref = book.latest_ref(ARGUMENT_BINDING, row)
        if ref is not None:
            return ref
        # A current four-field page with no exact caller row is a missing
        # binding. Do not recover it from an old caller-blind projection.
        if len(page.row_fields) == 4:
            return None
    row = (callee, int(formal_id), callsite_id)
    ref = book.latest_ref(ARGUMENT_BINDING, row)
    if ref is not None:
        return ref
    legacy = (callee, int(formal_id), "binding")
    if (legacy, int(callsite_id)) in page.cells:
        return Ref(ARGUMENT_BINDING, legacy, int(callsite_id))
    return None


def _argument_binding_fact(
    book: Any, callee_symbol: Any, formal_id: Any, callsite_id: Any,
    *, caller_symbol: Any = None,
) -> Any:
    """The exact caller-qualified binding fact, with legacy-row support."""

    ref = _argument_binding_ref(
        book, callee_symbol, formal_id, callsite_id,
        caller_symbol=caller_symbol,
    )
    if ref is None:
        return None
    return book.pages[ref.page.name].cell(ref.row, ref.column)


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
