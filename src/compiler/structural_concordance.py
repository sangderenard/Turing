"""Live structural views and boundary transport on the existing identity book.

The bootstrap's tuple mapping and the tire's returned Metrics field both lost
producer information at a consumer. The descriptors below are derived views
of source layouts, SSA values and registered storage descriptors. They do not
mint parallel identities or infer structure from a name or a scalar dtype.
"""

from .concordance_declarations import (
    BOUNDARY_TRANSPORT, CANONICAL_VALUE, RECORD_DESCRIPTOR,
    SEQUENCE_DESCRIPTOR, SOURCE_AGGREGATE_LAYOUT, SSA_VALUE,
    STRUCTURAL_CONCORDANCE, STRUCTURAL_VALUE, OPTIONAL_VALUE_STRUCTURE, BoundaryTransport,
    SourceAggregateLayout, StructuralValue, RETURN_SITE_SLOT, RETURN_OUTPUT_ARMS, CALL_EDGE,
    AGGREGATE_RETURN_LAYOUT, RECORD_SCALAR_SHAPE, SSAValueFact,
    VALUE_TYPE_ABI, CELL_SET, SSA_VALUE_PHYSICAL_FIELDS, INGESTION_VALUE,
)
from .identity_concordance import ConcordanceRefusal, Ref, current_identity_book


def _source_structure(read, scope, value_id):
    active = set()

    def visit(value_id, identity=None):
        if value_id in active:
            raise ConcordanceRefusal(f"recursive by-value source layout: {scope!r}/{value_id}")
        active.add(value_id)
        try:
            row = (scope, int(value_id))
            layout = read(SOURCE_AGGREGATE_LAYOUT, row)
            if isinstance(layout, SourceAggregateLayout):
                # The dependency reader exposes exact cells as well as facts.
                source = read.ref(SOURCE_AGGREGATE_LAYOUT, row)
                return StructuralValue(
                    "product", int(value_id), source,
                    tuple((index, visit(member.value_id, member.value_cell))
                          for index, member in enumerate(layout.members)),
                    storage=layout.aggregate_kind,
                )
            if identity is None:
                read(CANONICAL_VALUE, row)
                identity = read.ref(CANONICAL_VALUE, row)
            else:
                read(identity.page, identity.row)
                identity = read.ref(identity.page, identity.row)
            if identity is None:
                raise ConcordanceRefusal(f"source structure lacks identity: {row!r}")
            return StructuralValue("unresolved", int(value_id), identity)
        finally:
            active.remove(value_id)

    return visit(int(value_id))


def source_structure(scope, value_id, *, book=None):
    book = current_identity_book() if book is None else book
    row = ((SOURCE_AGGREGATE_LAYOUT.name, scope), int(value_id))
    cell = book.derive_current(
        STRUCTURAL_VALUE, row, _source_structure, (scope, int(value_id)),
        stage=STRUCTURAL_CONCORDANCE,
    )
    return book.page(STRUCTURAL_VALUE).cell(cell.row, cell.column), cell


def _return_slot_structure(read, scope, site, slot):
    identity = read(RETURN_SITE_SLOT, (scope, site, slot))
    if not isinstance(identity, Ref) or identity.page != CANONICAL_VALUE:
        raise ConcordanceRefusal(f"return slot lacks canonical identity: {(scope, site, slot)!r}")
    return _source_structure(read, identity.row[0], int(identity.row[1]))


def return_slot_structure(scope, site, slot, *, book=None):
    """Resolve the authored return operand, never its downstream merge output."""
    book = current_identity_book() if book is None else book
    row = ((RETURN_SITE_SLOT.name, scope, site), int(slot))
    cell = book.derive_current(
        STRUCTURAL_VALUE, row, _return_slot_structure, (scope, site, int(slot)),
        stage=STRUCTURAL_CONCORDANCE,
    )
    return book.page(STRUCTURAL_VALUE).cell(cell.row, cell.column), cell


def _ssa_physical_view(read, scope, value_id, fact, shape_scope=None, active=()):
    """The current physical view of existing concordance authorities.

    A source annotation, a shape state and a semantic mint are views of one
    identity. None is a fallback selected by a downstream callee's needs.
    """
    from .identity_concordance import SHAPE_STATE_PAGE, MINT_PAGE, shape_transformation_state
    from ..transmogrifier.dtype_layout import resolve

    fields = {}

    def merge(proposed):
        for name, value in proposed.items():
            if value is None or value is False:
                continue
            if name in fields and fields[name] != value:
                raise ConcordanceRefusal(
                    f"physical concordance disagreement for {(scope, value_id)!r}/{name}: "
                    f"{fields[name]!r} versus {value!r}"
                )
            fields[name] = value

    merge({name: getattr(fact, name) for name in SSA_VALUE_PHYSICAL_FIELDS})
    key = (scope, int(value_id))
    if key in active:
        return fields
    active = (*active, key)
    cell = read.ref(SSA_VALUE, key)
    pending = [cell]
    seen = set()
    origins = []
    while pending:
        source = pending.pop()
        if source is None or source in seen:
            continue
        seen.add(source)
        if source.page in (CANONICAL_VALUE, INGESTION_VALUE):
            if source.row[1] == int(value_id):
                abi = read(VALUE_TYPE_ABI, source.row)
                arms = tuple(getattr(abi, "physical_alternatives", ()))
                views = {(arm.storage, arm.dtype, tuple(arm.shape), arm.rank) for arm in arms}
                if len(views) == 1:
                    storage, dtype, shape, rank = next(iter(views))
                    if (storage != "none" and str(dtype) == str(fact.dtype)
                            and tuple(shape) == tuple(fact.shape or ())):
                        merge({"physical_dtype": str(dtype), "storage": storage, "rank": rank})
            continue
        # Revisions retain historical origin edges. Follow those exact refs,
        # not a guessed column zero or an unrelated equal numeric identifier.
        if source.page == SSA_VALUE and source.row == key:
            origins.append(source)
            pending.extend(read.sources(source))
        elif source.page == CELL_SET:
            read(CELL_SET, source.row)
            pending.extend(read.sources(source))
    if shape_scope is not None:
        state = read(SHAPE_STATE_PAGE, (tuple(shape_scope), int(value_id)))
        state = (shape_transformation_state(state[1]) if isinstance(state, tuple)
                 and state and state[0] == "resolved" else None)
        if state is not None:
            shape, dtype, rank, metadata, _row_shape = state
            layout = resolve(dtype)
            if (str(dtype) == str(fact.dtype) and layout is not None
                    and layout.kind in {"bool", "int", "uint", "float"}
                    and str(dtype) != "opaque_ref"
                    and fields.get("storage") in (None, "scalar", "span")):
                if (metadata == "static" and rank == len(shape)
                        and (tuple(shape) == tuple(fact.shape or ()) or not fact.shape)
                        and not (rank == 0 and fields.get("arena"))):
                    merge({"physical_dtype": str(dtype), "storage": "span" if rank else "scalar",
                           "rank": rank})
                elif metadata == "dynamic" and not shape and rank > 0 and not fact.shape:
                    merge({"physical_dtype": str(dtype), "storage": "span", "rank": rank})
    for origin in origins:
        mint = read(MINT_PAGE, (origin.key, int(value_id)))
        if not (isinstance(mint, tuple) and len(mint) == 2):
            continue
        transform, operands = mint
        name = getattr(transform, "name", "")
        if name == "control_const":
            layout = resolve(fact.dtype)
            if (not fact.shape and layout is not None
                    and layout.kind in {"bool", "int", "uint", "float"}
                    and str(fact.dtype) != "opaque_ref"
                    and fields.get("storage") in (None, "scalar")):
                merge({"physical_dtype": str(fact.dtype), "storage": "scalar", "rank": 0})
        elif name in {"phi_conditional", "phi_loop_header", "phi_loop_exit", "phi_return_merge"}:
            pending = list(operands)
            seen = set()
            incoming = []
            while pending:
                operand = pending.pop()
                if operand in seen:
                    continue
                seen.add(operand)
                if operand.page == CELL_SET:
                    read(CELL_SET, operand.row)
                    pending.extend(read.sources(operand))
                elif operand.page == SSA_VALUE:
                    value = read(SSA_VALUE, operand.row)
                    if isinstance(value, SSAValueFact):
                        physical = _ssa_physical_view(
                            read, *operand.row, value, shape_scope=shape_scope, active=active,
                        )
                        incoming.append((value, physical))
                        # The existing Phi law inherits storage/rank, not a
                        # numeric promotion. Before control pruning a rejected
                        # branch can still carry bool beside an integer seed.
                        # Such a mixed Phi has no inferred element contract;
                        # it requires settlement/edge conversion if it lives.
                        merge({field: item for field, item in physical.items()
                               if field != "physical_dtype"})
            if incoming and all(
                value.dtype == fact.dtype and tuple(value.shape) == tuple(fact.shape)
                for value, _physical in incoming
            ):
                for _value, physical in incoming:
                    merge({"physical_dtype": physical.get("physical_dtype")})
            # A typed scalar Phi declares its result slot's representation.
            # This does not admit mixed incoming edges: the emitter still
            # checks each edge's physical layout and requires explicit casts.
            # Storage/rank must already be proved independently above.
            layout = resolve(fact.dtype)
            if (fields.get("storage") == "scalar" and fields.get("rank") == 0
                    and layout is not None and layout.kind in {"bool", "int", "uint", "float"}
                    and str(fact.dtype) != "opaque_ref"):
                merge({"physical_dtype": str(fact.dtype)})
    return fields


def _ssa_structure(read, scope, value_id, shape_scope=None, storage_scope=None):
    row = (scope, int(value_id))
    fact = read(SSA_VALUE, row)
    cell = read.ref(SSA_VALUE, row)
    if cell is None or not isinstance(fact, SSAValueFact):
        raise ConcordanceRefusal(f"physical structure lacks SSA identity: {row!r}")
    physical = _ssa_physical_view(read, scope, value_id, fact, shape_scope)
    dtype = None if fact.dtype in (None, "", "unknown") else str(fact.dtype)
    shape = tuple(fact.shape or ())
    if shape_scope is not None:
        from .identity_concordance import SHAPE_STATE_PAGE, shape_transformation_state

        state = read(SHAPE_STATE_PAGE, (shape_scope, int(value_id)))
        if isinstance(state, tuple) and state and state[0] == "resolved":
            extents, element_type, rank, metadata_state, _row_shape = (
                shape_transformation_state(state[1])
            )
            # A runtime collection's empty static extent is not a scalar.
            # Only a complete static shape supplies the physical extent.
            # Graph extents complete a provisional physical value (the tire
            # result lease), but do not erase a separately published physical
            # reshape/view. In particular a graph scalar is not authority to
            # collapse an existing span to ().
            # Program: ``xs = [5] * (n % 4 + 1)`` (scratch c_k1).  Wrong
            # record: the heap column's ``ptr`` cell, posted shape (), took
            # the planner's (1,) extent here and its reserve's heap copy
            # refused the shaped "handle".  Rule: a declared reference is
            # one address; no graph extent shapes it.
            if (
                not shape and metadata_state == "static"
                and rank == len(extents)
                and physical.get("storage") != "reference"
            ):
                shape = extents
            if dtype is None and element_type not in ("", "unknown"):
                dtype = element_type
    if storage_scope is not None:
        scalar_shape = read(RECORD_SCALAR_SHAPE, (storage_scope, int(value_id)))
        if isinstance(scalar_shape, tuple) and len(scalar_shape) == 2 and scalar_shape[1] == ():
            shape = ()
    # Program: ``identity_spill._reindex_in_place``'s ``and row`` truth
    # test over a resident sequence.  Wrong record: the sequence's column
    # arena (posted ``arena=True``, dynamic extent so shape ``()``) was read
    # as a scalar, and its transport into the declared rank-1 span formal
    # refused.  Rule: an arena is resident rows -- never a scalar -- whatever
    # extent is known for it.
    kind = ("unresolved" if dtype is None or dtype == "ssa.aggregate"
            else "reference" if dtype in {"opaque_ref", "ptr"}
            else "span" if shape or bool(physical.get("arena", False))
            else "scalar")
    # Rule: the view reads the posted row.  A declared storage of scalar/span/
    # reference overrides the derived kind of a resolved leaf; sequence/table/
    # keyed/record are never reinterpreted from ``declared_storage``.
    declared = physical.get("storage")
    if kind != "unresolved" and declared in {"scalar", "span", "reference"}:
        kind = declared
    return StructuralValue(kind, int(value_id), cell, dtype=dtype,
                           shape=shape, storage=kind,
                           physical_dtype=physical.get("physical_dtype"),
                           rank=physical.get("rank"),
                           arena=bool(physical.get("arena", False)),
                           declared_storage=declared)


def _optional_structure(read, scope, value_id, shape_scope=None, storage_scope=None):
    row = (scope, int(value_id))
    optional = read(OPTIONAL_VALUE_STRUCTURE, row)
    if optional is None:
        raise ConcordanceRefusal(f"optional structure has no declaration: {row!r}")
    payload_id, presence_id = optional
    if int(payload_id) != int(value_id) or int(presence_id) == int(value_id):
        raise ConcordanceRefusal(f"invalid optional identities: {row!r}")
    payload = _ssa_structure(read, scope, int(payload_id), shape_scope, storage_scope)
    presence = _ssa_structure(read, scope, int(presence_id), shape_scope, storage_scope)
    if presence.kind != "scalar" or presence.dtype != "bool":
        raise ConcordanceRefusal(f"optional presence is not scalar bool: {row!r}")
    return StructuralValue("optional", int(value_id),
                           read.ref(OPTIONAL_VALUE_STRUCTURE, row),
                           (("payload", payload), ("presence", presence)))


def optional_structure(scope, value_id, *, shape_scope=None, storage_scope=None, book=None):
    """The semantic optional wraps two existing physical values, not one."""
    book = current_identity_book() if book is None else book
    authority = ((OPTIONAL_VALUE_STRUCTURE.name, scope)
                 if shape_scope is None and storage_scope is None else
                 (OPTIONAL_VALUE_STRUCTURE.name, scope, shape_scope, storage_scope))
    row = (authority, int(value_id))
    cell = book.derive_current(
        STRUCTURAL_VALUE, row, _optional_structure,
        ((scope, int(value_id)) if shape_scope is None and storage_scope is None
         else (scope, int(value_id), shape_scope, storage_scope)),
        stage=STRUCTURAL_CONCORDANCE,
    )
    return book.page(STRUCTURAL_VALUE).cell(cell.row, cell.column), cell


def ssa_structure(scope, value_id, *, shape_scope=None, storage_scope=None, book=None):
    book = current_identity_book() if book is None else book
    authority = ((SSA_VALUE.name, scope) if shape_scope is None and storage_scope is None else
                 (SSA_VALUE.name, scope, shape_scope, storage_scope))
    row = (authority, int(value_id))
    cell = book.derive_current(
        STRUCTURAL_VALUE, row, _ssa_structure,
        (scope, int(value_id), shape_scope, storage_scope),
        stage=STRUCTURAL_CONCORDANCE,
    )
    return book.page(STRUCTURAL_VALUE).cell(cell.row, cell.column), cell


def _ssa_view_structure(read, scope, value_id, view_shape, shape_scope=None,
                        storage_scope=None):
    from dataclasses import replace

    storage = _ssa_structure(read, scope, value_id, shape_scope, storage_scope)
    view_shape = tuple(map(int, view_shape))
    return replace(
        storage, shape=view_shape,
        kind="span" if view_shape and storage.kind != "unresolved"
        else storage.kind,
        storage="span" if view_shape and storage.kind != "unresolved"
        else storage.storage,
        declared_storage=None,
    )


def ssa_view_structure(scope, value_id, view_shape, *, shape_scope=None,
                       storage_scope=None, book=None):
    """The structure of a declared view of a storage: its own identity.

    Program: a passthrough call feeds a region formal of (8, 1, 1) with the
    ``ssa_storage_view`` rebinding of storage id 1, whose own row is (8,).
    Wrong record: the feed's producer endpoint was the storage's structural
    row, so the same-shape transport it claimed refused on (8,) vs (8, 1, 1).
    Rule: a declared view is keyed (storage_value_id, view_shape); this row
    is DERIVED from the storage's ``ssa_value`` row (and graph shape state)
    with the declared ``view_shape``; its source cell stays the storage's
    ``ssa_value`` row, which is what the call edge binds."""
    book = current_identity_book() if book is None else book
    view_shape = tuple(map(int, view_shape))
    authority = (SSA_VALUE.name, scope, shape_scope, storage_scope,
                 "ssa_storage_view", view_shape)
    row = (authority, int(value_id))
    cell = book.derive_current(
        STRUCTURAL_VALUE, row, _ssa_view_structure,
        (scope, int(value_id), view_shape, shape_scope, storage_scope),
        stage=STRUCTURAL_CONCORDANCE,
    )
    return book.page(STRUCTURAL_VALUE).cell(cell.row, cell.column), cell


def function_value_structure(function, value_id, *, book=None, view_shape=None):
    """Read a physical value in its owning SSA and graph-copy scopes.

    SSA type publication and graph shape publication are independent facts.
    In particular a newly leased span must not infer scalar storage from an
    SSA object's provisional empty shape.  The derived view tracks both
    revisions, including the absence of a shape row.
    """
    from .identity_concordance import shape_scope_of
    from .ssa_record_return_state import accepted_ssa_component, function_scope_of

    destination = current_identity_book() if book is None else book
    component = (function.metadata["accepted_ssa_component"]
                 if accepted_ssa_component(function) else None)
    owner = destination if component is None else component["identity_book"]
    if view_shape is not None:
        structure, cell = ssa_view_structure(
            function_scope_of(function), int(value_id), view_shape,
            shape_scope=shape_scope_of(function),
            storage_scope=str(function.name), book=owner,
        )
    else:
        structure, cell = ssa_structure(
            function_scope_of(function), int(value_id),
            shape_scope=shape_scope_of(function), storage_scope=str(function.name), book=owner,
        )
    if owner is not destination:
        cell = destination.import_cell(owner, cell, component["artifact_sha256"])
    return structure, cell


def function_optional_structure(function, value_id, *, book=None):
    """Optional payloads use the same owner and physical views as call values."""
    from .identity_concordance import shape_scope_of
    from .ssa_record_return_state import accepted_ssa_component, function_scope_of

    destination = current_identity_book() if book is None else book
    component = (function.metadata["accepted_ssa_component"]
                 if accepted_ssa_component(function) else None)
    owner = destination if component is None else component["identity_book"]
    structure, cell = optional_structure(
        function_scope_of(function), int(value_id),
        shape_scope=shape_scope_of(function), storage_scope=str(function.name), book=owner,
    )
    if owner is not destination:
        cell = destination.import_cell(owner, cell, component["artifact_sha256"])
    return structure, cell


def aggregate_return_layouts(function, *, book=None):
    """The ordered physical return expansions; metadata is not a fallback."""
    from .ssa_record_return_state import accepted_ssa_component, function_scope_of

    if book is None and accepted_ssa_component(function):
        book = function.metadata["accepted_ssa_component"]["identity_book"]
    book = current_identity_book() if book is None else book
    page = book.page(AGGREGATE_RETURN_LAYOUT)
    result = {}
    for row in page.scope_rows(function_scope_of(function)):
        cell = book.current_ref(AGGREGATE_RETURN_LAYOUT, row)
        if cell is not None:
            result[int(row[1])] = tuple(page.cell(row, cell.column))
    return result


def publish_aggregate_return_layout(function, output_id, members, *, sources, stage, book=None):
    """Publish a producer's expansion before exposing its metadata read view.

    The producer supplies the source layout/return-arm/optional cells that
    determine the order.  Member type facts alone cannot justify dropping or
    permuting return positions.
    """
    from .identity_concordance import Derived, Mode
    from .ssa_record_return_state import function_scope_of

    book = current_identity_book() if book is None else book
    members = tuple(map(int, members))
    causes = tuple(dict.fromkeys(cell for cell in sources if cell is not None))
    if not causes:
        raise ConcordanceRefusal("aggregate return expansion lacks its structural source")
    scope = function_scope_of(function)
    cells = []
    for member in members:
        cell = book.current_ref(SSA_VALUE, (scope, member))
        if cell is None:
            raise ConcordanceRefusal(f"aggregate return member has no SSA identity: {(scope, member)!r}")
        cells.append(cell)
    prior = book.current_ref(AGGREGATE_RETURN_LAYOUT, (scope, int(output_id)))
    cell = book.post(
        AGGREGATE_RETURN_LAYOUT, (scope, int(output_id)), members,
        stage=stage, provenance=Derived(tuple(dict.fromkeys(
            (*causes, *cells, *((prior,) if prior is not None else ()))
        ))), mode=Mode.REVISE,
    )
    function.metadata["aggregate_return_layouts"] = tuple(
        aggregate_return_layouts(function, book=book).items()
    )
    return cell


def optional_return_members(function, output_id, *, book=None):
    """Read an interleaved return as payload/presence pairs from the book."""
    from .ssa_record_return_state import accepted_ssa_component, function_scope_of
    from .identity_concordance import shape_scope_of

    if book is None and accepted_ssa_component(function):
        book = function.metadata["accepted_ssa_component"]["identity_book"]
    book = current_identity_book() if book is None else book
    layout = aggregate_return_layouts(function, book=book).get(int(output_id))
    if not layout or len(layout) % 2:
        return ()
    scope = function_scope_of(function)
    result = []
    for payload_id, presence_id in zip(layout[::2], layout[1::2]):
        row = (scope, payload_id)
        cell = book.current_ref(OPTIONAL_VALUE_STRUCTURE, row)
        if cell is None or book.page(OPTIONAL_VALUE_STRUCTURE).cell(row, cell.column) != (
            payload_id, presence_id,
        ):
            return ()
        structure, _cell = optional_structure(
            scope, payload_id, shape_scope=shape_scope_of(function),
            storage_scope=str(function.name), book=book,
        )
        result.append((payload_id, presence_id, dict(structure.children)["payload"]))
    return tuple(result)


def aggregate_return_source(function, output_id, *, book=None):
    """The authored layout reached through this return's publication chain."""
    from .ssa_record_return_state import accepted_ssa_component, function_scope_of

    if book is None and accepted_ssa_component(function):
        book = function.metadata["accepted_ssa_component"]["identity_book"]
    book = current_identity_book() if book is None else book
    cell = book.current_ref(
        AGGREGATE_RETURN_LAYOUT, (function_scope_of(function), int(output_id)),
    )
    frontier = [] if cell is None else [cell]
    visited = set()
    sources = set()
    while frontier:
        cell = frontier.pop()
        if cell in visited:
            continue
        visited.add(cell)
        for source, _stage in book.edges_into(cell):
            if source.page == AGGREGATE_RETURN_LAYOUT:
                frontier.append(source)
            elif source.page in {SOURCE_AGGREGATE_LAYOUT, RETURN_OUTPUT_ARMS}:
                layout_cell = book.current_ref(SOURCE_AGGREGATE_LAYOUT, source.row)
                if layout_cell is not None:
                    sources.add(layout_cell)
    if len(sources) > 1:
        raise ConcordanceRefusal(f"return expansion has competing authored layouts: {sources!r}")
    return next(iter(sources), None)


def _storage_structure(read, authority, owner, value_id, sequence_owner, active=(), column_scope=None):
    key = (authority, owner, value_id)
    if key in active:
        raise ConcordanceRefusal(f"recursive by-value storage descriptor: {key!r}")
    active = (*active, key)
    page = RECORD_DESCRIPTOR if authority == "record" else SEQUENCE_DESCRIPTOR
    row = (owner, int(value_id))
    descriptor = read(page, row)
    cell = read.ref(page, row)
    if descriptor is None:
        raise ConcordanceRefusal(f"withdrawn {authority} descriptor: {row!r}")
    if authority == "record":
        fields = []
        for field in descriptor.fields:
            storage = field.storage.value
            if storage in {"record", "sequence"}:
                nested_id = field.record_id if storage == "record" else field.sequence_id
                nested_owner = owner if storage == "record" else sequence_owner
                if nested_owner is None:
                    raise ConcordanceRefusal("nested sequence requires its registered table owner")
                if storage == "record":
                    # RECORD fields name independently stored records, not
                    # recursively embedded values. LiveScopeProjection's
                    # inherited field is a link whose target is not expanded
                    # in this table. Retain that identity and the dependency
                    # on its absent descriptor; later publication resolves it.
                    nested = read(RECORD_DESCRIPTOR, (nested_owner, int(nested_id)))
                    linked = ()
                    if nested is not None and (storage, nested_owner, nested_id) not in active:
                        linked = (("target", _storage_structure(
                            read, storage, nested_owner, nested_id, sequence_owner, active,
                            column_scope=column_scope,
                        )),)
                    child = StructuralValue(
                        "record_link", int(nested_id), cell, linked,
                        dtype=field.dtype, storage="record", mutable=field.writable,
                    )
                else:
                    child = _storage_structure(read, storage, nested_owner, nested_id, sequence_owner, active,
                                               column_scope=column_scope)
            else:
                members = tuple((index, StructuralValue(
                    "reference" if storage in {"reference", "function_table", "class_table"}
                    else "span" if storage == "span" else "scalar",
                    int(member), cell, dtype=field.dtype, storage=storage,
                    mutable=field.writable,
                )) for index, member in enumerate(field.value_ids))
                child = (members[0][1] if len(members) == 1 else
                         StructuralValue("product", int(value_id), cell, members,
                                         storage=storage, mutable=field.writable))
            fields.append((field.name, child))
        return StructuralValue("record", int(value_id), cell, tuple(fields), storage="record")
    columns = []
    for index, member in enumerate(descriptor.column_value_ids):
        dtype = str(descriptor.column_dtypes[index]) if descriptor.column_dtypes else None
        dtype = None if dtype in (None, "", "unknown", "None") else dtype
        source = cell
        if column_scope is not None:
            # A sequence's column is this exact SSA storage identity. Its
            # descriptor may predate a returned call's settled element type;
            # reading that old unknown as float64 made integer rows floating
            # Loads. Keep the schema and the current column as one live view.
            column = read(SSA_VALUE, (column_scope, int(member)))
            if isinstance(column, SSAValueFact):
                from ..transmogrifier.dtype_layout import resolve
                layout = resolve(column.dtype)
                current = (str(column.dtype) if layout is not None
                           and layout.kind in {"int", "uint", "bool", "float"}
                           else None)
                if dtype is not None and current is not None and dtype != current:
                    raise ConcordanceRefusal(
                        f"sequence column type disagreement for {(owner, value_id, index)!r}: "
                        f"descriptor={dtype!r}, column={current!r}"
                    )
                if current is not None:
                    dtype = current
                    source = read.ref(SSA_VALUE, (column_scope, int(member)))
        row_shape = tuple(descriptor.column_shapes[index]) if descriptor.column_shapes else ()
        columns.append((index, StructuralValue(
            "span", int(member), source, dtype=dtype, shape=row_shape,
            storage="span", mutable=descriptor.writable,
            physical_dtype=dtype, rank=1 + len(row_shape), arena=True,
            declared_storage="span",
        )))
    columns = tuple(columns)
    controls = tuple((role, StructuralValue(
        "unresolved", int(member), cell, storage=storage,
        mutable=descriptor.writable,
    )) for role, member, storage in (
        ("length", descriptor.length_address_id, "scalar"),
        ("capacity", descriptor.capacity_value_id, "scalar"),
        ("status", descriptor.status_address_id, "scalar"),
        ("live_flags", descriptor.live_flags_value_id, "span"),
        ("heap_owner", descriptor.heap_owner_value_id, "reference"),
    ) if member is not None)
    # Native heap members are a separate view of the same descriptor. In
    # the real two-batch caller, child-pool pointer formals lost resident
    # heap_element_dtype flags, although their exact descriptor retained
    # every member and owner. Consumers read this incumbent storage recipe,
    # rather than treating a bare ptr formal as a floating numeric column.
    from .ir_sequence_tables import _storage_values
    from ..transmogrifier.dtype_layout import resolve

    element_dtypes = tuple(descriptor.column_dtypes or ())
    if descriptor.child_table_pool is not None:
        element_dtypes += tuple(descriptor.child_table_pool.column_dtypes)
    known_heap_layout = (
        descriptor.heap_owner_value_id is None
        or (len(descriptor.column_dtypes or ()) == len(descriptor.column_value_ids)
            and (descriptor.child_table_pool is None or
                 len(descriptor.child_table_pool.column_dtypes) == len(descriptor.child_table_pool.column_value_ids))
            and all(resolve(dtype) is not None for dtype in element_dtypes))
    )
    native_members = []
    if known_heap_layout:
        for member in _storage_values(descriptor):
            accounting = dict(member.accounting or {})
            if accounting.get("heap_owner"):
                native = StructuralValue(
                    "reference", int(member.id), cell, dtype="ptr", shape=tuple(member.shape),
                    storage="heap_owner", physical_dtype="ptr", rank=len(member.shape),
                    declared_storage="reference", mutable=descriptor.writable,
                )
            elif accounting.get("heap_element_dtype") is not None:
                owner_id = accounting.get("heap_owner_value_id")
                if owner_id != descriptor.heap_owner_value_id:
                    raise ConcordanceRefusal("native sequence member has a different heap owner")
                element = StructuralValue(
                    "scalar", int(member.id), cell, dtype=str(accounting["heap_element_dtype"]),
                    physical_dtype=str(accounting["heap_element_dtype"]), rank=0,
                    storage="scalar", declared_storage="scalar",
                )
                owner_view = StructuralValue(
                    "reference", int(owner_id), cell, dtype="ptr", shape=(1,),
                    storage="heap_owner", physical_dtype="ptr", rank=1,
                    declared_storage="reference", mutable=descriptor.writable,
                )
                native = StructuralValue(
                    "reference", int(member.id), cell, (("element", element), ("owner", owner_view)),
                    dtype="ptr", shape=tuple(member.shape), storage="heap_cell",
                    physical_dtype="ptr", rank=len(member.shape),
                    declared_storage="reference", mutable=descriptor.writable,
                )
            else:
                layout = resolve(member.dtype)
                native = StructuralValue(
                    "unresolved" if layout is None else "span" if member.shape else "scalar",
                    int(member.id), cell, dtype=None if layout is None else str(member.dtype),
                    shape=tuple(member.shape), storage="span" if member.shape else "scalar",
                    physical_dtype=None if layout is None else str(member.dtype),
                    rank=len(member.shape), mutable=descriptor.writable,
                )
            native_members.append((int(member.id), native))
    native_storage = StructuralValue(
        "product" if known_heap_layout else "unresolved", int(value_id), cell,
        tuple(native_members), storage="native_storage", mutable=descriptor.writable,
    )
    return StructuralValue("sequence", int(value_id), cell,
                           (*columns, *controls, ("native_storage", native_storage)),
                           storage="sequence", mutable=descriptor.writable)


def storage_structure(authority, owner, value_id, *, sequence_owner=None, column_scope=None, book=None):
    if authority not in {"record", "sequence"}:
        raise ConcordanceRefusal(f"unknown descriptor authority {authority!r}")
    book = current_identity_book() if book is None else book
    row = ((authority, owner) if column_scope is None else
           (authority, owner, "ssa_columns", column_scope), int(value_id))
    cell = book.derive_current(
        STRUCTURAL_VALUE, row, _storage_structure,
        ((authority, owner, int(value_id), sequence_owner) if column_scope is None else
         (authority, owner, int(value_id), sequence_owner, (), column_scope)), stage=STRUCTURAL_CONCORDANCE,
    )
    return book.page(STRUCTURAL_VALUE).cell(cell.row, cell.column), cell


def _sequence_row_value(read, storage_row, column, origin):
    """An element Load is a projection, never an implicit conversion."""
    sequence = read(STRUCTURAL_VALUE, storage_row)
    element = dict(sequence.children).get(int(column))
    if element is None:
        raise ConcordanceRefusal(f"sequence row projection lacks column {column!r}")
    dtype = element.dtype
    return SSAValueFact(
        dtype, tuple(element.shape), origin,
        physical_dtype=element.physical_dtype,
        storage="span" if element.shape else "scalar", rank=len(element.shape),
    )


def register_sequence_row_projections(function, table, *, book=None):
    """Register the producer's exact element relation once, at construction."""
    from .ssa_record_return_state import function_scope_of
    book = table.book if book is None else book
    if table.book is not book:
        raise ConcordanceRefusal("sequence projection requires its registered descriptor book")
    scope = function_scope_of(function)
    for block in function.blocks.values():
        for instruction in block.instrs:
            attrs = instruction.attributes or {}
            if instruction.op != "Load" or instruction.res is None or attrs.get("binding") != "ssa_sequence_row_load":
                continue
            sequence_id, column = int(attrs["sequence_id"]), int(attrs["column"])
            _storage, storage = storage_structure(
                "sequence", table.owner, sequence_id, column_scope=scope, book=book,
            )
            row = (scope, int(instruction.res.id))
            fact = book.page(SSA_VALUE).latest(row)
            if not isinstance(fact, SSAValueFact):
                raise ConcordanceRefusal(f"sequence Load lacks SSA identity: {row!r}")
            book.derive_current(
                SSA_VALUE, row, _sequence_row_value,
                (storage.row, column, fact.origin), stage=STRUCTURAL_CONCORDANCE,
            )


def publish_module_structures(module):
    """Read registered tables under their own owners, including nested tables."""
    from .ssa_record_return_state import accepted_ssa_component, function_scope_of
    from .identity_concordance import shape_scope_of

    books = {}
    module_book = module.metadata.get("identity_book")
    for name, function in module.functions.items():
        if accepted_ssa_component(function):
            continue
        if module_book is not None:
            # Every live SSA identity enters the same concordance process.
            # Previously only optional values and storage tables were visited;
            # an ordinary scalar could reach emission without its shared
            # physical view ever being derived or its authorities compared.
            values = {
                int(value.id)
                for value in function.args
            }
            values.update(
                int(instruction.res.id)
                for block in function.blocks.values()
                for instruction in block.instrs
                if instruction.res is not None
            )
            for value_id in sorted(values):
                function_value_structure(function, value_id, book=module_book)
            for row in module_book.page(OPTIONAL_VALUE_STRUCTURE).scope_rows(function_scope_of(function)):
                optional_structure(*row, shape_scope=shape_scope_of(function),
                                   storage_scope=str(function.name), book=module_book)
            books[id(module_book)] = module_book
        sequences = module.sequence_tables.get(name)
        records = module.record_tables.get(name)
        if sequences is not None:
            for value_id in sequences.sequences:
                storage_structure("sequence", sequences.owner, value_id, book=sequences.book)
            books[id(sequences.book)] = sequences.book
        if records is not None:
            for value_id in records.records:
                storage_structure("record", records.owner, value_id,
                                  sequence_owner=None if sequences is None else sequences.owner,
                                  book=records.book)
            books[id(records.book)] = records.book
    for book in books.values():
        book.settle_derivations()


def _transport(read, producer_row, consumer_row, paths, binding_rows):
    producer = read(STRUCTURAL_VALUE, producer_row)
    consumer = read(STRUCTURAL_VALUE, consumer_row)
    if producer is None or consumer is None:
        raise ConcordanceRefusal("boundary transport lacks a structural endpoint")
    produced = dict(producer.leaves())
    consumed = dict(consumer.leaves())
    if (len(paths) != len(produced) or len(paths) != len(consumed)
            or {p for p, _ in paths} != set(produced)
            or {c for _, c in paths} != set(consumed)):
        raise ConcordanceRefusal(
            f"incomplete structural transport: produced={tuple(produced)!r}, "
            f"consumed={tuple(consumed)!r}, pairs={paths!r}"
        )
    members = []
    bindings = []
    for producer_path, consumer_path in paths:
        source, target = produced[producer_path], consumed[consumer_path]
        if source.dtype is None or target.dtype is None:
            raise ConcordanceRefusal(f"unresolved physical transport at {producer_path!r}")
        # Rule: a leaf pair is transportable only if the semantic dtype, the
        # PHYSICAL dtype, the shape and the declared storage (else derived
        # kind) all agree; a physical difference refuses like a dtype one.
        if (source.dtype, source.physical_dtype, source.shape,
                source.declared_storage or source.storage) != (
            target.dtype, target.physical_dtype, target.shape,
            target.declared_storage or target.storage
        ):
            raise ConcordanceRefusal(
                f"physical transport disagreement at {producer_path!r}: "
                f"producer={source!r}, consumer={target!r}"
            )
        binding_row = (target.source.row[0], target.value_id,
                       source.source.row[0], source.value_id)
        if binding_row not in binding_rows or read(CALL_EDGE, binding_row) != "argument":
            raise ConcordanceRefusal(f"transport lacks exact call binding: {binding_row!r}")
        bindings.append(read.ref(CALL_EDGE, binding_row))
        members.append((producer_path, source.source, source.value_id,
                        consumer_path, target.source, target.value_id))
    return BoundaryTransport(
        read.ref(STRUCTURAL_VALUE, producer_row),
        read.ref(STRUCTURAL_VALUE, consumer_row), tuple(members), tuple(bindings),
    )


def boundary_transport(owner, boundary, producer, consumer, paths, *, bindings=(), book=None):
    """Publish a total transport using explicit paths; never zip partial ABIs."""
    if not isinstance(producer, Ref) or not isinstance(consumer, Ref):
        raise ConcordanceRefusal("transport endpoints must be structural cells")
    if producer.page != STRUCTURAL_VALUE or consumer.page != STRUCTURAL_VALUE:
        raise ConcordanceRefusal("transport endpoints must name structural_value")
    if any(not isinstance(cell, Ref) or cell.page != CALL_EDGE for cell in bindings):
        raise ConcordanceRefusal("transport bindings must name existing call edges")
    book = current_identity_book() if book is None else book
    return book.derive_current(
        BOUNDARY_TRANSPORT, (owner, boundary), _transport,
        (producer.row, consumer.row, tuple(paths), tuple(cell.row for cell in bindings)),
        stage=STRUCTURAL_CONCORDANCE,
    )
