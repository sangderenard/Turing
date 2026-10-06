"""Resolve an immutable scalar field version for a physical return edge."""

import networkx as nx

from .monotonic_ids import GLOBAL_MONOTONIC_IDS


# ---------------------------------------------------------------------------
# The identity cell of a compiler-minted SSA value (plan 90, section 1).
#
# Every SSA value a record pass mints after reduction is a row on step 5's
# ``ssa_value`` page under the function's control scope -- the scope the
# control builder minted for the lowering and left in
# ``metadata["tensor_shape_concordance_scope"]`` -- posted NOVEL with the
# transform that made it and the cells it was made from.  Readers resolve a
# value's cell through ``ssa_value_identity_cell``; nothing is keyed by an
# instruction attribute (later frame rounds rebuild instructions and drop
# attributes; the row survives).
# ---------------------------------------------------------------------------


def function_scope_of(function):
    """The scope ``function``'s ``ssa_value`` rows are keyed by: the control
    scope its lowering minted, spelled as ``precompile_to_ssa`` spells it,
    else the function's name (a function no control lowering built)."""
    metadata = getattr(function, "metadata", None) or {}
    return str(metadata.get("tensor_shape_concordance_scope") or function.name)


def ssa_value_identity_cell(function, value_id, *, book=None):
    """The latest ``ssa_value`` cell of ``value_id`` in ``function``, or
    None when no pass has posted the value's identity.  ``book``: the book
    to read (a backend, after the compile closed, passes the module's
    attached book); default the active compile's."""
    from .concordance_declarations import SSA_VALUE
    from .identity_concordance import current_identity_book

    if value_id is None:
        return None
    if book is None:
        book = current_identity_book()
    return book.latest_ref(
        SSA_VALUE, (function_scope_of(function), int(value_id)),
    )


def ssa_block_identity_cell(function, label, *, book=None):
    """The latest ``ssa_block`` cell of block ``label`` of ``function``
    (row ``(function_scope_of(function), function.name, label)``), or None
    when no pass posted the block.  The fact is an ``SSABlockKind`` or
    ``Unresolved(SSA_BLOCK_OWNER_UNROUTED)``; either way the cell carries
    its edge back to the control cell that owns the block.  ``book`` as in
    ``ssa_value_identity_cell``."""
    from .concordance_declarations import SSA_BLOCK
    from .identity_concordance import current_identity_book

    if label is None:
        return None
    if book is None:
        book = current_identity_book()
    return book.latest_ref(
        SSA_BLOCK,
        (function_scope_of(function), str(function.name), str(label)),
    )


def identity_cells(function, *items):
    """The distinct cells named by ``items``: a Ref as itself, an int (or an
    object with ``.id``) as its ``ssa_value`` cell; None and values without
    a row are skipped."""
    from .identity_concordance import Ref

    found = []
    for item in items:
        if item is None:
            continue
        if isinstance(item, Ref):
            cell = item
        else:
            value_id = getattr(item, "id", item)
            if isinstance(value_id, bool) or not isinstance(value_id, int):
                continue
            cell = ssa_value_identity_cell(function, value_id)
        if cell is not None and cell not in found:
            found.append(cell)
    return tuple(found)


def post_dominance_rebinding(
    function, page, row, fact, *, original, replacement, blocks=(), stage,
    mode,
):
    """Post one dominance-proved operand substitution on the book.

    DERIVED(the original operand's ``ssa_value`` cell, the replacement's, and
    the ``ssa_block`` cell of every block the dominance proof names).  A
    substitution that can name no cell at all is written through the raw
    primitive, which the book tags ``Unsourced`` so it stays on the worklist
    (``_post_derived_or_raw``); it is never a metadata-only decision.
    """
    from .identity_concordance import current_identity_book
    from .precompile_to_ssa import _post_derived_or_raw

    cells = list(identity_cells(function, original, replacement))
    for label in blocks:
        cell = ssa_block_identity_cell(function, label)
        if cell is not None and cell not in cells:
            cells.append(cell)
    return _post_derived_or_raw(
        current_identity_book(), page, row, fact, tuple(cells),
        stage=stage, mode=mode,
    )


# ---------------------------------------------------------------------------
# The identity of a return SITE (plan 70, section 2).
#
# The reducer posts ``return_site_slot`` / ``return_site_field_state`` /
# ``return_site_container`` rows keyed by the return construct's cell and
# presents them, keyed by the returned expression's source span, through
# read views (``return_slot_values`` and its siblings); the view's span join
# (cell -> span) is the only place the two keys meet once the construct's
# own nodes have left the graph.  A site is that cell -- never the value it
# returns: three ``return m`` sites return one value and are three sites.
# ---------------------------------------------------------------------------


def return_site_span(expression):
    """The span key of the return whose returned expression is
    ``expression`` (``(lineno, col, end_lineno, end_col)``), or None."""
    if expression is None or getattr(expression, "lineno", None) is None:
        return None
    return (
        int(expression.lineno), int(getattr(expression, "col_offset", -1)),
        int(getattr(expression, "end_lineno", -1)),
        int(getattr(expression, "end_col_offset", -1)),
    )


def return_site_cells(metadata):
    """``span -> return-site cell`` for one graph's metadata mapping.

    Read from the reducer's return-site views (their cell -> span join);
    a site whose construct node still carries ``return_site_cell`` is
    found on the node too.  Empty when neither survives (the receipts were
    copied into plain dicts), in which case no site identity is claimed.
    """
    from .identity_concordance import Ref

    found = {}
    for key in ("return_slot_values", "return_record_field_states"):
        view = (metadata or {}).get(key)
        spans = getattr(view, "_spans", None)
        if not isinstance(spans, dict):
            continue
        for cell, span in spans.items():
            if isinstance(cell, Ref) and span is not None:
                found.setdefault(tuple(span), cell)
    return found


def return_site_cell_for(graph, expression):
    """The return-site cell of the return whose returned expression is
    ``expression`` in ``graph`` (a ProcessGraph or its networkx graph)."""
    span = return_site_span(expression)
    if span is None:
        return None
    nx_graph = getattr(graph, "G", graph)
    cell = return_site_cells(getattr(nx_graph, "graph", None)).get(span)
    if cell is not None:
        return cell
    for _node_id, data in nx_graph.nodes(data=True):
        cell = (data.get("attributes") or {}).get("return_site_cell")
        if cell is None:
            continue
        candidate = data.get("expr_obj")
        for item in (candidate, getattr(candidate, "value", None)):
            if return_site_span(item) == span:
                return cell
    return None


def mint_ssa_value(function, transform, operands, *, dtype=None, shape=(), stage):
    """Mint one SSA id for ``function`` through the book: a NOVEL
    ``ssa_value`` row on ``(function scope, NEW)`` with ``transform`` and the
    cells in ``operands`` (several become one ``cell_set`` row; none becomes
    the function root, as the control builder's ``fresh_value`` does)."""
    from .identity_concordance import current_identity_book
    from .precompile_to_ssa import _function_root_cell, _mint_ssa_id

    book = current_identity_book()
    scope = function_scope_of(function)
    cells = identity_cells(function, *operands)
    if not cells:
        cells = (_function_root_cell(book, scope),)
    return _mint_ssa_id(
        book, scope, transform, cells, dtype=dtype, shape=tuple(shape or ()),
        stage=stage,
    )


def record_descriptor_cell(table, record_id):
    """The latest ``record_descriptor`` cell of ``record_id`` in ``table``."""
    from .concordance_declarations import RECORD_DESCRIPTOR
    from .identity_concordance import current_identity_book

    if table is None or record_id is None:
        return None
    return current_identity_book().latest_ref(
        RECORD_DESCRIPTOR, (table.owner, int(record_id)),
    )


def record_member_cell(table, value_id):
    """The latest ``record_member`` cell of ``value_id`` in ``table``."""
    from .concordance_declarations import RECORD_MEMBER
    from .identity_concordance import current_identity_book

    if table is None or value_id is None:
        return None
    return current_identity_book().latest_ref(
        RECORD_MEMBER, (table.owner, int(value_id)),
    )


def post_record_return_layout(function, table, record_id, layout, *, stage):
    """Post ``record_return_layout`` row ``(function scope, record)`` =
    the layout tuple, REVISE, DERIVED(the record's descriptor cell, each
    layout id's identity cell).  A changed layout none of whose members has
    a cell yet cannot name its cause and is recorded
    ``Unsourced(LAYOUT_MEMBER_NOT_YET_DEFINED)``; an unchanged layout posts
    nothing.  ``metadata["record_return_layouts"]`` stays the readers'
    view and is written by the caller as before."""
    from .concordance_declarations import (
        LAYOUT_MEMBER_NOT_YET_DEFINED, RECORD_RETURN_LAYOUT,
    )
    from .identity_concordance import _post_or_unsourced, current_identity_book

    book = current_identity_book()
    row = (function_scope_of(function), int(record_id))
    fact = tuple(map(int, layout))
    stored = book.pages.get(RECORD_RETURN_LAYOUT.name)
    if stored is not None and stored.latest(row) == fact:
        return book.latest_ref(RECORD_RETURN_LAYOUT, row)
    cells = identity_cells(
        function, record_descriptor_cell(table, record_id), *fact,
    )
    return _post_or_unsourced(
        book, RECORD_RETURN_LAYOUT, row, fact, stage, cells,
        LAYOUT_MEMBER_NOT_YET_DEFINED,
    )


def assign_record_descriptor(table, record_id, descriptor, sources, *, stage):
    """``table.records[record_id] = descriptor`` with its cause: the revision
    is posted DERIVED from ``sources`` when the api admits it (a changed or
    different source), else written raw as before (tagged by the latch)."""
    from .identity_concordance import ConcordanceRefusal

    cells = tuple(cell for cell in sources if cell is not None)
    if cells:
        try:
            table.records.assign(
                int(record_id), descriptor, sources=cells, stage=stage,
            )
            return
        except ConcordanceRefusal:
            pass
    table.records[int(record_id)] = descriptor


def normalize_declared_scalar_record_shapes(module):
    """Enforce the physical shape promised by scalar record descriptors.

    Result-type propagation can refine a late call output after its record
    storage was first materialized.  The descriptor is the stronger physical
    ABI evidence: a field declared as scalar occupies one scalar slot even if
    the source value temporarily carries a singleton tensor shape.  Apply the
    invariant at the completed-module seam and retain the prior shape as
    provenance.  Repeated or equal evidence keeps the incumbent scalar form.
    """

    receipts = []
    for symbol, function in module.functions.items():
        table = module.record_tables.get(str(symbol))
        if table is None:
            continue
        values = {
            int(value.id): value for value in function.args
        }
        values.update({
            int(instruction.res.id): instruction.res
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
        })
        fields_by_value = {}
        for record_id, descriptor in table.records.items():
            for field in descriptor.fields:
                if str(getattr(field.storage, "value", field.storage)) != "scalar":
                    continue
                for value_id in field.value_ids:
                    fields_by_value.setdefault(int(value_id), []).append((
                        int(record_id), str(field.name),
                        str(field.storage_identity),
                    ))
        for value_id, owners in fields_by_value.items():
            value = values.get(value_id)
            prior_shape = () if value is None else tuple(value.shape or ())
            if value is None or not prior_shape:
                continue
            value.shape = ()
            value.accounting = {
                **dict(value.accounting or {}),
                "record_scalar_shape_normalization": True,
                "record_scalar_prior_shape": prior_shape,
                "record_scalar_shape_priority": "declared_record_storage",
                "record_scalar_shape_tie_policy": "incumbent",
            }
            receipts.append({
                "function": str(symbol),
                "value_id": int(value_id),
                "prior_shape": prior_shape,
                "record_fields": tuple(owners),
                "priority": "declared_record_storage",
                "tie_policy": "incumbent",
            })
    return tuple(receipts)


def publish_inout_scalar_return_snapshots(module):
    """Give returned snapshots the latest unambiguous in/out write version.

    An authored scalar record field may be both a callee formal and the source
    of a returned value.  Region projection lowering versions the write while
    the resident formal keeps the caller-owned address.  Returning the formal
    directly makes the native ABI treat the result as another in/out alias;
    later mutation of the resident field then also changes what should have
    been a value snapshot.  Replace that return position with the unique
    latest dominating write version.  Ambiguous maxima retain the incumbent.
    """
    from dataclasses import replace
    from .concordance_declarations import RECORD_RETURN_LAYOUT_STAGE

    receipts = []
    returned_layout_updates = {}
    returned_values_by_symbol = {}
    for symbol, function in module.functions.items():
        block_names = tuple(function.blocks)
        if not block_names:
            continue
        entry = "entry" if "entry" in function.blocks else block_names[0]
        cfg = nx.DiGraph()
        cfg.add_nodes_from(block_names)
        for name, block in function.blocks.items():
            cfg.add_edges_from(
                (name, successor) for successor in block.successors
                if successor in function.blocks
            )
        dominators = nx.immediate_dominators(cfg, entry)
        dominators[entry] = entry

        def block_dominates(owner, target):
            current = target
            while current in dominators:
                if current == owner:
                    return True
                parent = dominators[current]
                if parent == current:
                    break
                current = parent
            return False

        locations = {}
        writes = {}
        for block_name, block in function.blocks.items():
            for index, instruction in enumerate(block.instrs):
                if instruction.res is None:
                    continue
                locations[int(instruction.res.id)] = (block_name, index)
                accounting = instruction.res.accounting or {}
                source_id = accounting.get("source_value_id")
                if (
                    accounting.get("ssa_inout_write_version")
                    and source_id is not None
                    and not tuple(instruction.res.shape or ())
                ):
                    writes.setdefault(int(source_id), []).append(
                        instruction.res
                    )
        formals = {
            int(value.id): value for value in function.args
            if not tuple(value.shape or ())
        }

        def dominates(left, right_block, right_index):
            owner, index = locations[int(left.id)]
            return block_dominates(owner, right_block) and (
                owner != right_block or index < right_index
            )

        def later(left, right):
            """Whether left is strictly later than right on every path."""
            left_block, left_index = locations[int(left.id)]
            right_block, right_index = locations[int(right.id)]
            return block_dominates(right_block, left_block) and (
                right_block != left_block or right_index < left_index
            )

        old_and_new_returns = []
        for block_name, block in function.blocks.items():
            for return_index, operation in enumerate(block.instrs):
                if operation.op not in {"Ret", "ret", "Return", "return"}:
                    continue
                old_arguments = tuple(operation.args)
                arguments = list(old_arguments)
                for position, incumbent in enumerate(old_arguments):
                    source_id = int(incumbent.id)
                    if source_id not in formals:
                        continue
                    candidates = [
                        value for value in writes.get(source_id, ())
                        if value.dtype == incumbent.dtype
                        and dominates(value, block_name, return_index)
                    ]
                    maxima = [
                        value for value in candidates
                        if not any(
                            value is not other and later(other, value)
                            for other in candidates
                        )
                    ]
                    if len(maxima) != 1:
                        continue
                    selected = maxima[0]
                    arguments[position] = selected
                    receipts.append({
                        "function": str(symbol),
                        "block": str(block_name),
                        "return_position": int(position),
                        "source_value_id": source_id,
                        "snapshot_value_id": int(selected.id),
                        "priority": "exact_dominating_inout_write",
                        "replaced_priority": "resident_formal",
                        "tie_policy": "incumbent",
                    })
                if arguments != list(old_arguments):
                    operation.args = arguments
                    old_and_new_returns.append((
                        tuple(int(value.id) for value in old_arguments),
                        tuple(int(value.id) for value in arguments),
                    ))
        if not old_and_new_returns:
            continue
        # Full-native source functions have one settled return layout.  If
        # several returns disagree, leave derived record metadata untouched;
        # the ordinary return merge must settle that ambiguity first.
        new_returns = {new for _old, new in old_and_new_returns}
        if len(new_returns) == 1:
            returned_values_by_symbol[str(symbol)] = next(iter(new_returns))
        layouts = []
        table = module.record_tables.get(symbol)
        for record_id, layout in function.metadata.get(
            "record_return_layouts", ()
        ):
            updated = tuple(map(int, layout))
            for old_return, new_return in old_and_new_returns:
                starts = [
                    start for start in range(len(old_return) - len(updated) + 1)
                    if old_return[start:start + len(updated)] == updated
                ]
                if len(starts) == 1:
                    start = starts[0]
                    updated = new_return[start:start + len(updated)]
            layouts.append((int(record_id), updated))
            if table is None or int(record_id) not in table.records:
                continue
            record = table.records[int(record_id)]
            cursor = 0
            fields = []
            for field in record.fields:
                width = len(field.value_ids)
                field_ids = updated[cursor:cursor + width]
                cursor += width
                fields.append(replace(field, value_ids=tuple(field_ids)))
            # The re-sliced descriptor derives from the incumbent descriptor
            # and the snapshot versions that replaced the formals; the
            # returned layout row follows from the new descriptor cell.
            assign_record_descriptor(
                table, int(record_id), replace(record, fields=tuple(fields)),
                identity_cells(
                    function, record_descriptor_cell(table, record_id), *updated,
                ),
                stage=RECORD_RETURN_LAYOUT_STAGE,
            )
            post_record_return_layout(
                function, table, int(record_id), updated,
                stage=RECORD_RETURN_LAYOUT_STAGE,
            )
        if layouts:
            function.metadata["record_return_layouts"] = tuple(layouts)
            returned_layout_updates[str(symbol)] = tuple(layouts)

    for caller in module.functions.values():
        for block in caller.blocks.values():
            for operation in block.instrs:
                if operation.op not in {"Call", "call"}:
                    continue
                callee = str((operation.attributes or {}).get("callee") or "")
                returned = returned_values_by_symbol.get(callee)
                if returned is None:
                    continue
                attributes = operation.attributes or {}
                callee_ids = tuple(map(int, attributes.get(
                    "callee_output_ids", ()
                )))
                if len(callee_ids) == len(returned):
                    attributes["callee_output_ids"] = returned
                contract = tuple(attributes.get("native_result_contract", ()))
                if len(contract) == len(returned):
                    attributes["native_result_contract"] = tuple(
                        (returned[index], *tuple(item)[1:])
                        for index, item in enumerate(contract)
                    )
    if receipts:
        module.metadata["inout_scalar_return_snapshot_receipts"] = tuple(
            receipts
        )
    return len(receipts)


def reconcile_forwarded_record_results(module):
    """Rebind identity-return records after their input frame settles.

    ``forwarded_output_bindings`` is created when a call returns only its
    formal inputs.  Later record-frame reconciliation may replace those call
    inputs with stronger physical residents.  Carry the same formal-to-actual
    mapping into the returned descriptor and dominated consumers; otherwise a
    stale synthetic output survives even though the call has no output port
    capable of defining it.
    """
    from dataclasses import replace

    receipts = []
    for symbol, function in module.functions.items():
        block_names = tuple(function.blocks)
        if not block_names:
            continue
        entry = "entry" if "entry" in function.blocks else block_names[0]
        cfg = nx.DiGraph()
        cfg.add_nodes_from(block_names)
        for name, block in function.blocks.items():
            cfg.add_edges_from(
                (name, successor) for successor in block.successors
                if successor in function.blocks
            )
        immediate = nx.immediate_dominators(cfg, entry)
        immediate[entry] = entry

        def block_dominates(owner, target):
            current = target
            while current in immediate:
                if current == owner:
                    return True
                parent = immediate[current]
                if parent == current:
                    break
                current = parent
            return False

        call_records = {
            int(record.callsite_id): record
            for record in module.call_table.get(str(symbol), ())
        }
        candidates = {}
        table = module.record_tables.get(str(symbol))
        for block_name, block in function.blocks.items():
            for instruction_index, operation in enumerate(block.instrs):
                attributes = operation.attributes or {}
                forwarded = tuple(attributes.get(
                    "forwarded_output_bindings", ()
                ))
                if operation.op not in {"Call", "call"} or not forwarded:
                    continue
                callee_ids = tuple(map(int, attributes.get(
                    "callee_input_ids", ()
                )))
                if len(callee_ids) != len(operation.args):
                    continue
                actual_by_formal = dict(zip(callee_ids, operation.args))
                replacements = {}
                updated_bindings = []
                for callee_id, incumbent_id in forwarded:
                    actual = actual_by_formal.get(int(callee_id))
                    if actual is None:
                        updated_bindings.append((
                            int(callee_id), int(incumbent_id),
                        ))
                        continue
                    updated_bindings.append((int(callee_id), int(actual.id)))
                    if int(actual.id) != int(incumbent_id):
                        replacements.setdefault(int(incumbent_id), actual)
                if not replacements:
                    continue
                attributes["forwarded_output_bindings"] = tuple(
                    updated_bindings
                )
                site = int(attributes.get("plan_callsite_id", -1))
                record = call_records.get(site)
                result_record_ids = {
                    int(caller_id)
                    for _callee_id, caller_id in (
                        () if record is None else record.result_bindings
                    )
                }
                if table is not None:
                    for record_id in result_record_ids:
                        descriptor = table.records.get(record_id)
                        if descriptor is None:
                            continue
                        fields = tuple(
                            replace(field, value_ids=tuple(
                                int(replacements.get(
                                    int(value_id), value_id
                                ).id) if int(value_id) in replacements
                                else int(value_id)
                                for value_id in field.value_ids
                            ))
                            for field in descriptor.fields
                        )
                        if fields != descriptor.fields:
                            table.records[record_id] = replace(
                                descriptor, fields=fields
                            )
                for incumbent_id, actual in replacements.items():
                    candidates.setdefault(int(incumbent_id), []).append((
                        str(block_name), int(instruction_index), actual,
                        int(site), tuple(sorted(result_record_ids)),
                    ))

        def call_precedes(candidate, use_block, use_index):
            owner, index, _actual, _site, _records = candidate
            return block_dominates(owner, use_block) and (
                owner != use_block or index < use_index
            )

        def later(left, right):
            left_block, left_index, *_ = left
            right_block, right_index, *_ = right
            return block_dominates(right_block, left_block) and (
                right_block != left_block or right_index < left_index
            )

        for block_name, block in function.blocks.items():
            for instruction_index, operation in enumerate(block.instrs):
                arguments = list(operation.args)
                for position, incumbent in enumerate(tuple(arguments)):
                    eligible = [
                        candidate
                        for candidate in candidates.get(int(incumbent.id), ())
                        if call_precedes(
                            candidate, str(block_name), int(instruction_index)
                        )
                    ]
                    maxima = [
                        candidate for candidate in eligible
                        if not any(
                            candidate is not other and later(other, candidate)
                            for other in eligible
                        )
                    ]
                    if len(maxima) != 1:
                        continue
                    owner, _index, actual, site, record_ids = maxima[0]
                    arguments[position] = actual
                    receipts.append({
                        "function": str(symbol),
                        "callsite_id": int(site),
                        "call_block": str(owner),
                        "consumer_block": str(block_name),
                        "consumer_operation": str(operation.op),
                        "consumer_position": int(position),
                        "incumbent_value_id": int(incumbent.id),
                        "forwarded_value_id": int(actual.id),
                        "result_record_ids": record_ids,
                        "priority": "exact_settled_forwarded_input",
                        "replaced_priority": "provisional_forwarded_output",
                        "tie_policy": "incumbent",
                    })
                operation.args = arguments
    if receipts:
        module.metadata["forwarded_record_result_receipts"] = tuple(receipts)
    return len(receipts)


def reconcile_conditional_phi_continuations(module):
    """Carry settled conditional values into dominated continuation uses.

    Conditional lowering can schedule a later numerical region before it has
    replaced the region's captured arm value with the Phi that joins that arm.
    Value ids are not sufficient here: independently lowered projections may
    deliberately retain the same source id.  Match the exact SSAValue object,
    select the unique latest dominating conditional Phi, and leave an
    incumbent untouched when candidates are incomparable. An explicitly owned
    loop update is a snapshot of its selected producer, not a stale capture of
    a later conditional version.
    """
    receipts = []
    for symbol, function in module.functions.items():
        block_names = tuple(function.blocks)
        if not block_names:
            continue
        entry = "entry" if "entry" in function.blocks else block_names[0]
        cfg = nx.DiGraph()
        cfg.add_nodes_from(block_names)
        for block_name, block in function.blocks.items():
            cfg.add_edges_from(
                (block_name, successor)
                for successor in block.successors
                if successor in function.blocks
            )
        reachable = set(nx.descendants(cfg, entry)) | {entry}
        if not reachable:
            continue
        immediate = nx.immediate_dominators(cfg.subgraph(reachable), entry)
        immediate[entry] = entry

        def dominates(owner, target):
            if owner not in immediate or target not in immediate:
                return False
            current = target
            while True:
                if current == owner:
                    return True
                parent = immediate[current]
                if parent == current:
                    return False
                current = parent

        # An arm is keyed by Python identity, not by source-derived value id.
        candidates = {}
        definitions = {}
        for block_name, block in function.blocks.items():
            for instruction_index, operation in enumerate(block.instrs):
                if operation.res is not None:
                    definitions.setdefault(int(operation.res.id), []).append((
                        str(block_name), int(instruction_index), operation.res,
                    ))
                attributes = operation.attributes or {}
                if (
                    operation.op != "Phi"
                    or operation.res is None
                    or attributes.get("binding") != "conditional_carried"
                ):
                    continue
                candidate = (
                    str(block_name), int(instruction_index), operation.res,
                )
                for arm in operation.args:
                    if arm is operation.res:
                        continue
                    candidates.setdefault(id(arm), []).append(candidate)

        def candidate_available(candidate, target_block, use_index, phi_edge):
            owner, instruction_index, _result = candidate
            if not dominates(owner, target_block):
                return False
            if owner != target_block or phi_edge:
                return True
            return instruction_index < use_index

        def strictly_later(left, right):
            """Return whether left is strictly later than right."""
            left_block, left_index, _ = left
            right_block, right_index, _ = right
            if left_block == right_block:
                return left_index > right_index
            return dominates(right_block, left_block)

        for block_name, block in function.blocks.items():
            for instruction_index, operation in enumerate(block.instrs):
                arguments = list(operation.args)
                incoming = tuple((operation.attributes or {}).get(
                    "incoming_blocks", ()
                )) if operation.op == "Phi" else ()
                for position, incumbent in enumerate(tuple(arguments)):
                    target_block = (
                        str(incoming[position])
                        if position < len(incoming)
                        else str(block_name)
                    )
                    phi_edge = position < len(incoming)
                    owned_definitions = definitions.get(int(incumbent.id), ())
                    if (
                        phi_edge
                        and (operation.attributes or {}).get("binding")
                        == "loop_carried"
                        and (operation.attributes or {}).get("updated_value_id")
                        == int(incumbent.id)
                        and dominates(str(block_name), target_block)
                        and len(owned_definitions) == 1
                        and owned_definitions[0][2] is incumbent
                        and candidate_available(
                            owned_definitions[0], target_block,
                            int(instruction_index), True,
                        )
                    ):
                        # Dominance proves availability, not assignment to
                        # this logical loop binding. The raw update may also
                        # feed a conditional that controls another recurrence.
                        # Retain the exact producer chosen by local lowering.
                        receipt = {
                            "function": str(symbol),
                            "consumer_block": str(block_name),
                            "consumer_value_id": int(operation.res.id),
                            "consumer_position": int(position),
                            "updated_value_id": int(incumbent.id),
                            "producer_block": owned_definitions[0][0],
                            "priority": "exact_loop_carried_update",
                            "tie_policy": "incumbent",
                        }
                        prior = tuple(function.metadata.get(
                            "retained_loop_update_receipts", (),
                        ))
                        if receipt not in prior:
                            function.metadata["retained_loop_update_receipts"] = (
                                *prior, receipt,
                            )
                        continue
                    current = incumbent
                    chain = []
                    seen = {id(current)}
                    while True:
                        eligible = [
                            candidate
                            for candidate in candidates.get(id(current), ())
                            if candidate_available(
                                candidate,
                                target_block,
                                int(instruction_index),
                                phi_edge,
                            )
                        ]
                        maxima = [
                            candidate for candidate in eligible
                            if not any(
                                candidate is not other
                                and strictly_later(other, candidate)
                                for other in eligible
                            )
                        ]
                        if len(maxima) != 1:
                            break
                        selected = maxima[0]
                        replacement = selected[2]
                        if id(replacement) in seen:
                            break
                        chain.append(selected)
                        seen.add(id(replacement))
                        current = replacement
                    if current is incumbent:
                        continue
                    arguments[position] = current
                    receipts.append({
                        "function": str(symbol),
                        "consumer_block": str(block_name),
                        "consumer_operation": str(operation.op),
                        "consumer_position": int(position),
                        "incumbent_value_id": int(incumbent.id),
                        "continued_value_id": int(current.id),
                        "phi_blocks": tuple(item[0] for item in chain),
                        "priority": "unique_dominating_conditional_phi",
                        "tie_policy": "incumbent",
                    })
                operation.args = arguments
    if receipts:
        module.metadata["conditional_phi_continuation_receipts"] = tuple(
            receipts
        )
    return len(receipts)


def freshen_redefined_ssa_objects(module):
    """Give every repeated definition of one SSAValue object a fresh value.

    Some conditional construction paths reuse the exact carried-value object
    as a later branch projection result.  The two definitions then cannot be
    distinguished even by object-aware consumers.  Keep the first definition
    as the incumbent, clone each later definition, and rebind precisely the
    uses dominated by that later definition (including individual Phi edges).
    """
    from dataclasses import replace
    from .concordance_declarations import FRESHEN, RECORD_RETURN_REPAIR

    receipts = []
    for symbol, function in module.functions.items():
        block_names = tuple(function.blocks)
        if not block_names:
            continue
        entry = "entry" if "entry" in function.blocks else block_names[0]
        cfg = nx.DiGraph()
        cfg.add_nodes_from(block_names)
        for block_name, block in function.blocks.items():
            cfg.add_edges_from(
                (block_name, successor)
                for successor in block.successors
                if successor in function.blocks
            )
        reachable = set(nx.descendants(cfg, entry)) | {entry}
        immediate = nx.immediate_dominators(cfg.subgraph(reachable), entry)
        immediate[entry] = entry

        def dominates(owner, target):
            if owner not in immediate or target not in immediate:
                return False
            current = target
            while True:
                if current == owner:
                    return True
                parent = immediate[current]
                if parent == current:
                    return False
                current = parent

        definitions = []
        for block_name, block in function.blocks.items():
            for instruction_index, operation in enumerate(block.instrs):
                if operation.res is not None:
                    definitions.append((
                        str(block_name), int(instruction_index), operation,
                    ))
        first_by_object = {}
        for owner, definition_index, definition in definitions:
            original = definition.res
            object_key = id(original)
            incumbent = first_by_object.setdefault(
                object_key, (owner, definition_index, definition)
            )
            if incumbent[2] is definition:
                continue
            accounting = dict(original.accounting or {})
            accounting.update({
                "ssa_redefinition_freshened": True,
                "source_value_id": int(original.id),
            })
            # The clone is NOVEL(freshen) from the redefined value's own
            # identity cell (the function root when it has none).
            fresh = replace(
                original,
                id=mint_ssa_value(
                    function, FRESHEN, (original,),
                    dtype=original.dtype, shape=original.shape or (),
                    stage=RECORD_RETURN_REPAIR,
                ),
                accounting=accounting,
            )
            definition.res = fresh

            for use_block, block in function.blocks.items():
                for use_index, operation in enumerate(block.instrs):
                    if operation is definition:
                        continue
                    incoming = tuple((operation.attributes or {}).get(
                        "incoming_blocks", ()
                    )) if operation.op == "Phi" else ()
                    arguments = list(operation.args)
                    for position, argument in enumerate(tuple(arguments)):
                        if argument is not original:
                            continue
                        target = (
                            str(incoming[position])
                            if position < len(incoming)
                            else str(use_block)
                        )
                        phi_edge = position < len(incoming)
                        if not dominates(owner, target):
                            continue
                        if (
                            owner == target
                            and not phi_edge
                            and definition_index >= use_index
                        ):
                            continue
                        arguments[position] = fresh
                    operation.args = arguments
            receipts.append({
                "function": str(symbol),
                "definition_block": str(owner),
                "definition_operation": str(definition.op),
                "incumbent_definition_block": str(incumbent[0]),
                "source_value_id": int(original.id),
                "fresh_value_id": int(fresh.id),
                "priority": "later_definition_requires_unique_identity",
                "tie_policy": "incumbent_first_definition",
            })
    if receipts:
        module.metadata["redefined_ssa_object_receipts"] = tuple(receipts)
    return len(receipts)


def reconcile_nondominating_identity_cast_results(module):
    """Replace escaped branch-local no-op cast results with their call input.

    A planned region may return a schema-normalizing ``Cast`` whose input and
    result already have the same physical type.  If a correlated later guard
    reuses that projected result, ordinary SSA dominance cannot see the source
    correlation.  Prove the returned slot from the callee body, trace the exact
    caller aggregate projection, and substitute the dominating actual input
    only at uses the projection itself cannot dominate.
    """
    receipts = []

    def physical_identity(left, right):
        return (
            str(left.dtype) == str(right.dtype)
            and tuple(left.shape or ()) == tuple(right.shape or ())
            and getattr(left, "device", None) == getattr(right, "device", None)
        )

    # callee -> returned slot -> formal argument position
    identities = {}
    for callee_name, callee in module.functions.items():
        formal_positions = {id(value): index for index, value in enumerate(callee.args)}
        definitions = {}
        returns = []
        for block in callee.blocks.values():
            for operation in block.instrs:
                if operation.res is not None:
                    definitions.setdefault(id(operation.res), []).append(operation)
                if operation.op == "Ret":
                    returns.append(operation)
        if not returns:
            continue
        width = len(returns[0].args)
        if any(len(operation.args) != width for operation in returns):
            continue
        proven = {}
        for slot in range(width):
            positions = []
            for operation in returns:
                returned = operation.args[slot]
                defining = definitions.get(id(returned), ())
                if len(defining) != 1:
                    positions = []
                    break
                cast = defining[0]
                if cast.op.casefold() != "cast" or len(cast.args) != 1:
                    positions = []
                    break
                source = cast.args[0]
                position = formal_positions.get(id(source))
                if position is None or not physical_identity(source, returned):
                    positions = []
                    break
                positions.append(position)
            if positions and len(set(positions)) == 1:
                proven[slot] = positions[0]
        if proven:
            identities[str(callee_name)] = proven

    for symbol, function in module.functions.items():
        block_names = tuple(function.blocks)
        if not block_names:
            continue
        entry = "entry" if "entry" in function.blocks else block_names[0]
        cfg = nx.DiGraph()
        cfg.add_nodes_from(block_names)
        for block_name, block in function.blocks.items():
            cfg.add_edges_from(
                (block_name, successor)
                for successor in block.successors
                if successor in function.blocks
            )
        reachable = set(nx.descendants(cfg, entry)) | {entry}
        immediate = nx.immediate_dominators(cfg.subgraph(reachable), entry)
        immediate[entry] = entry

        def block_dominates(owner, target):
            if owner not in immediate or target not in immediate:
                return False
            current = target
            while True:
                if current == owner:
                    return True
                parent = immediate[current]
                if parent == current:
                    return False
                current = parent

        definitions = {}
        for block_name, block in function.blocks.items():
            for instruction_index, operation in enumerate(block.instrs):
                if operation.res is not None:
                    definitions.setdefault(id(operation.res), []).append((
                        str(block_name), int(instruction_index), operation,
                    ))
        formals = {id(value) for value in function.args}

        def value_dominates(value, target_block, use_index, phi_edge):
            if id(value) in formals:
                return target_block in reachable
            owners = definitions.get(id(value), ())
            if len(owners) != 1:
                return False
            owner, definition_index, _operation = owners[0]
            if not block_dominates(owner, target_block):
                return False
            if owner != target_block or phi_edge:
                return True
            return definition_index < use_index

        projections = {}
        for call_block, block in function.blocks.items():
            for call_index, call in enumerate(block.instrs):
                if call.op not in {"Call", "call"} or call.res is None:
                    continue
                attributes = call.attributes or {}
                callee_name = str(attributes.get("callee") or "")
                slots = identities.get(callee_name)
                output_ids = tuple(map(int, attributes.get("output_ids", ())))
                if not slots or not output_ids:
                    continue
                for slot, formal_position in slots.items():
                    if slot >= len(output_ids) or formal_position >= len(call.args):
                        continue
                    output_id = output_ids[slot]
                    actual = call.args[formal_position]
                    for _load_block, _load_index, load in (
                        item
                        for items in definitions.values()
                        for item in items
                    ):
                        load_attributes = load.attributes or {}
                        if (
                            load.op != "Load"
                            or load.res is None
                            or int(load_attributes.get("source_output_id", -1))
                            != output_id
                            or len(load.args) != 1
                        ):
                            continue
                        pointer_definitions = definitions.get(id(load.args[0]), ())
                        if len(pointer_definitions) != 1:
                            continue
                        pointer = pointer_definitions[0][2]
                        if (
                            pointer.op != "GetElementPtr"
                            or not pointer.args
                            or pointer.args[0] is not call.res
                        ):
                            continue
                        projections.setdefault(id(load.res), []).append((
                            load.res, actual, str(call_block), int(call_index),
                            callee_name, int(slot),
                        ))

        for block_name, block in function.blocks.items():
            for instruction_index, operation in enumerate(block.instrs):
                arguments = list(operation.args)
                incoming = tuple((operation.attributes or {}).get(
                    "incoming_blocks", ()
                )) if operation.op == "Phi" else ()
                for position, incumbent in enumerate(tuple(arguments)):
                    candidates = projections.get(id(incumbent), ())
                    if len(candidates) != 1:
                        continue
                    projected, actual, call_block, _call_index, callee, slot = (
                        candidates[0]
                    )
                    target_block = (
                        str(incoming[position])
                        if position < len(incoming)
                        else str(block_name)
                    )
                    phi_edge = position < len(incoming)
                    if value_dominates(
                        projected, target_block, int(instruction_index), phi_edge
                    ):
                        continue
                    if not value_dominates(
                        actual, target_block, int(instruction_index), phi_edge
                    ):
                        continue
                    arguments[position] = actual
                    receipts.append({
                        "function": str(symbol),
                        "consumer_block": str(block_name),
                        "consumer_operation": str(operation.op),
                        "consumer_position": int(position),
                        "callee": str(callee),
                        "return_slot": int(slot),
                        "incumbent_value_id": int(incumbent.id),
                        "forwarded_value_id": int(actual.id),
                        "call_block": str(call_block),
                        "priority": "exact_physical_identity_cast_result",
                        "tie_policy": "incumbent",
                    })
                operation.args = arguments
    if receipts:
        module.metadata["identity_cast_result_receipts"] = tuple(receipts)
    return len(receipts)


def repair_non_dominating_return_phi_inputs(function):
    """Recompute an exact pure return expression on its physical edge.

    Structured source guards can prove that a branch-local numerical region
    executed whenever a later synthesized return edge is selected. Repository
    SSA deliberately does not rely on that path correlation: every Phi input
    must have an ordinary dominating definition. When the return receipt names
    the same source value as a non-dominating planned-region projection, clone
    only that pure expression slice onto the return edge and select the fresh
    result. Existing dominating inputs are incumbents and are never replaced.

    The accepted clone is a strictly stronger placement (edge-local versus
    path-correlated), and the resulting operand dominates its edge, so a second
    pass makes no change.
    """
    from ..transmogrifier.ssa import Instr, SSAValue

    block_names = tuple(function.blocks)
    if not block_names:
        return ()
    entry = "entry" if "entry" in function.blocks else block_names[0]
    predecessors = {name: set() for name in block_names}
    for name, block in function.blocks.items():
        for successor in block.successors:
            if successor in predecessors:
                predecessors[successor].add(name)
    dominators = {
        name: ({name} if name == entry else set(block_names))
        for name in block_names
    }
    changed = True
    while changed:
        changed = False
        for name in block_names:
            if name == entry:
                continue
            incoming = predecessors[name]
            common = (
                set.intersection(*(dominators[parent] for parent in incoming))
                if incoming else set()
            )
            updated = {name} | common
            if updated != dominators[name]:
                dominators[name] = updated
                changed = True

    definitions = {}
    for block_name, block in function.blocks.items():
        for index, instruction in enumerate(block.instrs):
            if instruction.res is not None:
                value_id = int(instruction.res.id)
                definitions.setdefault(value_id, []).append(
                    (block_name, index, instruction)
                )
    formal_ids = {int(value.id) for value in function.args}

    def definition_dominates(value_id, edge_name, insertion_index):
        value_id = int(value_id)
        if value_id in formal_ids:
            return True
        locations = definitions.get(value_id, ())
        if len(locations) != 1:
            return False
        owner, index, _instruction = locations[0]
        return owner in dominators[edge_name] and (
            owner != edge_name or index < insertion_index
        )

    def cloneable(instruction):
        attributes = instruction.attributes or {}
        operation = str(instruction.op)
        if operation == "Call":
            return (
                attributes.get("region_index") is not None
                and attributes.get("result_convention") == "ssa.aggregate"
            )
        if operation == "GetElementPtr":
            return (
                attributes.get("region_index") is not None
                and attributes.get("source_output_id") is not None
            )
        if operation == "Load":
            return (
                attributes.get("region_index") is not None
                or attributes.get("binding") == "ssa_sequence_length"
            )
        if operation == "Cast":
            return attributes.get("binding") == "ssa_sequence_length"
        return operation in {"Const", "NoneValue"}

    receipts = []
    for exit_block in function.blocks.values():
        for phi in exit_block.instrs:
            attributes = phi.attributes or {}
            if (
                phi.op != "Phi"
                or attributes.get("binding") != "return_merge"
                or phi.res is None
            ):
                continue
            incoming_blocks = tuple(attributes.get("incoming_blocks", ()))
            if len(incoming_blocks) != len(phi.args):
                continue
            slot = attributes.get("return_slot_index")
            if not isinstance(slot, int):
                continue
            for operand_index, (edge_name, operand) in enumerate(zip(
                incoming_blocks, tuple(phi.args)
            )):
                edge = function.blocks.get(str(edge_name))
                if edge is None or not edge.instrs:
                    continue
                insertion_index = len(edge.instrs) - 1
                if definition_dominates(
                    int(operand.id), str(edge_name), insertion_index
                ):
                    continue
                source_slots = tuple(
                    (edge.instrs[-1].attributes or {}).get(
                        "return_source_value_ids", ()
                    )
                )
                if (
                    slot < 0
                    or slot >= len(source_slots)
                    or int(source_slots[slot]) != int(operand.id)
                ):
                    continue
                root_definitions = definitions.get(int(operand.id), ())
                if len(root_definitions) != 1:
                    continue
                source_block, _source_index, root = root_definitions[0]
                if not (
                    root.op == "Load"
                    and (root.attributes or {}).get("region_index") is not None
                    and int((root.attributes or {}).get(
                        "source_output_id", -1
                    )) == int(operand.id)
                ):
                    continue

                planned = []
                memo = {}
                failed = False

                def materialize(value):
                    nonlocal failed
                    value_id = int(value.id)
                    if definition_dominates(
                        value_id, str(edge_name), insertion_index
                    ):
                        return value
                    if value_id in memo:
                        return memo[value_id]
                    locations = definitions.get(value_id, ())
                    if len(locations) != 1:
                        failed = True
                        return value
                    _owner, _index, producer = locations[0]
                    if not cloneable(producer):
                        failed = True
                        return value
                    cloned_arguments = [
                        materialize(argument) for argument in producer.args
                    ]
                    if failed or producer.res is None:
                        failed = True
                        return value
                    cloned_result = SSAValue(
                        GLOBAL_MONOTONIC_IDS.mint(),
                        dtype=producer.res.dtype,
                        shape=producer.res.shape,
                        device=producer.res.device,
                        accounting={
                            **dict(producer.res.accounting or {}),
                            "return_edge_recomputed_from": int(
                                producer.res.id
                            ),
                            "return_edge_recomputation": True,
                        },
                    )
                    cloned_attributes = dict(producer.attributes or {})
                    cloned_attributes.update({
                        "return_edge_recomputation": True,
                        "return_edge_source_value_id": int(producer.res.id),
                    })
                    cloned = Instr(
                        producer.op,
                        cloned_arguments,
                        cloned_result,
                        arg_roles=list(producer.arg_roles),
                        attributes=cloned_attributes,
                        source_span=producer.source_span,
                    )
                    planned.append(cloned)
                    memo[value_id] = cloned_result
                    return cloned_result

                replacement = materialize(operand)
                if failed or replacement is operand or not planned:
                    continue
                edge.instrs[insertion_index:insertion_index] = planned
                phi.args[operand_index] = replacement
                for offset, instruction in enumerate(planned):
                    definitions[int(instruction.res.id)] = [(
                        str(edge_name), insertion_index + offset, instruction,
                    )]
                receipts.append({
                    "edge": str(edge_name),
                    "return_slot_index": int(slot),
                    "source_value_id": int(operand.id),
                    "physical_value_id": int(replacement.id),
                    "source_block": str(source_block),
                    "operation_count": len(planned),
                    "priority": "edge_local_definition",
                    "replaced_priority": "path_correlated_definition",
                    "tie_policy": "incumbent",
                })
    if receipts:
        function.metadata["return_edge_recomputations"] = tuple((
            *function.metadata.get("return_edge_recomputations", ()),
            *receipts,
        ))
    return tuple(receipts)


def _refuse_undefined_record_phi_fallback(
    function, result_id, fallback_id, block_name, use_index, position, target,
):
    """Post ``Unresolved(RECORD_PHI_FALLBACK_NOT_DEFINED)`` on the occurrence's
    ``record_phi_temporal_fallback_concordance`` row and raise."""
    from .concordance_declarations import (
        RECORD_PHI_FALLBACK_NOT_DEFINED, RECORD_PHI_TEMPORAL_FALLBACK,
        RECORD_RETURN_REPAIR,
    )
    from .identity_concordance import Mode, Unresolved

    read = identity_cells(function, result_id, fallback_id)
    post_dominance_rebinding(
        function, RECORD_PHI_TEMPORAL_FALLBACK,
        (
            str(function.name), int(result_id), str(block_name),
            int(use_index), int(position),
        ),
        Unresolved(RECORD_PHI_FALLBACK_NOT_DEFINED, read=read),
        original=result_id, replacement=fallback_id, blocks=(target,),
        stage=RECORD_RETURN_REPAIR, mode=Mode.CONCORD,
    )
    raise ValueError(
        f"record-field Phi %{result_id} is read at {block_name}[{use_index}] "
        f"operand {position} where its block does not dominate, and its "
        f"recorded initial %{fallback_id} has no definition in "
        f"{function.name!r}: there is no version to read instead"
    )


def repair_non_dominating_record_phi_uses(function):
    """Use a record-field Phi's authored initial field before its merge.

    Fieldwise record lowering records ``initial_value_id`` on every physical
    Phi. Late record/call reconciliation can correlate an earlier use with the
    eventual Phi result, but that result is unavailable before the merge. The
    recorded initial field is the exact predecessor identity; substitute it
    only where ordinary CFG dominance proves it available.
    """
    from .identity_concordance import current_identity_book

    block_names = tuple(function.blocks)
    if not block_names:
        return ()
    entry = "entry" if "entry" in function.blocks else block_names[0]
    cfg = nx.DiGraph()
    cfg.add_nodes_from(block_names)
    for name, block in function.blocks.items():
        cfg.add_edges_from(
            (str(name), str(successor))
            for successor in block.successors
            if successor in function.blocks
        )
    reachable = set(nx.descendants(cfg, entry)) | {entry}
    immediate = nx.immediate_dominators(cfg.subgraph(reachable), entry)
    immediate[entry] = entry

    def block_dominates(owner, target):
        current = str(target)
        while current in immediate:
            if current == str(owner):
                return True
            parent = immediate[current]
            if parent == current:
                break
            current = parent
        return False

    formals = {int(value.id): value for value in function.args}
    definitions = {}
    values = dict(formals)
    fallbacks = {}
    for block_name, block in function.blocks.items():
        for index, instruction in enumerate(block.instrs):
            if instruction.res is None:
                continue
            value_id = int(instruction.res.id)
            definitions.setdefault(value_id, []).append((
                str(block_name), int(index), instruction,
            ))
            values.setdefault(value_id, instruction.res)
            attrs = instruction.attributes or {}
            initial = attrs.get("initial_value_id")
            if (
                instruction.op == "Phi"
                and attrs.get("record_field_phi")
                and initial is not None
            ):
                fallbacks[value_id] = int(initial)

    def dominates(value_id, target, use_index, phi_edge):
        value_id = int(value_id)
        if value_id in formals:
            return True
        locations = definitions.get(value_id, ())
        if len(locations) != 1:
            return False
        owner, index, _instruction = locations[0]
        return block_dominates(owner, target) and (
            owner != target or phi_edge or index < use_index
        )

    from .concordance_declarations import (
        RECORD_PHI_TEMPORAL_FALLBACK, RECORD_RETURN_REPAIR,
    )
    from .identity_concordance import Mode

    receipts = []
    for block_name, block in function.blocks.items():
        for use_index, instruction in enumerate(block.instrs):
            incoming = tuple((instruction.attributes or {}).get(
                "incoming_blocks", ()
            )) if instruction.op == "Phi" else ()
            arguments = list(instruction.args)
            for position, argument in enumerate(tuple(arguments)):
                result_id = int(argument.id)
                fallback_id = fallbacks.get(result_id)
                if fallback_id is None:
                    continue
                fallback = values.get(fallback_id)
                target = (
                    str(incoming[position])
                    if position < len(incoming) else str(block_name)
                )
                phi_edge = position < len(incoming)
                if dominates(result_id, target, int(use_index), phi_edge):
                    continue
                if fallback is None:
                    # The Phi's result is read where it is not available and
                    # its recorded initial names a value this function does
                    # not define: there is nothing to substitute, and leaving
                    # the use reads an undefined value.  That is a refusal
                    # on the occurrence's row, never a skip.
                    _refuse_undefined_record_phi_fallback(
                        function, result_id, fallback_id, str(block_name),
                        int(use_index), int(position), target,
                    )
                if not dominates(fallback_id, target, int(use_index), phi_edge):
                    continue
                row = (
                    str(function.name), result_id, str(block_name),
                    int(use_index), int(position),
                )
                # The dominance evidence: where the fallback is defined (a
                # formal, or block and index), where the Phi result is
                # defined, and the use's target block.
                fallback_site = (
                    "formal" if fallback_id in formals
                    else definitions[fallback_id][0][:2]
                )
                result_sites = tuple(
                    site[:2] for site in definitions.get(result_id, ())
                )
                fact = (
                    fallback_id, target, "initial_record_field_version",
                    fallback_site, result_sites,
                )
                # CONCORD: a changed answer for the same occurrence raises.
                post_dominance_rebinding(
                    function, RECORD_PHI_TEMPORAL_FALLBACK, row, fact,
                    original=result_id, replacement=fallback_id,
                    blocks=(
                        target,
                        *(
                            () if fallback_site == "formal"
                            else (fallback_site[0],)
                        ),
                    ),
                    stage=RECORD_RETURN_REPAIR, mode=Mode.CONCORD,
                )
                arguments[position] = fallback
                receipts.append((result_id, fallback_id, block_name, use_index, position))
            instruction.args = arguments
    return tuple(receipts)


def publish_scalar_record_return_fields(module):
    """Publish against the module's own book, including after serialization.

    A pickled module carries its book on ``module.metadata``; replay must
    read the receipts' rows from THAT book, never an unrelated ambient one,
    and restore the caller's ambient book afterward."""
    from .identity_concordance import begin_identity_book, end_identity_book, identity_book

    _book, token = begin_identity_book(identity_book(module))
    try:
        return _publish_scalar_record_return_fields(module)
    finally:
        end_identity_book(token)


def _publish_scalar_record_return_fields(module):
    """Publish checked return versions after call signatures and CFG settle.

    Preserve physical return identities and recover initial storage from the
    record table, so repeated publication is idempotent.
    """
    from ..transmogrifier.ssa import Instr, SSAValue
    from .concordance_declarations import (
        RECORD_RETURN_FIELD_CONVERSION, RECORD_RETURN_FIELD_SELECTION,
        RECORD_RETURN_VERSION,
    )
    from .identity_concordance import current_identity_book

    changes = 0
    for symbol, function in module.functions.items():
        receipts = function.metadata.get('record_return_state_receipts', ())
        table = module.record_tables.get(symbol)
        if not receipts or table is None:
            continue
        graph = nx.DiGraph()
        graph.graph.update(
            return_slot_values={span: slots for span, slots, states in receipts},
            return_record_field_states={span: states for span, slots, states in receipts},
        )
        selection_scope = function.metadata.get('record_return_state_scope')
        if selection_scope is not None:
            graph.graph['lexical_read_scope'] = tuple(selection_scope)
        lookup = scalar_return_field_versions(function, graph, module.functions)

        def selection_cell(phi_cell, field_name, position, predecessor):
            if selection_scope is None or phi_cell is None:
                return None
            return current_identity_book().latest_ref(
                RECORD_RETURN_FIELD_SELECTION, (
                    tuple(selection_scope), phi_cell, str(field_name),
                    int(position), str(predecessor),
                ),
            )
        values = {int(value.id): value for value in function.args}
        for block in function.blocks.values():
            for operation in block.instrs:
                values.update((int(value.id), value) for value in operation.args)
                if operation.res is not None:
                    values[int(operation.res.id)] = operation.res
        conversions = {
            (name, int(op.args[0].id), op.res.dtype): op.res
            for name, block in function.blocks.items() for op in block.instrs
            if op.op == 'Cast' and op.args and op.res is not None
            and (op.attributes or {}).get('record_return_field_conversion')
        }
        for block in function.blocks.values():
            for operation in list(block.instrs):
                attrs = operation.attributes or {}
                if operation.op != 'Phi' or not attrs.get('record_return_scalar'):
                    continue
                receivers = attrs.get('record_return_receivers', ())
                predecessors = attrs.get('incoming_blocks', ())
                slot = attrs.get('return_slot_index')
                if not (len(receivers) == len(predecessors) == len(operation.args)):
                    continue
                arguments = list(operation.args)
                # The selection rows are keyed by the record return-merge
                # Phi this field Phi expands (its ``record_phi``), read
                # from the book, never from an attribute.
                phi_cell = ssa_value_identity_cell(
                    function,
                    attrs.get('record_phi')
                    if attrs.get('record_phi') is not None
                    else (operation.res.accounting or {}).get('record_phi'),
                )
                for index, (receiver, predecessor) in enumerate(zip(receivers, predecessors)):
                    edge = function.blocks.get(predecessor)
                    if edge is None or not edge.instrs:
                        continue
                    slots = (edge.instrs[-1].attributes or {}).get('return_source_value_ids', ())
                    if not isinstance(slot, int) or not 0 <= slot < len(slots):
                        continue
                    source = table.records.get(slots[slot])
                    physical = table.records.get(receiver)
                    if (source is None or physical is None or source.identity != physical.identity
                            or source.fields != physical.fields):
                        continue
                    field = next((item for item in physical.fields
                                  if item.name == attrs.get('record_field')), None)
                    if field is None or len(field.value_ids) != 1:
                        continue
                    fallback = values.get(int(field.value_ids[0]))
                    if fallback is None:
                        continue
                    selected = lookup(slots[slot], field.name, predecessor, fallback,
                                      alias_receivers=(receiver,),
                                      phi_cell=phi_cell, position=index)
                    if selected.dtype != fallback.dtype:
                        key = (predecessor, int(selected.id), fallback.dtype)
                        converted = conversions.get(key)
                        if converted is None:
                            # The Cast is NOVEL(record_return_field_conversion)
                            # from the selection cell that chose its operand
                            # (else the operand's own cell).
                            chosen_by = selection_cell(
                                phi_cell, field.name, index, predecessor,
                            )
                            converted = SSAValue(
                                mint_ssa_value(
                                    function, RECORD_RETURN_FIELD_CONVERSION,
                                    (selected,) if chosen_by is None else (chosen_by,),
                                    dtype=fallback.dtype,
                                    stage=RECORD_RETURN_VERSION,
                                ),
                                dtype=fallback.dtype,
                            )
                            edge.instrs.insert(-1, Instr('Cast', [selected], converted, attributes={
                                'record_return_field_conversion': field.name,
                                'source_field_value_id': int(selected.id),
                            }))
                            conversions[key] = converted
                        selected = converted
                    arguments[index] = selected
                if [int(value.id) for value in arguments] != [int(value.id) for value in operation.args]:
                    operation.args = arguments
                    attrs['initial_value_id'] = int(arguments[0].id)
                    changes += 1
    return changes


def scalar_return_field_versions(function, source_graph, functions=None):
    """Return a conservative lookup for recorded scalar return state.

    Only an unambiguous source receipt for the exact receiver is eligible.
    Missing/contradictory receipts retain the existing field; this is not a
    complete mutable-record lowering. In particular, it does not select
    dictionary handles, infer call aliases, or synthesize loop-header state.
    """
    from .concordance_declarations import (
        CANONICAL_RELABEL,
        INTERVENING_CALL_NOT_READONLY,
        INTERVENING_STORE,
        NO_RETURN_FIELD_RECEIPTS,
        PREDECESSOR_BLOCK_EMPTY,
        PREDECESSOR_NOT_A_RETURN_EDGE,
        RECORD_RETURN_FIELD_SELECTION,
        RECORD_RETURN_VERSION,
        REDUCER_FIELD_STATE,
        RETURN_SITE_FIELD_STATE,
        RETURN_SITE_SLOT,
        SITES_DISAGREE_ON_VERSION,
        SITE_WITHOUT_FIELD_STATE,
        SSA_FIELD_VERSION,
        VERSION_DOES_NOT_DOMINATE_RETURN,
        VERSION_IS_FORMAL_SHAPED_OR_MISTYPED,
        VERSION_NOT_ON_BOOK,
        VERSION_NOT_UNIQUELY_DEFINED,
        FieldState,
        FieldStateKind,
    )
    from .identity_concordance import (
        Derived, Mode, Ref, RowFieldKind, Unresolved, current_identity_book,
    )

    # Every exit of the lookup is a statement on ``record_return_field_selection``
    # at row (function scope, return-merge Phi cell, field, position,
    # predecessor): the selected ``ssa_field_version`` cell on success, else
    # ``Unresolved(reason, read=<the cells that were read>)``.  A row can be
    # keyed only when the caller names the Phi's identity cell and the graph
    # carries its reduction scope; without those the lookup decides as it
    # always did and the book records nothing (plan 70, section 6).
    scope = source_graph.graph.get('lexical_read_scope')
    scope = None if scope is None else tuple(scope)

    def return_site_cell(span):
        """The return construct's site cell for the receipt keyed by ``span``."""
        if span is None:
            return None
        joined = return_site_cells(source_graph.graph).get(tuple(span))
        if joined is not None:
            return joined
        if not hasattr(source_graph, 'nodes'):
            return None
        for _node_id, data in source_graph.nodes(data=True):
            cell = (data.get('attributes') or {}).get('return_site_cell')
            if cell is None:
                continue
            for candidate in (data.get('expr_obj'), getattr(data.get('expr_obj'), 'value', None)):
                if candidate is None or getattr(candidate, 'lineno', None) is None:
                    continue
                candidate_span = (
                    int(candidate.lineno), int(getattr(candidate, 'col_offset', -1)),
                    int(getattr(candidate, 'end_lineno', -1)),
                    int(getattr(candidate, 'end_col_offset', -1)),
                )
                if candidate_span == tuple(span):
                    return cell
        return None

    def site_state_cells(sites, receiver, field):
        """The ``return_site_field_state`` cells for ``(receiver, field)`` at
        each site, in site order; a site with no such row contributes none."""
        if scope is None:
            return ()
        book = current_identity_book()
        cells = []
        for span in sites:
            site = return_site_cell(span)
            if site is None:
                continue
            ref = book.latest_ref(
                RETURN_SITE_FIELD_STATE, (scope, site, int(receiver), str(field)),
            )
            if ref is not None:
                cells.append(ref)
        return tuple(cells)

    def version_cells(state_cells):
        """The ``ssa_field_version`` cell posted for each return-site state's
        field-state cell (the site row's fact), when it exists."""
        if scope is None:
            return ()
        book = current_identity_book()
        cells = []
        for state_cell in state_cells:
            stored = book.pages.get(state_cell.page.name)
            field_state = None if stored is None else stored.latest(state_cell.row)
            if not isinstance(field_state, Ref):
                continue
            ref = book.latest_ref(SSA_FIELD_VERSION, (scope, field_state))
            if ref is None:
                # Control SSA publishes a write's version at the field-state
                # cell stamped on the SetAttr (the reduction's ingestion
                # row); the return site names the canonical re-post of that
                # same state.  The book joins them: the canonical cell is
                # DERIVED from the ingestion cell at the canonical-relabel
                # stage.  Follow that edge, never an id match.
                ref = next((
                    found for source, stage in book.edges_into(field_state)
                    if stage == CANONICAL_RELABEL
                    and isinstance(source, Ref)
                    and source.page == REDUCER_FIELD_STATE
                    for found in (book.latest_ref(SSA_FIELD_VERSION, (scope, source)),)
                    if found is not None
                ), None)
            if ref is not None:
                cells.append(ref)
        return tuple(cells)

    def decide(reason, value, read, *, field, predecessor, phi_cell, position,
               fact=None):
        """Return ``value`` after posting the decision that produced it.

        ``reason`` None is success: ``read`` is (site state cell, version
        cell) and the fact is that version cell -- or, for a site that
        records no write of the field, ``fact`` is the entered version's
        cell and ``read`` is (the site's slot cell, that cell).  Otherwise
        the fact is ``Unresolved(reason, read)`` derived from what was
        read, or ``Unsourced(reason)`` when nothing on the book was read.
        """
        if scope is None or not isinstance(phi_cell, Ref) or position is None:
            return value
        row = (scope, phi_cell, str(field), int(position), str(predecessor))
        read = tuple(cell for cell in read if isinstance(cell, Ref))
        if reason is None:
            if fact is None:
                versions = [cell for cell in read if cell.page == SSA_FIELD_VERSION]
                if len(read) < 2 or not versions:
                    # The lookup found a version through the graph's receipt
                    # view but the book holds no site row or version cell to
                    # derive it from; nothing can be posted as a sourced
                    # selection.
                    return value
                fact = versions[-1]
        else:
            if not read:
                # An exit that read no site or version row (no receipts, an
                # empty predecessor, a predecessor that is not a return
                # edge) still read the Phi whose row it keys: the Phi's
                # identity cell is what was looked at.
                read = (phi_cell,)
            fact = Unresolved(reason, read=read)
        book = current_identity_book()
        stored = book.pages.get(RECORD_RETURN_FIELD_SELECTION.name)
        if stored is not None and stored.latest(row) == fact:
            return value
        book.post(
            RECORD_RETURN_FIELD_SELECTION, row, fact,
            stage=RECORD_RETURN_VERSION,
            provenance=Derived(read),
            mode=Mode.REVISE,
        )
        return value

    receipts = source_graph.graph.get('return_record_field_states') or {}
    if not receipts:
        def no_receipts(receiver, field, predecessor, fallback, *,
                        alias_receivers=(), phi_cell=None, position=None):
            return decide(
                NO_RETURN_FIELD_RECEIPTS, fallback, (),
                field=field, predecessor=predecessor,
                phi_cell=phi_cell, position=position,
            )
        return no_receipts
    function.metadata['record_return_state_receipts'] = tuple(
        (span, tuple((source_graph.graph.get('return_slot_values') or {}).get(span, ())), tuple(states))
        for span, states in receipts.items()
    )
    if scope is not None:
        # The reduction scope the receipts' rows are keyed by, so the later
        # publication over the receipt view keys the same selection rows.
        function.metadata['record_return_state_scope'] = scope
    formal_ids = {int(value.id) for value in function.args}
    definitions = {}
    instructions = {}
    for name, block in function.blocks.items():
        for instruction in block.instrs:
            if instruction.res is not None:
                definitions.setdefault(int(instruction.res.id), []).append((name, instruction.res))
                instructions[int(instruction.res.id)] = instruction
    cfg = nx.DiGraph()
    cfg.add_nodes_from(function.blocks)
    for name, block in function.blocks.items():
        if not block.instrs:
            continue
        terminal = block.instrs[-1]
        attrs = terminal.attributes or {}
        targets = ((attrs.get('target'),) if terminal.op == 'Br' else
                   (attrs.get('true_target'), attrs.get('false_target'))
                   if terminal.op == 'CondBr' else ())
        cfg.add_edges_from((name, target) for target in targets if target in function.blocks)
    entry = 'entry' if 'entry' in function.blocks else next(iter(function.blocks), None)
    dominators = nx.immediate_dominators(cfg, entry) if entry is not None else {}
    if entry is not None:
        # The repository's NetworkX adapter omits the root self-entry.
        dominators[entry] = entry

    def argument_readonly(callee_name, position, visiting):
        callee = (functions or {}).get(callee_name)
        key = (callee_name, position)
        if callee is None or key in visiting or position >= len(callee.args):
            return False
        formal = callee.args[position]
        if (formal.accounting or {}).get('program_abi_field_written'):
            return False
        operations = [op for body in callee.blocks.values() for op in body.instrs]
        aliases = {int(formal.id)}
        changed = True
        while changed:
            changed = False
            for op in operations:
                if (op.res is not None and op.op in {'GetElementPtr', 'BitCast', 'Cast', 'Identity'}
                        and any(int(arg.id) in aliases for arg in op.args)
                        and int(op.res.id) not in aliases):
                    aliases.add(int(op.res.id))
                    changed = True
        for op in operations:
            if op.res is not None and int(op.res.id) == int(formal.id):
                return False
            if op.op == 'Store':
                if len(op.args) != 2 or int(op.args[1].id) in aliases:
                    return False
            elif 'store' in op.op.lower() or 'atomic' in op.op.lower():
                if any(int(arg.id) in aliases for arg in op.args):
                    return False
            elif op.op == 'Call':
                target = (op.attributes or {}).get('callee')
                target_function = (functions or {}).get(target)
                for index, arg in enumerate(op.args):
                    if int(arg.id) not in aliases:
                        continue
                    if (target_function is None or len(op.args) != len(target_function.args)
                            or not argument_readonly(target, index, visiting | {key})):
                        return False
        return True

    def boolean_phi_tree(value, visiting):
        if value.shape:
            return False
        if value.dtype == 'bool':
            return True
        value_id = int(value.id)
        instruction = instructions.get(value_id)
        if (value_id in visiting or len(definitions.get(value_id, ())) != 1
                or instruction is None or instruction.op != 'Phi'
                or (instruction.attributes or {}).get('binding') != 'conditional_carried'
                or not instruction.args):
            return False
        return all(boolean_phi_tree(argument, visiting | {value_id})
                   for argument in instruction.args)

    def cell_fact(cell):
        """The fact stored at ``cell`` (its own column, not the row's latest)."""
        stored = current_identity_book().pages.get(cell.page.name)
        if stored is None:
            return None
        return dict(stored.history(cell.row)).get(cell.column)

    def cell_value_id(cell):
        """The VALUE_ID element of a node identity cell's row."""
        if not isinstance(cell, Ref):
            return None
        for declared, item in zip(cell.page.row_fields, cell.row):
            if declared.kind is RowFieldKind.VALUE_ID and isinstance(item, int):
                return int(item)
        return None

    def site_state_value_id(state_cell):
        """The value id a ``return_site_field_state`` cell names: its field
        state's value cell, read from the book; None when the site's state
        is Unresolved."""
        field_state_cell = cell_fact(state_cell)
        if not isinstance(field_state_cell, Ref):
            return None
        field_state = cell_fact(field_state_cell)
        if not isinstance(field_state, FieldState):
            return None
        return cell_value_id(field_state.value)

    def observed_formal(state_cell):
        """The ProgramABI formal an OBSERVED site state names, joined
        through the book: the formal's ``ssa_value`` cell must derive
        (transitively, along posted edges) from the state's value cell --
        the authored read of the incoming field.  Never an id match: the
        state's value cell is a source-graph identity, the formal an SSA
        one.  ``(formal, formal cell, field-state cell)`` or None."""
        field_state_cell = cell_fact(state_cell)
        if not isinstance(field_state_cell, Ref):
            return None
        field_state = cell_fact(field_state_cell)
        if (not isinstance(field_state, FieldState)
                or field_state.kind is not FieldStateKind.OBSERVED):
            return None
        book = current_identity_book()
        for formal in function.args:
            formal_cell = ssa_value_identity_cell(function, int(formal.id))
            if formal_cell is None:
                continue
            frontier, seen = [formal_cell], {formal_cell}
            for _depth in range(8):
                if field_state.value in seen:
                    return formal, formal_cell, field_state_cell
                frontier = [
                    source for cell in frontier
                    for source, _stage in book.edges_into(cell)
                    if isinstance(source, Ref) and source not in seen
                ]
                if not frontier:
                    break
                seen.update(frontier)
            if field_state.value in seen:
                return formal, formal_cell, field_state_cell
        return None

    def derives_from(cell, target, depth=8):
        """Whether ``cell`` reaches ``target`` along posted edges (DERIVED
        sources and NOVEL mint operands), at most ``depth`` steps back."""
        book = current_identity_book()
        frontier, seen = [cell], {cell}
        for _depth in range(depth):
            if target in seen:
                return True
            step = []
            for current in frontier:
                sources = [source for source, _stage in book.edges_into(current)]
                mint = book.mint_of(current)
                if mint is not None:
                    sources.extend(mint[1])
                step.extend(
                    source for source in sources
                    if isinstance(source, Ref) and source not in seen
                )
            if not step:
                break
            frontier = step
            seen.update(step)
        return target in seen

    def publish_joined_version(read, value):
        """The ``ssa_field_version`` cells published for ``value`` at each
        return-site state in ``read`` whose value cell the SSA value's
        identity cell derives from; () unless every state joins."""
        if scope is None:
            return ()
        value_cell = ssa_value_identity_cell(function, int(value.id))
        states = [cell for cell in read
                  if isinstance(cell, Ref) and cell.page == RETURN_SITE_FIELD_STATE]
        if value_cell is None or not states:
            return ()
        joins = []
        for state_cell in states:
            field_state_cell = cell_fact(state_cell)
            if not isinstance(field_state_cell, Ref):
                return ()
            field_state = cell_fact(field_state_cell)
            if (not isinstance(field_state, FieldState)
                    or not isinstance(field_state.value, Ref)
                    or not derives_from(value_cell, field_state.value)):
                return ()
            joins.append(field_state_cell)
        book = current_identity_book()
        return tuple(
            book.post(
                SSA_FIELD_VERSION, (scope, field_state_cell), int(value.id),
                stage=RECORD_RETURN_VERSION,
                provenance=Derived((field_state_cell, value_cell)),
                mode=Mode.CONCORD,
            )
            for field_state_cell in joins
        )

    def entered_version(site, receiver, slots, fallback, exit_with):
        """The version current at a site that records no write of the
        field: the receiver's entered field value -- its ProgramABI formal,
        which is the parameter's descriptor value (``fallback``) only when
        that value is a formal.  The selection derives from the site's slot
        cell carrying the receiver and the formal's identity cell.  A
        descriptor value that is not a formal is another site's write
        folded into the descriptor; nothing is selected from it."""
        book = current_identity_book()
        slot_cell = next((
            book.latest_ref(RETURN_SITE_SLOT, (scope, site, index))
            for index, value in enumerate(slots)
            if value is not None and int(value) == int(receiver)
        ), None)
        read = tuple(cell for cell in (slot_cell,) if cell is not None)
        if int(fallback.id) not in formal_ids:
            return exit_with(SITE_WITHOUT_FIELD_STATE, read)
        formal_cell = ssa_value_identity_cell(function, int(fallback.id))
        if slot_cell is None or formal_cell is None:
            return exit_with(SITE_WITHOUT_FIELD_STATE, read)
        return exit_with(None, (slot_cell, formal_cell), fact=formal_cell)

    def lookup(receiver, field, predecessor, fallback, *, alias_receivers=(),
               phi_cell=None, position=None):
        def exit_with(reason, read=(), *, fact=None):
            return decide(
                reason, fallback, read, field=field, predecessor=predecessor,
                phi_cell=phi_cell, position=position, fact=fact,
            )

        block = function.blocks.get(predecessor)
        if block is None or not block.instrs:
            return exit_with(PREDECESSOR_BLOCK_EMPTY)
        terminal_attributes = block.instrs[-1].attributes or {}
        slots = terminal_attributes.get('return_source_value_ids')
        if slots is None:
            return exit_with(PREDECESSOR_NOT_A_RETURN_EDGE)
        site = terminal_attributes.get('return_site_cell')
        if isinstance(site, Ref) and scope is not None:
            # The edge names its own return SITE (the construct's cell the
            # control builder stamped beside ``return_source_value_ids``):
            # the field state is that site's ``return_site_field_state``
            # row, never another site's that happens to return the same
            # value.
            state_cell = current_identity_book().latest_ref(
                RETURN_SITE_FIELD_STATE, (scope, site, int(receiver), str(field)),
            )
            if state_cell is None:
                # The site records no write of this field: the version
                # current there is the entered one (the scope ladder).
                return entered_version(site, receiver, slots, fallback, exit_with)
            version_id = site_state_value_id(state_cell)
            if version_id is None:
                return exit_with(SITE_WITHOUT_FIELD_STATE, (state_cell,))
            versions = version_cells((state_cell,))
            read = (state_cell, *versions)
            if versions:
                # The site's version IS the SSA value posted on that
                # state's ``ssa_field_version`` row; the field state's own
                # value cell names the source graph's value, which is not
                # an SSA identity.
                posted = cell_fact(versions[-1])
                if isinstance(posted, int):
                    version_id = int(posted)
            candidates = definitions.get(int(version_id), ())
            if not candidates and not versions:
                # The site OBSERVED the incoming field (an authored read, no
                # write on this path): its version is the formal that read
                # became.  Publish that as the state's ``ssa_field_version``
                # DERIVED(field-state cell, formal cell), so the selection
                # is a version cell like every written site's.
                entered = observed_formal(state_cell)
                if entered is not None:
                    formal, formal_cell, field_state_cell = entered
                    current_identity_book().post(
                        SSA_FIELD_VERSION, (scope, field_state_cell), int(formal.id),
                        stage=RECORD_RETURN_VERSION,
                        provenance=Derived((field_state_cell, formal_cell)),
                        mode=Mode.CONCORD,
                    )
                    versions = version_cells((state_cell,))
                    read = (state_cell, *versions)
                    version_id = int(formal.id)
            if not candidates and versions:
                # A published version that no instruction defines is a
                # formal of this function (both are SSA ids of one
                # function): it is defined at entry.
                formal = next((arg for arg in function.args
                               if int(arg.id) == int(version_id)), None)
                if formal is not None:
                    candidates = ((entry, formal),)
        else:
            sites = [span for span, values in
                     (source_graph.graph.get('return_slot_values') or {}).items()
                     if tuple(values) == tuple(slots)]
            # An edge that names no site: equal return slot identities can
            # occur at different authored sites, and missing field state
            # at even one such site is not a proof.
            states = [dict(((int(r), str(f)), int(v)) for r, f, v in receipts.get(span, ()))
                      for span in sites]
            key = (int(receiver), str(field))
            state_cells = site_state_cells(sites, receiver, field)
            if not states or any(key not in state for state in states):
                return exit_with(SITE_WITHOUT_FIELD_STATE, state_cells)
            candidates = {state[key] for state in states}
            if len(candidates) != 1:
                return exit_with(SITES_DISAGREE_ON_VERSION, state_cells)
            read = (*state_cells, *version_cells(state_cells))
            candidates = definitions.get(next(iter(candidates)), ())
        if len(candidates) != 1:
            return exit_with(VERSION_NOT_UNIQUELY_DEFINED, read)
        owner, value = candidates[0]
        definition = instructions.get(int(value.id))
        # A version the book records for this exact state (an
        # ``ssa_field_version`` cell in ``read``) is the published version
        # itself, whatever instruction produced it; the Const / carried-Phi
        # admission only guards a version recovered without one.
        published = any(isinstance(cell, Ref) and cell.page == SSA_FIELD_VERSION
                        for cell in read)
        if definition is None and not published:
            return exit_with(VERSION_NOT_UNIQUELY_DEFINED, read)
        # A version found by id (the receipt view or the field state's value
        # id) with no ``ssa_field_version`` cell is admitted only when the
        # book joins it: the SSA value's identity cell derives, along posted
        # edges, from each site state's value cell.  The join is published
        # as the state's version (DERIVED(field-state cell, value cell)) and
        # the selection derives from it.  No join -- including a graph with
        # no reduction scope, where nothing can be joined or keyed -- is
        # Unresolved(VERSION_NOT_ON_BOOK): a selection with no edge is not a
        # selection.  (408155a7 admitted any Const or conditional-carried
        # Phi here by rule; lane A, 2026-10-03, removed it.)
        joined = ()
        if not published:
            joined = publish_joined_version(read, value)
            if not joined:
                return exit_with(VERSION_NOT_ON_BOOK, read)
            read = (*read, *joined)
        if published or joined:
            # The published version begins at its authored assignment effect
            # (the Store stamped with the version row's field-state cell),
            # not where its right-hand side was evaluated: a Const RHS may
            # live at entry, and the write itself is not an intervening one.
            version_state = next(
                cell.row[1] for cell in reversed(read)
                if isinstance(cell, Ref) and cell.page == SSA_FIELD_VERSION
            )
            effects = [
                (name, operation)
                for name, body in function.blocks.items()
                for operation in body.instrs
                if operation.op == 'Store'
                and (operation.attributes or {}).get('field_state_cell') == version_state
            ]
            if len(effects) == 1:
                owner, definition = effects[0]
        if ((int(value.id) in formal_ids and not published) or value.shape
                or (value.dtype != fallback.dtype and not (
                    fallback.dtype == 'bool' and boolean_phi_tree(value, set())))):
            # What this decision read beyond the site and version rows: the
            # version's SSA value and the descriptor's field value (their
            # shape, dtype and formal-ness).  Record materialization revisits
            # the same row inside one fixed point -- the expansion pass
            # decides before the field Phi exists, the ``record_return_scalar``
            # revisit after -- and those two values are what changes between
            # the passes (orbital step_with_dt_control_used, hard_failure at
            # if_merge.18: a conditional_carried Phi typed float64 against a
            # bool field, then typed bool).  With only (site, version) as
            # sources the later decision was a REVISE without a changed
            # source and the book refused it.
            return exit_with(
                VERSION_IS_FORMAL_SHAPED_OR_MISTYPED,
                (*read, *identity_cells(function, value, fallback)),
            )
        current = predecessor
        while current in dominators:
            if current == owner:
                # A later effect through this field's storage invalidates
                # the receipt. Follow explicit pointer/value aliases
                # conservatively; do not infer that such a call is read-only.
                aliases = {int(fallback.id), int(value.id)}
                # A record receiver is not an alias of each of its fields.
                # Whole-record calls remain barriers, but projections of
                # unrelated fields must not taint the selected scalar slot.
                record_aliases = {int(receiver), *map(int, alias_receivers)}
                changed = True
                while changed:
                    changed = False
                    for operation in instructions.values():
                        if (operation.op in {'GetElementPtr', 'BitCast', 'Cast', 'Identity'}
                                and any(int(arg.id) in aliases for arg in operation.args)
                                and int(operation.res.id) not in aliases):
                            aliases.add(int(operation.res.id))
                            changed = True
                # Re-executing the definition kills the previous iteration's
                # version. Do not treat earlier effects in the next iteration
                # as intervening writes to the version reaching this return.
                tail_cfg = cfg.copy()
                tail_cfg.remove_edges_from(tuple(tail_cfg.in_edges(owner)))
                relevant = (nx.descendants(tail_cfg, owner) | {owner}) & (
                    nx.ancestors(tail_cfg, predecessor) | {predecessor})
                for block_name in relevant:
                    operations = function.blocks[block_name].instrs
                    if block_name == owner and definition is not None:
                        operations = operations[operations.index(definition) + 1:]
                    for operation in operations:
                        writes_slot = (operation.res is not None
                                       and int(operation.res.id) == int(fallback.id))
                        touches_alias = any(
                            int(arg.id) in aliases | record_aliases or
                            (arg.accounting or {}).get('ssa_storage_alias') in aliases | record_aliases
                            for arg in operation.args
                        )
                        if touches_alias and operation.op == 'Call':
                            target = (operation.attributes or {}).get('callee')
                            target_function = (functions or {}).get(target)
                            if (target_function is None or len(operation.args) != len(target_function.args)
                                    or any(not argument_readonly(target, index, set())
                                           for index, arg in enumerate(operation.args)
                                           if int(arg.id) in aliases | record_aliases or
                                           (arg.accounting or {}).get('ssa_storage_alias') in aliases | record_aliases)):
                                # The call this decision read: its result's
                                # cell, else the aliased argument's.  The
                                # instructions between definition and return
                                # change as record Phis are expanded within
                                # the same fixed point; the row's next
                                # revision must derive from what it read.
                                return exit_with(INTERVENING_CALL_NOT_READONLY, (
                                    *read,
                                    *identity_cells(
                                        function,
                                        operation.res if operation.res is not None
                                        else next((
                                            arg for arg in operation.args
                                            if int(arg.id) in aliases | record_aliases
                                            or (arg.accounting or {}).get('ssa_storage_alias')
                                            in aliases | record_aliases
                                        ), None),
                                    ),
                                ))
                        store_target = (operation.args[1:] if operation.op == 'Store'
                                        and len(operation.args) == 2 else operation.args)
                        if writes_slot or (('store' in operation.op.lower() or 'atomic' in operation.op.lower())
                                           and any(int(arg.id) in aliases | record_aliases for arg in store_target)):
                            # The storing operation this decision read: its
                            # result's cell when it has one, else the aliased
                            # store target's (orbital step_with_dt_control_used,
                            # div_inf at if_merge.12: INTERVENING_STORE on the
                            # expansion pass, INTERVENING_CALL_NOT_READONLY on
                            # the revisit, from the same (site, version) rows).
                            return exit_with(INTERVENING_STORE, (
                                *read,
                                *identity_cells(
                                    function,
                                    operation.res if operation.res is not None
                                    else next((
                                        arg for arg in store_target
                                        if int(arg.id) in aliases | record_aliases
                                    ), None),
                                ),
                            ))
                # Success: the selection IS the version cell, derived from
                # the return-site state that named it and that version cell.
                return decide(
                    None, value, read, field=field, predecessor=predecessor,
                    phi_cell=phi_cell, position=position,
                )
            parent = dominators[current]
            if parent == current:
                break
            current = parent
        return exit_with(VERSION_DOES_NOT_DOMINATE_RETURN, read)

    return lookup
