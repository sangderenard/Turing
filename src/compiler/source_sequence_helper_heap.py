"""Producer receipts for existing registered-helper heap operations only.

These facts seal the factory's literal/layout/owner choices. They do not
prove a copy's live bounds, source independence, or public ownership escape.
"""
from .concordance_declarations import (
    CONTROL_BLOCK, SSA_BLOCK, SSABlockKind, SSA_VALUE,
    SSA_CONSTANT_MATERIALIZATION, SEQUENCE_HELPER_HEAP_OPERATION,
    SequenceHelperHeapOperation,
)


def _fact(book, cell):
    return book.page(cell.page).cell(cell.row, cell.column)


def _heap_fact(book, function, label, operation, owner, block_cell):
    from .ssa_heap_allocation import validate_heap_allocation, validate_heap_copy
    scope = str(function.name)

    def value_cell(value):
        cell = book.latest_ref(SSA_VALUE, (scope, int(value.id)))
        if cell is None:
            raise ValueError('unregistered helper heap value')
        declared = _fact(book, cell)
        if declared.dtype != value.dtype or tuple(declared.shape) != tuple(value.shape or ()):
            raise ValueError('helper heap value differs from its declaration')
        return cell

    attrs = operation.attributes or {}
    if operation.op == 'Alloca':
        _count, allocation_owner, layout = validate_heap_allocation(operation)
        carriers = ((value_cell(operation.res), value_cell(allocation_owner), layout.name),)
        allocation_kind, element_dtype = attrs['allocation_kind'], attrs['element_dtype']
        zero = attrs.get('zero_initialize', False)
        if not isinstance(zero, bool):
            raise ValueError('heap zero initialization must be an explicit boolean')
        callee = native_abi = None
    elif operation.op == 'Call':
        destination, source, _count, layout = validate_heap_copy(operation)
        values = {int(v.id): v for v in function.args}
        values.update({int(i.res.id): i.res for b in function.blocks.values()
                       for i in b.instrs if i.res is not None})
        carriers = tuple((value_cell(v), value_cell(values[int(v.accounting['heap_owner_value_id'])]),
                          layout.name) for v in (destination, source))
        allocation_kind = element_dtype = zero = None
        callee, native_abi = attrs['callee'], attrs['native_heap_abi']
    else:
        raise ValueError('not an existing heap allocation or element copy')
    return SequenceHelperHeapOperation(
        owner, block_cell, operation.op, tuple(value_cell(v) for v in operation.args),
        None if operation.res is None else value_cell(operation.res),
        allocation_kind, element_dtype, zero, callee, native_abi, carriers,
    )


def post_sequence_helper_semantics(book, function, owner, stage):
    """At the existing registration seam, before helpers are handed to passes.

    Source-less utility lowering keeps its old behavior but gains no receipt.
    No caller can create these facts by merely annotating a final native Call.
    """
    from .identity_concordance import Derived, Mode
    if owner.page != CONTROL_BLOCK or book.latest_ref(owner.page, owner.row) != owner:
        return
    scope = str(function.name)
    for label, block in function.blocks.items():
        block_cell = book.latest_ref(SSA_BLOCK, (scope, function.name, str(label)))
        if (block_cell is None or _fact(book, block_cell) != SSABlockKind.SEQUENCE_HELPER
                or owner not in {p for p, _ in book.edges_into(block_cell)}):
            continue
        for index, operation in enumerate(block.instrs):
            if operation.op == 'Const' and operation.res is not None:
                cell = book.latest_ref(SSA_VALUE, (scope, int(operation.res.id)))
                if cell is not None:
                    # Preserve the factory's actual native width at this
                    # producer, including helper generations replaced before
                    # scalar settlement visits them. This is not Python-int
                    # inference and never widens a declared machine literal.
                    from ..transmogrifier.dtype_layout import resolve
                    from .concordance_declarations import SCALAR_INTEGER_WIDTH, IntegerWidthFact, IntegerWidthDecision
                    layout = resolve(operation.res.dtype)
                    payload = operation.attributes.get('value')
                    if type(payload) is int and layout is not None and layout.kind in {'int', 'uint'}:
                        bits = layout.byte_size * 8
                        lower = -(1 << (bits - 1)) if layout.kind == 'int' else 0
                        upper = (1 << (bits - (1 if layout.kind == 'int' else 0))) - 1
                        if not lower <= payload <= upper:
                            raise ValueError('registered helper literal exceeds its declared native width')
                        book.post(SCALAR_INTEGER_WIDTH, (scope, int(operation.res.id)),
                                  IntegerWidthFact(IntegerWidthDecision.KEPT_DECLARED,
                                                   operation.res.dtype, operation.res.dtype),
                                  stage=stage, provenance=Derived((cell,)), mode=Mode.CONCORD)
                    book.post(SSA_CONSTANT_MATERIALIZATION,
                              (scope, function.name, str(label), int(operation.res.id)),
                              ('Const', str(operation.res.dtype), repr(operation.attributes.get('value'))),
                              stage=stage, provenance=Derived((owner, block_cell, cell)), mode=Mode.CONCORD)
            attrs = operation.attributes or {}
            if not (operation.op == 'Alloca' and attrs.get('allocation_kind') == 'owned_heap'
                    or operation.op == 'Call' and attrs.get('native_heap_abi') == 'copy_elements'):
                continue
            fact = _heap_fact(book, function, label, operation, owner, block_cell)
            causes = (owner, block_cell, *fact.argument_cells,
                      *((fact.result_cell,) if fact.result_cell is not None else ()),
                      *(cell for carrier, allocation_owner, _ in fact.carrier_layouts
                        for cell in (carrier, allocation_owner)))
            book.post(SEQUENCE_HELPER_HEAP_OPERATION,
                      (scope, function.name, str(label), index), fact,
                      stage=stage, provenance=Derived(tuple(dict.fromkeys(causes))), mode=Mode.CONCORD)


def sequence_helper_heap_operation_sources(module, function_name, label, index):
    """Match a live operation to its exact registration receipt, not a grant."""
    try:
        return _validate(module, str(function_name), str(label), int(index))
    except (ValueError, KeyError, TypeError, AttributeError, IndexError):
        return None


def _validate(module, name, label, index):
    book = dict(module.metadata or {}).get('identity_book')
    if book is None:
        return None
    function = module.functions[name]
    cell = book.latest_ref(SEQUENCE_HELPER_HEAP_OPERATION, (name, name, label, index))
    if cell is None:
        return None
    fact = _fact(book, cell)
    if not isinstance(fact, SequenceHelperHeapOperation):
        return None
    if (fact.control_cell.page != CONTROL_BLOCK
            or book.latest_ref(CONTROL_BLOCK, fact.control_cell.row) != fact.control_cell
            or fact.block_cell != book.latest_ref(SSA_BLOCK, (name, name, label))
            or _fact(book, fact.block_cell) != SSABlockKind.SEQUENCE_HELPER
            or fact.control_cell not in {p for p, _ in book.edges_into(fact.block_cell)}):
        return None
    operation = function.blocks[label].instrs[index]
    current = _heap_fact(book, function, label, operation, fact.control_cell, fact.block_cell)
    # The frame binder may publish the caller's exact physical span view on
    # a registered pointer formal. Preserve the original factory receipt;
    # accept only that adjacent, occurrence-proven declaration revision.
    old_refs = (*fact.argument_cells, *((fact.result_cell,) if fact.result_cell else ()),
                *(c for carrier, owner, _ in fact.carrier_layouts for c in (carrier, owner)))
    new_refs = (*current.argument_cells, *((current.result_cell,) if current.result_cell else ()),
                *(c for carrier, owner, _ in current.carrier_layouts for c in (carrier, owner)))
    if len(old_refs) != len(new_refs):
        return None
    rebinding, revision_proof = {}, set()
    for original, latest in zip(old_refs, new_refs):
        if original == latest:
            continue
        owners = {(o, d) for c, o, d in current.carrier_layouts if c == latest}
        if len(owners) != 1:
            return None
        owner, dtype = next(iter(owners))
        proof = _frame_formal_revision(module, name, original, latest, owner, dtype)
        if proof is None or latest in rebinding and rebinding[latest] != original:
            return None
        rebinding[latest] = original
        revision_proof.update(proof)
    from dataclasses import replace
    normalized = replace(current,
        argument_cells=tuple(rebinding.get(c, c) for c in current.argument_cells),
        result_cell=rebinding.get(current.result_cell, current.result_cell),
        carrier_layouts=tuple((rebinding.get(c, c), rebinding.get(o, o), d)
                             for c, o, d in current.carrier_layouts))
    if normalized != fact:
        return None
    expected = {fact.control_cell, fact.block_cell, *fact.argument_cells,
                *(c for carrier, owner, _ in fact.carrier_layouts for c in (carrier, owner))}
    if fact.result_cell is not None:
        expected.add(fact.result_cell)
    if {p for p, _ in book.edges_into(cell)} != expected:
        return None
    return tuple(sorted({cell, *expected, *revision_proof}, key=repr))


def _frame_formal_revision(module, name, original, current, owner, element_dtype):
    from .concordance_declarations import FRAME_BINDING, FORMAL_ACTUAL_OCCURRENCE
    from .ssa_record_return_state import function_scope_of
    book = module.metadata['identity_book']
    if original.page != SSA_VALUE or current.page != SSA_VALUE or original.row != current.row:
        return None
    versions = sorted(c for r, c in book.page(SSA_VALUE).cells if r == current.row)
    if versions[-2:] != [original.column, current.column]:
        return None
    edges = tuple(book.edges_into(current))
    parents = {p for p, _ in edges}
    if original not in parents or len(parents) != 2 or any(s != FRAME_BINDING for _, s in edges):
        return None
    actual, = parents - {original}
    if actual.page != SSA_VALUE or book.latest_ref(SSA_VALUE, actual.row) != actual:
        return None
    before, after, source = (_fact(book, c) for c in (original, current, actual))
    stable = ('dtype', 'shape', 'origin', 'physical_dtype', 'rank', 'arena', 'device')
    if any(getattr(before, k) != getattr(after, k) for k in stable):
        return None
    physical = ('physical_dtype', 'storage', 'rank', 'arena', 'device')
    if any(getattr(after, k) != getattr(source, k) for k in physical):
        return None
    # This is a caller-owned pointer CELL, not a borrowed payload span. The
    # ordinary span-layout reader intentionally cannot invent its capacity.
    roots = _registered_pointer_root_sources(module, name, int(current.row[-1]), int(owner.row[-1]), element_dtype)
    if roots is None:
        return None
    callee = module.functions[name]
    positions = [i for i, v in enumerate(callee.args) if int(v.id) == int(current.row[-1])]
    if len(positions) != 1:
        return None
    position, = positions
    occurrences = []
    for caller_name, caller in module.functions.items():
        for label, block in caller.blocks.items():
            for index, call in enumerate(block.instrs):
                if call.op != 'Call' or call.attributes.get('callee') != name or len(call.args) != len(callee.args):
                    continue
                argument = call.args[position]
                if (function_scope_of(caller), int(argument.id)) != actual.row:
                    continue
                row = (name, int(current.row[-1]), caller_name, str(label), index, position, function_scope_of(caller))
                occurrence = book.latest_ref(FORMAL_ACTUAL_OCCURRENCE, row)
                if (occurrence is not None and _fact(book, occurrence) == int(argument.id)
                        and actual in {p for p, _ in book.edges_into(occurrence)}):
                    occurrences.append(occurrence)
    return (original, current, actual, *roots, *occurrences) if occurrences else None


def _registered_pointer_root_sources(module, name, carrier, owner, element_dtype, seen=frozenset()):
    """Follow both carrier AND owner through actual call occurrences to a root.

    No shape, numeric-ID coincidence or pointer accounting is root authority.
    Every incoming use must agree; recursive/unknown borrowed roots refuse.
    """
    from .concordance_declarations import FORMAL_ACTUAL_OCCURRENCE
    from .source_sequence_pointer_cells import sequence_pointer_cell, sequence_child_pointer_cell
    from .ssa_record_return_state import function_scope_of
    from ..transmogrifier.dtype_layout import resolve
    key = (name, carrier, owner)
    if key in seen:
        return None
    seen = seen | {key}
    book = module.metadata['identity_book']
    function = module.functions[name]
    roles = [[v for v in function.args if int(v.id) == value] for value in (carrier, owner)]
    if any(len(values) != 1 for values in roles):
        return None
    data_value, owner_value = (values[0] for values in roles)
    if (data_value.dtype != 'ptr' or tuple(data_value.shape or ())
            or owner_value.dtype != 'ptr' or tuple(owner_value.shape or ()) != (1,)):
        return None
    data_cell = book.latest_ref(SSA_VALUE, (function_scope_of(function), carrier))
    if data_cell is None or resolve(_fact(book, data_cell).physical_dtype) != resolve(element_dtype):
        return None
    table = getattr(module, 'sequence_tables', {}).get(name)
    if table is not None:
        if table.book is not book:
            return None
        descriptors = [d for d in table.sequences.values()
                       if d.heap_owner_value_id == owner and (carrier in d.column_value_ids
                       or d.child_table_pool is not None and carrier in (
                           *d.child_table_pool.column_value_ids, d.child_table_pool.length_value_id,
                           d.child_table_pool.status_value_id))]
        if len(descriptors) != 1:
            return None
        proof = sequence_pointer_cell(module, name, carrier) or sequence_child_pointer_cell(module, name, carrier)
        if proof is not None:
            return proof.proof_cells
        # Imported helper tables describe the mapped view but are not local
        # allocation roots. Only registered helpers may continue along the
        # exact incoming carrier/owner pair to the original source root.
        if any((c := book.latest_ref(SSA_BLOCK, (function_scope_of(function), name, str(label)))) is None
               or _fact(book, c) != SSABlockKind.SEQUENCE_HELPER for label in function.blocks):
            return None
    positions = [[i for i, v in enumerate(function.args) if int(v.id) == value]
                 for value in (carrier, owner)]
    if any(len(p) != 1 for p in positions):
        return None
    data_position, owner_position = (p[0] for p in positions)
    proof, incoming = set(), 0
    for caller_name, caller in module.functions.items():
        scope = function_scope_of(caller)
        for label, block in caller.blocks.items():
            for index, call in enumerate(block.instrs):
                if call.op != 'Call' or call.attributes.get('callee') != name:
                    continue
                if len(call.args) != len(function.args):
                    return None
                actuals = []
                for position in (data_position, owner_position):
                    value_id = int(call.args[position].id)
                    row = (name, int(function.args[position].id), caller_name, str(label), index, position, scope)
                    occurrence = book.latest_ref(FORMAL_ACTUAL_OCCURRENCE, row)
                    actual = book.latest_ref(SSA_VALUE, (scope, value_id))
                    if (occurrence is None or actual is None or _fact(book, occurrence) != value_id
                            or actual not in {p for p, _ in book.edges_into(occurrence)}):
                        return None
                    proof.update((occurrence, actual))
                    actuals.append(value_id)
                root = _registered_pointer_root_sources(module, caller_name, *actuals, element_dtype, seen)
                if root is None:
                    return None
                proof.update(root)
                incoming += 1
    return tuple(proof) if incoming else None


def registered_helper_integer_materialization(module, row, materialization_cell):
    """Consume the existing KEPT_DECLARED width decision, never infer a width.

    Limited to exact source-linked helper Const receipts. Like the incumbent
    typed-geometry materialization audit, this validates historical producer
    facts too. It is NOT a live instruction/retirement or helper writer proof.
    An unrelated width row or widening cannot override Python semantics.
    """
    import ast
    from .concordance_declarations import SCALAR_INTEGER_WIDTH, IntegerWidthDecision
    from ..transmogrifier.dtype_layout import resolve
    try:
        book = module.metadata['identity_book']
        scope, name, label, value_id = row
        if (materialization_cell.page != SSA_CONSTANT_MATERIALIZATION or materialization_cell.row != row
                or book.latest_ref(SSA_CONSTANT_MATERIALIZATION, row) != materialization_cell):
            return False
        operation, dtype, spelling = _fact(book, materialization_cell)
        literal = ast.literal_eval(spelling)
        if operation != 'Const' or type(literal) is not int or scope != name:
            return False
        layout = resolve(dtype)
        if layout is None or layout.kind not in {'int', 'uint'}:
            return False
        bits = layout.byte_size * 8
        lower = -(1 << (bits - 1)) if layout.kind == 'int' else 0
        upper = (1 << (bits - (1 if layout.kind == 'int' else 0))) - 1
        if not lower <= literal <= upper:
            return False
        width = book.latest_ref(SCALAR_INTEGER_WIDTH, (scope, value_id))
        if width is None:
            return False
        decision = _fact(book, width)
        if (decision.decision is not IntegerWidthDecision.KEPT_DECLARED
                or decision.source_dtype != dtype or decision.dtype != dtype):
            return False
        parents = {p for p, _ in book.edges_into(materialization_cell)}
        block = book.latest_ref(SSA_BLOCK, (scope, name, label))
        controls = {p for p in parents if p.page == CONTROL_BLOCK}
        if (block is None or block not in parents or _fact(book, block) != SSABlockKind.SEQUENCE_HELPER
                or len(controls) != 1 or not controls <= {p for p, _ in book.edges_into(block)}):
            return False
        control, = controls
        if book.latest_ref(CONTROL_BLOCK, control.row) != control:
            return False
        declared = {p for p, _ in book.edges_into(width)}
        if (len(declared) != 1 or not declared <= parents
                or any(p.page != SSA_VALUE or p.row != (scope, value_id) for p in declared)):
            return False
        declaration, = declared
        if parents != {control, block, declaration}:
            return False
        from .source_owned_record_writes import _mint_causes
        from .concordance_declarations import CONTROL_CONST
        minted = _mint_causes(book, declaration, CONTROL_CONST)
        if minted is None or minted[0] != {control}:
            return False
        if _fact(book, declaration).dtype != dtype or tuple(_fact(book, declaration).shape):
            return False
        current = book.latest_ref(SSA_VALUE, (scope, value_id))
        from .source_owned_record_writes import _ssa_history
        if current is None or declaration not in _ssa_history(book, current) or _fact(book, current).dtype != dtype:
            return False
        function = module.functions.get(name)
        if function is None:
            return True  # Historical domain only; no current callable is admitted.
        definitions = [i for b in function.blocks.values() for i in b.instrs
                       if i.res is not None and int(i.res.id) == value_id]
        if not definitions:
            # An orphaned current use is not a retired historical literal.
            return not (any(int(v.id) == value_id for v in function.args)
                        or any(int(v.id) == value_id for b in function.blocks.values()
                               for i in b.instrs for v in i.args))
        return (len(definitions) == 1 and definitions[0].op == 'Const'
                and not definitions[0].args and definitions[0].res.dtype == dtype
                and repr(definitions[0].attributes.get('value')) == spelling)
    except (KeyError, TypeError, ValueError, AttributeError, SyntaxError):
        return False
