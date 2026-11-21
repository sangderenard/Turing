"""Audit exact owned-record public projection before admitting native ownership."""
from dataclasses import replace


def validated_record_owned_return_structure(module, function_name, public_sequence_id):
    from .concordance_declarations import (
        RECORD_OWNED_RETURN_PROJECTION, RecordOwnedReturnProjection,
        SSA_VALUE, SSA_VALUE_PHYSICAL_FIELDS, SSA_BLOCK_GEOMETRY, SEQUENCE_DESCRIPTOR,
        RECORD_DESCRIPTOR, RECORD_RETURN_LAYOUT, CELL_SET, RECORD_OWNED_RETURN_PROJECTION_VALUE,
    )
    from .source_owned_record_returns import OwnedRecordReturnSource, validate_owned_record_return_source
    from .source_owned_record_projection import public_source_entry
    from .source_owned_record_elements import owned_record_primitive_element
    from .source_sequence_selection import local_selection_lifetimes
    from .ssa_record_return_state import function_scope_of, map_sequence_descriptor
    from ..transmogrifier.ssa import SSARecordFieldStorage
    name, public_id = str(function_name), int(public_sequence_id)
    function = module.functions.get(name)
    sequences = getattr(module, 'sequence_tables', {}).get(name)
    records = getattr(module, 'record_tables', {}).get(name)
    book = dict(getattr(module, 'metadata', {}) or {}).get('identity_book')
    if function is None or sequences is None or records is None or book is None:
        return None
    page = book.page(RECORD_OWNED_RETURN_PROJECTION)
    cell = book.latest_ref(RECORD_OWNED_RETURN_PROJECTION, (name, public_id))
    if cell is None:
        return None
    def value(ref):
        return book.page(ref.page).cell(ref.row, ref.column)
    try:
        fact = value(cell)
        if not isinstance(fact, RecordOwnedReturnProjection):
            return None
        # Initial ABI supports one owned field, not overlapping or separately
        # retained projections of the same original allocation.
        if len([row for row in page.rows() if row[0] == name]) != 1:
            return None
        entry = public_source_entry(module, name)
        if entry is None or entry[0] != fact.entry_cell:
            return None
        source = OwnedRecordReturnSource(fact.source_record_id, fact.field_name,
            fact.source_sequence_id, fact.source_return_slot_cell,
            fact.source_constructor_operands_cell, fact.source_cell,
            fact.source_descriptor_cell, fact.source_record_cell, fact.source_owner_id)
        if not validate_owned_record_return_source(module, name, source):
            return None
        element = owned_record_primitive_element(module, name, source)
        if (element is None or (element.field_cell, element.binding_cell, element.element_cell)
                != (fact.element_declaration_cell, fact.element_binding_cell, fact.element_class_cell)):
            return None
        selected = local_selection_lifetimes(module, name, source.sequence_id)
        if selected is None or selected.terminal_cells != fact.terminal_lifetime_cells:
            return None
        original = sequences.sequences[source.sequence_id]
        expected_dtype = {'builtins.int':'int64', 'builtins.float':'float64', 'builtins.bool':'bool'}.get(element.semantic_class)
        if tuple(original.column_dtypes) != (expected_dtype,):
            return None
        pairs = fact.member_pairs
        count = len(selected.selection.member_ids)
        scope = function_scope_of(function)
        if len(pairs) != count or len(fact.snapshot_cells) != count:
            return None
        old_ids = tuple(int(left.row[-1]) for left, _ in pairs)
        new_ids = tuple(int(right.row[-1]) for _, right in pairs)
        snapshots = tuple(int(ref.row[-1]) for ref in fact.snapshot_cells)
        null_id = int(fact.null_cell.row[-1])
        if (old_ids != selected.selection.member_ids or new_ids[0] != public_id
                or len(set(new_ids)) != count or set(old_ids) & set((*new_ids, *snapshots, null_id))
                or len(set((*new_ids, *snapshots, null_id))) != 2*count+1):
            return None
        refs = (*[ref for pair in pairs for ref in pair], *fact.snapshot_cells, fact.null_cell)
        if any(ref.page != SSA_VALUE or ref.row[0] != scope
               or book.latest_ref(SSA_VALUE, ref.row) != ref for ref in refs):
            return None
        def exact_projection_mint(ref, expected):
            minted = book.mint_of(ref)
            if (minted is None or minted[0] != RECORD_OWNED_RETURN_PROJECTION_VALUE
                    or len(minted[1]) != 1):
                return False
            bundle = minted[1][0]
            if bundle.page != CELL_SET:
                return False
            parents = {p for p,_ in book.edges_into(bundle)}
            return parents == set(expected) and set(value(bundle)) == {p.key for p in parents}
        member_sources = (fact.entry_cell, fact.source_return_slot_cell,
            fact.source_constructor_operands_cell, fact.source_cell,
            fact.source_descriptor_cell, fact.source_record_cell, fact.source_return_layout_cell,
            fact.element_declaration_cell, fact.element_binding_cell, fact.element_class_cell,
            *fact.terminal_lifetime_cells, *[left for left,_ in pairs], *fact.writer_cells)
        if (any(not exact_projection_mint(right, member_sources) for _,right in pairs)
                or any(not exact_projection_mint(snapshot, pair)
                       for snapshot,pair in zip(fact.snapshot_cells,pairs))
                or not exact_projection_mint(fact.null_cell, pairs[-1])):
            return None
        remap = dict(zip(old_ids, new_ids))
        descriptor = sequences.sequences.get(public_id)
        if descriptor != map_sequence_descriptor(original, remap, sequence_id=public_id, child_table_pool=None):
            return None
        if (fact.public_descriptor_cell != book.latest_ref(SEQUENCE_DESCRIPTOR, (sequences.owner, public_id))
                or value(fact.public_descriptor_cell) != descriptor):
            return None
        arguments = {int(arg.id):arg for arg in function.args}
        public_ports = (new_ids[0], remap[original.length_address_id],
                        remap[original.capacity_value_id], new_ids[-1])
        if any(port not in arguments for port in public_ports):
            return None
        if (not (arguments[new_ids[-1]].accounting or {}).get('heap_owner')
                or (arguments[public_id].accounting or {}).get('heap_owner_value_id') != new_ids[-1]):
            return None
        for index, (left, right) in enumerate(pairs):
            old, new, snapshot = value(left), value(right), value(fact.snapshot_cells[index])
            dtype = 'ptr' if index in (0,count-1) else old.dtype
            if ((old.dtype, old.shape) != (new.dtype, new.shape)
                    or any(getattr(old, field) != getattr(new, field)
                           for field in SSA_VALUE_PHYSICAL_FIELDS)
                    or snapshot.dtype != dtype or tuple(snapshot.shape or ())
                    or snapshot.physical_dtype != dtype or snapshot.storage != 'scalar'
                    or snapshot.rank != 0 or snapshot.arena):
                return None
        null = value(fact.null_cell)
        if (null.dtype != 'ptr' or tuple(null.shape or ()) or null.physical_dtype != 'ptr'
                or null.storage != 'scalar' or null.rank != 0 or null.arena):
            return None
        original_record = value(fact.source_record_cell)
        expected_fields = tuple(replace(field, storage=SSARecordFieldStorage.SEQUENCE,
            value_ids=new_ids, sequence_id=public_id) if field.name==fact.field_name else field
            for field in original_record.fields)
        record = records.records.get(source.record_id)
        if (record != replace(original_record, fields=expected_fields)
                or fact.public_record_cell != book.latest_ref(RECORD_DESCRIPTOR,(records.owner,source.record_id))
                or value(fact.public_record_cell) != record):
            return None
        layout_row = (scope, source.record_id)
        if (fact.source_return_layout_cell.page != RECORD_RETURN_LAYOUT
                or fact.source_return_layout_cell.row != layout_row
                or fact.public_return_layout_cell != book.latest_ref(RECORD_RETURN_LAYOUT,layout_row)):
            return None
        old_layout = tuple(value(fact.source_return_layout_cell))
        if old_layout.count(source.sequence_id) != 1:
            return None
        layout = tuple(public_id if item == source.sequence_id else item for item in old_layout)
        if tuple(value(fact.public_return_layout_cell)) != layout:
            return None
        rets = [(label,block,index,op) for label,block in function.blocks.items()
                for index,op in enumerate(block.instrs) if op.op=='Ret' and not op.attributes.get('fault_exit')]
        if len(rets) != 1:
            return None
        label, block, position, ret = rets[0]
        if position != len(block.instrs)-1 or tuple(int(v.id) for v in ret.args) != layout:
            return None
        # Fresh descriptor cells must not escape before ownership moves.
        # Source-member auditing intentionally does not cover these new IDs.
        fresh_ids = set((*new_ids, *snapshots, null_id))
        argument_ids = [int(arg.id) for arg in function.args]
        if (any(argument_ids.count(value_id) != 1 for value_id in new_ids)
                or set((*snapshots, null_id)).intersection(argument_ids)):
            return None
        start = position - (2 * count + 2)
        for other_label, other_block in function.blocks.items():
            for index, instruction in enumerate(other_block.instrs):
                touched = {int(v.id) for v in instruction.args}
                if instruction.res is not None:
                    touched.add(int(instruction.res.id))
                if fresh_ids.intersection(touched) and not (
                        other_label == label and start <= index <= position):
                    return None
        # A full contiguous recipe leaves no opportunity to clear before the
        # snapshots, replace the chosen owner, or store into an unrelated cell.
        ops = block.instrs[position-(2*count+2):position]
        if len(ops) != 2*count+2:
            return None
        for index in range(count):
            load, store = ops[index], ops[count+index]
            if (load.op!='Load' or load.attributes or tuple(int(v.id) for v in load.args)!=(old_ids[index],)
                    or load.res is None or int(load.res.id)!=snapshots[index]
                    or store.op!='Store' or store.attributes or store.res is not None
                    or tuple(int(v.id) for v in store.args)!=(snapshots[index],new_ids[index])):
                return None
        zero, clear = ops[-2:]
        if (zero.op!='Const' or zero.args or zero.res is None or int(zero.res.id)!=null_id
                or set(zero.attributes)!={'value'} or type(zero.attributes['value']) is not int
                or zero.attributes['value']!=0 or clear.op!='Store' or clear.attributes
                or clear.res is not None or tuple(int(v.id) for v in clear.args)!=(null_id,old_ids[-1])):
            return None
        geometry = (tuple((str(op.op),tuple(int(v.id) for v in op.args),
            None if op.res is None else int(op.res.id),tuple(op.attributes.get('incoming_blocks') or ()))
            for op in block.instrs),tuple(block.successors))
        if (fact.block_geometry_cell.page!=SSA_BLOCK_GEOMETRY
                or tuple(fact.block_geometry_cell.row)[-2:]!=(name,str(label))
                or book.latest_ref(SSA_BLOCK_GEOMETRY,fact.block_geometry_cell.row)!=fact.block_geometry_cell
                or value(fact.block_geometry_cell)!=geometry):
            return None
        if any(d.heap_owner_value_id is not None and int(d.sequence_id)!=public_id
               and any(int(column) in layout for column in d.column_value_ids)
               for d in sequences.sequences.values()):
            return None
        causes={fact.entry_cell,fact.source_return_slot_cell,fact.source_constructor_operands_cell,
            fact.source_cell,fact.source_descriptor_cell,fact.source_record_cell,fact.source_return_layout_cell,
            fact.element_declaration_cell,fact.element_binding_cell,fact.element_class_cell,
            fact.public_descriptor_cell,fact.public_record_cell,fact.public_return_layout_cell,
            *refs,fact.block_geometry_cell,*fact.terminal_lifetime_cells,*fact.writer_cells}
        if {parent for parent,_ in book.edges_into(cell)} != causes:
            return None
        return cell,fact
    except (KeyError,TypeError,ValueError,AttributeError,IndexError):
        return None



def validated_record_owned_return_suffix(module, function_name, public_sequence_id):
    """Return exact verified transfer operation positions, not an escape grant.

    Writer auditing can exclude only this recipe from its use census. This
    structural reader never calls writer auditing, avoiding circular authority.
    """
    structural = validated_record_owned_return_structure(module, function_name, public_sequence_id)
    if structural is None:
        return None
    _, fact = structural
    function = module.functions[str(function_name)]
    label = str(fact.block_geometry_cell.row[-1])
    block = function.blocks[label]
    # The structure proof established one terminal Ret and the contiguous
    # load/store, null, owner-clear suffix immediately preceding it.
    start = len(block.instrs) - (2 * len(fact.member_pairs) + 3)
    return label, start, tuple(block.instrs[start:])


def validated_record_owned_return_projection(module, function_name, public_sequence_id):
    """Grant only structure AND a closed actual writer-element proof."""
    structural = validated_record_owned_return_structure(module, function_name, public_sequence_id)
    if structural is None:
        return None
    from .source_owned_record_returns import OwnedRecordReturnSource
    from .source_owned_record_elements import owned_record_primitive_element
    from .source_owned_record_writes import owned_record_primitive_writes
    cell, fact = structural
    source = OwnedRecordReturnSource(fact.source_record_id, fact.field_name,
        fact.source_sequence_id, fact.source_return_slot_cell,
        fact.source_constructor_operands_cell, fact.source_cell,
        fact.source_descriptor_cell, fact.source_record_cell, fact.source_owner_id)
    element = owned_record_primitive_element(module, str(function_name), source)
    if element is None:
        return None
    writers = owned_record_primitive_writes(module, str(function_name), source, element)
    if writers is None or tuple(writers) != fact.writer_cells:
        return None
    return structural


def record_owned_return_projection_findings(module):
    from .concordance_declarations import RECORD_OWNED_RETURN_PROJECTION
    from .identity_concordance import Finding
    book=dict(getattr(module,'metadata',{}) or {}).get('identity_book')
    if book is None:
        return ()
    result=[]
    for row in book.page(RECORD_OWNED_RETURN_PROJECTION).rows():
        if validated_record_owned_return_projection(module,row[0],row[1]) is None:
            result.append(Finding('owned-record-return-projection',str(row[0]),int(row[1]),
                'source/public mapping, selected local owner or exact native transfer is unproven'))
    return tuple(result)


def validated_record_public_owned_return(module, cell):
    """Read public ports only after the complete record-specific source proof."""
    from .concordance_declarations import (
        PUBLIC_OWNED_SEQUENCE_RETURN, RECORD_OWNED_RETURN_PROJECTION,
        PublicOwnedSequenceReturn,
    )
    from ..transmogrifier.ssa import SSASequenceCapacityPolicy
    book = dict(getattr(module, 'metadata', {}) or {}).get('identity_book')
    if book is None or cell.page != PUBLIC_OWNED_SEQUENCE_RETURN:
        return None
    try:
        if book.latest_ref(cell.page, cell.row) != cell:
            return None
        fact = book.page(cell.page).cell(cell.row, cell.column)
        if (not isinstance(fact, PublicOwnedSequenceReturn)
                or fact.source_cell.page != RECORD_OWNED_RETURN_PROJECTION
                or fact.source_cell.row != cell.row):
            return None
        validated = validated_record_owned_return_projection(module, cell.row[0], cell.row[1])
        if validated is None or validated[0] != fact.source_cell:
            return None
        _, projection = validated
        if (fact.descriptor_cell != projection.public_descriptor_cell
                or fact.owner_cell != projection.member_pairs[-1][1]
                or {p for p,_ in book.edges_into(cell)} != {
                    fact.source_cell, fact.descriptor_cell, fact.owner_cell}):
            return None
        function = module.functions[str(cell.row[0])]
        descriptor = module.sequence_tables[str(cell.row[0])].sequences[int(cell.row[1])]
        if descriptor.capacity_policy is not SSASequenceCapacityPolicy.DYNAMIC:
            return None
        owner = next(arg for arg in function.args if int(arg.id) == int(descriptor.heap_owner_value_id))
        if owner.dtype != 'ptr' or owner.shape != (1,) or not (owner.accounting or {}).get('heap_owner'):
            return None
        return fact, projection
    except (KeyError,TypeError,ValueError,AttributeError,IndexError,StopIteration):
        return None


def publish_record_owned_public_returns(module, function_name):
    """Publish the incumbent public-owned ABI, never an annotation-only grant."""
    from .concordance_declarations import (
        RECORD_OWNED_RETURN_PROJECTION, PUBLIC_OWNED_SEQUENCE_RETURN,
        PublicOwnedSequenceReturn, SHELL_HANDOFF,
    )
    from .identity_concordance import CorrelationTable, Derived, Mode
    book = dict(getattr(module, 'metadata', {}) or {}).get('identity_book')
    if book is None or CorrelationTable.build(module).findings(module):
        return ()
    pending = []
    for row in book.page(RECORD_OWNED_RETURN_PROJECTION).rows():
        if row[0] != str(function_name):
            continue
        validated = validated_record_owned_return_projection(module, row[0], row[1])
        if validated is None:
            return ()
        source, projection = validated
        pending.append((row, PublicOwnedSequenceReturn(source,
            projection.public_descriptor_cell, projection.member_pairs[-1][1])))
    result = []
    for row, fact in pending:
        cell = book.post(PUBLIC_OWNED_SEQUENCE_RETURN, row, fact,
            stage=SHELL_HANDOFF, provenance=Derived((fact.source_cell,
                fact.descriptor_cell,fact.owner_cell)),mode=Mode.CONCORD)
        if validated_record_public_owned_return(module, cell) is None:
            raise ValueError('record public-owned return failed its final validation')
        result.append(cell)
    return tuple(result)
