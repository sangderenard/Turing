"""Preflight exact owned-record escapes before any SSA ownership mutation.

The plan is structural only. Semantic element proof and public projection
validation are separate prerequisites; no owner is detached by this reader.
"""
from dataclasses import dataclass
from .source_owned_record_returns import (
    OwnedRecordReturnSource, owned_record_return_sources,
    validate_owned_record_return_source,
)
from .source_sequence_selection import LocalSelectionLifetimes, local_selection_lifetimes
from .source_owned_record_projection import public_source_entry
from .identity_concordance import Ref


@dataclass(frozen=True)
class OwnedRecordTransferPlan:
    source: OwnedRecordReturnSource
    entry_cell: Ref
    lifetimes: LocalSelectionLifetimes
    return_block: str
    return_position: int
    member_cells: tuple[Ref, ...]
    source_return_layout_cell: Ref


def owned_record_transfer_plans(module, function_name):
    """Prove a unique root return and non-overlapping local original owners.

    No partial plan is offered if an independent exported field overlaps an
    original owner or lacks lifetime proof. This prevents moving a shared
    owner twice while still pretending each returned field owns its data.
    """
    name = str(function_name)
    entry = public_source_entry(module, name)
    if entry is None:
        return ()
    function = module.functions[name]
    returns = [(label, index, op) for label, block in function.blocks.items()
               for index, op in enumerate(block.instrs)
               if op.op == 'Ret' and not op.attributes.get('fault_exit')]
    if len(returns) != 1:
        return ()
    label, index, ret = returns[0]
    if index != len(function.blocks[label].instrs) - 1:
        return ()
    from .ssa_record_return_state import ssa_value_identity_cell, function_scope_of
    from .concordance_declarations import RECORD_RETURN_LAYOUT
    returned = tuple(int(value.id) for value in ret.args)
    sources = owned_record_return_sources(module, name)
    source_ids = {source.sequence_id for source in sources}
    table = module.sequence_tables.get(name)
    if table is None:
        return ()
    for descriptor in table.sequences.values():
        if descriptor.heap_owner_value_id is not None and any(
            int(value) in returned for value in descriptor.column_value_ids
        ) and int(descriptor.sequence_id) not in source_ids:
            # An unproven additional escape might share an original owner.
            return ()
    owners, plans = set(), []
    for source in sources:
        if not validate_owned_record_return_source(module, name, source):
            return ()
        if returned.count(source.sequence_id) != 1:
            return ()
        lifetimes = local_selection_lifetimes(module, name, source.sequence_id)
        if lifetimes is None or owners.intersection(lifetimes.original_owner_ids):
            return ()
        owners.update(lifetimes.original_owner_ids)
        cells = tuple(ssa_value_identity_cell(function, value, book=module.metadata['identity_book'])
                      for value in lifetimes.selection.member_ids)
        if any(cell is None for cell in cells):
            return ()
        book = module.metadata['identity_book']
        layout_cell = book.latest_ref(RECORD_RETURN_LAYOUT, (function_scope_of(function),source.record_id))
        if layout_cell is None:
            return ()
        layout = book.page(layout_cell.page).cell(layout_cell.row,layout_cell.column)
        if tuple(layout).count(source.sequence_id) != 1:
            return ()
        plans.append(OwnedRecordTransferPlan(source, entry[0], lifetimes,
                                            str(label), index, cells, layout_cell))
    return tuple(plans)


def materialize_owned_record_transfers(module, function_name):
    """Install exact root-only primitive transfers; no public ABI grant here.

    The separate projection validator must verify these receipts before a
    backend can expose any returned allocation to its caller.
    """
    from .source_owned_record_elements import owned_record_primitive_element
    from .identity_concordance import begin_identity_book, end_identity_book
    from .source_owned_record_writes import owned_record_primitive_writes
    name = str(function_name)
    plans = owned_record_transfer_plans(module, name)
    if len(plans) != 1:
        # Multiple fields require one aggregate final-layout receipt.
        return ()
    elements = tuple(owned_record_primitive_element(module, name, plan.source)
                     for plan in plans)
    if any(element is None for element in elements):
        return ()
    writers = tuple(owned_record_primitive_writes(module, name, plan.source, element)
                    for plan, element in zip(plans, elements))
    if any(writer is None for writer in writers):
        return ()
    # Validate every field before the first mutation. An unsupported sibling
    # field must not cause a half-projected return frame.
    table = module.sequence_tables[name]
    expected = {'builtins.int':'int64', 'builtins.float':'float64', 'builtins.bool':'bool'}
    for plan, element in zip(plans, elements):
        descriptor = table.sequences[plan.source.sequence_id]
        if tuple(descriptor.column_dtypes) != (expected.get(element.semantic_class),):
            return ()
    book, token = begin_identity_book(module.metadata['identity_book'])
    try:
        return _materialize_verified_plans(module, name, plans, elements, writers, book)
    finally:
        end_identity_book(token)


def _materialize_verified_plans(module, name, plans, elements, writers, book):
    from dataclasses import replace
    from ..transmogrifier.ssa import SSAValue, Instr, SSARecordFieldStorage
    from .concordance_declarations import (
        RECORD_OWNED_RETURN_PROJECTION_VALUE, RECORD_OWNED_RETURN_PROJECTION,
        RecordOwnedReturnProjection, RECORD_RETURN_LAYOUT_STAGE,
        SEQUENCE_DESCRIPTOR, RECORD_DESCRIPTOR, SSA_VALUE,
    )
    from .identity_concordance import Derived, Mode
    from .ssa_record_return_state import (
        mint_ssa_value, ssa_value_identity_cell, function_scope_of,
        map_sequence_descriptor, assign_record_descriptor,
        post_record_return_layout, record_return_layouts, _posted_physical_fields,
    )
    from .precompile_to_ssa import _post_ssa_block_geometry
    from .source_owned_record_elements import owned_record_primitive_element
    function = module.functions[name]
    sequences = module.sequence_tables[name]
    records = module.record_tables[name]
    values = {int(value.id):value for value in function.args}
    for block in function.blocks.values():
        for op in block.instrs:
            values.update((int(value.id),value) for value in op.args)
            if op.res is not None:
                values[int(op.res.id)] = op.res
    stage = RECORD_RETURN_LAYOUT_STAGE
    pending = []
    from .source_owned_record_writes import owned_record_primitive_writes
    for plan, element, writer_cells in zip(plans, elements, writers):
        source = plan.source
        # Recheck the actual semantic reader immediately before the mint;
        # callers cannot inject a convenient class label or naked Ref pair.
        if owned_record_primitive_element(module, name, source) != element:
            raise ValueError('owned record element proof changed before transfer')
        if owned_record_primitive_writes(module, name, source, element) != writer_cells:
            raise ValueError('owned record writer proof changed before transfer')
        old_layout = book.page(plan.source_return_layout_cell.page).cell(
            plan.source_return_layout_cell.row,plan.source_return_layout_cell.column)
        if (book.latest_ref(plan.source_return_layout_cell.page,plan.source_return_layout_cell.row)
                != plan.source_return_layout_cell or tuple(old_layout).count(source.sequence_id) != 1):
            raise ValueError('owned record return physical layout changed before transfer')
        descriptor = sequences.sequences[source.sequence_id]
        member_ids = plan.lifetimes.selection.member_ids
        causes = (plan.entry_cell, source.return_slot_cell, source.constructor_operands_cell,
                  source.source_cell, source.descriptor_cell, source.record_cell,
                  plan.source_return_layout_cell, element.field_cell, element.binding_cell, element.element_cell,
                  *plan.lifetimes.terminal_cells, *plan.member_cells, *writer_cells)
        mapped, snapshots = {}, []
        for position, source_id in enumerate(member_ids):
            original = values[source_id]
            public_id = mint_ssa_value(function, RECORD_OWNED_RETURN_PROJECTION_VALUE,
                causes, dtype=original.dtype, shape=original.shape, stage=stage,
                physical=_posted_physical_fields(function, original))
            # A fresh frame slot takes no source alias, parameter, record or
            # lexical-value labels. As in descriptor name_member, accounting
            # is only a view of the newly minted identity and physical fact.
            physical = _posted_physical_fields(function, original) or {}
            accounting = {'compiler_frame_storage':name,
                          'compiler_frame_member':position}
            for field, label in (('storage','program_abi_storage'),
                                 ('rank','program_abi_rank'),
                                 ('physical_dtype','physical_dtype')):
                if physical.get(field) is not None:
                    accounting[label] = physical[field]
            if physical.get('arena'):
                accounting['sequence_arena'] = True
            public = SSAValue(public_id, dtype=original.dtype, shape=original.shape,
                              accounting=accounting)
            mapped[source_id] = public
            pointer = position == 0 or position == len(member_ids)-1
            dtype = 'ptr' if pointer else original.dtype
            snapshot_id = mint_ssa_value(function, RECORD_OWNED_RETURN_PROJECTION_VALUE,
                (plan.member_cells[position], ssa_value_identity_cell(function, public_id, book=book)),
                dtype=dtype, shape=(), stage=stage,
                physical={'physical_dtype':dtype, 'storage':'scalar', 'rank':0, 'arena':False})
            snapshots.append(SSAValue(snapshot_id, dtype=dtype, shape=()))
        public_data = mapped[member_ids[0]]
        public_owner = mapped[descriptor.heap_owner_value_id]
        public_data.accounting.update(heap_owner_value_id=public_owner.id,
                                      heap_element_dtype=descriptor.column_dtypes[0],
                                      sequence_row_shape=descriptor.column_shapes[0])
        public_owner.accounting.update(heap_owner=True)
        for public in mapped.values():
            public.accounting['compiler_frame_sequence_id'] = public_data.id
        remap = {source_id:int(public.id) for source_id,public in mapped.items()}
        public_descriptor = map_sequence_descriptor(descriptor, remap,
            sequence_id=public_data.id, child_table_pool=None)
        public_cells = tuple(ssa_value_identity_cell(function, mapped[i].id, book=book) for i in member_ids)
        sequences.register(public_descriptor, sources=(*causes,*public_cells), stage=stage)
        public_descriptor_cell = book.latest_ref(SEQUENCE_DESCRIPTOR,(sequences.owner,public_data.id))
        # Every storage cell is explicit in the native frame. Status retains
        # its private compiler-frame classification; only four ports are public.
        for source_id in member_ids:
            function.args.append(mapped[int(source_id)])
        null_id = mint_ssa_value(function, RECORD_OWNED_RETURN_PROJECTION_VALUE,
            (plan.member_cells[-1],public_cells[-1]), dtype='ptr', shape=(), stage=stage,
            physical={'physical_dtype':'ptr','storage':'scalar','rank':0,'arena':False})
        null = SSAValue(null_id,dtype='ptr',shape=())
        ops = [Instr('Load',[values[source_id]],snapshot)
               for source_id,snapshot in zip(member_ids,snapshots)]
        ops.extend(Instr('Store',[snapshot,mapped[source_id]],None)
                   for source_id,snapshot in zip(member_ids,snapshots))
        ops.extend((Instr('Const',[],null,attributes={'value':0}),
                    Instr('Store',[null,values[descriptor.heap_owner_value_id]],None)))
        block = function.blocks[plan.return_block]
        ret = block.instrs[-1]
        block.instrs[-1:-1] = ops
        ret.args = [public_data if int(value.id)==source.sequence_id else value for value in ret.args]
        record = records.records[source.record_id]
        fields = tuple(replace(field,storage=SSARecordFieldStorage.SEQUENCE,
                               value_ids=tuple(mapped[i].id for i in member_ids),sequence_id=public_data.id)
                       if field.name==source.field_name else field for field in record.fields)
        assign_record_descriptor(records,source.record_id,replace(record,fields=fields),
                                 (*causes,public_descriptor_cell,*public_cells),stage=stage)
        record_cell = book.latest_ref(RECORD_DESCRIPTOR,(records.owner,source.record_id))
        layout = tuple(public_data.id if value==source.sequence_id else value for value in old_layout)
        layout_cell = post_record_return_layout(function,records,source.record_id,layout,stage=stage)
        pending.append((plan,element,writer_cells,public_descriptor_cell,record_cell,layout_cell,
                        tuple(zip(plan.member_cells,public_cells)),
                        tuple(ssa_value_identity_cell(function,value.id,book=book) for value in snapshots),
                        ssa_value_identity_cell(function,null_id,book=book),public_data.id))
    from .structural_concordance import publish_module_structures
    publish_module_structures(module)
    geometry = _post_ssa_block_geometry(book,function_scope_of(function),function)
    result=[]
    for plan,element,writer_cells,descriptor_cell,record_cell,layout_cell,pairs,snapshots,null_cell,public_id in pending:
        source=plan.source
        fact=RecordOwnedReturnProjection(
            plan.entry_cell,source.record_id,source.field_name,source.sequence_id,source.owner_id,
            source.return_slot_cell,source.constructor_operands_cell,source.source_cell,
            source.descriptor_cell,source.record_cell,plan.source_return_layout_cell,element.field_cell,element.binding_cell,
            element.element_cell,descriptor_cell,record_cell,layout_cell,pairs,snapshots,null_cell,
            geometry[plan.return_block],plan.lifetimes.terminal_cells,tuple(writer_cells))
        causes=(plan.entry_cell,source.return_slot_cell,source.constructor_operands_cell,
                source.source_cell,source.descriptor_cell,source.record_cell,plan.source_return_layout_cell,
                element.field_cell,element.binding_cell,element.element_cell,
                descriptor_cell,record_cell,layout_cell,*[cell for pair in pairs for cell in pair],
                *snapshots,null_cell,geometry[plan.return_block],*plan.lifetimes.terminal_cells,*writer_cells)
        result.append(book.post(RECORD_OWNED_RETURN_PROJECTION,(name,public_id),fact,
                                stage=stage,provenance=Derived(causes),mode=Mode.CONCORD))
    return tuple(result)
