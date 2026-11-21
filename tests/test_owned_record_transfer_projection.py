"""Source-gated ownership transfer IR is not yet a backend output grant."""
from tests.test_owned_record_transfer_plan import _lower, SOURCE
from src.compiler.source_owned_record_transfer import materialize_owned_record_transfers
from src.compiler.source_owned_record_returns import (
    owned_record_return_sources, validate_owned_record_return_source)
from src.compiler.identity_concordance import CorrelationTable
from src.compiler.source_owned_record_projection_validation import validated_record_owned_return_projection


def test_projection_keeps_original_identity_and_moves_selected_owner_address():
    module, name = _lower(SOURCE)
    original = owned_record_return_sources(module, name)[0]
    refs = materialize_owned_record_transfers(module, name)
    assert len(refs) == 1
    book = module.metadata['identity_book']
    fact = book.page(refs[0].page).cell(refs[0].row, refs[0].column)
    assert validated_record_owned_return_projection(module, name, refs[0].row[1]) == (refs[0],fact)
    assert fact.source_sequence_id == original.sequence_id
    assert refs[0].row[1] != original.sequence_id
    assert validate_owned_record_return_source(module, name, original)
    assert owned_record_return_sources(module, name) == ()
    assert CorrelationTable.build(module).findings(module) == []
    function = module.functions[name]
    operations = function.blocks[fact.block_geometry_cell.row[-1]].instrs
    owner_source, owner_public = fact.member_pairs[-1]
    owner_snapshot = fact.snapshot_cells[-1]
    owner_load = next(i for i, op in enumerate(operations)
                      if op.op == 'Load' and op.res.id == owner_snapshot.row[-1])
    owner_copy = next(i for i, op in enumerate(operations)
                      if op.op == 'Store' and [v.id for v in op.args] == [owner_snapshot.row[-1],owner_public.row[-1]])
    owner_clear = next(i for i, op in enumerate(operations)
                      if op.op == 'Store' and [v.id for v in op.args] == [fact.null_cell.row[-1],owner_source.row[-1]])
    assert owner_load < owner_copy < owner_clear < len(operations)-1
    assert operations[-1].op == 'Ret'
    assert operations[-1].args[0].id == refs[0].row[1]
    descriptor = module.sequence_tables[name].sequences[refs[0].row[1]]
    assert descriptor.heap_owner_value_id == owner_public.row[-1]
    assert descriptor.status_address_id is not None
    assert descriptor.status_address_id in {v.id for v in function.args}
    public_ids = {right.row[-1] for left,right in fact.member_pairs}
    for arg in function.args:
        if arg.id in public_ids:
            assert arg.accounting['compiler_frame_sequence_id'] == refs[0].row[1]
            assert not {'program_abi_parameter','program_abi_record','program_abi_field',
                        'record_constructor_value','source_value_id','alias_of'}.intersection(arg.accounting)


def test_borrowed_source_cannot_trigger_projection():
    module, name = _lower('''from dataclasses import dataclass
@dataclass
class Result:
    items: list[int]
def root(items: list[int]) -> Result:
    return Result(items=items)
''')
    assert materialize_owned_record_transfers(module, name) == ()


def test_projection_rejects_forged_recipe_owner_null_and_layout():
    from dataclasses import replace
    module, name = _lower(SOURCE)
    refs = materialize_owned_record_transfers(module, name)
    assert len(refs) == 1
    book = module.metadata['identity_book']
    cell = refs[0]
    page = book.page(cell.page)
    key = (cell.row,cell.column)
    fact = page.cells[key]
    check = lambda: validated_record_owned_return_projection(module,name,cell.row[1])
    assert check() == (cell,fact)
    function = module.functions[name]
    block = function.blocks[fact.block_geometry_cell.row[-1]]
    original_ops = list(block.instrs)
    geometry_page = book.page(fact.block_geometry_cell.page)
    geometry_key = (fact.block_geometry_cell.row,fact.block_geometry_cell.column)
    original_geometry = geometry_page.cells[geometry_key]
    def refresh_forged_geometry():
        geometry_page.cells[geometry_key] = (
            tuple((str(op.op),tuple(int(v.id) for v in op.args),
                   None if op.res is None else int(op.res.id),
                   tuple(op.attributes.get('incoming_blocks') or ())) for op in block.instrs),
            tuple(block.successors))
    # Even a forged geometry snapshot cannot authorize clearing before copy.
    try:
        block.instrs[-2],block.instrs[-4] = block.instrs[-4],block.instrs[-2]
        refresh_forged_geometry()
        assert check() is None
    finally:
        block.instrs[:] = original_ops
        geometry_page.cells[geometry_key] = original_geometry
    # Null is exact native zero, never an arbitrary pointer-valued literal.
    zero = block.instrs[-3]
    old_attributes = zero.attributes
    zero.attributes = {'value':1}
    try:
        assert check() is None
    finally:
        zero.attributes = old_attributes
    # Clearing an original branch owner instead of the selected address is wrong.
    clear = block.instrs[-2]
    old_args = clear.args
    selected = fact.member_pairs[-1][0].row[-1]
    wrong = next(arg for arg in function.args
                 if (arg.accounting or {}).get('heap_owner') and arg.id != selected
                 and arg.id != fact.member_pairs[-1][1].row[-1])
    clear.args = [old_args[0],wrong]
    try:
        refresh_forged_geometry()
        assert check() is None
    finally:
        clear.args = old_args
        geometry_page.cells[geometry_key] = original_geometry
    # Captured source/public lineage is not replaceable by another member.
    for replacement in (replace(fact,source_owner_id=fact.source_sequence_id),
                        replace(fact,source_return_layout_cell=fact.public_return_layout_cell)):
        page.cells[key] = replacement
        try:
            assert check() is None
        finally:
            page.cells[key] = fact
    assert check() == (cell,fact)


def test_misleading_primitive_field_cannot_bless_actual_string_writes():
    source = SOURCE.replace('out.append(i)', "out.append('x')").replace(
        'out.append(-7)', "out.append('seed')").replace('out.append(-i)', "out.append('y')")
    module, name = _lower(source)
    function = module.functions[name]
    before_args = tuple(v.id for v in function.args)
    before_ir = tuple((label, tuple(repr(op) for op in block.instrs))
                      for label, block in function.blocks.items())
    before_sequences = dict(module.sequence_tables[name].sequences)
    assert materialize_owned_record_transfers(module, name) == ()
    assert tuple(v.id for v in function.args) == before_args
    assert tuple((label, tuple(repr(op) for op in block.instrs))
                 for label, block in function.blocks.items()) == before_ir
    assert module.sequence_tables[name].sequences == before_sequences
