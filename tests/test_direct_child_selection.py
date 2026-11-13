"""Direct child storage is explicit opt-in, not a public ownership grant."""
from dataclasses import replace
import pytest
from tests.test_sequence_child_storage_lifetime import lowered
from src.compiler.source_sequence_selection import sequence_selection, local_selection_lifetimes
from src.compiler.source_sequence_child_lifetime import validated_sequence_child_storage
from src.compiler.concordance_declarations import SEQUENCE_DESCRIPTOR, SEQUENCE_MEMBER


def test_child_leaf_remains_refused_by_primitive_callers(lowered):
    module, name, sid = lowered
    assert sequence_selection(module, name, sid) is None
    assert local_selection_lifetimes(module, name, sid) is None


def test_opted_in_child_leaf_has_complete_distinct_storage_and_proof(lowered):
    module, name, sid = lowered
    d = module.sequence_tables[name].sequences[sid]
    p = d.child_table_pool
    tree = sequence_selection(module, name, sid, include_child_pool=True)
    assert tree is not None and not tree.arms and tree.geometry_cell is None
    assert tree.member_ids == (*d.column_value_ids, d.length_address_id, d.capacity_value_id,
        *((d.status_address_id,) if d.status_address_id is not None else ()),
        *p.column_value_ids, p.length_value_id,
        *((p.status_value_id,) if p.status_value_id is not None else ()),
        p.capacity_value_id, p.row_stride_value_id, d.heap_owner_value_id)
    assert len(tree.member_ids) == len(set(tree.member_ids))
    lifetime = local_selection_lifetimes(module, name, sid, include_child_pool=True)
    assert lifetime is not None and lifetime.selection == tree
    assert lifetime.original_owner_ids == (d.heap_owner_value_id,)
    assert lifetime.child_storage_cells == validated_sequence_child_storage(module, name, sid)
    assert len(lifetime.terminal_cells) == 1
    # Primitive lists keep their exact incumbent vector/proof by default.
    plain = next(k for k, v in module.sequence_tables[name].sequences.items()
                 if v.heap_owner_value_id is not None and v.child_table_pool is None)
    assert sequence_selection(module, name, plain) is not None
    assert local_selection_lifetimes(module, name, plain) is not None
    assert sequence_selection(module, name, plain) == sequence_selection(module, name, plain, include_child_pool=True)
    assert local_selection_lifetimes(module, name, plain) == local_selection_lifetimes(module, name, plain, include_child_pool=True)


@pytest.mark.parametrize('change', ['missing-role', 'swapped-extents', 'aliased-role', 'foreign-owner'])
def test_child_leaf_lifetime_does_not_infer_role_or_owner(lowered, monkeypatch, change):
    module, name, sid = lowered
    table = module.sequence_tables[name]
    book, d = table.book, table.sequences[sid]
    p = d.child_table_pool
    assert local_selection_lifetimes(module, name, sid, include_child_pool=True) is not None
    if change == 'missing-role':
        cell = book.latest_ref(SEQUENCE_MEMBER, (table.owner, p.length_value_id))
        monkeypatch.setitem(book.page(cell.page).cells, (cell.row, cell.column), ())
    else:
        if change == 'swapped-extents':
            altered = replace(d, child_table_pool=replace(p,
                capacity_value_id=p.row_stride_value_id, row_stride_value_id=p.capacity_value_id))
        elif change == 'aliased-role':
            altered = replace(d, child_table_pool=replace(p, capacity_value_id=p.row_stride_value_id))
        else:
            other = next(v.heap_owner_value_id for k,v in table.sequences.items()
                         if k != sid and v.heap_owner_value_id is not None)
            altered = replace(d, heap_owner_value_id=other)
        cell = book.latest_ref(SEQUENCE_DESCRIPTOR, (table.owner, sid))
        monkeypatch.setitem(book.page(cell.page).cells, (cell.row, cell.column), altered)
    assert local_selection_lifetimes(module, name, sid, include_child_pool=True) is None


def test_child_address_phi_is_not_a_terminal_scalar_proof(lowered):
    from src.transmogrifier.ssa import Instr, SSAValue
    module, name, sid = lowered
    function = module.functions[name]
    p = module.sequence_tables[name].sequences[sid].child_table_pool
    # Adversarial structural probe only: no invented valid producer or native run.
    block = next(iter(function.blocks.values()))
    op = Instr('Phi', [SSAValue(p.capacity_value_id, 'int64'), SSAValue(p.row_stride_value_id, 'int64')],
        SSAValue(p.capacity_value_id, 'int64'), attributes={'incoming_blocks': ('left', 'right')})
    block.instrs.append(op)
    try:
        assert sequence_selection(module, name, sid, include_child_pool=True) is None
        assert local_selection_lifetimes(module, name, sid, include_child_pool=True) is None
    finally:
        block.instrs.pop()
    assert local_selection_lifetimes(module, name, sid, include_child_pool=True) is not None
