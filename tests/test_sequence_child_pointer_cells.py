"""Child payloads stay dynamic; only exact local native pointer cells are sized."""
from dataclasses import replace
import pytest
from tests.test_sequence_child_storage_lifetime import lowered
from src.compiler.source_sequence_pointer_cells import sequence_child_pointer_cell
from src.compiler.concordance_declarations import SEQUENCE_DESCRIPTOR, SEQUENCE_MEMBER, SSA_VALUE
from src.compiler.ssa_storage_requirements import function_storage_requirements


def test_exact_child_pointer_cells_keep_payload_dynamic(lowered):
    module,name,sid=lowered
    descriptor=module.sequence_tables[name].sequences[sid];pool=descriptor.child_table_pool
    before=function_storage_requirements(module,name)
    for value in (*pool.column_value_ids,pool.length_value_id,pool.status_value_id):
        proof=sequence_child_pointer_cell(module,name,value)
        assert proof is not None and proof.element_count==1 and proof.proof_cells
        assert before[value].dynamic and before[value].element_count is None
    assert function_storage_requirements(module,name)==before
    for value in (pool.capacity_value_id,pool.row_stride_value_id,descriptor.heap_owner_value_id,
                  descriptor.column_value_ids[0],descriptor.length_address_id):
        assert sequence_child_pointer_cell(module,name,value) is None
    owned_ids={*pool.column_value_ids,pool.length_value_id,pool.status_value_id}
    for other in module.sequence_tables[name].sequences.values():
        if other.child_table_pool is None:
            for value in other.column_value_ids:
                if value not in owned_ids:
                    assert sequence_child_pointer_cell(module,name,value) is None


@pytest.mark.parametrize('change',['owner','role','cause','layout','missing','native-shape','native-type'])
def test_child_pointer_slot_needs_complete_owner_and_role(lowered,monkeypatch,change):
    module,name,sid=lowered;table=module.sequence_tables[name];book=table.book
    descriptor=table.sequences[sid];value=descriptor.child_table_pool.column_value_ids[0]
    proof=sequence_child_pointer_cell(module,name,value);assert proof
    dc=book.latest_ref(SEQUENCE_DESCRIPTOR,(table.owner,sid))
    cell=next(c for c in proof.proof_cells if c.page==SSA_VALUE and c.row[-1]==value
              and c in {p for p,_ in book.edges_into(dc)})
    if change=='owner':
        other=next(d.heap_owner_value_id for i,d in table.sequences.items()
                   if i!=sid and d.heap_owner_value_id is not None)
        monkeypatch.setitem(book.page(SEQUENCE_DESCRIPTOR).cells,(dc.row,dc.column),replace(descriptor,heap_owner_value_id=other))
    elif change=='role':
        role=book.latest_ref(SEQUENCE_MEMBER,(table.owner,value))
        monkeypatch.setitem(book.page(SEQUENCE_MEMBER).cells,(role.row,role.column),((sid,('child_pool','length_value_id')),))
    elif change=='cause':
        original=book.edges_into
        monkeypatch.setattr(book,'edges_into',lambda c:tuple(e for e in original(c) if e[0]!=cell) if c==dc else original(c))
    elif change=='layout':
        page=book.page(SSA_VALUE);key=(cell.row,cell.column)
        monkeypatch.setitem(page.cells,key,replace(page.cells[key],physical_dtype='float64'))
    elif change=='missing':
        original=book.latest_ref
        monkeypatch.setattr(book,'latest_ref',lambda page,row:None if page==SEQUENCE_DESCRIPTOR and row==dc.row else original(page,row))
    else:
        formal=next(v for v in module.functions[name].args if v.id==value)
        monkeypatch.setattr(formal,'shape' if change=='native-shape' else 'dtype',(1,) if change=='native-shape' else 'int64')
    assert sequence_child_pointer_cell(module,name,value) is None

def test_c_wrapper_uses_child_pointer_slots_without_sizing_payload(lowered):
    from src.compiler.ssa_c_backend import emit_ssa_module_to_c
    module,name,sid=lowered
    pool=module.sequence_tables[name].sequences[sid].child_table_pool
    ids=(*pool.column_value_ids,pool.length_value_id,pool.status_value_id)
    before=function_storage_requirements(module,name)
    emitted=emit_ssa_module_to_c(module,name)
    assert not [s for s in emitted.shortfalls if s.operation=='storage'
                and any(f'root formal {i} ' in s.reason for i in ids)], emitted.shortfalls
    after=function_storage_requirements(module,name)
    assert all(after[i]==before[i] and after[i].dynamic for i in ids)
