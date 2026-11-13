"""Source-born child arenas retain exact roles under their local owner."""
from dataclasses import replace
from pathlib import Path
import pytest
from src.compiler.concordance_declarations import SEQUENCE_DESCRIPTOR,SEQUENCE_MEMBER,SSA_VALUE
from src.compiler.source_sequence_child_lifetime import validated_sequence_child_storage
from src.compiler.source_sequence_lifetime import validated_sequence_lifetime

@pytest.fixture(scope='module')
def lowered():
    from tests.test_borrowed_record_string_copy import SOURCE
    from tests.owned_record_test_support import lower_before_owned_record_boundary
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.identity_concordance import CorrelationTable
    text=SOURCE.replace('    return len(out)','    spare = []\n    for other in nodes:\n        spare.append(7)\n    return len(out) + len(spare)')
    module,_,exports=lower_before_owned_record_boundary(text,'root',
        extraction_contract=ExtractionContract(Path(__file__).parents[1]/'extraction_contracts/program_extraction.yaml'),
        progress=lambda _:None,progress_bars=False)
    assert CorrelationTable.build(module).findings(module)==[]
    name=exports[0];table=module.sequence_tables[name]
    sid=next(sid for sid,d in table.sequences.items() if d.child_table_pool is not None and d.heap_owner_value_id is not None)
    return module,name,sid

def test_complete_source_child_storage_and_immutable_roles(lowered,monkeypatch):
    module,name,sid=lowered;table=module.sequence_tables[name];book=table.book
    descriptor=table.sequences[sid];pool=descriptor.child_table_pool
    check=lambda:validated_sequence_child_storage(module,name,sid)
    expected=check();assert expected is not None
    assert validated_sequence_lifetime(module,name,sid) is not None
    dc=book.latest_ref(SEQUENCE_DESCRIPTOR,(table.owner,sid));page=book.page(SEQUENCE_DESCRIPTOR);key=(dc.row,dc.column)
    wrong_owner=next(d.heap_owner_value_id for i,d in table.sequences.items() if i!=sid and d.heap_owner_value_id is not None)
    for altered in (replace(descriptor,child_table_pool=replace(pool,capacity_value_id=pool.row_stride_value_id,row_stride_value_id=pool.capacity_value_id)),
                    replace(descriptor,child_table_pool=replace(pool,column_value_ids=(pool.length_value_id,),length_value_id=pool.column_value_ids[0])),
                    replace(descriptor,heap_owner_value_id=wrong_owner)):
        with monkeypatch.context() as patch:
            patch.setitem(page.cells,key,altered)
            assert check() is None
        assert check()==expected
    for value_id in (*pool.column_value_ids,pool.length_value_id,pool.capacity_value_id,pool.row_stride_value_id,pool.status_value_id,descriptor.heap_owner_value_id):
        role=book.latest_ref(SEQUENCE_MEMBER,(table.owner,value_id))
        with monkeypatch.context() as patch:
            patch.setitem(book.page(SEQUENCE_MEMBER).cells,(role.row,role.column),())
            assert check() is None
        actual=next(c for c in expected if c.page==SSA_VALUE and c.row[-1]==value_id and c in {p for p,_ in book.edges_into(dc)})
        original_edges=book.edges_into
        with monkeypatch.context() as patch:
            patch.setattr(book,'edges_into',lambda c: tuple(edge for edge in original_edges(c) if edge[0]!=actual) if c==dc else original_edges(c))
            assert check() is None
        assert check()==expected

def test_child_layout_must_be_exact_current_native_layout(lowered,monkeypatch):
    module,name,sid=lowered;table=module.sequence_tables[name];book=table.book
    check=lambda:validated_sequence_child_storage(module,name,sid)
    expected=check();assert expected is not None
    pool=table.sequences[sid].child_table_pool
    cell=next(c for c in expected if c.page==SSA_VALUE and c.row[-1]==pool.column_value_ids[0] and book.page(SSA_VALUE).cell(c.row,c.column).dtype=='ptr')
    page=book.page(SSA_VALUE);key=(cell.row,cell.column);fact=page.cells[key]
    for altered in (replace(fact,dtype='int64'),replace(fact,physical_dtype='float64'),replace(fact,shape=(2,)),replace(fact,arena=False),replace(fact,device='cuda')):
        with monkeypatch.context() as patch:
            patch.setitem(page.cells,key,altered)
            assert check() is None
        assert check()==expected
    for value_id in (pool.capacity_value_id,pool.row_stride_value_id):
        cell=next(c for c in expected if c.page==SSA_VALUE and c.row[-1]==value_id)
        key=(cell.row,cell.column);fact=page.cells[key]
        from src.compiler.source_evidence_resolver import require_physical_layout,PhysicalLayoutEvidence
        assert isinstance(require_physical_layout(cell.row[0],value_id,book=book),PhysicalLayoutEvidence)
        with pytest.raises(ValueError,match='no native physical storage kind'):
            replace(fact,storage='unknown')
        for altered in (replace(fact,storage='span'),replace(fact,rank=1),replace(fact,storage=None),replace(fact,rank=None),replace(fact,storage=None,rank=None),replace(fact,device='cuda')):
            with monkeypatch.context() as patch:
                patch.setitem(page.cells,key,altered)
                assert check() is None
            assert check()==expected
