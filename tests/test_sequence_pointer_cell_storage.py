"""A native arena pointer cell is not a one-element logical sequence."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.source_sequence_pointer_cells import sequence_pointer_cell
from src.compiler.ssa_storage_requirements import function_storage_requirements
from src.compiler.concordance_declarations import SEQUENCE_DESCRIPTOR

SOURCE="""def root(n: int):
    a=[]
    b=[]
    for i in range(n):
        a.append(i)
        b.append(i*2)
    return len(a)+len(b)
"""

@pytest.fixture(scope="module")
def lowered():
    m,_,names=lower_ast_source_to_ssa(SOURCE,'root',extraction_contract=ExtractionContract(
        Path(__file__).resolve().parents[1]/'extraction_contracts/program_extraction.yaml'),progress=lambda _:None,progress_bars=False)
    return m,names[0]


def test_native_dynamic_list_cells_keep_dynamic_payload(lowered,tmp_path):
    from src.compiler.ssa_c_backend import emit_ssa_module_to_c
    from src.compiler.identity_concordance import CorrelationTable
    m,name=lowered
    table=m.sequence_tables[name]
    columns=[d.column_value_ids[0] for d in table.sequences.values() if d.heap_owner_value_id is not None]
    assert len(columns)==2
    before=function_storage_requirements(m,name)
    for column in columns:
        assert before[column].dynamic and before[column].element_count is None
        assert sequence_pointer_cell(m,name,column).element_count==1
    artifact=emit_ssa_module_to_c(m,name)
    assert artifact.complete,artifact.shortfalls
    assert CorrelationTable.build(m).findings(m)==[]
    artifact.compile(tmp_path/'c')
    param=dict(m.functions[name].metadata['parameter_names'])['n']
    output=next(i.args[0].id for block in m.functions[name].blocks.values() for i in block.instrs if i.op=='Ret')
    for n in (0,1,17,4097):
        execution=artifact.prepare_execution({param:np.array([n],dtype=np.int64)})
        for _ in range(2):
            execution.run()
            assert execution.buffers[output].item()==2*n
    after=function_storage_requirements(m,name)
    assert all(after[c]==before[c] for c in columns)


@pytest.mark.parametrize('change',['column','owner','missing'])
def test_wrong_or_absent_descriptor_refused(lowered,monkeypatch,change):
    m,name=lowered;table=m.sequence_tables[name];book=table.book
    a,b=[d for d in table.sequences.values() if d.heap_owner_value_id is not None]
    cell=book.latest_ref(SEQUENCE_DESCRIPTOR,(table.owner,a.sequence_id))
    if change=='missing':
        original=book.latest_ref
        monkeypatch.setattr(book,'latest_ref',lambda page,row:None if page==SEQUENCE_DESCRIPTOR and row==cell.row else original(page,row))
    else:
        changed=replace(a,**({'column_value_ids':b.column_value_ids} if change=='column' else {'heap_owner_value_id':b.heap_owner_value_id}))
        monkeypatch.setitem(book.page(SEQUENCE_DESCRIPTOR).cells,(cell.row,cell.column),changed)
    assert sequence_pointer_cell(m,name,a.column_value_ids[0]) is None


@pytest.mark.parametrize('change',['native-shape','native-type','ambiguous-descriptors'])
def test_native_cell_shape_and_unique_descriptor_are_required(lowered,monkeypatch,change):
    m,name=lowered;table=m.sequence_tables[name]
    a,b=[d for d in table.sequences.values() if d.heap_owner_value_id is not None]
    column=a.column_value_ids[0]
    formal=next(v for v in m.functions[name].args if int(v.id)==column)
    if change=='native-shape':
        monkeypatch.setattr(formal,'shape',(2,))
    elif change=='native-type':
        monkeypatch.setattr(formal,'dtype','int64')
    else:
        descriptors=dict(table.sequences)
        descriptors[b.sequence_id]=replace(b,column_value_ids=a.column_value_ids)
        monkeypatch.setattr(table,'sequences',descriptors)
    assert sequence_pointer_cell(m,name,column) is None
