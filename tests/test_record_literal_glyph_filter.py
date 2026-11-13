"""A literal's exact glyph recipe participates in full-content filtering."""
from pathlib import Path
import numpy as np
import pytest
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable
from src.compiler.string_table import string_token

@pytest.mark.parametrize('operator', ['==', '!='])
@pytest.mark.parametrize('literal', ['master', 'λ', ''])
def test_record_literal_filter_preserves_selected_rows(tmp_path, operator, literal):
    from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution
    source = f'''from dataclasses import dataclass
@dataclass
class Node:
    layer: str
    score: int

def root(nodes: list[Node]):
    selected = [node.score for node in nodes if node.layer {operator} {literal!r}]
    total = 0
    for score in selected:
        total += score
    return total
'''
    contract = ExtractionContract(Path(__file__).resolve().parents[1] / 'extraction_contracts/program_extraction.yaml')
    module, _, exports = lower_ast_source_to_ssa(source, 'root', extraction_contract=contract, progress=lambda _: None, progress_bars=False)
    assert CorrelationTable.build(module).findings(module) == []
    fn = module.functions[exports[0]]
    seq = module.sequence_tables[exports[0]].sequences[dict(fn.metadata['parameter_names'])['nodes']]
    fields = {arg.accounting['program_abi_field']: arg for arg in fn.args if 'program_abi_field' in (arg.accounting or {})}
    emitted = emit_ssa_function_to_llvm(module, exports[0])
    assert emitted.shortfalls == ()
    artifact = compile_artifact(emitted, directory=tmp_path / 'llvm')
    out = next(i.args[0].id for b in fn.blocks.values() for i in b.instrs if i.op == 'Ret' and not i.attributes.get('fault_exit'))
    words = ['master', 'mast', 'λ', '', 'masterx', 'MASTER']
    for handles, scores in [([2,0,3,1,5,4], [7,11,19,23,37,41]), ([], []), ([0,0,2], [2,5,13])]:
        feeds = {seq.column_value_ids[0]:np.array(handles,dtype=np.int64),seq.column_value_ids[1]:np.array(scores,dtype=np.int64),
                 seq.length_address_id:np.array([len(handles)],dtype=np.int64),seq.capacity_value_id:np.array([len(handles)],dtype=np.int64),
                 seq.status_address_id:np.array([0],dtype=np.int64)}
        stride=max(map(len,words))
        data={'column':[string_token(w[i]) if i<len(w) else 0 for w in words for i in range(stride)],
              'lengths':list(map(len,words)), 'row_stride':[stride], 'capacity':[len(words)]}
        for member,values in data.items():feeds[fields[f'Node.layer.{member}'].id]=np.array(values,dtype=np.int64)
        before={k:v.copy() for k,v in feeds.items()}
        expected=sum(score for handle,score in zip(handles,scores) if ((words[handle]==literal) if operator=='==' else (words[handle]!=literal)))
        for _ in range(2):
            result=prepare_artifact_execution(artifact,feeds).run()
            assert int(result.buffers[out].reshape(-1)[0])==expected
            for k in feeds:np.testing.assert_array_equal(feeds[k],before[k])

@pytest.mark.parametrize('damage', ['absent', 'wrong_component', 'ungrounded'])
def test_literal_recipe_requires_exact_source_evidence(monkeypatch, damage):
    import src.compiler.glsl_deployment_strategy as planner
    from src.compiler.concordance_declarations import SOURCE_GLYPH_SEQUENCE
    from src.compiler.identity_concordance import current_identity_book
    from dataclasses import replace
    original = planner._resident_string_comparison_cells
    checked = []
    def inspect(graph, node_id):
        data=graph.G.nodes[node_id]
        if str(data.get('op') or data.get('type')).lower()=='equal':
            for parent,_ in data.get('parents',()):
                row=(graph.G.nodes[parent].get('attributes') or {}).get('source_literal_glyph_sequence_row')
                if row is None:continue
                book=current_identity_book();page=book.page(SOURCE_GLYPH_SEQUENCE)
                cell=book.latest_ref(SOURCE_GLYPH_SEQUENCE,tuple(row))
                recipe=page.latest(tuple(row))
                if cell is None:continue
                with monkeypatch.context() as changed:
                    if damage=='absent':
                        latest=book.latest_ref
                        changed.setattr(book,'latest_ref',lambda p,r,*args,**kw:None if p==SOURCE_GLYPH_SEQUENCE and r==tuple(row) else latest(p,r,*args,**kw))
                    elif damage=='wrong_component':
                        latest=page.latest
                        broken=replace(recipe,components=(replace(recipe.components[0],component_value_id=-100),))
                        changed.setattr(page,'latest',lambda r:broken if r==tuple(row) else latest(r))
                    else:
                        edges=book.edges_into
                        changed.setattr(book,'edges_into',lambda c:() if c==cell else edges(c))
                    assert original(graph,node_id)==()
                    checked.append(True)
                    raise RuntimeError("literal proof checked")
        return original(graph,node_id)
    monkeypatch.setattr(planner,'_resident_string_comparison_cells',inspect)
    contract=ExtractionContract(Path(__file__).resolve().parents[1]/'extraction_contracts/program_extraction.yaml')
    with pytest.raises(RuntimeError, match="literal proof checked"):
        lower_ast_source_to_ssa("def root(a: str):\n    return a == 'master'\n",'root',extraction_contract=contract,progress=lambda _:None,progress_bars=False)
    assert checked
