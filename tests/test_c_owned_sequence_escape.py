"""C may not export an owned arena pointer as an ordinary one-cell array."""
from pathlib import Path
import numpy as np
import pytest
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_module_to_c


def test_selected_owned_record_return_requires_public_escape(tmp_path):
    from tests.test_owned_record_return_source import SOURCE
    from tests.owned_record_test_support import lower_before_owned_record_boundary
    m,_,names=lower_before_owned_record_boundary(SOURCE,'root',extraction_contract=ExtractionContract(
        Path(__file__).resolve().parents[1]/'extraction_contracts/program_extraction.yaml'),progress=lambda _:None,progress_bars=False)
    artifact=emit_ssa_module_to_c(m,names[0])
    assert not artifact.complete
    assert any('no validated public ownership escape contract' in f.reason for f in artifact.shortfalls)
    with pytest.raises(ValueError,match='ownership escape contract'):
        artifact.compile(tmp_path/'must-not-build')
    assert not (tmp_path/'must-not-build').exists()


@pytest.mark.parametrize('owned',[False,True],ids=['borrowed-alias','validated-owned-concat'])
def test_existing_bytes_returns_keep_their_native_contract(owned,tmp_path):
    expression='a+b' if owned else 'a'
    params=('a','b') if owned else ('a',)
    contract=ExtractionContract(Path(__file__).resolve().parents[1]/'extraction_contracts/program_extraction.yaml').with_program_abi({
        'values':[{'function':'root','parameter':p,'storage':'span','dtype':'int64','rank':1,'python_type':'builtins.bytes'} for p in params],
        'records':{},'bindings':[]})
    source='def root('+','.join(p+': bytes' for p in params)+') -> bytes:\n    return '+expression+'\n'
    m,_,names=lower_ast_source_to_ssa(source,'root',extraction_contract=contract,progress=lambda _:None,progress_bars=False)
    name=names[0]
    artifact=emit_ssa_module_to_c(m,name)
    assert artifact.complete,artifact.shortfalls
    assert bool(artifact.output_heap_sequences)==owned
    artifact.compile(tmp_path/'native')
    parameters=dict(m.functions[name].metadata['parameter_names'])
    output=next(i.args[0].id for block in m.functions[name].blocks.values() for i in block.instrs if i.op=='Ret')
    for a,b in ((b'',b''),(b'\x00\xff',b'xyz'),(bytes(range(256)),b'x'*257)):
        feeds={parameters['a']:np.frombuffer(a,dtype=np.uint8)}
        if owned:feeds[parameters['b']]=np.frombuffer(b,dtype=np.uint8)
        execution=artifact.prepare_execution(feeds)
        kept=[]
        for _ in range(2):
            execution.run()
            result=execution.buffers[output]
            assert bytes(result.tolist())==(a+b if owned else a)
            kept.append(result)
        assert all(bytes(r.tolist())==(a+b if owned else a) for r in kept)
