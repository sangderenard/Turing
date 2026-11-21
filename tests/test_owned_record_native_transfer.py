"""Native lifetime regression, gated before touching any produced pointer."""
import numpy as np
import pytest
from tests.test_owned_record_transfer_plan import _lower, SOURCE
from src.compiler.source_owned_record_transfer import materialize_owned_record_transfers
from src.compiler.ssa_storage_requirements import public_owned_sequence_returns
from src.compiler.source_owned_record_projection_validation import publish_record_owned_public_returns


@pytest.mark.parametrize('backend',['llvm','c'])
def test_selected_record_owner_retains_copies_and_releases_on_host_copy_failure(backend,tmp_path,monkeypatch):
    module,name = _lower(SOURCE)
    assert len(materialize_owned_record_transfers(module,name)) == 1
    assert len(publish_record_owned_public_returns(module,name)) == 1
    ports = public_owned_sequence_returns(module,name)
    assert len(ports) == 1, 'no validated public ownership grant: do not execute native result'
    port = ports[0]
    if backend == 'llvm':
        from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm,compile_artifact,prepare_artifact_execution
        artifact=emit_ssa_function_to_llvm(module,name)
        assert artifact.complete,artifact.shortfalls
        artifact=compile_artifact(artifact,directory=tmp_path/'llvm')
        prepare=lambda feeds:prepare_artifact_execution(artifact,feeds)
    else:
        from src.compiler.ssa_c_backend import emit_ssa_module_to_c
        artifact=emit_ssa_module_to_c(module,name)
        assert artifact.complete,artifact.shortfalls
        artifact.compile(tmp_path/'c')
        prepare=artifact.prepare_execution
    parameters=dict(module.functions[name].metadata['parameter_names'])
    execution=prepare({parameters['n']:np.array(0,dtype=np.int64),
                       parameters['choose']:np.array(True,dtype=np.bool_),
                       parameters['fallback']:np.array(False,dtype=np.bool_)})
    retained=[]
    expected=[]
    for n,choose,fallback in ((0,True,False),(4,False,False),(2,True,False),
                              (0,False,False),(3,False,True),(1,False,False),
                              (129,True,False),(129,False,False),(32,False,True),
                              (-2,True,False),(-2,False,False)):
        for key,value in (('n',n),('choose',choose),('fallback',fallback)):
            execution.buffers[parameters[key]][...]=value
        execution.run()
        expected=list(range(n)) if choose or fallback else [-7,*(-i for i in range(n))]
        result=execution.buffers[port.data_id]
        assert result.tolist()==expected
        assert execution._owned_result_cells[port.owner_id].item()==0
        assert execution._owned_result_cells[port.data_id].item()==0
        retained.append((result,expected))
        for old,values in retained:
            assert old.tolist()==values
    pointers=execution.pointers if backend=='llvm' else execution._pointers
    artifact.release_owned_results(pointers)
    artifact.release_owned_results(pointers)
    from src.compiler import runtime_abi
    def refuse_copy(*args):
        raise ValueError('deliberate owned record host-copy failure')
    with monkeypatch.context() as patch:
        patch.setattr(runtime_abi,'copy_owned_sequence_results',refuse_copy)
        with pytest.raises(ValueError,match='deliberate owned record host-copy failure'):
            execution.run()
    assert execution._owned_result_cells[port.owner_id].item()==0
    execution.run()
    assert execution.buffers[port.data_id].tolist()==expected
    artifact.release_owned_results(pointers)
    del execution
    for old,values in retained:
        assert old.tolist()==values
