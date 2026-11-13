"""Explicit native null values are not arbitrary integer-to-pointer casts."""
import numpy as np
import pytest
from tests.test_native_owner_pointer_snapshot import _snapshot_module
from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution


@pytest.mark.parametrize('backend', ['llvm', 'c'])
@pytest.mark.parametrize('payload', [0])
def test_explicit_scalar_pointer_null_clears_owner_without_reloading_snapshot(backend, payload, tmp_path):
    module, name = _snapshot_module()
    operation = module.functions[name].blocks['body'].instrs[1]
    operation.op = 'Const'
    operation.args = []
    operation.attributes = {'value':payload}
    if backend == 'llvm':
        artifact = emit_ssa_function_to_llvm(module,name)
        assert artifact.complete, artifact.shortfalls
        assert 'store ptr null,' in artifact.llvm_ir
        native = compile_artifact(artifact,directory=tmp_path/'null')
        prepare = lambda feeds: prepare_artifact_execution(native,feeds)
    else:
        from src.compiler.ssa_c_backend import emit_ssa_module_to_c
        artifact=emit_ssa_module_to_c(module,name)
        assert artifact.complete,artifact.shortfalls
        artifact.compile(tmp_path/'null')
        prepare=artifact.prepare_execution
    for value in (0,0x12345678,0x87654321):
        source=np.array([value],dtype=np.uint64)
        destination=np.zeros(1,dtype=np.uint64)
        prepare({1:source,2:destination,5:np.zeros(1,dtype=np.uint64)}).run()
        assert source[0] == 0 and destination[0] == value


@pytest.mark.parametrize('payload', [None, 1, -1, 0.0, '0'])
def test_non_null_or_coerced_pointer_literal_is_not_admitted(payload):
    module,name=_snapshot_module()
    op=module.functions[name].blocks['body'].instrs[1]
    op.op='Const';op.args=[];op.attributes={'value':payload}
    artifact=emit_ssa_function_to_llvm(module,name)
    assert any('explicit null and scalar physical receipt' in x.reason for x in artifact.shortfalls)


def test_null_without_scalar_pointer_receipt_is_not_admitted():
    module,name=_snapshot_module(receipt=False)
    op=module.functions[name].blocks['body'].instrs[1]
    op.op='Const';op.args=[];op.attributes={'value':0}
    artifact=emit_ssa_function_to_llvm(module,name)
    assert any('explicit null and scalar physical receipt' in x.reason for x in artifact.shortfalls)


def test_python_none_is_not_a_native_null_pointer_literal_in_c():
    from src.compiler.ssa_c_backend import emit_ssa_module_to_c
    module,name=_snapshot_module()
    op=module.functions[name].blocks['body'].instrs[1]
    op.op='Const';op.args=[];op.attributes={'value':None}
    artifact=emit_ssa_module_to_c(module,name)
    assert any('None' in x.reason for x in artifact.shortfalls)
