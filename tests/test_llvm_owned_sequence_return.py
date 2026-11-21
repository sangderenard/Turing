"""LLVM preserves the existing direct owned sequence ABI across host materialization."""
from pathlib import Path
import numpy as np
import pytest
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_storage_requirements import public_owned_sequence_returns
from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution
from src.compiler.identity_concordance import CorrelationTable


def test_direct_bytes_owned_result_lifetime(tmp_path, monkeypatch):
    source = 'def root(a: bytes, b: bytes) -> bytes:\n    return a + b\n'
    contract = ExtractionContract(Path(__file__).resolve().parents[1]/'extraction_contracts/program_extraction.yaml').with_program_abi({
        'values': [{'function':'root','parameter':key,'storage':'span','dtype':'int64','rank':1,'python_type':'builtins.bytes'} for key in ('a','b')],
        'records':{}, 'bindings':[]})
    module, _, names = lower_ast_source_to_ssa(source, 'root', extraction_contract=contract, progress=lambda _:None, progress_bars=False)
    (ports,) = public_owned_sequence_returns(module, names[0])
    artifact = emit_ssa_function_to_llvm(module, names[0])
    assert artifact.complete, artifact.shortfalls
    assert artifact.output_heap_sequences == ((ports.data_id, ports.length_id, ports.capacity_id, ports.owner_id, ports.element_dtype),)
    assert CorrelationTable.build(module).findings(module) == []
    from src.compiler.concordance_declarations import EMISSION_ARTIFACT, ArtifactPart, Backend
    from src.transmogrifier.dtype_layout import DTYPES
    api = artifact.emission.book.page(EMISSION_ARTIFACT).latest((artifact.name, Backend.LLVM_MODULE, ArtifactPart.API_CONTRACT))
    pointer_buffers = [buffer for buffer in api.location[2] if buffer.dtype == 'uintp']
    assert pointer_buffers
    assert all(buffer.itemsize == DTYPES['ptr'].byte_size for buffer in pointer_buffers)
    assert 'uintp' not in DTYPES
    # Receipt tampering cannot turn an owner pointer into an ordinary output.
    from dataclasses import replace
    from src.compiler.concordance_declarations import PUBLIC_OWNED_SEQUENCE_RETURN
    page = module.metadata['identity_book'].page(PUBLIC_OWNED_SEQUENCE_RETURN)
    key = (ports.public_cell.row, ports.public_cell.column)
    original = page.cells[key]
    page.cells[key] = replace(original, owner_cell=ports.descriptor_cell)
    try:
        refused = emit_ssa_function_to_llvm(module, names[0])
        assert not refused.complete
        assert any('ownership escape contract' in item.reason for item in refused.shortfalls)
    finally:
        page.cells[key] = original
    # The host checks the exact runtime ABI before touching native buffers.
    from src.compiler.identity_concordance import ConcordanceRefusal
    forged = replace(artifact, output_heap_sequences=((ports.data_id, ports.length_id,
                     ports.capacity_id, ports.length_id, ports.element_dtype),))
    with pytest.raises((ConcordanceRefusal, ValueError)):
        forged.runtime_abi()
    artifact = compile_artifact(artifact, directory=tmp_path/'native')
    parameters = dict(module.functions[names[0]].metadata['parameter_names'])
    for a,b in ((b'',b''),(b'\x00\xff',b'abc'),(bytes(range(256)), b'x'*4097)):
        execution = prepare_artifact_execution(artifact, {parameters[k]:np.frombuffer(data,dtype=np.uint8) for k,data in (('a',a),('b',b))})
        retained = []
        for _ in range(3):
            execution.run()
            result = execution.buffers[ports.data_id]
            assert bytes(result.tolist()) == a+b
            assert execution.buffers[ports.length_id].item() == len(a+b)
            assert execution._owned_result_cells[ports.owner_id].item() == 0
            assert execution._owned_result_cells[ports.data_id].item() == 0
            retained.append(result)
        artifact.release_owned_results(execution.pointers)
        assert all(bytes(x.tolist()) == a+b for x in retained)
    from src.compiler import runtime_abi
    def fail(*args):
        raise ValueError('copy failed deliberately')
    with monkeypatch.context() as patch:
        patch.setattr(runtime_abi,'copy_owned_sequence_results',fail)
        with pytest.raises(ValueError,match='copy failed deliberately'):
            execution.run()
    assert execution._owned_result_cells[ports.owner_id].item() == 0
    execution.run()
    assert bytes(execution.buffers[ports.data_id].tolist()) == a+b


def test_borrowed_bytes_alias_return_keeps_incoming_storage(tmp_path):
    source = 'def root(a: bytes) -> bytes:\n    return a\n'
    contract = ExtractionContract(Path(__file__).resolve().parents[1]/'extraction_contracts/program_extraction.yaml').with_program_abi({
        'values':[{'function':'root','parameter':'a','storage':'span','dtype':'int64','rank':1,'python_type':'builtins.bytes'}],
        'records':{},'bindings':[]})
    module, _, names = lower_ast_source_to_ssa(source, 'root', extraction_contract=contract,
                                              progress=lambda _:None, progress_bars=False)
    artifact = emit_ssa_function_to_llvm(module, names[0])
    assert artifact.complete, artifact.shortfalls
    assert artifact.output_heap_sequences == ()
    assert artifact.release_entry_name is None
    artifact = compile_artifact(artifact, directory=tmp_path/'borrowed')
    parameter = dict(module.functions[names[0]].metadata['parameter_names'])['a']
    for values in ([], [0,255,128,37]):
        data = np.array(values,dtype=np.int64)
        execution = prepare_artifact_execution(artifact,{parameter:data})
        execution.run()
        np.testing.assert_array_equal(execution.buffers[parameter],data)
        assert np.shares_memory(execution.buffers[parameter],data) or len(data)==0
