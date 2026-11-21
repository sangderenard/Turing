from pathlib import Path
import numpy as np
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution
from src.compiler.ssa_storage_requirements import public_sequence_capacity_sources

SOURCE = """def root(packet: bytes, at: int, count: int) -> tuple[bytes, bytes, bytes]:
    return packet[:at], packet[at:at + count], packet[at + count:]
"""

def test_llvm_public_byte_slice_capacity_and_written_lengths(tmp_path):
    contract = ExtractionContract(Path(__file__).resolve().parents[1] / 'extraction_contracts/program_extraction.yaml').with_program_abi({
        'values': [{'function':'root','parameter':'packet','storage':'span','dtype':'int64','rank':1,'python_type':'builtins.bytes'},
                   *({'function':'root','parameter':name,'storage':'scalar','dtype':'int64','rank':0,'shape':[],'python_type':'builtins.int'} for name in ('at','count'))],
        'records':{}, 'bindings':[]})
    module, _, exports = lower_ast_source_to_ssa(SOURCE, 'root', extraction_contract=contract, progress=lambda _:None, progress_bars=False)
    name = exports[0]
    relations = public_sequence_capacity_sources(module, name)
    assert len(relations) == 3
    artifact = emit_ssa_function_to_llvm(module, name)
    assert artifact.shortfalls == ()
    from src.compiler.identity_concordance import CorrelationTable
    assert CorrelationTable.build(module).findings(module) == []
    artifact = compile_artifact(artifact, directory=tmp_path/'llvm')
    parameters = dict(module.functions[name].metadata['parameter_names'])
    for packet, at, count in ((b'',0,0),(bytes(range(32)),3,10),(bytes(range(9)),99,5),(bytes(range(9)),-4,2),(bytes((0,127,128,255)),1,2)):
        execution = prepare_artifact_execution(artifact, {parameters['packet']:np.frombuffer(packet,dtype=np.uint8), parameters['at']:at, parameters['count']:count})
        for _ in range(2):
            execution.run()
            assert tuple(bytes(execution.buffers[r.output_data_id].tolist()) for r in relations) == (packet[:at],packet[at:at+count],packet[at+count:])
    # Reusing one native pointer table may shrink and then grow results.
    feeds = {parameters['packet']:np.arange(31,dtype=np.int64),
             parameters['at']:np.array(2,dtype=np.int64),
             parameters['count']:np.array(3,dtype=np.int64)}
    execution = prepare_artifact_execution(artifact, feeds)
    retained = None
    for at, count in ((2,3),(19,8),(0,0),(31,0),(1,29)):
        execution.buffers[parameters['at']][...] = at
        execution.buffers[parameters['count']][...] = count
        execution.run()
        packet = bytes(range(31))
        assert tuple(bytes(execution.buffers[r.output_data_id].tolist()) for r in relations) == (packet[:at],packet[at:at+count],packet[at+count:])
        retained = tuple(execution.buffers[r.output_data_id] for r in relations)
    del execution
    assert tuple(bytes(a.tolist()) for a in retained) == (packet[:1],packet[1:30],packet[30:])
    import pytest
    with pytest.raises(ValueError, match='capacity override disagrees'):
        prepare_artifact_execution(artifact, feeds, shapes={relations[0].output_data_id:(1,)})
    original = artifact.output_length_sources
    artifact.output_length_sources = ((original[0][0],original[1][1]), (original[1][0],original[0][1]), *original[2:])
    from src.compiler.identity_concordance import ConcordanceRefusal
    import pytest
    with pytest.raises(ConcordanceRefusal): artifact.runtime_abi()
