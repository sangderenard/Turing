"""Input-bounded keyed records retain separate capacity and logical length."""
from pathlib import Path
import numpy as np
import pytest
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable
from src.compiler.ssa_storage_requirements import public_sequence_capacity_sources

SOURCE = '''from dataclasses import dataclass
@dataclass
class Result:
    registry: dict[int, int]

def root(xs: list[int]) -> Result:
    entries = {x: x * 2 for x in xs if x > 0}
    return Result(entries)
'''

@pytest.mark.parametrize('backend', ['llvm', 'c'])
def test_keyed_result_capacity_is_not_its_length(tmp_path, backend):
    module, _, exports = lower_ast_source_to_ssa(
        SOURCE, 'root', extraction_contract=ExtractionContract(
            Path(__file__).resolve().parents[1]/'extraction_contracts/program_extraction.yaml'),
        progress=lambda _: None, progress_bars=False)
    name = exports[0]
    function = module.functions[name]
    relations = public_sequence_capacity_sources(module, name)
    assert len(relations) == 2
    assert relations[0].output_length_id == relations[1].output_length_id
    source_id = dict(function.metadata['parameter_names'])['xs']
    incoming = module.sequence_tables[name].sequences[source_id]
    if backend == 'llvm':
        from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution
        artifact = emit_ssa_function_to_llvm(module, name)
        assert artifact.shortfalls == ()
        artifact = compile_artifact(artifact, directory=tmp_path/'llvm')
        prepare = lambda feeds: prepare_artifact_execution(artifact, feeds)
    else:
        from src.compiler.ssa_c_backend import emit_ssa_module_to_c
        artifact = emit_ssa_module_to_c(module, name)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path/'c')
        prepare = artifact.prepare_execution
    assert CorrelationTable.build(module).findings(module) == []
    for data in ([], [-4, 0, -2], [1, 2, 3, 4], [3, -1, 3, 2, 0, 2]):
        values = np.array(data, dtype=np.int64)
        feeds = {source_id: values,
                 incoming.length_address_id: np.array([len(data)], dtype=np.int64),
                 incoming.capacity_value_id: np.array([len(data)], dtype=np.int64),
                 incoming.status_address_id: np.array([0], dtype=np.int64)}
        execution = prepare(feeds)
        for length in (len(data), 0, len(data)):
            execution.buffers[incoming.length_address_id][...] = length
            execution.run()
            keys = execution.buffers[relations[0].output_data_id]
            items = execution.buffers[relations[1].output_data_id]
            expected = {x: x * 2 for x in data[:length] if x > 0}
            assert dict(zip(keys.tolist(), items.tolist())) == expected
            assert len(keys) == len(items) == len(expected)
            np.testing.assert_array_equal(values, np.array(data, dtype=np.int64))
        retained = keys, items
        del execution
        assert dict(zip(retained[0].tolist(), retained[1].tolist())) == expected
    # The bound is not permission for an unknown runtime callee to mutate
    # the returned arena. The consumer independently closes column uses.
    from src.transmogrifier.ssa import Instr
    block = next(iter(function.blocks.values()))
    column = next(arg for arg in function.args if arg.id == relations[0].output_data_id)
    escape = Instr('Call', [column], None, attributes={'callee': 'unknown_mutator'})
    block.instrs.append(escape)
    try:
        assert public_sequence_capacity_sources(module, name) == ()
    finally:
        block.instrs.remove(escape)
