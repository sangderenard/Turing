"""Owned record results must publish data length and lifetime, not a pointer cell.

The ABI assertions deliberately precede native execution: a missing escape
contract is a deterministic regression, not permission to dereference a freed
address. Native retention assertions become reachable when that contract exists.
"""
from pathlib import Path
import numpy as np
import pytest
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_storage_requirements import public_owned_sequence_returns

from tests.test_owned_record_return_source import SOURCE


@pytest.mark.parametrize("backend", ["llvm", "c"])
def test_nested_owned_record_return_has_length_and_retained_native_values(tmp_path, backend):
    module, outputs, exports = lower_ast_source_to_ssa(
        SOURCE, 'root', extraction_contract=ExtractionContract(
            Path(__file__).resolve().parents[1]/'extraction_contracts/program_extraction.yaml'),
        progress=lambda _: None, progress_bars=False)
    name = exports[0]
    ports = public_owned_sequence_returns(module, name)
    assert len(ports) == 1, "selected owned record field lacks exact public length/owner escape contract"
    port = ports[0]
    assert len({port.data_id, port.length_id, port.capacity_id, port.owner_id}) == 4
    assert port.data_id in [value.id for value in outputs[name]]
    book = module.metadata['identity_book']
    projection = book.page(port.source_cell.page).cell(port.source_cell.row, port.source_cell.column)
    assert projection.source_sequence_id not in [value.id for value in outputs[name]]
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
    from src.compiler.identity_concordance import CorrelationTable
    assert CorrelationTable.build(module).findings(module) == []
    parameters = dict(module.functions[name].metadata['parameter_names'])
    execution = prepare({
        parameters['n']: np.array(0, dtype=np.int64),
        parameters['choose']: np.array(True, dtype=np.bool_),
        parameters['fallback']: np.array(False, dtype=np.bool_),
    })
    retained = []
    for n, choose, fallback in ((0, True, False), (4, False, False),
                                (2, True, False), (0, False, False),
                                (3, False, True), (1, False, False),
                                (129, True, False), (129, False, False), (-2, True, False)):
        for key, value in (('n', n), ('choose', choose), ('fallback', fallback)):
            execution.buffers[parameters[key]][...] = value
        execution.run()
        expected = list(range(n)) if choose or fallback else [-7, *(-i for i in range(n))]
        result = execution.buffers[port.data_id]
        assert result.tolist() == expected
        retained.append((result, expected))
        for previous, values in retained:
            assert previous.tolist() == values
    del execution
    for previous, values in retained:
        assert previous.tolist() == values
