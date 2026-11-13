"""Nested branch aliases select a resident descriptor, not only its data pointer."""
from types import SimpleNamespace
import numpy as np
import pytest
from src.compiler.precompile_to_ssa import _ControlSSABuilder
from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.extraction_contract import ExtractionContract


def test_descriptor_selection_requires_exact_planning_and_operand_identity():
    builder = object.__new__(_ControlSSABuilder)
    builder.function_name = "probe"
    builder.concorded_value_aliases = {97: 104}
    builder.sequence_descriptors = {104: SimpleNamespace(column_value_ids=(104,))}
    assert builder._conditional_sequence_descriptor_id(97, SimpleNamespace(id=104)) == 104
    assert builder._conditional_sequence_descriptor_id(98, SimpleNamespace(id=104)) is None
    assert builder._conditional_sequence_descriptor_id(97, SimpleNamespace(id=106)) is None
    assert 97 not in builder.sequence_descriptors
    builder.concorded_value_aliases = {97: 98, 98: 97}
    with pytest.raises(ValueError, match="cyclic durable"):
        builder._conditional_sequence_descriptor_id(97, SimpleNamespace(id=104))


@pytest.mark.parametrize("mutate", [False, True])
def test_nested_branch_descriptor_native_length_and_mutation(tmp_path, mutate):
    source = """def root(n: int, choose: bool) -> int:
    if choose:
        out = [n, 2, 3]
    else:
        out = [n]
        if n < 0:
            out = [4, 5]
    MUTATION
    return len(out) * 100 + out[0]
""".replace("    MUTATION", "    out[0] = out[0] + 7" if mutate else "")
    contract = ExtractionContract('extraction_contracts/program_extraction.yaml').with_execution_file('extraction_contracts/vehicle_full_native_execution.yaml')
    module, _, exports = lower_ast_source_to_ssa(source, 'root', extraction_contract=contract, progress=lambda _: None, progress_bars=False)
    artifact = emit_ssa_function_to_llvm(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    from src.compiler.identity_concordance import CorrelationTable
    assert CorrelationTable.build(module).findings(module) == []
    artifact = compile_artifact(artifact, directory=tmp_path/'native')
    parameters = dict(module.functions[exports[0]].metadata['parameter_names'])
    output = next(i.args[0].id for b in module.functions[exports[0]].blocks.values() for i in b.instrs if i.op == 'Ret')
    namespace = {}; exec(source, namespace)
    for n, choose in ((3, True), (3, False), (-2, False), (-2, True), (0, False)):
        execution = prepare_artifact_execution(artifact, {parameters['n']: n, parameters['choose']: choose})
        for _ in range(2):
            execution.run()
            assert execution.buffers[output].item() == namespace['root'](n, choose)
