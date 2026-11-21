"""Folded For/While seeds retain their posted scalar physical domains."""
from pathlib import Path
import numpy as np
import pytest
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable


@pytest.mark.parametrize('backend', ['c', 'llvm'])
@pytest.mark.parametrize('loop', ['for', 'while'])
@pytest.mark.parametrize('annotation,seed,update', [
    ('int', '4294967296', 'value + 1'),
    ('float', '1.25', 'value + 0.5'),
    ('bool', 'False', 'i == 0'),
])
def test_folded_loop_seed_native(tmp_path, backend, loop, annotation, seed, update):
    if loop == 'for':
        body = f'    for i in range(n):\n        value = {update}\n'
    else:
        body = f'    i = 0\n    while i < n:\n        value = {update}\n        i += 1\n'
    source = f'def root(n: int) -> {annotation}:\n    value = {seed}\n' + body + '    return value\n'
    module, _, exports = lower_ast_source_to_ssa(source, 'root', extraction_contract=ExtractionContract(
        Path(__file__).parents[1] / 'extraction_contracts/program_extraction.yaml'),
        progress=lambda _: None, progress_bars=False)
    name = exports[0]
    function = module.functions[name]
    assert CorrelationTable.build(module).findings(module) == []
    from src.compiler.concordance_declarations import SSA_CONSTANT_MATERIALIZATION
    book = module.metadata['identity_book']
    receipts = [book.page(SSA_CONSTANT_MATERIALIZATION).latest(row)
                for row in book.page(SSA_CONSTANT_MATERIALIZATION).rows() if row[1] == name]
    expected_dtype = {'int': 'int64', 'float': 'float64', 'bool': 'bool'}[annotation]
    assert ('Const', expected_dtype, seed) in receipts
    if backend == 'c':
        from src.compiler.ssa_c_backend import emit_ssa_module_to_c
        artifact = emit_ssa_module_to_c(module, name)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path / 'c')
        prepare = artifact.prepare_execution
    else:
        from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution
        emitted = emit_ssa_function_to_llvm(module, name)
        assert emitted.shortfalls == ()
        artifact = compile_artifact(emitted, directory=tmp_path / 'llvm')
        prepare = lambda feeds: prepare_artifact_execution(artifact, feeds)
    parameter = dict(function.metadata['parameter_names'])['n']
    result = next(op.args[0].id for block in function.blocks.values() for op in block.instrs
                  if op.op == 'Ret' and not op.attributes.get('fault_exit'))
    for n in (0, 1, 2, 3, 0):
        execution = prepare({parameter: np.array([n], dtype=np.int64)}).run()
        expected = (4294967296+n if annotation == 'int' else
                    1.25+0.5*n if annotation == 'float' else n == 1)
        assert execution.buffers[result].reshape(-1)[0] == expected


def test_captured_recipe_does_not_publish_current_block_layout():
    from src.compiler.precompile_to_ssa import _ControlSSABuilder
    from src.transmogrifier.ssa import SSAValue
    class Captured:
        _captured_instruction_sink = []
        def _book(self):
            raise AssertionError('captured recipe must not publish to current block')
    builder = Captured()
    assert _ControlSSABuilder._publish_scalar_constant_layout(builder, SSAValue(1, 'int64'), 0) is False
    assert _ControlSSABuilder._publish_folded_seed_layout(builder, SSAValue(1, 'int64'), 0, 1, None) is None
