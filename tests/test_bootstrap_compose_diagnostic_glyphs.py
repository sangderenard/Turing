"""Bootstrap id_space.compose owns its validation format-spec recipes."""
from pathlib import Path

import numpy as np
import pytest

from src.compiler.fortran_c_shell import (
    lower_ast_source_to_ssa, compile_fortran_module_c_shell,
)
from src.compiler.id_space import MINTED, SHARED, SERIAL_BITS
from src.compiler.identity_concordance import CorrelationTable
from src.compiler.ssa_c_backend import emit_ssa_to_c
from src.compiler.ssa_llvm_backend import (
    emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution,
)
from src.compiler.ssa_fortran_backend import (
    emit_module, FortranCoreNative, fortran_compiler,
)


SOURCE = '''from src.compiler.id_space import compose

def root(serial: int, flags: int):
    return compose(serial, flags)
'''


NESTED_FORMAT_SPEC = '''def root(serial: int, flags: int):
    if serial < 0:
        raise ValueError(f"{serial:{flags}}")
    return serial | flags
'''


@pytest.fixture(scope='module', params=[
    (SOURCE, ((-1, 0), (SERIAL_BITS + 1, 0), (0, 1))),
    (NESTED_FORMAT_SPEC, ((-1, 0),)),
], ids=['real-compose', 'nested-format-spec'])
def compose_case(request):
    source, invalid_inputs = request.param
    module, outputs, exports = lower_ast_source_to_ssa(
        source, 'root', name='bootstrap_compose_diagnostic',
        extraction_contract=(Path(__file__).resolve().parents[1]
                             / 'extraction_contracts/program_extraction.yaml'),
        progress=lambda _: None, progress_bars=False,
    )
    assert CorrelationTable.build(module).findings(module) == []
    root = module.functions[exports[0]]
    parameters = dict(root.metadata['parameter_names'])
    returns = [instruction.args for block in root.blocks.values()
               for instruction in block.instrs if instruction.op == 'Ret']
    assert len(returns) == 1 and len(returns[0]) == 1
    return module, outputs, exports, parameters, returns[0][0].id, invalid_inputs


@pytest.mark.parametrize('lane', ['C', 'LLVM', 'Fortran'])
def test_compose_diagnostic_format_spec_has_no_native_arena(compose_case, lane, tmp_path):
    # Each lane consumes the very same canonical module held by the fixture.
    module, outputs, exports, parameters, result_id, invalid_inputs = compose_case
    if lane == 'C':
        artifact = emit_ssa_to_c(module, exports[0])
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path / 'native')
        prepare_feeds = artifact.prepare_execution
    elif lane == 'LLVM':
        artifact = emit_ssa_function_to_llvm(module, exports[0])
        assert artifact.complete, artifact.shortfalls
        built = compile_artifact(artifact, directory=tmp_path / 'native')
        prepare_feeds = lambda feeds: prepare_artifact_execution(built, feeds)
    else:
        if fortran_compiler() is None:
            pytest.skip('native Fortran compiler unavailable')
        artifact = emit_module(module, name='bootstrap_compose_native',
                               outputs={name: outputs[name] for name in exports},
                               extra_roots=exports,
                               progress=lambda _: None)
        assert artifact.complete, artifact.shortfalls
        built = compile_fortran_module_c_shell(
            artifact, {}, tmp_path / 'native', entrypoint=exports[0], library=True,
        )
        core = FortranCoreNative(
            built.executable_path, artifact.api.entry_point(exports[0]), artifact.api,
        )
        prepare_feeds = core.prepare_execution
        # Fortran retains its terminal native guard. Executing ERROR STOP in
        # the pytest process would terminate it, so invalid execution is
        # checked on C/LLVM below; Fortran executes all valid exact cases.
        assert 'error stop' in artifact.source.lower()

    def prepare(serial, flags):
        return prepare_feeds({
            parameters['serial']: np.array([serial], dtype=np.int64),
            parameters['flags']: np.array([flags], dtype=np.int64),
        })

    # These native outputs must retain every bit, including a serial that
    # binary64 cannot represent. This calls the real authored compose body.
    for serial, flags in ((2**53 + 1, MINTED),
                          (SERIAL_BITS, MINTED | SHARED), (0, 0)):
        buffers = prepare(serial, flags).run().buffers
        assert buffers[result_id].dtype == np.dtype('int64')
        assert buffers[result_id].item() == serial | flags
    # Removing diagnostic-only glyph work must not remove the owning guard.
    if lane != 'Fortran':
        for serial, flags in invalid_inputs:
            with pytest.raises(RuntimeError, match='validation failed with code'):
                prepare(serial, flags).run()


@pytest.mark.parametrize('selection', ['outputs', 'extra_roots'])
def test_fortran_explicit_root_does_not_export_uncalled_control_heap_return(selection):
    from src.transmogrifier.ssa import SSAValue, Instr, Function, BasicBlock, IRModule

    value = SSAValue(0, 'int64')
    root = Function('selected_entry', [value], {
        'entry': BasicBlock('entry', [Instr('Ret', [value], None)]),
    }, metadata={'control_ir': True})
    arena = SSAValue(0, 'ptr', accounting={'heap_element_dtype': 'int64'})
    uncalled = Function('uncalled_heap_return', [arena], {
        'entry': BasicBlock('entry', [Instr('Ret', [arena], None)]),
    }, metadata={'control_ir': True})
    module = IRModule({
        root.name: root, uncalled.name: uncalled,
    })
    kwargs = ({'outputs': {root.name: (value,)}}
              if selection == 'outputs' else {'extra_roots': (root.name,)})
    selected = emit_module(module, name='selected_surface', progress=lambda _: None, **kwargs)
    assert selected.complete, selected.shortfalls
    assert {subroutine.name for subroutine in selected.subroutines} == {root.name}

    # With no selected public interface, retain the existing implicit whole
    # control surface and its exact heap-escape refusal. An empty outputs map
    # also preserves that control-root fallback; it does not select an entry.
    for implicit in ({}, {'outputs': {}}):
        whole = emit_module(module, name='whole_surface', progress=lambda _: None, **implicit)
        assert {subroutine.name for subroutine in whole.subroutines} == {root.name, uncalled.name}
        assert not whole.complete
        assert any('entry releases its heap' in item.reason for item in whole.shortfalls)
