"""An imported constructor's exact lexical array callable owns its arena."""
from pathlib import Path
import sys
import pytest

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import concordance_report


@pytest.mark.parametrize('backend', ['c', 'llvm', 'fortran'])
def test_imported_constructor_retains_array_owner(backend, tmp_path, monkeypatch):
    authored = tmp_path / 'imported_array_birth.py'
    authored.write_text('''from array import array
class Buffer:
    def __init__(self):
        self.domain = array("B")
    def put(self, value: int):
        self.domain.append(value)
    def count(self):
        return len(self.domain)
''')
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, 'imported_array_birth', raising=False)
    contract = ExtractionContract(str(Path(__file__).resolve().parents[1] /
        'extraction_contracts/program_extraction.yaml')).with_sources((('imported_array_birth', authored),))
    source = '''from imported_array_birth import Buffer
def root():
    buffer = Buffer()
    buffer.put(255)
    buffer.put(128)
    return buffer.count()
'''
    module, outputs, exports = lower_ast_source_to_ssa(source, 'root', name='imported_array_owner',
        extraction_contract=contract, progress=lambda _: None, progress_bars=False)
    report = concordance_report(module)
    assert '0 finding(s)' in report, report
    ctor = next(name for name in module.functions if '____init__' in name)
    owned = [d for d in module.sequence_tables[ctor].sequences.values() if d.column_dtypes == ('uint8',)]
    assert len(owned) == 1
    assert owned[0].heap_owner_value_id is not None and owned[0].capacity_policy.value == 'dynamic'
    entry = next(name for name in exports if name.endswith('__root'))
    if backend == 'c':
        from src.compiler.ssa_c_backend import emit_ssa_to_c
        artifact = emit_ssa_to_c(module, entry)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path / 'native')
        execution = artifact.prepare_execution({})
    elif backend == 'llvm':
        from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution
        artifact = emit_ssa_function_to_llvm(module, entry)
        assert artifact.complete, artifact.shortfalls
        execution = prepare_artifact_execution(compile_artifact(artifact, directory=tmp_path / 'native'), {})
    else:
        from src.compiler.ssa_fortran_backend import emit_module, FortranCoreNative, fortran_compiler
        from src.compiler.fortran_c_shell import compile_fortran_module_c_shell
        if fortran_compiler() is None:
            pytest.skip('Fortran compiler unavailable')
        artifact = emit_module(module, name='imported_array_owner', outputs={entry: outputs[entry]}, extra_roots=(entry,), progress=lambda _: None)
        assert artifact.complete, artifact.shortfalls
        built = compile_fortran_module_c_shell(artifact, {}, tmp_path / 'native', entrypoint=entry, library=True)
        execution = FortranCoreNative(built.executable_path, artifact.api.entry_point(entry), artifact.api).prepare_execution({})
    for _ in range(3):
        execution.run()
        assert execution.buffers[outputs[entry][0].id].item() == 2
