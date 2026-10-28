"""Repeated primitive class domains keep per-copy class/ABI relations."""
from pathlib import Path
import pytest

SOURCE = '''def validate(offsets: list[int]):
    previous=0
    for offset in offsets:
        if isinstance(offset,bool):
            return -1
        previous=offset
    return previous

def root():
    first=validate([0,1])
    second=validate([False,True])
    third=validate([0,2])
    return first+second+third
'''


@pytest.mark.parametrize("backend", ["c", "llvm", "fortran"])
def test_three_calls_retain_current_class_projection_native(backend, tmp_path):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa, compile_fortran_module_c_shell
    from src.compiler.identity_concordance import CorrelationTable
    module, outputs, exports = lower_ast_source_to_ssa(
        SOURCE, "root", name="multicall_class_projection",
        extraction_contract=Path(__file__).resolve().parents[1] / "extraction_contracts/program_extraction.yaml",
        progress=lambda _: None, progress_bars=False,
    )
    assert CorrelationTable.build(module).findings(module) == []
    root = exports[0]
    if backend == "c":
        from src.compiler.ssa_c_backend import emit_ssa_to_c
        artifact = emit_ssa_to_c(module, root)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path)
        execution = artifact.prepare_execution({})
    elif backend == "llvm":
        from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution
        artifact = emit_ssa_function_to_llvm(module, root)
        assert artifact.complete, artifact.shortfalls
        execution = prepare_artifact_execution(compile_artifact(artifact, directory=tmp_path), {})
    else:
        from src.compiler.ssa_fortran_backend import emit_module, FortranCoreNative, fortran_compiler
        if fortran_compiler() is None:
            pytest.skip("Fortran compiler unavailable")
        artifact = emit_module(module, name="multicall_class_projection", outputs={root: outputs[root]}, extra_roots=(root,), progress=lambda _: None)
        assert artifact.complete, artifact.shortfalls
        native = compile_fortran_module_c_shell(artifact, {}, tmp_path, entrypoint=root, library=True)
        execution = FortranCoreNative(native.executable_path, artifact.api.entry_point(root), artifact.api).prepare_execution({})
    execution.run()
    assert execution.buffers[outputs[root][0].id].item() == 2
