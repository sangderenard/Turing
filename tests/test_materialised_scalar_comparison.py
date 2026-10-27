"""Builtin bool arithmetic keeps Python's integer result ABI on native targets."""

import itertools
from pathlib import Path

import numpy as np
import pytest

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import (
    compile_fortran_module_c_shell,
    lower_ast_source_to_ssa,
)
from src.compiler.ssa_c_backend import emit_ssa_to_c
from src.compiler.ssa_fortran_backend import (
    FortranCoreNative,
    emit_module as emit_fortran_module,
    fortran_compiler,
)
from src.compiler.ssa_llvm_backend import (
    compile_artifact as compile_llvm_artifact,
    emit_ssa_function_to_llvm,
    prepare_artifact_execution as prepare_llvm_execution,
)


@pytest.mark.parametrize("backend", ["c", "llvm", "fortran"])
def test_materialised_bool_literal_numeric_comparison(tmp_path, backend):
    from src.compiler.identity_concordance import CorrelationTable
    source = "def root(n: int) -> bool:\n    return True != n\n"
    module, outputs, exports = lower_ast_source_to_ssa(
        source, "root", name="materialised_numeric_comparison",
        extraction_contract="extraction_contracts/program_extraction.yaml",
        progress=lambda _: None, progress_bars=False,
    )
    assert CorrelationTable.build(module).findings(module) == []
    casts = [ins for fn in module.functions.values()
             for block in fn.blocks.values() for ins in block.instrs
             if ins.op == "Cast" and ins.args[0].dtype == "bool"]
    assert casts and all(ins.res.dtype == "int64" for ins in casts)
    comparisons = [ins for fn in module.functions.values()
                   for block in fn.blocks.values() for ins in block.instrs
                   if ins.op == "Ne"]
    assert comparisons
    assert all(tuple(v.dtype for v in ins.args) == ("int64", "int64")
               for ins in comparisons)
    root = module.functions[exports[0]]
    parameter = dict(root.metadata["parameter_names"])["n"]
    output_id = outputs[exports[0]][0].id
    if backend == "c":
        artifact = emit_ssa_to_c(module, exports[0])
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path / "c-native")
        execute = lambda feeds: artifact.prepare_execution(feeds).run()
    elif backend == "llvm":
        artifact = emit_ssa_function_to_llvm(module, exports[0])
        assert artifact.complete, artifact.shortfalls
        artifact = compile_llvm_artifact(artifact, directory=tmp_path / "llvm-native")
        execute = lambda feeds: prepare_llvm_execution(artifact, feeds).run()
    else:
        if fortran_compiler() is None:
            pytest.skip("native Fortran compiler unavailable")
        artifact = emit_fortran_module(
            module, outputs={name: outputs[name] for name in exports},
            extra_roots=exports, progress=lambda _: None,
        )
        assert artifact.complete, artifact.shortfalls
        built = compile_fortran_module_c_shell(
            artifact, {}, tmp_path / "fortran-native",
            entrypoint=exports[0], library=True,
        )
        core = FortranCoreNative(
            built.executable_path, artifact.api.entry_point(exports[0]), artifact.api,
        )
        execute = lambda feeds: core.prepare_execution(feeds).run()
    for n, expected in ((0, True), (1, False), (2, True), (-1, True)):
        result = execute({parameter: np.asarray([n], dtype="int64")})
        assert bool(result.buffers[output_id].item()) is expected
