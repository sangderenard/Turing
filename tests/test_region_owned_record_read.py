"""Region-owned arithmetic retains the temporal identity of its record reads."""


import numpy as np
import pytest

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
def test_region_owned_expression_does_not_move_late_record_read(tmp_path, backend):
    from src.compiler.identity_concordance import CorrelationTable
    source = """class Clock:
    def __init__(self):
        self.epoch = 0
    def add(self, value: int) -> int:
        self.epoch = self.epoch + value
        return value

def root(n: int) -> int:
    clock = Clock()
    index = 0
    while index < n:
        value = clock.add(1)
        if value != 1:
            return -1
        index += 1
    before = clock.epoch
    value = clock.add(0)
    if value != 0:
        return -2
    if clock.epoch != before or before != n:
        return -3
    if clock.epoch != n:
        return -4
    if before != n:
        return -5
    return before + clock.epoch * 1000003
"""
    module, outputs, exports = lower_ast_source_to_ssa(
        source, "root", name="temporal_record_region",
        extraction_contract="extraction_contracts/program_extraction.yaml",
        progress=lambda _: None, progress_bars=False,
    )
    assert CorrelationTable.build(module).findings(module) == []
    # This multiplication belongs to the planned region. Recovery must
    # not manufacture a duplicate at function_exit that captures its field
    # read and strands the real earlier region Call's argument.
    assert not any(ins.op == "Mul" and ins.attributes.get("structural_operation")
                   for block in module.functions[exports[0]].blocks.values()
                   for ins in block.instrs)
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
    for n in (0, 1, 5):
        result = execute({parameter: np.asarray([n], dtype="int64")})
        assert result.buffers[output_id].item() == n * 1000004
