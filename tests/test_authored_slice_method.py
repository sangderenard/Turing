"""A resolved method called slice is a call, never an AST slice literal."""
from pathlib import Path

import numpy as np
import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts/program_extraction.yaml"
SOURCES = {
    "runtime": '''
class Box:
    def slice(self, token: int, begin: int, end: int):
        return 0, token + begin + end

def root(n: int):
    box = Box()
    status, token = box.slice(n, 2, 3)
    return status + token
''',
    "literal": '''
class Box:
    def slice(self, token: int, begin: int, end: int):
        return 0, token + begin + end

def root(n: int):
    box = Box()
    status, token = box.slice(3, 2, 3)
    return n + status + token
''',
    "builtin": '''
def root(n: int):
    values = [3, 5, 7]
    return values[1:2][0] + n
''',
}


@pytest.mark.parametrize("case", SOURCES)
@pytest.mark.parametrize("backend", ("c", "llvm"))
def test_slice_method_keeps_its_authored_call_identity(case, backend, tmp_path):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.identity_concordance import CorrelationTable
    module, _, exports = lower_ast_source_to_ssa(
        SOURCES[case], "root", name="authored_slice", extraction_contract=str(CONTRACT), progress=lambda _: None)
    assert CorrelationTable.build(module).findings(module) == []
    root = module.functions[exports[0]]
    outputs = [int(v.id) for block in root.blocks.values() for ins in block.instrs
               if ins.op == "Ret" for v in ins.args]
    parameters = dict(root.metadata["parameter_names"])
    if backend == "c":
        from src.compiler.ssa_c_backend import emit_ssa_to_c
        artifact = emit_ssa_to_c(module, exports[0])
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path)
        prepare = artifact.prepare_execution
    else:
        from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution
        emitted = emit_ssa_function_to_llvm(module, exports[0])
        assert not emitted.shortfalls, emitted.shortfalls
        artifact = compile_artifact(emitted, directory=tmp_path)
        prepare = lambda feeds: prepare_artifact_execution(artifact, feeds)
    for n in (0, 7, -3):
        execution = prepare({parameters["n"]: np.asarray([n], dtype=np.int64)}).run()
        actual = int(execution.buffers[outputs[0]].reshape(-1)[0])
        assert actual == n + (8 if case == "literal" else 5)
