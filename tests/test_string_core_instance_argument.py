"""An instance passed to an unannotated helper that calls its method.

Shrunk from the string-core structure driver ``structure_grow``:
``fold_cover(vocabulary, d, token)`` calls ``vocabulary.cover(token)`` and
folds the returned ``(status, pieces)``.  The construction ``V()`` stamped
its node's ``class_ref`` but posted no class row, so the call-actual edge
had no class to carry to the unannotated formal: ``vocabulary.cover``
resolved to no method and ``status`` had no settled dtype.  The program runs
natively (C and LLVM) and must equal Python exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "extraction_contracts" / "program_extraction.yaml"
)

HELPER_METHOD_CALL = '''
class V:
    def __init__(self):
        self.k = 3

    def cover(self, token):
        if token == 0:
            return 9, []
        return 0, [token, token + self.k]


def fold_cover(vocabulary, d, token):
    status, cover = vocabulary.cover(token)
    d = d * 7 + status
    for piece in cover:
        d = d * 7 + piece
    return d


def root(n: int) -> int:
    v = V()
    return fold_cover(v, n % 3, n % 4)
'''

PROGRAMS = {
    "helper_method_call": (HELPER_METHOD_CALL, (0, 1, 2, 5, 8)),
}


def _python(source, n):
    namespace: dict = {"__name__": "instance_argument"}
    exec(compile(source, "<instance_argument>", "exec"), namespace)
    return namespace["root"](n)


def _lowered(name, source):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name=f"instance_argument_{name}",
        extraction_contract=str(CONTRACT), progress=lambda _message: None,
    )
    root = module.functions[exports[0]]
    outputs = [
        int(value.id)
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret" and not instruction.attributes.get("fault_exit")
        for value in instruction.args
    ]
    return module, exports[0], dict(root.metadata["parameter_names"]), outputs


@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_instance_passed_to_helper_matches_python_natively(name, tmp_path):
    from src.compiler.ssa_c_backend import emit_ssa_to_c
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )

    source, inputs = PROGRAMS[name]
    module, export, parameters, outputs = _lowered(name, source)
    artifact = emit_ssa_to_c(module, export)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "c")
    emitted = emit_ssa_function_to_llvm(module, export)
    assert not emitted.shortfalls, emitted.shortfalls
    llvm = compile_artifact(emitted, directory=tmp_path / "llvm")
    for n in inputs:
        feeds = {parameters["n"]: np.array([n], dtype=np.int64)}
        expected = _python(source, n)
        c_value = artifact.prepare_execution(feeds).run().buffers[outputs[0]]
        llvm_value = prepare_artifact_execution(llvm, feeds).run().buffers[outputs[0]]
        assert int(c_value.reshape(-1)[0]) == expected, (name, "c", n)
        assert int(llvm_value.reshape(-1)[0]) == expected, (name, "llvm", n)

