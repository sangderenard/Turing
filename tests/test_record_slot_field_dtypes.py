"""A record's scalar field slot takes the dtype its field declares.

Shrunk from the string-core structure driver ``structure_grow``, which
passes ``vocabulary.epoch`` of a local ``Vocabulary()`` to an unannotated
``fold``.  The field read was consumed only as a call actual, and only a
receiver PARAMETER's record fields were read for a dtype, so the slot ABI
refused the local record's field as having none.  The program runs natively
(C and LLVM) and must equal Python exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "extraction_contracts" / "program_extraction.yaml"
)

LOCAL_FIELD_ACTUAL = '''
from array import array


class V:
    def __init__(self):
        self.epoch = 0
        self.created = array("Q")

    def add(self, x):
        self.epoch += 1
        self.created.append(self.epoch)
        return 0, x + len(self.created)


def fold(d, v):
    return d * 7 + v + 1


def root(n: int) -> int:
    v = V()
    d = 0
    for i in range(n % 5):
        status, t = v.add(i)
        d = fold(d, t + status)
    return fold(d, v.epoch)
'''

PROGRAMS = {
    "local_field_actual": (LOCAL_FIELD_ACTUAL, (0, 1, 3, 4, 9)),
}


def _python(source, n):
    namespace: dict = {"__name__": "record_slot_field_dtypes"}
    exec(compile(source, "<record_slot_field_dtypes>", "exec"), namespace)
    return namespace["root"](n)


def _lowered(name, source):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name=f"record_slot_{name}",
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
def test_record_field_slot_matches_python_natively(name, tmp_path):
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
