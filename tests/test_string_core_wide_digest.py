"""The structure drivers' digest over unsigned rows stays exact natively.

Shrunk from the string-core structure differential drivers, whose fold is
``d = (d * 1000003 + v + 7) % 4611686018427387847`` with ``v`` a token read
from an ``array("Q")`` row.  ``d * 1000003 + v`` is int128 wide; no
interval proves the sum (an ``array("Q")`` row has none), so ``... + 7``
fell to the int64 operator law and was truncated to 64 bits on C and LLVM
with no finding.  An operand wider than 64 bits now joins by exact value
domain, and the remainder takes its divisor's layout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "extraction_contracts" / "program_extraction.yaml"
)

UNSIGNED_ROW_DIGEST = '''
from array import array


def root(n: int) -> int:
    xs = array("Q")
    xs.append(5)
    xs.append(9223372036854775783)
    d = n % 1000
    d = (d * 1000003 + 9) % 4611686018427387847
    d = (d * 1000003 + 9) % 4611686018427387847
    d = (d * 1000003 + 9) % 4611686018427387847
    p = xs[n % 2]
    d = (d * 1000003 + p + 7) % 4611686018427387847
    return d
'''

PROGRAMS = {
    "unsigned_row_digest": (UNSIGNED_ROW_DIGEST, (0, 1, 2, 13)),
}


def _python(source, n):
    namespace: dict = {"__name__": "wide_digest"}
    exec(compile(source, "<wide_digest>", "exec"), namespace)
    return namespace["root"](n)


def _lowered(name, source):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name=f"wide_digest_{name}",
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
def test_wide_digest_matches_python_natively(name, tmp_path):
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
