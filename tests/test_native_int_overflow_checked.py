"""Python ``int`` arithmetic never wraps natively.

``a * b`` on ints no interval bounds is int64 arithmetic.  It wrapped at
int64 and returned a value Python never computes.  Until an overflowing
result continues as Precision limbs, both native lanes report it
(``native_faults.INT_OVERFLOW``) instead of returning the wrapped value.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"

SOURCE = '''
def mul(a: int, b: int) -> int:
    return a * b


def root(n: int):
    return mul(n, n) % 1000
'''


def _lowered():
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, exports = lower_ast_source_to_ssa(
        SOURCE, "root", name="int_overflow_checked",
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


def _run_c(tmp_path, n):
    from src.compiler.ssa_c_backend import emit_ssa_to_c

    module, export, parameters, outputs = _lowered()
    artifact = emit_ssa_to_c(module, export)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "native")
    execution = artifact.prepare_execution(
        {parameters["n"]: np.array([n], dtype=np.int64)}
    ).run()
    return int(execution.buffers[outputs[0]].reshape(-1)[0])


def _run_llvm(tmp_path, n):
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )

    module, export, parameters, outputs = _lowered()
    emitted = emit_ssa_function_to_llvm(module, export)
    assert not emitted.shortfalls, emitted.shortfalls
    artifact = compile_artifact(emitted, directory=tmp_path)
    execution = prepare_artifact_execution(
        artifact, {parameters["n"]: np.array([n], dtype=np.int64)}
    ).run()
    return int(execution.buffers[outputs[0]].reshape(-1)[0])


@pytest.mark.parametrize("run", [_run_c, _run_llvm], ids=["c", "llvm"])
def test_in_range_product_matches_python(tmp_path, run):
    assert run(tmp_path, 3037000499) == (3037000499 * 3037000499) % 1000


@pytest.mark.parametrize("run", [_run_c, _run_llvm], ids=["c", "llvm"])
def test_product_past_int64_is_reported_not_wrapped(tmp_path, run):
    from src.compiler.native_faults import NativeIntOverflow

    with pytest.raises(NativeIntOverflow):
        run(tmp_path, 3037000500)
