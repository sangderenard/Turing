"""The authored int(float) operation reports a catchable Python fault code."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.native_faults import NativeOverflowError, NativeValueError


CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "extraction_contracts" / "program_extraction.yaml"
)


def _lowered():
    module, _outputs, exports = lower_ast_source_to_ssa(
        "def root(x: float) -> int:\n    return int(x)\n",
        "root", name="native_float_int_status",
        extraction_contract=CONTRACT,
        progress=lambda _message: None,
        progress_bars=False,
    )
    root = module.functions[exports[0]]
    parameter = dict(root.metadata["parameter_names"])["x"]
    returns = tuple(
        instruction for block in root.blocks.values()
        for instruction in block.instrs if instruction.op == "Ret"
    )
    assert len(returns) == 1 and len(returns[0].args) == 1
    return module, exports[0], parameter, int(returns[0].args[0].id)


@pytest.mark.parametrize("backend", ["c", "llvm"])
def test_float_int_native_status_matches_python_class(backend, tmp_path):
    module, export, parameter, output = _lowered()
    if backend == "c":
        from src.compiler.ssa_c_backend import emit_ssa_to_c

        artifact = emit_ssa_to_c(module, export)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path / "c")
        run = lambda value: artifact.prepare_execution({
            parameter: np.array([value], dtype=np.float64),
        }).run()
    else:
        from src.compiler.ssa_llvm_backend import (
            compile_artifact, emit_ssa_function_to_llvm,
            prepare_artifact_execution,
        )

        emitted = emit_ssa_function_to_llvm(module, export)
        assert not emitted.shortfalls, emitted.shortfalls
        artifact = compile_artifact(emitted, directory=tmp_path / "llvm")
        run = lambda value: prepare_artifact_execution(artifact, {
            parameter: np.array([value], dtype=np.float64),
        }).run()
    assert int(run(1.9).buffers[output].reshape(-1)[0]) == 1
    assert int(run(-1.9).buffers[output].reshape(-1)[0]) == -1
    with pytest.raises(NativeValueError, match="NaN"):
        run(float("nan"))
    with pytest.raises(NativeOverflowError, match="infinity"):
        run(float("inf"))
