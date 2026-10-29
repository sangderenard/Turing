"""The authored int(float) operation reports a catchable Python fault code."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.native_faults import NativeOverflowError, NativeValueError
from src.transmogrifier.ssa import BasicBlock, Function, IRModule, Instr, SSAValue


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


@pytest.mark.parametrize("backend", ["c", "llvm"])
@pytest.mark.parametrize("dynamic", [False, True], ids=["literal", "ssa-code"])
def test_native_status_take_returns_and_clears_the_reported_code(
    backend, dynamic, tmp_path,
):
    status = SSAValue(100, dtype="int64")
    code = SSAValue(99, dtype="int64")
    report = (
        Instr("Call", [code], None, attributes={
            "callee": "turing_validation_error",
        }) if dynamic else
        Instr("Call", [], None, attributes={
            "callee": "turing_validation_error", "error_code": 97,
        })
    )
    function = Function("take_status", [], {
        "entry": BasicBlock("entry", [
            *((Instr("Const", [], code, attributes={"value": 97}),)
              if dynamic else ()),
            report,
            Instr("Call", [], status, attributes={
                "callee": "turing_validation_error_take",
            }),
            Instr("Ret", [status], None),
        ]),
    }, metadata={"control_ir": True})
    module = IRModule({function.name: function})
    if backend == "c":
        from src.compiler.ssa_c_backend import emit_ssa_to_c

        artifact = emit_ssa_to_c(module, function.name)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path / "c")
        execution = artifact.prepare_execution({})
    else:
        from src.compiler.ssa_llvm_backend import (
            compile_artifact, emit_ssa_function_to_llvm,
            prepare_artifact_execution,
        )

        emitted = emit_ssa_function_to_llvm(module, function.name)
        assert not emitted.shortfalls, emitted.shortfalls
        artifact = compile_artifact(emitted, directory=tmp_path / "llvm")
        execution = prepare_artifact_execution(artifact, {})
    assert int(execution.run().buffers[100].reshape(-1)[0]) == 97
    assert artifact.take_validation_error() == 0
