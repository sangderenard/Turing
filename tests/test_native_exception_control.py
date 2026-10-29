"""Execute protected conversion failure edges in both native backends."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


SOURCE = """\
def root(x: float, n: int) -> int:
    y = 0
    for i in range(n):
        try:
            y = int(x)
        except (ValueError, OverflowError):
            raise TypeError("caught")
    return y
"""

SOURCE_CALL = """\
def fail_if_zero(x: int) -> int:
    if x == 0:
        raise ValueError("bad")
    return x

def root(n: int) -> int:
    x = 0
    for i in range(n):
        try:
            x = fail_if_zero(i)
        except ValueError:
            raise TypeError("caught")
    return x
"""


@pytest.mark.parametrize("backend", ("c", "llvm"))
def test_protected_int_cast_routes_python_fault_and_propagates_width_limit(
    backend: str, tmp_path: Path,
) -> None:
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.identity_concordance import CorrelationTable
    from src.compiler.native_faults import NativeIntOverflow, NativeTypeError
    from src.compiler.ssa_c_backend import emit_ssa_to_c
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm,
        prepare_artifact_execution,
    )

    module, _outputs, exports = lower_ast_source_to_ssa(
        SOURCE, "root", name=f"native_try_int_cast_{backend}",
        extraction_contract="extraction_contracts/program_extraction.yaml",
        progress=lambda _message: None,
    )
    assert CorrelationTable.build(module).findings(module) == []
    function = module.functions[exports[0]]
    parameters = dict(function.metadata["parameter_names"])
    output = next(
        int(instruction.args[0].id)
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret"
    )
    if backend == "c":
        artifact = emit_ssa_to_c(module, exports[0])
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path / "native")

        def execute(value: float):
            return artifact.prepare_execution({
                parameters["x"]: np.array([value]),
                parameters["n"]: np.array([1]),
            }).run()

    else:
        artifact = emit_ssa_function_to_llvm(module, exports[0])
        assert artifact.complete, artifact.shortfalls
        artifact = compile_artifact(artifact, directory=tmp_path / "llvm")

        def execute(value: float):
            return prepare_artifact_execution(artifact, {
                parameters["x"]: np.array([value]),
                parameters["n"]: np.array([1]),
            }).run()

    assert execute(2.4).buffers[output].reshape(-1).tolist() == [2]
    with pytest.raises(NativeTypeError):
        execute(float("nan"))
    with pytest.raises(NativeTypeError):
        execute(float("inf"))
    # Python's int(1e30) succeeds with an arbitrary-precision integer.
    # The native-width refusal is not an authored OverflowError catch edge.
    with pytest.raises(NativeIntOverflow):
        execute(1e30)


@pytest.mark.parametrize("backend", ("c", "llvm"))
def test_standalone_helper_raise_has_concorded_transfer(
    backend: str, tmp_path: Path,
) -> None:
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.identity_concordance import CorrelationTable
    from src.compiler.native_faults import NativeValueError
    from src.compiler.ssa_c_backend import emit_ssa_to_c
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm,
        prepare_artifact_execution,
    )

    module, _outputs, exports = lower_ast_source_to_ssa(
        SOURCE_CALL, "fail_if_zero", name=f"native_raise_helper_{backend}",
        extraction_contract="extraction_contracts/program_extraction.yaml",
        progress=lambda _message: None,
    )
    assert CorrelationTable.build(module).findings(module) == []
    function = module.functions[exports[0]]
    parameter = dict(function.metadata["parameter_names"])["x"]
    if backend == "c":
        artifact = emit_ssa_to_c(module, exports[0])
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path / "native")

        def execute(value: int):
            return artifact.prepare_execution({
                parameter: np.array([value]),
            }).run()

    else:
        artifact = emit_ssa_function_to_llvm(module, exports[0])
        assert artifact.complete, artifact.shortfalls
        artifact = compile_artifact(artifact, directory=tmp_path / "llvm")

        def execute(value: int):
            return prepare_artifact_execution(artifact, {
                parameter: np.array([value]),
            }).run()

    execute(1)
    with pytest.raises(NativeValueError):
        execute(0)
