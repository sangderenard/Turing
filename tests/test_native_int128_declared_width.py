"""An int128-declared integer result is computed in int128.

The interval solver declares int128 for a result whose proven range leaves
int64 (``python_int_range_dtype``).  LLVM ran such an Add/Sub/Mul/Neg in the
operands' i64 and sign-extended the wrapped value; C negated in int64 before
widening.  Each program below proves an int128 result from int64 operands,
then folds it back into int64 with an exact ``%`` so the result is readable;
native must equal Python.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"

P = 1000003
# 2**62 + 1, spelled as a literal: ``a % M`` is proven in [0, 2**62].
M = 4611686018427387905

SOURCES = {
    # x, y in [0, 2**40): the product reaches 2**80, declared int128.
    "mul": f"""
def root(a: int, b: int):
    x = a % 1099511627776
    y = b % 1099511627776
    return (x * y) % {P}
""",
    # x in [-2**63, 3] (int64): -x in [-3, 2**63], declared int128.
    "neg": f"""
def root(a: int, b: int):
    x = b % 4 - a % {M} - a % {M}
    return (-x) % {P}
""",
    # x, y in [-2**63, 0] (int64): x + y in [-2**64, 0], declared int128.
    "add": f"""
def root(a: int, b: int):
    x = 0 - a % {M} - a % {M}
    y = 0 - b % {M} - b % {M}
    return (x + y) % {P}
""",
}

CASES = {
    "mul": [(2 ** 40 - 1, 2 ** 40 - 3), (3037000500, 3037000500)],
    "neg": [(2 ** 62, 0), (2 ** 62, 3), (12345, 2)],
    "add": [(2 ** 62, 2 ** 62), (2 ** 62, 7), (5, 11)],
}


def _python(name, a, b):
    namespace = {}
    exec(SOURCES[name], namespace)
    return namespace["root"](a, b)


def _lowered(name):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, exports = lower_ast_source_to_ssa(
        SOURCES[name], "root", name=f"int128_width_{name}",
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


def _inputs(parameters, a, b):
    return {
        parameters["a"]: np.array([a], dtype=np.int64),
        parameters["b"]: np.array([b], dtype=np.int64),
    }


def _run_c(tmp_path, name, a, b):
    from src.compiler.ssa_c_backend import emit_ssa_to_c

    module, export, parameters, outputs = _lowered(name)
    artifact = emit_ssa_to_c(module, export)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "native")
    execution = artifact.prepare_execution(_inputs(parameters, a, b)).run()
    return int(execution.buffers[outputs[0]].reshape(-1)[0])


def _run_llvm(tmp_path, name, a, b):
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )

    module, export, parameters, outputs = _lowered(name)
    emitted = emit_ssa_function_to_llvm(module, export)
    assert not emitted.shortfalls, emitted.shortfalls
    artifact = compile_artifact(emitted, directory=tmp_path)
    execution = prepare_artifact_execution(
        artifact, _inputs(parameters, a, b)
    ).run()
    return int(execution.buffers[outputs[0]].reshape(-1)[0])


@pytest.mark.parametrize("name", sorted(SOURCES))
def test_program_declares_int128_from_int64_operands(name):
    """Guard: the program's op really widens int64 operands to int128."""
    module, _export, _parameters, _outputs = _lowered(name)
    op = {"mul": "Mul", "neg": "Neg", "add": "Add"}[name]
    widened = [
        instruction
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == op
        and instruction.res is not None
        and str(instruction.res.dtype) == "int128"
        and all(str(arg.dtype) == "int64" for arg in instruction.args)
    ]
    assert widened, name


@pytest.mark.parametrize("run", [_run_c, _run_llvm], ids=["c", "llvm"])
@pytest.mark.parametrize("name", sorted(SOURCES))
def test_int128_result_matches_python(tmp_path, run, name):
    for index, (a, b) in enumerate(CASES[name]):
        got = run(tmp_path / str(index), name, a, b)
        assert got == _python(name, a, b), (name, a, b, got)
