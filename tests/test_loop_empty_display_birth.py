"""An empty list display a loop re-executes empties its list each time.

Shrunk from the string-core intern differential driver (``values = []``
then six ``values.append`` per batch, inside the batch loop).  An empty
display had no birth at its site, only the frame's zero-length initializer
at function entry, so the second iteration appended after the first's rows:
``len(values)`` read 2, 4, 6 and ``values[0]`` the first iteration's row.
Silent wrong on C and LLVM.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"

WHILE_BODY = '''
def root(seed, n):
    d = 0
    i = 0
    while i < n:
        values = []
        k = 0
        while k < 2:
            values.append(seed + k + 10 * i)
            k += 1
        d = d * 1000 + len(values) * 100 + values[0] * 10 + values[len(values) - 1]
        i += 1
    return d
'''

REBOUND_FOR_BODY = '''
def root(seed, n):
    d = 0
    values = []
    for i in range(n):
        values = []
        for k in range(1 + i % 3):
            values.append(seed + k + 10 * i)
        d = d * 1000 + len(values) * 100 + values[0]
    return d * 10 + len(values)
'''

CASES = ((0, 0), (0, 2), (2, 3), (1, 5))


def _python(source, seed, n):
    namespace: dict = {}
    exec(compile(source, "<program>", "exec"), namespace)
    return namespace["root"](seed, n)


def _lowered(source):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name="loop_empty_display_birth",
        extraction_contract=str(CONTRACT), progress=lambda _message: None,
        root_calls=[{"seed": seed, "n": n} for seed, n in CASES],
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


def _feeds(parameters, seed, n):
    return {
        parameters["seed"]: np.array([seed], dtype=np.int64),
        parameters["n"]: np.array([n], dtype=np.int64),
    }


def _run_c(tmp_path, lowered, seed, n):
    from src.compiler.ssa_c_backend import emit_ssa_to_c

    module, export, parameters, outputs = lowered
    artifact = emit_ssa_to_c(module, export)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / f"c_{seed}_{n}")
    execution = artifact.prepare_execution(_feeds(parameters, seed, n)).run()
    return int(execution.buffers[outputs[0]].reshape(-1)[0])


def _run_llvm(tmp_path, lowered, seed, n):
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )

    module, export, parameters, outputs = lowered
    emitted = emit_ssa_function_to_llvm(module, export)
    assert not emitted.shortfalls, emitted.shortfalls
    artifact = compile_artifact(emitted, directory=tmp_path / f"llvm_{seed}_{n}")
    execution = prepare_artifact_execution(
        artifact, _feeds(parameters, seed, n)
    ).run()
    return int(execution.buffers[outputs[0]].reshape(-1)[0])


@pytest.mark.parametrize("source", [WHILE_BODY, REBOUND_FOR_BODY], ids=["while_body", "rebound_for_body"])
@pytest.mark.parametrize("run", [_run_c, _run_llvm], ids=["c", "llvm"])
def test_loop_empty_display_matches_python(tmp_path, run, source):
    lowered = _lowered(source)
    for seed, n in CASES:
        assert run(tmp_path, lowered, seed, n) == _python(source, seed, n), (seed, n)
