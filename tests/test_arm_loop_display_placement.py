"""A display guarded by an early return is born inside its own loop.

Shrunk from the string-core intern differential driver (``if n == 0:
return d`` before the batch loop).  The loop's displays are guarded by the
early return's implicit ``orelse``; their births were ordered among that
arm's top-level children, so they landed after the loop (``while_exit``)
and the loop's call read rows never written: final concordance refused
"Call reads a value defined at while_body".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"

EARLY_RETURN = '''
def make(values, offsets):
    count = len(offsets) - 1
    statuses = [0] * count
    tokens = [0] * count
    item = 0
    while item < count:
        tokens[item] = values[offsets[item]] + 10
        item += 1
    return 0, statuses, tokens


def root(seed, n):
    d = 71 + seed
    if n == 0:
        return d
    i = 0
    while i < n:
        env, statuses, tokens = make([seed % 3, i % 3, 1], [0, 2, 3])
        d = d * 31 + env + statuses[0] + tokens[0] + tokens[1]
        i += 1
    return d
'''

EARLY_RETURN_FOR = '''
def total(values):
    s = 0
    for v in values:
        s = s * 7 + v
    return s


def root(seed, n):
    if n < 0:
        return -1
    d = 0
    for i in range(n):
        d = d * 100 + total([seed, i, seed + i])
    return d
'''

CASES = ((0, 0), (0, 2), (2, 3), (1, 5))


def _python(source, seed, n):
    namespace: dict = {}
    exec(compile(source, "<program>", "exec"), namespace)
    return namespace["root"](seed, n)


def _lowered(source):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name="arm_loop_placement",
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


@pytest.mark.parametrize("source", [EARLY_RETURN, EARLY_RETURN_FOR], ids=["early_return_while", "early_return_for"])
@pytest.mark.parametrize("run", [_run_c, _run_llvm], ids=["c", "llvm"])
def test_guarded_loop_display_matches_python(tmp_path, run, source):
    lowered = _lowered(source)
    for seed, n in CASES:
        assert run(tmp_path, lowered, seed, n) == _python(source, seed, n), (seed, n)
