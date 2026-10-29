"""A tuple holding a list, unpacked beside another call.

Program: ``env, t = batch(n, 2)`` in ``root`` next to ``helper(n)`` (the
shape of string-core ``intern_batch`` called beside other calls).  Wrong
record: the aggregate unpack read the callee's sequences from a table bound
by an earlier loop over other calls (the last callee it linked), so the
returned list was not recognised as a caller arena and was also loaded out
of the aggregate; the load retired the arena formal and the call read
undefined operands (refused by the full-native execution contract).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"
EXECUTION = Path(__file__).resolve().parents[1] / "extraction_contracts" / "vehicle_full_native_execution.yaml"

SOURCE = '''
def batch(d, count):
    tokens = [0] * count
    tokens[0] = d + 1
    return 0, tokens


def helper(d):
    return d * 2


def root(n: int):
    env, t = batch(n, 2)
    return env + t[0] + helper(n) * 1000
'''

CASES = (0, 1, 2, 9)


def _python(n):
    namespace = {}
    exec(SOURCE, namespace)
    return namespace["root"](n)


@pytest.fixture(scope="module")
def lowered():
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    contract = ExtractionContract(CONTRACT).with_execution_file(EXECUTION)
    module, _outputs, exports = lower_ast_source_to_ssa(
        SOURCE, "root", name="aggregate_unpack_callee_table",
        extraction_contract=contract, progress=lambda _message: None,
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


def test_matches_python_on_c(tmp_path, lowered):
    from src.compiler.ssa_c_backend import emit_ssa_to_c

    module, export, parameters, outputs = lowered
    artifact = emit_ssa_to_c(module, export)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "native")
    for n in CASES:
        execution = artifact.prepare_execution(
            {parameters["n"]: np.array([n], dtype=np.int64)}
        ).run()
        assert int(execution.buffers[outputs[0]].reshape(-1)[0]) == _python(n)


def test_matches_python_on_llvm(tmp_path, lowered):
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )

    module, export, parameters, outputs = lowered
    emitted = emit_ssa_function_to_llvm(module, export)
    assert not emitted.shortfalls, emitted.shortfalls
    artifact = compile_artifact(emitted, directory=tmp_path)
    for n in CASES:
        execution = prepare_artifact_execution(
            artifact, {parameters["n"]: np.array([n], dtype=np.int64)}
        ).run()
        assert int(execution.buffers[outputs[0]].reshape(-1)[0]) == _python(n)
