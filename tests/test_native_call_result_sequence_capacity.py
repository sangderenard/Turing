"""A list returned by a linked call starts empty in every caller frame.

``mid`` receives ``make``'s fresh ``out[:k]``.  The list's heap column is
the Call result, a frame local that is null on each ``mid`` invocation,
while its capacity cell is workspace the caller keeps.  The second ``mid``
call saw the first call's capacity, skipped the allocation and stored
through the null column: SIGSEGV on C and LLVM (string core
``_intern_validated_span`` -> ``_cover_span`` for rounds >= 2).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"

SOURCE = '''
def make(atoms: list[int], begin: int, end: int) -> list[int]:
    count = end - begin
    out = [0] * count
    k = 0
    i = 0
    while i < count:
        out[k] = atoms[begin + i]
        k += 1
        i += 2
    return out[:k]


def mid(atoms: list[int], begin: int, end: int) -> int:
    cover = make(atoms, begin, end)
    s = 0
    for v in cover:
        s += v
    return s


def root(rounds: int) -> int:
    atoms = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
    total = 0
    r = 0
    while r < rounds:
        total += mid(atoms, r % 3, 6 + r % 5)
        r += 1
    return total
'''

ROUNDS = (0, 1, 2, 3, 9)


def _expected(rounds):
    namespace = {}
    exec(SOURCE, namespace)
    return namespace["root"](rounds)


@pytest.fixture(scope="module")
def lowered():
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, exports = lower_ast_source_to_ssa(
        SOURCE, "root", name="call_result_sequence_capacity",
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


def test_c_repeated_call_result_list(tmp_path, lowered):
    from src.compiler.ssa_c_backend import emit_ssa_to_c

    module, export, parameters, outputs = lowered
    artifact = emit_ssa_to_c(module, export)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "native")
    for rounds in ROUNDS:
        execution = artifact.prepare_execution(
            {parameters["rounds"]: np.array([rounds], dtype=np.int64)}
        ).run()
        got = int(execution.buffers[outputs[0]].reshape(-1)[0])
        assert got == _expected(rounds), rounds


def test_llvm_repeated_call_result_list(tmp_path, lowered):
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )

    module, export, parameters, outputs = lowered
    emitted = emit_ssa_function_to_llvm(module, export)
    assert not emitted.shortfalls, emitted.shortfalls
    artifact = compile_artifact(emitted, directory=tmp_path)
    for rounds in ROUNDS:
        execution = prepare_artifact_execution(
            artifact, {parameters["rounds"]: np.array([rounds], dtype=np.int64)}
        ).run()
        got = int(execution.buffers[outputs[0]].reshape(-1)[0])
        assert got == _expected(rounds), rounds
