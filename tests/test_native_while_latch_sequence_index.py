"""A while-loop index reloaded from a list: ``i = chosen[i]``.

The checked subscript lowers to an in-range block that the loop latch
consumes, and that block is appended after the latch in the function's block
dict.  The C emitter bound results in dict order, so the latch's use of the
loaded index had no value.  Blocks are now emitted in reverse post-order,
which places every definition before the blocks it dominates.
(string_core ``_cover_span``: ``i = chosen_next[i]``.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "extraction_contracts"

SOURCE = '''
def root(n: int) -> int:
    count = n % 7
    best = [count + 1] * (count + 1)
    chosen = [0] * count
    best[count] = 0
    i = count - 1
    while i >= 0:
        best[i] = 1 + best[i + 1]
        chosen[i] = i + 1
        i -= 1
    cover = [0] * count
    c = 0
    i = 0
    while i < count:
        cover[c] = chosen[i] * 10
        c += 1
        i = chosen[i]
    out = cover[:c]
    total = 0
    for v in out:
        total += v
    return total + best[0] * 1000 + len(out)
'''


def _python(n: int) -> int:
    namespace: dict = {}
    exec(SOURCE, namespace)
    return namespace["root"](n)


@pytest.fixture(scope="module")
def lowered():
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    contract = ExtractionContract(CONTRACTS / "program_extraction.yaml")
    contract = contract.with_execution_file(
        CONTRACTS / "vehicle_full_native_execution.yaml"
    )
    module, _outputs, exports = lower_ast_source_to_ssa(
        SOURCE, "root", name="while_latch_sequence_index",
        extraction_contract=contract, progress=lambda _message: None,
    )
    root = module.functions[exports[0]]
    outputs = [
        int(value.id)
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret" and not instruction.attributes.get("fault_exit")
        for value in instruction.args
    ] + [
        int(value.id)
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret"
        for value in instruction.args
    ]
    return module, exports[0], dict(root.metadata["parameter_names"]), outputs


def _result(execution, outputs) -> int:
    held = [key for key in outputs if key in execution.buffers]
    return int(execution.buffers[held[0]].reshape(-1)[0])


def _feeds(parameters, n):
    return {parameters["n"]: np.array([n], dtype=np.int64)}


def test_c_matches_python(lowered, tmp_path):
    from src.compiler.ssa_c_backend import emit_ssa_to_c

    module, export, parameters, outputs = lowered
    artifact = emit_ssa_to_c(module, export)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "c")
    for n in (0, 1, 3, 6):
        execution = artifact.prepare_execution(_feeds(parameters, n)).run()
        assert _result(execution, outputs) == _python(n)


def test_llvm_matches_python(lowered, tmp_path):
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )

    module, export, parameters, outputs = lowered
    emitted = emit_ssa_function_to_llvm(module, export)
    assert not emitted.shortfalls, emitted.shortfalls
    artifact = compile_artifact(emitted, directory=tmp_path / "llvm")
    for n in (0, 1, 3, 6):
        execution = prepare_artifact_execution(
            artifact, _feeds(parameters, n)
        ).run()
        assert _result(execution, outputs) == _python(n)
