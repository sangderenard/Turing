"""A literal list with runtime elements, indexed by a loop-carried index.

Program: ``commit([1 + n, 3], 2)`` where ``commit`` reads
``fingerprints[i]`` in ``while i < count``.  Wrong record: inside a loop the
graph reads the carried ``i`` through its initial literal ``i = 0``, so
binding the display by its members took ``fingerprints[i]`` for the literal
projection ``fingerprints[0]`` and every iteration read element 0 (200 for
400; silent on C and LLVM).  The same misreading turned ``r[i]`` over a
call's returned tuple into ``r[0]``.  A loop-carried initial is never a
literal index.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"
EXECUTION = Path(__file__).resolve().parents[1] / "extraction_contracts" / "vehicle_full_native_execution.yaml"

SOURCES = {
    "while_index": '''
def commit(fingerprints, count: int):
    total = 0
    i = 0
    while i < count:
        fingerprint = fingerprints[i]
        total += fingerprint * 100
        i += 1
    return total


def root(n: int):
    return commit([1 + n, 3], 2)
''',
    "three_rows": '''
def commit(values, count: int):
    total = 0
    i = 0
    while i < count:
        total = total * 10 + values[i]
        i += 1
    return total


def root(n: int):
    return commit([1 + n, 3, n], 3)
''',
}

CASES = (0, 1, 7)


def _python(source, n):
    namespace = {}
    exec(source, namespace)
    return namespace["root"](n)


def _lower(source, name):
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    contract = ExtractionContract(CONTRACT).with_execution_file(EXECUTION)
    return lower_ast_source_to_ssa(
        source, "root", name=name,
        extraction_contract=contract, progress=lambda _message: None,
    )


@pytest.fixture(scope="module", params=sorted(SOURCES))
def lowered(request):
    source = SOURCES[request.param]
    module, _outputs, exports = _lower(
        source, f"display_runtime_elements_{request.param}",
    )
    root = module.functions[exports[0]]
    outputs = [
        int(value.id)
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret" and not instruction.attributes.get("fault_exit")
        for value in instruction.args
    ]
    return source, module, exports[0], dict(root.metadata["parameter_names"]), outputs


def test_matches_python_on_c(tmp_path, lowered):
    from src.compiler.ssa_c_backend import emit_ssa_to_c

    source, module, export, parameters, outputs = lowered
    artifact = emit_ssa_to_c(module, export)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "native")
    for n in CASES:
        execution = artifact.prepare_execution(
            {parameters["n"]: np.array([n], dtype=np.int64)}
        ).run()
        assert int(execution.buffers[outputs[0]].reshape(-1)[0]) == _python(source, n)


def test_matches_python_on_llvm(tmp_path, lowered):
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )

    source, module, export, parameters, outputs = lowered
    emitted = emit_ssa_function_to_llvm(module, export)
    assert not emitted.shortfalls, emitted.shortfalls
    artifact = compile_artifact(emitted, directory=tmp_path)
    for n in CASES:
        execution = prepare_artifact_execution(
            artifact, {parameters["n"]: np.array([n], dtype=np.int64)}
        ).run()
        assert int(execution.buffers[outputs[0]].reshape(-1)[0]) == _python(source, n)


def test_call_result_read_by_carried_index_is_never_member_zero():
    """``r[i]`` over a returned tuple must not lower as ``r[0]``.

    It is not lowered natively yet; it must be refused rather than wrong.
    """

    source = '''
def pair(d):
    return d + 1, d * 3


def root(n: int):
    r = pair(n)
    total = 0
    i = 0
    while i < 2:
        total += r[i] * (10 ** i)
        i += 1
    return total
'''
    from src.compiler.ssa_c_backend import emit_ssa_to_c

    try:
        module, _outputs, exports = _lower(source, "display_runtime_call_result")
    except Exception:
        return
    artifact = emit_ssa_to_c(module, exports[0])
    assert not artifact.complete
