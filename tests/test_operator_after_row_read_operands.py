"""An operator region is placed after the row reads it consumes.

Program: ``env, s = batch(n, 2); return env * 1000 + s[0] * 100 + s[1]``
(the shape of string-core ``intern`` after ``intern_batch``).  Wrong record:
the ``+`` region completed at its own start (``env``), so lexical placement
put it ahead of the region reading ``s[1]`` and it consumed that row before
it was loaded (refused: "Call reads a value defined at").  An operator
executes after its operands, so it now completes after each operand ends.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"
EXECUTION = Path(__file__).resolve().parents[1] / "extraction_contracts" / "vehicle_full_native_execution.yaml"

SOURCES = {
    "tuple_result": '''
def batch(d, count):
    statuses = [0] * count
    statuses[1] = d + 1
    return d + 3, statuses


def root(n: int):
    env, s = batch(n, 2)
    return env * 1000 + s[0] * 100 + s[1]
''',
    "tuple_result_loop": '''
def batch(d, count):
    statuses = [0] * count
    i = 0
    while i < count:
        statuses[i] = i + d
        i += 1
    return 0, statuses


def root(n: int):
    env, s = batch(n, 2)
    return env * 1000 + s[0] * 100 + s[1]
''',
}

CASES = (0, 1, 2, 9)


def _python(source, n):
    namespace = {}
    exec(source, namespace)
    return namespace["root"](n)


@pytest.fixture(scope="module", params=sorted(SOURCES))
def lowered(request):
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    source = SOURCES[request.param]
    contract = ExtractionContract(CONTRACT).with_execution_file(EXECUTION)
    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name=f"operator_after_row_reads_{request.param}",
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
