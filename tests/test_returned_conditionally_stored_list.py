"""A list stored into by a loop under a condition, then returned.

Program: ``tokens = [0] * count; if status == 0: while ...: tokens[i] =
...`` then ``return status, tokens`` (the shape of string-core
``intern_batch``).  Wrong record: the returned version was the conditional's
merge of the loop's version and the untouched list, so the walk to the
stored list stopped at the merge and the caller's view of ``tokens`` named
no storage (refused before lowering).  A merge whose arms are versions of
one list is now that list's one storage; arms that reach different storage
stay a real merge.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"
EXECUTION = Path(__file__).resolve().parents[1] / "extraction_contracts" / "vehicle_full_native_execution.yaml"

SOURCES = {
    "loop_in_body": '''
def batch(d, count):
    status = 0
    if d > 100:
        status = 3
    tokens = [0] * count
    if status == 0:
        i = 0
        while i < count:
            tokens[i] = d + i
            i += 1
    return status, tokens


def root(n: int):
    status, t = batch(n, 2)
    return status * 1000 + t[1] * 10 + t[0]
''',
    "loop_in_orelse": '''
def batch(d, count):
    status = 0
    if d > 100:
        status = 3
    tokens = [0] * count
    if status != 0:
        pass
    else:
        i = 0
        while i < count:
            tokens[i] = d + 2 * i
            i += 1
    return status, tokens


def root(n: int):
    status, t = batch(n, 2)
    return status * 1000 + t[1] * 10 + t[0]
''',
}

CASES = (0, 1, 2, 101)


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
        source, "root", name=f"returned_conditionally_stored_list_{request.param}",
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
