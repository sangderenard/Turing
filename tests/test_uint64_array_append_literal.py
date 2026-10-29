"""``array("Q")`` holds the full unsigned 64-bit range on every backend.

Program: ``values = array("Q"); values.append(18446744073709551615)``.
Wrong record (LLVM only): the heap cell's element address took its
signedness from the base pointer, so the uint64 element was treated as int64
and the store's narrowing check faulted 86 (INT_OVERFLOW) on a value Python
accepts; C was correct.  The element address now reads the heap cell's
element class for signedness as it already did for width.  A negative
append still faults through the typed-array row range check.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"
EXECUTION = Path(__file__).resolve().parents[1] / "extraction_contracts" / "vehicle_full_native_execution.yaml"

SOURCE = '''
from array import array


def root(n: int):
    values = array("Q")
    values.append(18446744073709551615)
    values.append(n)
    if values[0] == 18446744073709551615:
        return len(values) * 10 + values[1] % 7
    return len(values) * 10 + 100
'''

CASES = (0, 1, 5, 9)
NEGATIVE = -1


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
        SOURCE, "root", name="uint64_array_append_literal",
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


def _runner(backend, lowered, directory):
    module, export, parameters, outputs = lowered
    if backend == "c":
        from src.compiler.ssa_c_backend import emit_ssa_to_c

        artifact = emit_ssa_to_c(module, export)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(directory / "native")

        def prepare(feeds):
            return artifact.prepare_execution(feeds)
    else:
        from src.compiler.ssa_llvm_backend import (
            compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
        )

        emitted = emit_ssa_function_to_llvm(module, export)
        assert not emitted.shortfalls, emitted.shortfalls
        artifact = compile_artifact(emitted, directory=directory)

        def prepare(feeds):
            return prepare_artifact_execution(artifact, feeds)

    def run(n):
        execution = prepare({parameters["n"]: np.array([n], dtype=np.int64)}).run()
        return int(execution.buffers[outputs[0]].reshape(-1)[0])

    return run


@pytest.mark.parametrize("backend", ["c", "llvm"])
def test_full_range_literal_matches_python(tmp_path, lowered, backend):
    run = _runner(backend, lowered, tmp_path)
    for n in CASES:
        assert run(n) == _python(n)


@pytest.mark.parametrize("backend", ["c", "llvm"])
def test_negative_append_still_faults(tmp_path, lowered, backend):
    # Python raises OverflowError; the typed-array row check reports its
    # range code (74) rather than storing a wrapped value.
    with pytest.raises(OverflowError):
        _python(NEGATIVE)
    run = _runner(backend, lowered, tmp_path)
    with pytest.raises(RuntimeError, match="code 74"):
        run(NEGATIVE)
