"""A local mapping of lists grown by ``setdefault(k, []).append(x)``.

string_core ``token_orchard/core.py`` keeps ``candidates`` / ``prefix``
mappings of lists: rows are grown by ``d.setdefault(key, []).append(token)``
and read back by ``for t in d.get(key, ())`` and ``len(d[key])``.  Each
program here lowers under the full-native execution contract and must match
Python on C and on LLVM.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

# Write-only rows: the pool element comes from the appends alone.
WRITE_ONLY = '''
def root(n: int) -> int:
    d = {}
    i = 0
    while i < n:
        d.setdefault(i % 3, []).append(100 + i)
        i += 1
    return len(d)
'''

# Tuple keys, read back through ``get`` with an empty tuple default (the
# string-core shape): iterating a row or ``()`` yields ints.
GET_TUPLE_DEFAULT = '''
def root(n: int) -> int:
    candidates = {}
    i = 0
    while i < n:
        key = (1, i % 3, 7, 2)
        candidates.setdefault(key, []).append(100 + i)
        i += 1
    total = 0
    for token in candidates.get((1, 1, 7, 2), ()):
        total += token
    for token in candidates.get((1, 9, 7, 2), ()):
        total += 1000
    return total
'''

# ``len`` of a keyed row.
ROW_LENGTH = '''
def root(n: int) -> int:
    d = {}
    count = 1
    while count < n:
        key = (1, count)
        d.setdefault(key, []).append(count)
        count += 1
    d.setdefault((1, 2), []).append(9)
    return len(d) * 100 + len(d[(1, 2)]) * 10
'''

# Iterating a keyed row.
ROW_ITERATION = '''
def root(n: int) -> int:
    d = {}
    i = 0
    while i < n:
        d.setdefault(i % 3, []).append(100 + i)
        i += 1
    d.setdefault(1, []).append(9)
    total = 0
    for t in d[1]:
        total += t
    return total
'''

CASES = {
    "write_only": (WRITE_ONLY, (0, 1, 5)),
    "get_tuple_default": (GET_TUPLE_DEFAULT, (0, 1, 5, 17)),
    "row_length": (ROW_LENGTH, (0, 2, 4, 5)),
    "row_iteration": (ROW_ITERATION, (0, 1, 5)),
}


def _python(source, n):
    namespace: dict = {}
    exec(source, namespace)
    return namespace["root"](n)


def _lowered(name, source):
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    contract = ExtractionContract(
        ROOT / "extraction_contracts" / "program_extraction.yaml"
    ).with_execution_file(
        ROOT / "extraction_contracts" / "vehicle_full_native_execution.yaml"
    )
    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name=f"mapping_list_rows_{name}",
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


def _run_c(tmp_path, lowered, values):
    from src.compiler.ssa_c_backend import emit_ssa_to_c

    module, export, parameters, outputs = lowered
    artifact = emit_ssa_to_c(module, export)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "c")
    results = []
    for n in values:
        execution = artifact.prepare_execution(
            {parameters["n"]: np.array([n], dtype=np.int64)}
        ).run()
        results.append(int(execution.buffers[outputs[0]].reshape(-1)[0]))
    return results


def _run_llvm(tmp_path, lowered, values):
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )

    module, export, parameters, outputs = lowered
    emitted = emit_ssa_function_to_llvm(module, export)
    assert not emitted.shortfalls, emitted.shortfalls
    artifact = compile_artifact(emitted, directory=tmp_path / "llvm")
    results = []
    for n in values:
        execution = prepare_artifact_execution(
            artifact, {parameters["n"]: np.array([n], dtype=np.int64)}
        ).run()
        results.append(int(execution.buffers[outputs[0]].reshape(-1)[0]))
    return results


@pytest.mark.parametrize("name", sorted(CASES))
def test_mapping_of_lists_matches_python_on_c_and_llvm(tmp_path, name):
    source, values = CASES[name]
    expected = [_python(source, n) for n in values]
    lowered = _lowered(name, source)
    assert _run_c(tmp_path, lowered, values) == expected
    assert _run_llvm(tmp_path, lowered, values) == expected
