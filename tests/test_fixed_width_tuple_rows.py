"""A list of fixed-width tuples is a sequence of fixed-width rows.

Program: string core's ``intern_preflighted_batch`` reads
``fingerprint = fingerprints[i]`` inside ``while i < count``, checks
``len(fingerprint) != 2`` and reads ``fingerprint[0]`` and
``fingerprint[1]``.  Wrong records: the caller's display ``[(1, n), (3,
4)]`` kept a width of one, so each two-value row was an impossible
insertion; an unannotated formal had no row width, so the call bound no
second column; the all-literal rows became a 2x2 tensor constant against a
two-column sequence; a mutated list's row reads had no class; and
``len`` of a row read by a loop-carried index became a formal no caller
fills.  Every row of such a sequence is its tuple width, read column by
column; an out-of-range row still raises.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"
EXECUTION = Path(__file__).resolve().parents[1] / "extraction_contracts" / "vehicle_full_native_execution.yaml"

_COMMIT = '''
def commit(fingerprints{annotation}, count: int):
    total = 0
    i = 0
    while i < count:
        fingerprint = fingerprints[i]
        if len(fingerprint) != 2:
            raise ValueError("invalid preflight fingerprint")
        hi = fingerprint[0]
        lo = fingerprint[1]
        total += hi * 100 + lo
        i += 1
    return total
'''

SOURCES = {
    "annotated_display": _COMMIT.format(annotation=": list[tuple[int, int]]") + '''

def root(n: int):
    return commit([(1, n), (3, 4)], 2)
''',
    "unannotated_literal_rows": _COMMIT.format(annotation="") + '''

def root(n: int):
    return commit([(1, 2), (3, 4)], 2) + n
''',
    "tuple_of_rows": _COMMIT.format(annotation="") + '''

def root(n: int):
    return commit(((1, n), (3, 4)), 2)
''',
    "appended_rows": _COMMIT.format(annotation="") + '''

def root(n: int):
    xs = [(1, n), (3, 4)]
    xs.append((n, 5))
    return commit(xs, 3)
''',
    "float_column": '''
def commit(rows: list[tuple[int, float]], count: int):
    total = 0.0
    i = 0
    while i < count:
        row = rows[i]
        total += row[0] * row[1]
        i += 1
    return total


def root(n: int):
    return commit([(1, 2.5), (n, 0.5)], 2)
''',
    "row_out_of_range": _COMMIT.format(annotation=": list[tuple[int, int]]") + '''

def root(n: int):
    return commit([(1, n), (3, 4)], 3)
''',
}

CASES = (0, 1, 7)


def _python(source, n):
    namespace = {}
    exec(source, namespace)
    try:
        return namespace["root"](n), None
    except Exception as error:  # the native lane must fault too
        return None, type(error)


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
    module, _outputs, exports = _lower(source, f"tuple_rows_{request.param}")
    root = module.functions[exports[0]]
    outputs = [
        int(value.id)
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret" and not instruction.attributes.get("fault_exit")
        for value in instruction.args
    ]
    return source, module, exports[0], dict(root.metadata["parameter_names"]), outputs


def _check(run, source, parameters, outputs):
    for n in CASES:
        expected, raised = _python(source, n)
        feeds = {parameters["n"]: np.array([n], dtype=np.int64)}
        if raised is not None:
            with pytest.raises(Exception):
                run(feeds)
            continue
        got = run(feeds).buffers[outputs[0]].reshape(-1)[0].item()
        assert type(got) is type(expected) and got == expected, (n, got, expected)


def test_matches_python_on_c(tmp_path, lowered):
    from src.compiler.ssa_c_backend import emit_ssa_to_c

    source, module, export, parameters, outputs = lowered
    artifact = emit_ssa_to_c(module, export)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "native")
    _check(
        lambda feeds: artifact.prepare_execution(feeds).run(),
        source, parameters, outputs,
    )


def test_matches_python_on_llvm(tmp_path, lowered):
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )

    source, module, export, parameters, outputs = lowered
    emitted = emit_ssa_function_to_llvm(module, export)
    assert not emitted.shortfalls, emitted.shortfalls
    artifact = compile_artifact(emitted, directory=tmp_path)
    _check(
        lambda feeds: prepare_artifact_execution(artifact, feeds).run(),
        source, parameters, outputs,
    )
