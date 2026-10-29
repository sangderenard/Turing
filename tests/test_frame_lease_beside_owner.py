"""A caller's own call-frame storage is not the lease it hands a callee.

Program: string core's batch shape -- ``env, t = batch(n, 2)`` in the
root and again inside ``one(n)``, which the root also calls.  Wrong
record: the root owns the storage of its own ``batch`` frame and leases
``one`` the storage of ``one``'s inner ``batch`` frame.  Both carry the
same compiler-frame key (``batch.seq4.handle``), so final concordance
reported the lease as duplicate storage and refused the program.  Rule:
an owner linked to one call frame of the caller and a lease linked to
another are two storage instances; same-frame duplication is still
reported.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"
EXECUTION = Path(__file__).resolve().parents[1] / "extraction_contracts" / "vehicle_full_native_execution.yaml"

_BATCH = '''
def batch(d, count):
    tokens = [0] * count
    tokens[0] = d + 1
    return 0, tokens
'''

SOURCES = {
    "same_callee_in_caller_and_callee": _BATCH + '''

def one(d):
    env, t = batch(d, 2)
    return env + t[0]


def root(n: int):
    env, t = batch(n, 2)
    return env + t[0] * 100 + one(n) * 1000
''',
    "two_levels_and_a_loop": _BATCH + '''

def one(d):
    env, t = batch(d, 3)
    return env + t[0] + len(t)


def two(d):
    env, t = batch(d + 1, 2)
    return one(d) * 10 + t[0]


def root(n: int):
    env, t = batch(n, 2)
    total = 0
    i = 0
    while i < len(t):
        total = total * 10 + t[i]
        i += 1
    return total + two(n) * 1000 + one(n) * 100000
''',
}

CASES = (0, 1, 7)


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
    module, _outputs, exports = _lower(source, f"frame_lease_{request.param}")
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
    namespace = {}
    exec(source, namespace)
    for n in CASES:
        expected = namespace["root"](n)
        feeds = {parameters["n"]: np.array([n], dtype=np.int64)}
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
