"""Copies minted for different aggregate actuals have their own numeric scope.

Program: string core's ``intern(1, [x])`` beside ``intern(1, [65, x])``
(driver ia2).  Wrong record: the two callsite copies differ only by the
member layout of the list actual (one int member against two), which the
numeric-identity receipt left out, so both shared one numeric scope and
``len(atoms)`` disagreed: ``source_int_range_concordance disagreement ...
recorded=(1, 1), proposed=(2, 2)``.  The aggregate descriptors a copy is
minted for are part of its numeric identity, as its tensor descriptors are.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"
EXECUTION = Path(__file__).resolve().parents[1] / "extraction_contracts" / "vehicle_full_native_execution.yaml"

_FUNCTIONS = '''
def batch(domain, atoms, n):
    total = 0
    i = 0
    while i < n:
        total = total * 31 + atoms[i] + domain
        i += 1
    return total


def intern(domain, atoms):
    return batch(domain, atoms, len(atoms)) * 10 + len(atoms)
'''

_CLASS = '''
class V:
    def __init__(self):
        self.calls = 0

    def batch(self, domain, atoms, n):
        total = 0
        i = 0
        while i < n:
            total = total * 31 + atoms[i] + domain
            i += 1
        self.calls += 1
        return total

    def intern(self, domain, atoms):
        return self.batch(domain, atoms, len(atoms)) * 10 + len(atoms)
'''

SOURCES = {
    "functions": _FUNCTIONS + '''

def root(a: int, b: int):
    d = intern(1, [a]) + intern(1, [65, a]) * 1000
    return d * 7 + intern(2, [b, a, 3])
''',
    "methods": _CLASS + '''

def root(a: int, b: int):
    v = V()
    d = v.intern(1, [a]) + v.intern(1, [65, a]) * 1000
    return d * 7 + v.intern(2, [b, a, 3]) + v.calls
''',
}

CASES = ((0, 0), (1, 1), (3, 5), (-4, 9))


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
    module, _outputs, exports = _lower(source, f"aggregate_scope_{request.param}")
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
    for a, b in CASES:
        expected = namespace["root"](a, b)
        feeds = {
            parameters["a"]: np.array([a], dtype=np.int64),
            parameters["b"]: np.array([b], dtype=np.int64),
        }
        got = run(feeds).buffers[outputs[0]].reshape(-1)[0].item()
        assert type(got) is type(expected) and got == expected, (a, b, got, expected)


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
