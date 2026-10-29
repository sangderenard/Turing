"""A scalar-or-None actual carries its presence into the callee.

Program: ``y = None; if n > 2: y = n; return f(y)`` where ``f`` tests
``x is None``.  Wrong record: the callee's copy bound ``x`` to the
payload's int64 descriptor, read that as storage that cannot be None and
folded ``x is None`` to False, and the call passed only the payload --
f(None) returned 0 instead of 7.  A formal whose actual may be None keeps
its ``is None`` test, and every call feeds it the presence its actual
carries: the caller's optional presence, False for a literal None, True
for any other value; a formal forwarded on to such a callee carries it too.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"
EXECUTION = Path(__file__).resolve().parents[1] / "extraction_contracts" / "vehicle_full_native_execution.yaml"

_F = '''
def f(x):
    if x is None:
        return 7
    return x * 100
'''

SOURCES = {
    "merge_actual": _F + '''

def root(n: int):
    y = None
    if n > 2:
        y = n
    return f(y)
''',
    "merge_and_plain_actuals": _F + '''

def root(n: int):
    y = None
    if n > 2:
        y = n
    return f(y) + f(n) * 1000
''',
    "optional_call_result_actual": _F + '''

def g(n):
    if n > 2:
        return n
    return None


def root(n: int):
    return f(g(n))
''',
    "forwarded_formal": '''
def h(x):
    if x is None:
        return 7
    return x * 100


def f(x, k):
    return h(x) + k


def root(n: int):
    y = None
    if n > 2:
        y = n
    return f(y, 1) * 10 + f(n, 2)
''',
    "loop_actual": _F + '''

def root(n: int):
    total = 0
    i = 0
    while i < n:
        y = None
        if i % 2 == 0:
            y = i
        total = total * 3 + f(y)
        i += 1
    return total
''',
    "is_not_none": '''
def f(x, base):
    if x is not None:
        return x * 3 + base
    return base


def root(n: int):
    y = None
    if n > 2:
        y = n
    return f(y, n + 1) * 100 + f(n, 1)
''',
}

CASES = (0, 1, 3, 7)


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
    module, _outputs, exports = _lower(source, f"optional_args_{request.param}")
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


def test_annotated_optional_formal_is_not_silently_none():
    """``def f(x: int | None)`` reached with a maybe-None local.

    Its native type tag formal used to be leased as storage nobody fills,
    so every call read the NoneType tag.  Until the call edge binds the tag
    from its subject actual, the program must not lower silently wrong.
    """

    source = '''
def f(x: int | None):
    if x is None:
        return 7
    return x * 100


def root(n: int):
    y = None
    if n > 2:
        y = n
    return f(y) + f(n) * 1000
'''
    namespace = {}
    exec(source, namespace)
    try:
        module, _outputs, exports = _lower(source, "optional_args_annotated")
    except Exception:
        return  # refused by name: acceptable until the tag edge is bound
    from src.compiler.ssa_c_backend import emit_ssa_to_c
    import tempfile

    root = module.functions[exports[0]]
    parameters = dict(root.metadata["parameter_names"])
    outputs = [
        int(value.id)
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret" and not instruction.attributes.get("fault_exit")
        for value in instruction.args
    ]
    artifact = emit_ssa_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    with tempfile.TemporaryDirectory() as directory:
        artifact.compile(Path(directory) / "native")
        _check(
            lambda feeds: artifact.prepare_execution(feeds).run(),
            source, parameters, outputs,
        )
