"""Functions that return a list, run natively on C and LLVM against Python.

``def f(out, x): out[0] = x; return out`` returns its argument: the caller's
result IS the caller's list, so a store through either name is seen through
the other.  The returned name is the store's version of ``out``
(``IndexedStore``), and the readers that recognise a returned formal, its
element class and the native output record did not follow a store version
back to the list it writes, so the program was refused.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"

# Unannotated callee, mutated argument returned; the caller mutates through
# both names afterwards.
STORED_ARGUMENT = '''
def f(out, x):
    out[0] = x
    return out


def root(n: int) -> int:
    a = [1, 2, 3]
    t = f(a, n)
    t[1] = n + 5
    a[2] = n * 3
    return a[0] + a[1] * 10 + a[2] * 100 + t[0] * 1000 + t[2] * 10000
'''

# The annotated form of the same callee.
STORED_ARGUMENT_ANNOTATED = '''
def f(out: list[int], x: int) -> list[int]:
    out[0] = x
    return out


def root(n: int) -> int:
    a = [1, 2, 3]
    t = f(a, n)
    return t[0] + a[0] * 10
'''

PROGRAMS = {
    "stored_argument": STORED_ARGUMENT,
    "stored_argument_annotated": STORED_ARGUMENT_ANNOTATED,
}
INPUTS = (0, 4, -7)


def _python(source, n):
    namespace: dict = {"__name__": "list_returns"}
    exec(compile(source, "<list_returns>", "exec", dont_inherit=True), namespace)
    return namespace["root"](n)


def _lowered(source, name):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name=name,
        extraction_contract=str(CONTRACT), progress=lambda _message: None,
    )
    root = module.functions[exports[0]]
    # The first Ret carries the output identities (a fault exit included).
    outputs = [
        int(value.id)
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret"
        for value in instruction.args
    ][:1]
    return module, exports[0], dict(root.metadata["parameter_names"]), outputs


def _run_c(tmp_path, source, name):
    from src.compiler.ssa_c_backend import emit_ssa_to_c

    module, export, parameters, outputs = _lowered(source, name)
    artifact = emit_ssa_to_c(module, export)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "native")
    return lambda n: int(artifact.prepare_execution(
        {parameters["n"]: np.array([n], dtype=np.int64)}
    ).run().buffers[outputs[0]].reshape(-1)[0])


def _run_llvm(tmp_path, source, name):
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )

    module, export, parameters, outputs = _lowered(source, name)
    emitted = emit_ssa_function_to_llvm(module, export)
    assert not emitted.shortfalls, emitted.shortfalls
    artifact = compile_artifact(emitted, directory=tmp_path)
    return lambda n: int(prepare_artifact_execution(
        artifact, {parameters["n"]: np.array([n], dtype=np.int64)}
    ).run().buffers[outputs[0]].reshape(-1)[0])


@pytest.mark.parametrize("name", sorted(PROGRAMS))
@pytest.mark.parametrize("run", [_run_c, _run_llvm], ids=["c", "llvm"])
def test_returned_list_matches_python(tmp_path, run, name):
    source = PROGRAMS[name]
    native = run(tmp_path, source, f"list_returns_{name}")
    for n in INPUTS:
        assert native(n) == _python(source, n), (name, n)
