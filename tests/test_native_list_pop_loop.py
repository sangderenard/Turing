"""A callee that pops its list argument in a ``while`` whose test reads it.

``while len(out) > 1 and out[len(out) - 1] == 0: out.pop()`` in a callee
given the caller's list.  Lowering refused it three ways:

* the trailing ``if len(out) == 1: out[0] = 0``: its predicate's region sat
  after the conditional in the scheduled run, so the arena walk ordered the
  predicate's read of ``out`` after the arm's store while the predicate
  value ordered it before ("control effect order conflicts with value
  dependencies");
* the latch re-tested the loop with the ENTRY test's ``len(out)`` loads --
  stale after a pop, and the one under the ``and`` defined only in the entry
  test's short-circuit arm ("Call reads a value defined at if_true");
* with ``out`` unannotated, the loop's ``pop`` stayed an opaque effect, and
  ``out[...]`` read a raw span while ``pop`` made ``out`` a heap sequence.

Both native lanes must match Python, with and without trailing pops.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"

TRIM_THEN_ZERO = '''
def _trim(out):
    while len(out) > 1 and out[len(out) - 1] == 0:
        out.pop()
    if len(out) == 1:
        out[0] = 0


def root(n: int):
    out = [1, n, 0, 0]
    _trim(out)
    return len(out) + out[0]
'''

TRIM_ONLY = '''
def _trim(out):
    while len(out) > 1 and out[len(out) - 1] == 0:
        out.pop()


def root(n: int):
    out = [1, n, 0, 0]
    _trim(out)
    return len(out) + out[0]
'''

# ``n = 0`` pops down to ``[0]``, where only a length read again at the
# latch stops the loop (the entry's 3 would pop on into an empty list).
TRIM_TO_ONE = '''
def _trim(out):
    while len(out) > 1 and out[len(out) - 1] == 0:
        out.pop()


def root(n: int):
    out = [n, 0, 0]
    _trim(out)
    return len(out) * 10 + out[0]
'''

GUARDED_POP = '''
def _trim(out):
    if out[-1] == 0:
        out.pop()


def root(n: int):
    out = [1, 2, n]
    _trim(out)
    return len(out) * 10 + out[len(out) - 1]
'''

PROGRAMS = {
    "trim_then_zero": TRIM_THEN_ZERO,
    "trim_only": TRIM_ONLY,
    "trim_to_one": TRIM_TO_ONE,
    "guarded_pop": GUARDED_POP,
}
INPUTS = (0, 7, -3)


def _python(source, n):
    namespace: dict = {}
    exec(compile(source, "<pop-loop>", "exec", dont_inherit=True), namespace)
    return namespace["root"](n)


@functools.lru_cache(maxsize=None)
def _lowered(name):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, exports = lower_ast_source_to_ssa(
        PROGRAMS[name], "root", name=f"pop_loop_{name}",
        extraction_contract=str(CONTRACT), progress=lambda _message: None,
    )
    root = module.functions[exports[0]]
    # A bounds check leaves through a fault-exit return carrying the same
    # output slots; the first return names them (as the corpus reads them).
    outputs = [
        int(value.id)
        for value in next(
            instruction.args
            for block in root.blocks.values()
            for instruction in block.instrs
            if instruction.op == "Ret"
        )
    ]
    return module, exports[0], dict(root.metadata["parameter_names"]), outputs


def _run_c(tmp_path, name):
    from src.compiler.ssa_c_backend import emit_ssa_to_c

    module, export, parameters, outputs = _lowered(name)
    artifact = emit_ssa_to_c(module, export)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "native")
    results = []
    for n in INPUTS:
        execution = artifact.prepare_execution(
            {parameters["n"]: np.array([n], dtype=np.int64)}
        ).run()
        results.append(int(execution.buffers[outputs[0]].reshape(-1)[0]))
    return results


def _run_llvm(tmp_path, name):
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )

    module, export, parameters, outputs = _lowered(name)
    emitted = emit_ssa_function_to_llvm(module, export)
    assert not emitted.shortfalls, emitted.shortfalls
    artifact = compile_artifact(emitted, directory=tmp_path)
    results = []
    for n in INPUTS:
        execution = prepare_artifact_execution(
            artifact, {parameters["n"]: np.array([n], dtype=np.int64)}
        ).run()
        results.append(int(execution.buffers[outputs[0]].reshape(-1)[0]))
    return results


@pytest.mark.parametrize("name", sorted(PROGRAMS))
@pytest.mark.parametrize("run", [_run_c, _run_llvm], ids=["c", "llvm"])
def test_list_popped_by_callee_matches_python(tmp_path, run, name):
    expected = [_python(PROGRAMS[name], n) for n in INPUTS]
    assert run(tmp_path, name) == expected
