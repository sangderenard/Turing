"""An empty list that appends grow, rebound on one branch, is carried.

Shrunk from the string-core intern differential driver.  ``atoms = []``
grown by appends, then rebound to a new ``[]`` grown by appends on one
branch, was refused as "bound to lists of different shapes on two
branches": an empty initial display was taken to have no element type even
though its appends type it.  Once carried, the rebound arm's list was also
minted as a scalar arm value -- its truth test ``len(atoms) > 0`` under the
list's own value id -- so the appends' storage handle became a bool.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"

REBOUND_GROWN = '''
def root(seed, n):
    atoms = []
    j = 0
    while j < n:
        atoms.append(j + seed)
        j += 1
    if seed % 2 == 0:
        atoms = []
        j = 0
        while j < 3:
            atoms.append(j * 7)
            j += 1
    total = len(atoms) * 1000
    j = 0
    while j < len(atoms):
        total += atoms[j]
        j += 1
    return total
'''

REBOUND_ONE_APPEND = '''
def root(seed, n):
    atoms = []
    for j in range(n):
        atoms.append(j * 3 + seed)
    if n > 2:
        atoms = []
        atoms.append(seed + 100)
    return len(atoms) * 1000 + atoms[len(atoms) - 1] if atoms else -1
'''

REBOUND_UNGROWN = '''
def root(seed, n):
    atoms = []
    if n > 1:
        atoms = [seed, n, 7]
    total = len(atoms) * 1000
    for a in atoms:
        total += a
    return total
'''

CASES = ((0, 0), (0, 2), (2, 3), (1, 5))


def _python(source, seed, n):
    namespace: dict = {}
    exec(compile(source, "<program>", "exec"), namespace)
    return namespace["root"](seed, n)


def _lowered(source):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name="rebound_empty_list",
        extraction_contract=str(CONTRACT), progress=lambda _message: None,
        root_calls=[{"seed": seed, "n": n} for seed, n in CASES],
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


def _feeds(parameters, seed, n):
    return {
        parameters["seed"]: np.array([seed], dtype=np.int64),
        parameters["n"]: np.array([n], dtype=np.int64),
    }


def _run_c(tmp_path, lowered, seed, n):
    from src.compiler.ssa_c_backend import emit_ssa_to_c

    module, export, parameters, outputs = lowered
    artifact = emit_ssa_to_c(module, export)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / f"c_{seed}_{n}")
    execution = artifact.prepare_execution(_feeds(parameters, seed, n)).run()
    return int(execution.buffers[outputs[0]].reshape(-1)[0])


def _run_llvm(tmp_path, lowered, seed, n):
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )

    module, export, parameters, outputs = lowered
    emitted = emit_ssa_function_to_llvm(module, export)
    assert not emitted.shortfalls, emitted.shortfalls
    artifact = compile_artifact(emitted, directory=tmp_path / f"llvm_{seed}_{n}")
    execution = prepare_artifact_execution(
        artifact, _feeds(parameters, seed, n)
    ).run()
    return int(execution.buffers[outputs[0]].reshape(-1)[0])


@pytest.mark.parametrize("source", [REBOUND_GROWN, REBOUND_ONE_APPEND, REBOUND_UNGROWN], ids=["rebound_grown", "rebound_one_append", "rebound_ungrown"])
@pytest.mark.parametrize("run", [_run_c, _run_llvm], ids=["c", "llvm"])
def test_rebound_empty_list_matches_python(tmp_path, run, source):
    lowered = _lowered(source)
    for seed, n in CASES:
        assert run(tmp_path, lowered, seed, n) == _python(source, seed, n), (seed, n)
