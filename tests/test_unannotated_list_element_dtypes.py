"""Rows of unannotated local lists keep their proven class.

Shrunk from the string-core intern differential driver (a span history kept
in plain lists and an unannotated helper).  Two refusals:

* ``pick(hist[seed % 2], n)`` with ``def pick(domain, atom)``: the element
  read had no class cell and no descriptor at the callsite, so the callee's
  formal had no dtype ("region value ... has no settled dtype"), although the
  caller's planner proves the read is int64.
* ``d = 1; if ...: d = hist[k]; hist.append(d)``: the list's rows join its
  appends, the append writes ``d``, ``d`` merges a row read; the cycle had no
  seed, so the phi refused ("no planned scalar dtype").  It now assumes the
  narrowest consistent class (bool < int64 < float64), checked against the
  writers; an int|float cycle is still refused by name.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"

LCG = "(s * 1103515245 + 12345) % 2147483648"

CALL_ELEMENT = '''
def pick(domain, atom):
    return domain * 10 + atom


def root(seed: int, n: int) -> int:
    hist = [1, 2]
    return pick(hist[seed % 2], n)
'''

CYCLE = f'''
def pick(domain, atom):
    if domain == 2:
        return 100 + atom
    return atom


def root(seed: int, n: int) -> int:
    hist = []
    s = seed
    d = 0
    i = 0
    while i < n:
        s = {LCG}
        domain = 1
        if s % 3 == 0:
            domain = 2
        if len(hist) > 0 and s % 4 == 0:
            domain = hist[(s // 4) % len(hist)]
        hist.append(domain)
        d = (d * 3 + pick(domain, s % 50)) % 1000000007
        i += 1
    return d
'''

FLOAT_CYCLE = f'''
def root(seed: int, n: int) -> int:
    hist = []
    s = seed
    d = 0.0
    i = 0
    while i < n:
        s = {LCG}
        v = 0.5
        if s % 3 == 0:
            v = 1.25
        if len(hist) > 0 and s % 4 == 0:
            v = hist[(s // 4) % len(hist)]
        hist.append(v)
        d = d * 0.5 + v
        i += 1
    return int(d * 1000)
'''

MIXED_CYCLE = f'''
def root(seed: int, n: int) -> int:
    hist = []
    s = seed
    d = 0
    i = 0
    while i < n:
        s = {LCG}
        v = s % 7
        if s % 5 == 0:
            v = 0.5
        if len(hist) > 0 and s % 4 == 0:
            v = hist[(s // 4) % len(hist)]
        hist.append(v)
        d = (d * 3 + int(v * 2)) % 1000000007
        i += 1
    return d
'''


def _python(source, seed, n):
    namespace: dict = {}
    exec(compile(source, "<program>", "exec"), namespace)
    return namespace["root"](seed, n)


def _lowered(source, name):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name=name,
        extraction_contract=str(CONTRACT), progress=lambda _message: None,
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


CASES = ((0, 0), (1, 1), (2, 10), (7, 60))


@pytest.mark.parametrize("run", [_run_c, _run_llvm], ids=["c", "llvm"])
@pytest.mark.parametrize(
    "name, source",
    [("call_element", CALL_ELEMENT), ("cycle", CYCLE), ("float_cycle", FLOAT_CYCLE)],
)
def test_list_rows_match_python(tmp_path, run, name, source):
    lowered = _lowered(source, f"list_rows_{name}")
    for seed, n in CASES:
        assert run(tmp_path, lowered, seed, n) == _python(source, seed, n), (seed, n)


def test_int_float_cycle_is_refused_by_name():
    with pytest.raises(ValueError, match="no planned scalar dtype"):
        _lowered(MIXED_CYCLE, "list_rows_mixed")
