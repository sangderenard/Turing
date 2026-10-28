"""``dst.extend(values[b:e])`` with run-time bounds matches Python natively.

Two wrong records, one statement (the token-orchard core's cover append):

* The extend was ordered at its call's start, before the slice
  ``values[b:e]`` (a mutation authored later on the same line) filled its
  arena, so it copied an empty sequence: ``lst`` kept its old rows on C and
  LLVM (silently wrong).  A method-call mutation now completes where its
  call ends, after its arguments.
* In a method, ``self.cover.extend(values[b:e])`` over a parameter list was
  refused: the typed-array receiver was not a builtin collection, so the
  slice counted as escaping, and its caller-owned source had no bound, so
  the slice was "a private sequence with no proven capacity".  The slice now
  takes the dynamic contract; its append reserves the ``e - b`` rows (clipped
  to ``len(values)``) it copies.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"

LOCAL = '''
from array import array


def root(m: int) -> int:
    n = m % 16
    values = []
    i = 0
    while i < n:
        values.append(i * 5 + 2)
        i += 1
    arr = array("{code}")
    b = n // 3
    e = n - n // 4 + 2
    arr.extend(values[b:e])
    lst = [1]
    lst.extend(values[b:e])
    t = 0
    j = 0
    while j < len(arr):
        t = t * 3 + int(arr[j])
        j += 1
    return t * 100 + len(lst) * 10000000 + sum(lst)
'''

FIELD = '''
from array import array


class Store:
    def __init__(self):
        self.cover = array("{code}")

    def add(self, values: list[int], b: int, e: int) -> int:
        self.cover.extend(values[b:e])
        size = len(self.cover)
        if size == 0:
            return 0
        return size * 1000000 + int(self.cover[size - 1])


def root(n: int) -> int:
    s = Store()
    i = 0
    total = 0
    while i < n:
        total = total * 7 + s.add([65 + i, 128578, 7, i], i % 3, 2 + i % 4)
        i += 1
    return total + len(s.cover)
'''


def _python(source, n):
    namespace = {}
    exec(compile(source, "<extend-slice>", "exec"), namespace)
    return namespace["root"](n)


def _native(source, name, tmp_path, ns):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_c_backend import emit_ssa_to_c
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )

    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name=name,
        extraction_contract=str(CONTRACT), progress=lambda _message: None,
    )
    export = exports[0]
    root = module.functions[export]
    parameter = next(iter(dict(root.metadata["parameter_names"]).values()))
    output = next(
        int(value.id)
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret" and not instruction.attributes.get("fault_exit")
        for value in instruction.args
    )
    artifact = emit_ssa_to_c(module, export)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "c")
    emitted = emit_ssa_function_to_llvm(module, export)
    assert not emitted.shortfalls, emitted.shortfalls
    llvm = compile_artifact(emitted, directory=tmp_path / "llvm")
    results = {}
    for n in ns:
        feeds = {parameter: np.array([n], dtype=np.int64)}
        results[n] = (
            int(artifact.prepare_execution(feeds).run().buffers[output].reshape(-1)[0]),
            int(prepare_artifact_execution(llvm, feeds).run().buffers[output].reshape(-1)[0]),
        )
    return results


@pytest.mark.parametrize("code", ["I", "Q"])
def test_local_extend_of_dynamic_slice(tmp_path, code):
    source = LOCAL.format(code=code)
    ns = (0, 1, 5, 12, 15)
    for n, (c_result, llvm_result) in _native(
        source, f"extend_slice_local_{code}", tmp_path, ns,
    ).items():
        expected = _python(source, n)
        assert (c_result, llvm_result) == (expected, expected), n


@pytest.mark.parametrize("code", ["I", "Q"])
def test_field_extend_of_parameter_slice(tmp_path, code):
    source = FIELD.format(code=code)
    ns = (0, 1, 4, 9)
    for n, (c_result, llvm_result) in _native(
        source, f"extend_slice_field_{code}", tmp_path, ns,
    ).items():
        expected = _python(source, n)
        assert (c_result, llvm_result) == (expected, expected), n
