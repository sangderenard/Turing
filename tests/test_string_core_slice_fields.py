"""Field slices and ``list(...)`` copies Token Orchard's core relies on.

Shrunk from the string-core structure differential drivers.  Each program
runs natively (C and LLVM) and must equal Python exactly:

* ``for x in list(xs)`` over a grown list ran ONCE (the copy's one
  argument was read as its one member) and returned a wrong value silently;
* ``rows = list(self.values[b : b + 2]); rows[0]`` read the whole slice as
  row 0 (C/LLVM shortfalls: elementwise arithmetic over a span);
* a slice of an ``array`` field had no row contract and no arena.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "extraction_contracts" / "program_extraction.yaml"
)

LIST_SNAPSHOT_LOOP = '''
def root(n: int) -> int:
    xs = [7, 9, 11]
    xs.append(n)
    total = 0
    for x in list(xs):
        total = total * 10 + x
    return total
'''

ARRAY_FIELD_SLICE = '''
from array import array


class V:
    def __init__(self):
        self.atoms = array("I")


def root(n: int) -> int:
    v = V()
    v.atoms.append(7)
    v.atoms.append(9)
    v.atoms.append(11)
    xs = v.atoms[0:n + 1]
    return len(xs) * 100 + xs[0]
'''

ARRAY_FIELD_SLICE_LIST_LOOP = '''
from array import array


class V:
    def __init__(self):
        self.atoms = array("I")

    def f(self, t):
        total = 0
        for x in list(self.atoms[0:t]):
            total = total * 10 + x
        return total


def root(n: int) -> int:
    v = V()
    v.atoms.append(7)
    v.atoms.append(9)
    v.atoms.append(11)
    return v.f(n)
'''

FIELD_SLICE_LIST_ROWS = '''
from array import array


class Store:
    def __init__(self):
        self.values = array("Q")

    def add(self, v):
        self.values.append(v)

    def pair(self, begin):
        rows = list(self.values[begin : begin + 2])
        return rows[0] * 1000 + rows[1]


def root(n: int) -> int:
    store = Store()
    i = 0
    while i < n + 3:
        store.add(i * 3 + 1)
        i += 1
    return store.pair(1) % 1000000007
'''

PROGRAMS = {
    "list_snapshot_loop": (LIST_SNAPSHOT_LOOP, (0, 1, 4, -3)),
    "array_field_slice": (ARRAY_FIELD_SLICE, (0, 1, 2, 3)),
    "array_field_slice_list_loop": (ARRAY_FIELD_SLICE_LIST_LOOP, (0, 1, 2, 3)),
    "field_slice_list_rows": (FIELD_SLICE_LIST_ROWS, (0, 1, 5)),
}


def _python(source, n):
    namespace: dict = {"__name__": "slice_fields"}
    exec(compile(source, "<slice_fields>", "exec"), namespace)
    return namespace["root"](n)


def _lowered(name, source):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name=f"slice_fields_{name}",
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


@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_slice_and_copy_match_python_natively(name, tmp_path):
    from src.compiler.ssa_c_backend import emit_ssa_to_c
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )

    source, inputs = PROGRAMS[name]
    module, export, parameters, outputs = _lowered(name, source)
    artifact = emit_ssa_to_c(module, export)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "c")
    emitted = emit_ssa_function_to_llvm(module, export)
    assert not emitted.shortfalls, emitted.shortfalls
    llvm = compile_artifact(emitted, directory=tmp_path / "llvm")
    for n in inputs:
        feeds = {parameters["n"]: np.array([n], dtype=np.int64)}
        expected = _python(source, n)
        c_value = artifact.prepare_execution(feeds).run().buffers[outputs[0]]
        llvm_value = prepare_artifact_execution(llvm, feeds).run().buffers[outputs[0]]
        assert int(c_value.reshape(-1)[0]) == expected, (name, "c", n)
        assert int(llvm_value.reshape(-1)[0]) == expected, (name, "llvm", n)
