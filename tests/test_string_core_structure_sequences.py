"""Sequence shapes Token Orchard's ``Vocabulary.expand`` / ``slice`` rely on.

Shrunk from the string-core structure differential drivers.  Each program
runs natively (C and LLVM) and must equal Python exactly:

* ``for x in list(xs)`` over a grown list ran ONCE (its members were read
  as the materializer's arguments) and returned a wrong value silently;
* a slice of an ``array("I")`` field had no row contract and no arena;
* ``list(<array slice>)`` read its slice as a one-member display;
* ``status, atoms = v.expand(t)`` typed ``atoms`` only from the whole
  tuple's joined class (int64 with uint32 rows: none), so it was refused;
* a function returning ``[]`` on some paths and ``list(...)`` on another was
  refused as divergent storage, though ``list(...)`` is born there.
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

RETURNED_ARRAY_SLICE_LIST = '''
from array import array


class V:
    def __init__(self):
        self.atoms = array("I")

    def expand(self, t):
        return 0, list(self.atoms[0:t])


def root(n: int) -> int:
    v = V()
    v.atoms.append(7)
    v.atoms.append(9)
    v.atoms.append(11)
    st, xs = v.expand(n)
    total = st * 100 + len(xs)
    for x in xs:
        total = total * 10 + x
    return total
'''

DIVERGENT_LIST_RETURNS = '''
from array import array


class V:
    def __init__(self):
        self.atoms = array("I")

    def expand(self, t):
        if t == 0:
            return 3, []
        if t == 1:
            return 0, []
        return 0, list(self.atoms[0:t - 1])


def root(n: int) -> int:
    v = V()
    v.atoms.append(7)
    v.atoms.append(9)
    v.atoms.append(11)
    st, xs = v.expand(n)
    total = st * 100 + len(xs)
    for x in xs:
        total = total * 10 + x
    return total
'''

PROGRAMS = {
    "list_snapshot_loop": (LIST_SNAPSHOT_LOOP, (0, 1, 4, -3)),
    "array_field_slice": (ARRAY_FIELD_SLICE, (0, 1, 2, 3)),
    "array_field_slice_list_loop": (ARRAY_FIELD_SLICE_LIST_LOOP, (0, 1, 2, 3)),
    "returned_array_slice_list": (RETURNED_ARRAY_SLICE_LIST, (0, 1, 2, 3)),
    "divergent_list_returns": (DIVERGENT_LIST_RETURNS, (0, 1, 2, 3, 4)),
}


def _python(source, n):
    namespace: dict = {"__name__": "structure_sequences"}
    exec(compile(source, "<structure>", "exec"), namespace)
    return namespace["root"](n)


def _lowered(name, source):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name=f"structure_{name}",
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
def test_sequence_shape_matches_python_natively(name, tmp_path):
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
