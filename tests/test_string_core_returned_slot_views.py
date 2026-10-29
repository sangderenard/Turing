"""Returned list slots whose views disagreed on storage or row layout.

Shrunk from the string-core structure differential drivers.  Each program
runs natively (C and LLVM) and must equal Python exactly:

* ``divergent_displays``: ``f`` returns ``[]`` or ``[t, t + 1]`` from one
  slot.  The caller's binding and the callee's returned arena are one
  frame sequence; the duplicate-storage key now names that sequence, so
  the two views no longer read as leased storage beside an owner.
* ``uint32_len_only``: ``expand`` returns ``list(self.atoms[0 : t - 1])``
  over ``array("I")`` and the caller reads only ``len(xs)``.  Only
  int64/float64/bool rows named a component's element before, so the
  callee's uint32 rows named none and the caller's view was born with an
  unknown row; any native scalar layout is a row class now.
* ``kind_guarded_len_only``: Token Orchard ``cover(n)`` over uint64 rows
  with ``n`` unbounded; ``[token]`` is proven within uint64 by
  ``self.kind(token) == 2``, and the caller reads only ``len``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "extraction_contracts" / "program_extraction.yaml"
)

DIVERGENT_DISPLAYS = '''
def f(t):
    if t == 0:
        return 3, []
    return 0, [t, t + 1]


def root(n: int) -> int:
    st, xs = f(n)
    return st * 1000 + len(xs) * 100
'''

UINT32_LEN_ONLY = '''

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
    return st * 1000 + len(xs) * 100
'''

KIND_GUARDED_LEN_ONLY = '''

from array import array

ATOM_BASE = 1
ATOM_LIMIT = 300
AGG_BASE = 1000


class V:
    def __init__(self):
        self.cover_begin = array("Q")
        self.cover_count = array("Q")
        self.cover_tokens = array("Q")

    def add(self, a, b):
        self.cover_begin.append(len(self.cover_tokens))
        self.cover_count.append(2)
        self.cover_tokens.append(a)
        self.cover_tokens.append(b)
        return AGG_BASE + len(self.cover_begin) - 1

    def kind(self, token):
        if token == 0:
            return 1
        if ATOM_BASE <= token < ATOM_LIMIT:
            return 2
        row = token - AGG_BASE
        if 0 <= row < len(self.cover_begin):
            return 3
        return 0

    def cover(self, token):
        kind = self.kind(token)
        if kind == 0:
            return 9, []
        if kind == 1:
            return 0, []
        if kind == 2:
            return 0, [token]
        row = token - AGG_BASE
        begin = self.cover_begin[row]
        count = self.cover_count[row]
        return 0, list(self.cover_tokens[begin : begin + count])


def root(n: int) -> int:
    v = V()
    v.add(5, 7)
    v.add(11, 13)
    st, cover = v.cover(n)
    d = st * 1000 + len(cover)
    return d
'''

PROGRAMS = {
    "divergent_displays": (DIVERGENT_DISPLAYS, (0, 1, 2, -3)),
    "uint32_len_only": (UINT32_LEN_ONLY, (0, 1, 2, 4)),
    "kind_guarded_len_only": (
        KIND_GUARDED_LEN_ONLY, (0, 1, 5, 299, 300, 1000, 1001, -4),
    ),
}


def _python(source, n):
    namespace: dict = {"__name__": "returned_slot_views"}
    exec(compile(source, "<returned_slot_views>", "exec"), namespace)
    return namespace["root"](n)


def _lowered(name, source):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name=f"returned_slot_views_{name}",
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
def test_returned_slot_views_match_python_natively(name, tmp_path):
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

