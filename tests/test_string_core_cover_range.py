"""Token Orchard ``cover``: one returned slot, ``[token]`` and uint64 rows.

Shrunk from the string-core structure differential drivers.  ``cover``
returns ``(STATUS_OK, [token])`` for an atom and ``(STATUS_OK,
list(self.cover_tokens[begin : begin + count]))`` over ``array("Q")`` for
an aggregate, from the same slot.  Each program runs natively (C and LLVM)
and must equal Python exactly:

* the slot's rows settle by proven range: a token proven within uint64's
  domain (an interval or a literal at the callsite) beside uint64 rows is
  a uint64 row, so the slot's one arena has one row layout;
* two copies of one method minted for different actual intervals no longer
  share one numeric scope (their formal's interval row disagreed);
* a slice whose bounds are uint64 rows (``begin``, ``count`` read from
  ``array("Q")``) is clipped in the bounds' own domain; it could not be
  emitted before (mixed signed/unsigned sign tests).

* ``cover(n)`` with ``n`` unbounded: ``[token]`` is built only where
  ``kind == 2`` held, and ``kind`` is ``self.kind(token)``, which returns 2
  only under ``ATOM_BASE <= token < ATOM_LIMIT``.  That read is proven
  there (the callee's guarded formal at its returns of 2), so the row is
  uint64 too.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "extraction_contracts" / "program_extraction.yaml"
)

VOCABULARY = '''
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
'''

ROOT = '''

def root(n: int) -> int:
    v = V()
    v.add(5, 7)
    v.add(11, 13)
    d = n % 3
    status, parts = v.cover(FIRST)
    d = (d * 1000003 + status * 10 + len(parts) + 7) % 4611686018427387847
    for p in parts:
        d = (d * 1000003 + p + 7) % 4611686018427387847
    status, parts = v.cover(SECOND)
    d = (d * 1000003 + status * 10 + len(parts) + 7) % 4611686018427387847
    for p in parts:
        d = (d * 1000003 + p + 7) % 4611686018427387847
    return d
'''

PROVEN_INTERVALS = VOCABULARY + ROOT.replace(
    "FIRST", "n % 7 + 3"
).replace("SECOND", "AGG_BASE + n % 2")
LITERAL_TOKENS = VOCABULARY + ROOT.replace("FIRST", "5").replace(
    "SECOND", "1001"
)
UNPROVEN_TOKEN = VOCABULARY + ROOT.replace("FIRST", "n").replace(
    "SECOND", "AGG_BASE + n % 2"
)

UNSIGNED_SLICE_BOUNDS = '''
from array import array


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

    def part(self, row):
        begin = self.cover_begin[row]
        count = self.cover_count[row]
        return list(self.cover_tokens[begin : begin + count])


def root(n: int) -> int:
    v = V()
    v.add(5, 7)
    v.add(11, 13)
    parts = v.part(n % 2)
    d = len(parts)
    for p in parts:
        d = (d * 1000003 + p + 7) % 4611686018427387847
    return d
'''

PROGRAMS = {
    "proven_intervals": (PROVEN_INTERVALS, (0, 1, 2, -5, 13)),
    "literal_tokens": (LITERAL_TOKENS, (0, 1)),
    "unsigned_slice_bounds": (UNSIGNED_SLICE_BOUNDS, (0, 1, 2)),
    "token_proven_by_kind": (
        UNPROVEN_TOKEN, (0, 1, 5, 299, 300, -4, 1001, 123456789),
    ),
}


def _python(source, n):
    namespace: dict = {"__name__": "cover_range"}
    exec(compile(source, "<cover_range>", "exec"), namespace)
    return namespace["root"](n)


def _lowered(name, source):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name=f"cover_range_{name}",
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
def test_cover_slot_matches_python_natively(name, tmp_path):
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

