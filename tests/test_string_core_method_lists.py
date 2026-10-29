"""Methods returning a built list, alone or with a status, to the caller.

Shrunk from the string-core structure differential drivers (Token Orchard's
``count, rows = store.delta(k)``).  Each program runs natively (C and LLVM)
and must equal Python exactly:

* ``count, rows = store.delta(k); rows[j]``: the caller's ``rows[j]`` had
  no class (a member of a method's returned tuple was read only from the
  whole tuple's joined class);
* the returned list is the loop's version of ``out = [0] * count``: the
  caller's arena (heap owner, length, capacity) was never bound to the
  callee's, so the two views disagreed (code 75 and heap corruption);
* ``rows = store.delta(k)`` with a bare list: a method call is named by
  its method, so the call rule never read the callee's return record;
* a method's uint64 arithmetic result is ``int128`` wide, which is no
  NumPy dtype.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "extraction_contracts" / "program_extraction.yaml"
)

STORE = '''
from array import array


class Store:
    def __init__(self):
        self.values = array("Q")

    def add(self, v):
        self.values.append(v)

    def delta(self, first):
        count = len(self.values) - first
        out = [0] * count
        i = 0
        while i < count:
            out[i] = int(self.values[first + i]) + 1
            i += 1
        return RETURNED
'''

CALLER = '''

def root(n: int) -> int:
    store = Store()
    i = 0
    while i < n:
        store.add(i * 3 + 1)
        i += 1
    BIND
    total = count
    j = 0
    while j < count:
        total = (total * 7 + rows[j]) % 1000003
        j += 1
    return total
'''

STATUS_AND_LIST = (
    STORE.replace("RETURNED", "count, out")
    + CALLER.replace("BIND", "count, rows = store.delta(n // 2)")
)
BARE_LIST = (
    STORE.replace("RETURNED", "out")
    + CALLER.replace(
        "BIND", "rows = store.delta(n // 2)\n    count = len(rows)"
    )
)

# Once the method's return record types ``store.pair(1)``, the uint64 rows
# times an int join to the internal ``int128`` layout, which NumPy was asked
# to promote (the lowering crashed: "data type 'int128' not understood").
WIDE_METHOD_RESULT = '''
from array import array


class Store:
    def __init__(self):
        self.values = array("Q")

    def add(self, v):
        self.values.append(v)

    def pair(self, begin):
        return self.values[begin] * 1000 + self.values[begin + 1]


def root(n: int) -> int:
    store = Store()
    i = 0
    while i < n + 3:
        store.add(i * 3 + 1)
        i += 1
    return store.pair(1) % 1000000007
'''

PROGRAMS = {
    "status_and_list": (STATUS_AND_LIST, (0, 1, 2, 9, 300)),
    "bare_list": (BARE_LIST, (0, 1, 5, 64)),
    "wide_method_result": (WIDE_METHOD_RESULT, (0, 1, 5)),
}


def _python(source, n):
    namespace: dict = {"__name__": "method_lists"}
    exec(compile(source, "<method_lists>", "exec"), namespace)
    return namespace["root"](n)


def _lowered(name, source):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name=f"method_lists_{name}",
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
def test_method_returned_lists_match_python_natively(name, tmp_path):
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
