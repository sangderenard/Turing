"""A returned list that is the merge of a loop's version and its birth.

Shrunk from Token Orchard ``intern_batch``: ``statuses = [STATUS_OK] *
item_count; if envelope_status == STATUS_OK: while ...: statuses[i] = ...;
return envelope_status, statuses, tokens``.  The returned value is the
``if`` merge of the loop's version of the list and its birth, which is one
storage.  Before, that merge was no store or loop version:

* the caller's ``rows[j]`` had no element class (the method's returned
  member was the merge, not the list);
* the callee's descriptor was never found for the caller's view, whose
  capacity and status cells nothing defined (refused by the final
  concordance), and the two views' capacity contracts were never joined
  (code 75).

The program runs natively (C and LLVM) and must equal Python exactly,
including the ``status != 0`` path that skips the fill.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "extraction_contracts" / "program_extraction.yaml"
)

MERGED_RETURN = '''
from array import array


class Store:
    def __init__(self):
        self.values = array("Q")

    def add(self, v):
        self.values.append(v)

    def envelope(self, first):
        if first > len(self.values):
            return 4, 0
        return 0, len(self.values) - first

    def delta(self, first):
        status, count = self.envelope(first)
        out = [0] * count
        if status == 0:
            i = 0
            while i < count:
                out[i] = int(self.values[first + i]) + 1
                i += 1
        return status, out


def root(n: int) -> int:
    store = Store()
    i = 0
    while i < n:
        store.add(i * 3 + 1)
        i += 1
    status, rows = store.delta(n // 2 + (n % 4 == 3) * (n + 2))
    total = status * 100 + len(rows)
    j = 0
    while j < len(rows):
        total = (total * 7 + rows[j]) % 1000003
        j += 1
    return total
'''

PROGRAMS = {
    "merged_return": (MERGED_RETURN, (0, 1, 2, 3, 5, 7, 9)),
}


def _python(source, n):
    namespace: dict = {"__name__": "returned_merge"}
    exec(compile(source, "<returned_merge>", "exec"), namespace)
    return namespace["root"](n)


def _lowered(name, source):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name=f"returned_merge_{name}",
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
def test_returned_merge_matches_python_natively(name, tmp_path):
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

