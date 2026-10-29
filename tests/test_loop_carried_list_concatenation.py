"""A loop that rebinds a list to ``xs + [...]`` carries the list's rows.

Shrunk from the string-core structure driver ``structure_queries``, whose
cover rebuild is ``rebuilt = [7]; for c in cover: ...; rebuilt = rebuilt +
ca``.  The ``+`` of two lists had no SOURCE_VALUE_CLASS row, so the loop
promotion (which reads residency from that page) carried the rebound list
as a scalar handle: ``len(xs)`` after the loop had no producer, and the
structure driver's merge named a sequence and a span.  Once carried, the
concatenation must read the carried list's rows (not the initial display's
one leaf), the arena must grow past the display's one row, and a later
``for x in xs`` must iterate the carried list.  Several trip counts run, so
a repeated initial element or a short arena would show.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "extraction_contracts" / "program_extraction.yaml"
)

COUNTED = '''
def root(n: int) -> int:
    xs = [7]
    for i in range(n % 6):
        xs = xs + [i, i * 2]
    d = len(xs)
    for x in xs:
        d = d * 7 + x
    return d
'''

COVER = '''
class V:
    def __init__(self):
        self.k = 3

    def expand(self, t):
        if t == 0:
            return 1, []
        return 0, [t, t + self.k]

    def cover(self, t):
        if t == 0:
            return 1, []
        return 0, [t, t + 1]


def root(n: int) -> int:
    v = V()
    st, cover = v.cover(n % 5)
    d = st
    if st == 0:
        rebuilt = [7]
        for c in cover:
            cs, ca = v.expand(c)
            if cs != 0:
                return -5
            rebuilt = rebuilt + ca
        d = d * 7 + len(rebuilt)
        for r in rebuilt:
            d = d * 7 + r
    return d
'''

PROGRAMS = {
    "counted": (COUNTED, (0, 1, 3, 5)),
    "cover": (COVER, (0, 1, 3, 4, 9)),
}


def _python(source, n):
    namespace: dict = {"__name__": "loop_carried_list_concatenation"}
    exec(compile(source, "<loop_carried_list_concatenation>", "exec"), namespace)
    return namespace["root"](n)


@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_loop_carried_list_concatenation_matches_python_natively(name, tmp_path):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_c_backend import emit_ssa_to_c
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )

    source, inputs = PROGRAMS[name]
    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name=f"loop_concat_{name}",
        extraction_contract=str(CONTRACT), progress=lambda _message: None,
    )
    root = module.functions[exports[0]]
    parameters = dict(root.metadata["parameter_names"])
    outputs = [
        int(value.id)
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret" and not instruction.attributes.get("fault_exit")
        for value in instruction.args
    ]
    artifact = emit_ssa_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "c")
    emitted = emit_ssa_function_to_llvm(module, exports[0])
    assert not emitted.shortfalls, emitted.shortfalls
    llvm = compile_artifact(emitted, directory=tmp_path / "llvm")
    for n in inputs:
        feeds = {parameters["n"]: np.array([n], dtype=np.int64)}
        expected = _python(source, n)
        c_run = artifact.prepare_execution(feeds).run().buffers
        llvm_run = prepare_artifact_execution(llvm, feeds).run().buffers
        c_value = next(c_run[o] for o in outputs if o in c_run)
        llvm_value = next(llvm_run[o] for o in outputs if o in llvm_run)
        assert int(c_value.reshape(-1)[0]) == expected, (name, "c", n)
        assert int(llvm_value.reshape(-1)[0]) == expected, (name, "llvm", n)
