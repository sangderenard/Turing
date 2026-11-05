"""``[] + xs`` is a sequence concatenation, lowered natively.

Shrunk from the string-core structure driver ``structure_queries``, which
rebuilds a cover's atoms as ``rebuilt = []; ...; rebuilt = rebuilt + ca``.
The folded ``[]`` is a constant no tensor flattens, so the ``+`` was
classified coordinator metadata and dropped: its result had no producer
and the program read the list back empty (silently).  The element class of
``[]`` is the one its declaration posts (``list[int]``): the annotated
program runs natively (C and LLVM) and must equal Python.  The unannotated
one has no posted class, and its concatenation refuses by name.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "extraction_contracts" / "program_extraction.yaml"
)

EXPAND = '''
def expand(t):
    if t == 0:
        return 1, []
    return 0, [t, t + 3]
'''

ANNOTATED = EXPAND + '''

def root(n: int) -> int:
    rebuilt: list[int] = []
    cs, ca = expand(n % 5)
    rebuilt = rebuilt + ca
    d = cs * 7 + len(rebuilt)
    for r in rebuilt:
        d = d * 7 + r
    return d
'''

UNANNOTATED = ANNOTATED.replace("rebuilt: list[int] = []", "rebuilt = []")

PROGRAMS = {
    "annotated": (ANNOTATED, (0, 1, 3, 4, 9)),
}

UNANNOTATED_LEN = EXPAND + '''

def root(n: int) -> int:
    rebuilt = []
    cs, ca = expand(n % 5)
    rebuilt = rebuilt + ca
    return cs * 7 + len(rebuilt)
'''


def _python(source, n):
    namespace: dict = {"__name__": "empty_list_concatenation"}
    exec(compile(source, "<empty_list_concatenation>", "exec"), namespace)
    return namespace["root"](n)


def _lowered(name, source):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name=f"empty_concat_{name}",
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
def test_empty_list_concatenation_matches_python_natively(name, tmp_path):
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


@pytest.mark.parametrize(("source", "refusal"), [
    (UNANNOTATED_LEN, r"list concatenation \d+ .* is planned by no region"),
    (UNANNOTATED, r"input 'r' in function 'root'\) has no settled dtype"),
])
def test_unannotated_empty_list_concatenation_refuses_by_name(source, refusal):
    with pytest.raises(ValueError, match=refusal):
        _lowered("unannotated", source)
