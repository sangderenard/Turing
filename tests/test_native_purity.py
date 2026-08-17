"""A native artifact must contain no unlowered source-language residue.

`ast_process_graph` records syntax it cannot lower as `opaque_python`
nodes, carrying only an `ast_type` and an `ast.dump` string. They are
diagnostic markers, not an execution mechanism -- nothing can run one, and
no backend has an emission for one. So the danger is not that Python gets
wrapped into a native build; it is that an unlowered node reaches a
backend and the backend says nothing useful, or worse, says nothing.

Native production has no room for either. These tests hold the invariant
explicitly instead of leaving it to the absence of a handler:

* an op with no emission is REFUSED, by name, as a shortfall -- and the
  artifact reports itself incomplete rather than partially emitted;
* an artifact that does emit is self-contained, declaring nothing beyond
  LLVM intrinsics -- no libpython, no callback into a host interpreter.

The second is the one worth stating out loud. `shortfalls == ()` says the
compiler believes it emitted everything; it does not say the result can
run without an interpreter behind it. Those are different claims and only
the link step can tell them apart.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from src.compiler.ssa_llvm_backend import (
    compile_artifact,
    emit_ssa_function_to_llvm,
    prepare_artifact_execution,
)
from src.transmogrifier.ssa import BasicBlock, Function, IRModule, Instr, SSAValue


def _module(op: str) -> tuple:
    source = SSAValue(1000, dtype="float64", shape=())
    result = SSAValue(1001, dtype="float64", shape=())
    function = Function(
        "program",
        [source],
        {
            "entry": BasicBlock("entry", [
                Instr(op, [source], result),
                Instr("Ret", [result], None),
            ])
        },
    )
    return IRModule({"program": function}), source, result


def test_an_unlowered_node_is_refused_by_name():
    """`opaque_python` must not emit as anything at all."""
    module, _source, _result = _module("opaque_python")
    artifact = emit_ssa_function_to_llvm(module, "program")

    assert artifact.shortfalls != (), (
        "opaque_python emitted without complaint; an unlowered node reaching "
        "a backend must be reported, never silently rendered"
    )
    assert not artifact.complete
    reported = " ".join(str(item) for item in artifact.shortfalls)
    assert "opaque_python" in reported, (
        f"the shortfall does not name the offending op: {reported!r}"
    )


def test_a_lowerable_program_stays_complete():
    """The refusal above must be about the op, not about the shape.

    Without this, `test_an_unlowered_node_is_refused_by_name` would still
    pass if emission were broken for every op, which would make it a test
    of nothing.
    """
    module, _source, _result = _module("Abs")
    artifact = emit_ssa_function_to_llvm(module, "program")
    assert artifact.shortfalls == (), artifact.shortfalls
    assert artifact.complete


def test_emitted_native_module_is_self_contained(tmp_path):
    """No external symbol beyond LLVM intrinsics -- nothing to link Python to.

    A host-interpreter callback would appear here as a `declare` of a
    symbol that is not an `llvm.*` intrinsic. Checking the emitted module
    is what separates "the compiler thinks it emitted everything" from
    "the result can run without an interpreter behind it".
    """
    module, source, result = _module("Abs")
    artifact = emit_ssa_function_to_llvm(module, "program")
    native = compile_artifact(artifact, directory=tmp_path / "pure")

    text = "\n".join(
        path.read_text()
        for path in (tmp_path / "pure").glob("*.ll")
    )
    assert text, "no LLVM text was written; nothing was checked"
    declared = {
        match.group(1)
        for match in re.finditer(r"^declare[^@]*@([\w.$]+)", text, re.M)
    }
    foreign = {name for name in declared if not name.startswith("llvm.")}
    assert not foreign, (
        f"the emitted module declares non-intrinsic external symbols: "
        f"{sorted(foreign)}. A native artifact must not depend on a host "
        "runtime."
    )

    execution = prepare_artifact_execution(
        native, {int(source.id): np.asarray([-2.5], dtype=np.float64)},
    )
    execution.run()
    produced = float(
        np.asarray(execution.buffers[int(result.id)], dtype=float).reshape(-1)[0]
    )
    assert produced == pytest.approx(2.5)
