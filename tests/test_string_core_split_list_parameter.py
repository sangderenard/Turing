"""A list display passed to an unannotated parameter read by index and len.

Shrunk from Token Orchard ``intern(domain, atoms)``, which calls
``self.intern_batch(domain, atoms, [0, len(atoms)])``.  ``atoms[0]`` splits
the unannotated parameter into member formals; the whole parameter was then
never bound to the caller's display, so the callee's view of it (kept for
``len``) was leased a fresh arena nothing sized (refused as
dynamic-growth-unavailable), and the callee initialized that view empty, so
``len(atoms)`` read 0 once storage existed.  Each program runs natively (C
and LLVM) and must equal Python exactly:

* ``free_function``: ``one([n % 5 + 1, 2])`` reads ``atoms[0]`` and
  ``len(atoms)``;
* ``method``: the same through ``v.one([...])``;
* ``mutated``: the callee appends to the caller's list, which the caller
  then reads;
* ``batch_two_calls_down``: the ``intern`` shape -- a method passes
  ``[0, len(atoms)]`` on to a batch method that returns lists filled under
  a status test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "extraction_contracts" / "program_extraction.yaml"
)

FREE_FUNCTION = '''
def one(atoms):
    return atoms[0] * 10 + len(atoms)


def root(n: int) -> int:
    return one([n % 5 + 1, 2])
'''

METHOD = '''
class V:
    def __init__(self):
        self.k = 3

    def one(self, atoms):
        return atoms[0] * 10 + len(atoms) + self.k


def root(n: int) -> int:
    v = V()
    return v.one([n % 5 + 1, 2])
'''

MUTATED = '''
def one(atoms):
    atoms.append(3)
    return atoms[0] * 10 + len(atoms)


def root(n: int) -> int:
    xs = [n % 5 + 1, 2]
    r = one(xs)
    return r * 100 + len(xs) * 10 + xs[2]
'''

BATCH_TWO_CALLS_DOWN = '''
class V:
    def __init__(self):
        self.k = 3

    def envelope(self, offsets):
        if len(offsets) < 1:
            return 4, 0
        return 0, len(offsets) - 1

    def batch(self, values, offsets):
        envelope_status, item_count = self.envelope(offsets)
        statuses = [0] * item_count
        tokens = [0] * item_count
        if envelope_status == 0:
            i = 0
            while i < item_count:
                begin = offsets[i]
                statuses[i] = begin % 2
                tokens[i] = values[begin] + self.k
                i += 1
        return envelope_status, statuses, tokens

    def one(self, atoms):
        b, statuses, tokens = self.batch(atoms, [0, len(atoms)])
        if b != 0:
            return b, 0
        return statuses[0], tokens[0]


def root(n: int) -> int:
    v = V()
    s, t = v.one([n % 5 + 1, 2])
    return s * 1000 + t
'''

PROGRAMS = {
    "free_function": (FREE_FUNCTION, (0, 1, 2, 5)),
    "method": (METHOD, (0, 1, 2, 5)),
    "mutated": (MUTATED, (0, 1, 2, 5)),
    "batch_two_calls_down": (BATCH_TWO_CALLS_DOWN, (0, 1, 2, 5)),
}


def _python(source, n):
    namespace: dict = {"__name__": "split_list_parameter"}
    exec(compile(source, "<split_list_parameter>", "exec"), namespace)
    return namespace["root"](n)


def _lowered(name, source):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name=f"split_list_parameter_{name}",
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
def test_split_list_parameter_matches_python_natively(name, tmp_path):
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
