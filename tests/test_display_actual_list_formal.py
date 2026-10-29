"""An unannotated formal fed a list display, forwarded to a ``list[int]`` formal.

Program: ``one(atoms)`` called as ``v.one([97, 98])`` passes ``atoms`` to
``batch(self, values: list[int], offsets)``, which reads
``values[offsets[i]]`` in a loop (the shape of string-core ``intern`` calling
``intern_preflighted_batch``).  Wrong record: only an aggregate-parameter
actual gave an unannotated formal its sequence record, so ``atoms`` stayed a
bare span with no length member; the length, capacity and status members of
``values`` were leased as fresh storage that no caller wrote, and the bounds
check read garbage (IndexError where Python returns 97).

A list display actual now gives the formal the same record.  A parameter
member that is still leased from nothing is refused by name when the callee
reads it, and left alone when it does not.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"
EXECUTION = Path(__file__).resolve().parents[1] / "extraction_contracts" / "vehicle_full_native_execution.yaml"

SOURCES = {
    "loop_read": '''
class V:
    def __init__(self):
        self.base = 0

    def batch(self, values: list[int], offsets):
        count = len(offsets) - 1
        total = 0
        i = 0
        while i < count:
            total += values[offsets[i]]
            i += 1
        return total

    def one(self, atoms):
        return self.batch(atoms, [0, len(atoms)])


def root(n: int):
    v = V()
    return v.one([97, 98]) + n
''',
    "static_read": '''
class V:
    def __init__(self):
        self.base = 0

    def batch(self, values: list[int], offsets):
        return 0, values[offsets[0]] + len(offsets) * 1000

    def one(self, atoms):
        env, token = self.batch(atoms, [0, len(atoms)])
        return token


def root(n: int):
    v = V()
    return v.one([97, 98]) + n
''',
}

CASES = (0, 1, 2, 9)


def _python(source, n):
    namespace = {}
    exec(source, namespace)
    return namespace["root"](n)


def _lower(source, name):
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    contract = ExtractionContract(CONTRACT).with_execution_file(EXECUTION)
    return lower_ast_source_to_ssa(
        source, "root", name=name,
        extraction_contract=contract, progress=lambda _message: None,
    )


@pytest.fixture(scope="module", params=sorted(SOURCES))
def lowered(request):
    source = SOURCES[request.param]
    module, _outputs, exports = _lower(
        source, f"display_actual_list_formal_{request.param}",
    )
    root = module.functions[exports[0]]
    outputs = [
        int(value.id)
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret" and not instruction.attributes.get("fault_exit")
        for value in instruction.args
    ]
    return source, module, exports[0], dict(root.metadata["parameter_names"]), outputs


def test_matches_python_on_c(tmp_path, lowered):
    from src.compiler.ssa_c_backend import emit_ssa_to_c

    source, module, export, parameters, outputs = lowered
    artifact = emit_ssa_to_c(module, export)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "native")
    for n in CASES:
        execution = artifact.prepare_execution(
            {parameters["n"]: np.array([n], dtype=np.int64)}
        ).run()
        assert int(execution.buffers[outputs[0]].reshape(-1)[0]) == _python(source, n)


def test_matches_python_on_llvm(tmp_path, lowered):
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )

    source, module, export, parameters, outputs = lowered
    emitted = emit_ssa_function_to_llvm(module, export)
    assert not emitted.shortfalls, emitted.shortfalls
    artifact = compile_artifact(emitted, directory=tmp_path)
    for n in CASES:
        execution = prepare_artifact_execution(
            artifact, {parameters["n"]: np.array([n], dtype=np.int64)}
        ).run()
        assert int(execution.buffers[outputs[0]].reshape(-1)[0]) == _python(source, n)


def test_unbound_parameter_member_that_is_read_is_refused_by_name():
    source = SOURCES["loop_read"].replace(
        "return v.one([97, 98]) + n",
        "return v.one([97 + k for k in range(2)]) + n",
    )
    with pytest.raises(ValueError, match="parameter member has no caller binding"):
        _lower(source, "display_actual_list_formal_comprehension")
