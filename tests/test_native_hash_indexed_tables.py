"""A keyed heap table answers lookups through its hash index, exactly.

The string core's ``Vocabulary`` keeps dict-of-lists fields that gain one key
per minted token; a linear scan per lookup made native time quadratic in the
rounds.  The index (``ir_sequence_tables.hash_index_carrier``) must change
only the cost: every hit, miss, re-store and growth across reserves matches
Python on both native lanes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"

SOURCE = '''
class S:
    def __init__(self):
        self.d = {}

    def put(self, a: int, b: int, v: int):
        self.d.setdefault((a, b), []).append(v)

    def total(self, a: int, b: int) -> int:
        t = 0
        for x in self.d.get((a, b), ()):
            t = t * 3 + x
        return t


def root(n: int) -> int:
    s = S()
    check = 0
    i = 0
    while i < n:
        s.put(i % 5, i, i + 1)
        if i % 3 == 0:
            s.put(i % 5, i // 2, i + 7)
        check = (check * 31 + s.total(i % 5, i) + s.total(i % 5, i // 2)
                 + s.total(i % 5, i + 1)) % 1000000007
        i += 1
    return check * 1000 + len(s.d)
'''

INPUTS = (0, 1, 2, 7, 64, 300)


def python_root(n):
    namespace = {}
    exec(SOURCE, namespace)
    return namespace["root"](n)


@pytest.fixture(scope="module")
def lowered():
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, exports = lower_ast_source_to_ssa(
        SOURCE, "root", name="hash_indexed_tables",
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


def test_dict_of_lists_field_is_hash_indexed(lowered):
    module = lowered[0]
    blocks = {name for function in module.functions.values() for name in function.blocks}
    assert any(name.endswith("lookup_index_valid") for name in blocks)
    assert any(name.endswith("store_index_valid") for name in blocks)
    assert any("reserve_rebuild" in name for name in blocks)


def test_c_matches_python(lowered, tmp_path):
    from src.compiler.ssa_c_backend import emit_ssa_to_c

    module, export, parameters, outputs = lowered
    artifact = emit_ssa_to_c(module, export)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "native")
    for n in INPUTS:
        execution = artifact.prepare_execution(
            {parameters["n"]: np.array([n], dtype=np.int64)}
        ).run()
        assert int(execution.buffers[outputs[0]].reshape(-1)[0]) == python_root(n), n


def test_llvm_matches_python(lowered, tmp_path):
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )

    module, export, parameters, outputs = lowered
    emitted = emit_ssa_function_to_llvm(module, export)
    assert not emitted.shortfalls, emitted.shortfalls
    artifact = compile_artifact(emitted, directory=tmp_path)
    for n in INPUTS:
        execution = prepare_artifact_execution(
            artifact, {parameters["n"]: np.array([n], dtype=np.int64)}
        ).run()
        assert int(execution.buffers[outputs[0]].reshape(-1)[0]) == python_root(n), n
