"""A value producer pulled ahead by the access walk brings its arena's
earlier writes with it (c244, c373 raised fault 71 on C and LLVM)."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "extraction_contracts" / "program_extraction.yaml"
CORPUS = ROOT / "tests" / "native_semantics_corpus"
PROGRAMS = [
    "c244_sort_beside_unrolled_literal_loop",
    "c373_augmented_store_at_computed_index",
]


def _program(stem):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    source = (CORPUS / f"{stem}.py").read_text()
    namespace: dict = {"__name__": f"walk_{stem}"}
    exec(compile(source, stem, "exec", dont_inherit=True), namespace)
    inputs = ast.literal_eval(source.split("# INPUTS", 1)[1].strip())
    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name=f"walk_{stem}",
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
    return (
        module, exports[0], dict(root.metadata["parameter_names"]),
        outputs, [(feeds, namespace["root"](**feeds)) for feeds in inputs],
    )


def _run(execution, parameters, outputs, feeds):
    run = execution({
        parameters[name]: np.array([value], dtype=np.int64)
        for name, value in feeds.items()
    }).run()
    return int(run.buffers[outputs[0]].reshape(-1)[0])


@pytest.mark.parametrize("stem", PROGRAMS)
def test_c_matches_python(tmp_path, stem):
    from src.compiler.ssa_c_backend import emit_ssa_to_c

    module, export, parameters, outputs, cases = _program(stem)
    artifact = emit_ssa_to_c(module, export)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "native")
    for feeds, expected in cases:
        assert _run(artifact.prepare_execution, parameters, outputs, feeds) == expected


@pytest.mark.parametrize("stem", PROGRAMS)
def test_llvm_matches_python(tmp_path, stem):
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )

    module, export, parameters, outputs, cases = _program(stem)
    emitted = emit_ssa_function_to_llvm(module, export)
    assert not emitted.shortfalls, emitted.shortfalls
    artifact = compile_artifact(emitted, directory=tmp_path)
    for feeds, expected in cases:
        assert _run(
            lambda buffers: prepare_artifact_execution(artifact, buffers),
            parameters, outputs, feeds,
        ) == expected
