"""Graph-native scalar relation spellings use the LLVM comparison table."""

from __future__ import annotations

import math

from src.compiler.ssa_llvm_backend import (
    compile_artifact,
    emit_ssa_function_to_llvm,
    prepare_artifact_execution,
)
from src.transmogrifier.ssa import BasicBlock, Function, IRModule, Instr, SSAValue


def test_graph_native_scalar_comparisons_emit_and_execute(tmp_path):
    left = SSAValue(0, "float64")
    right = SSAValue(1, "float64")
    comparisons = tuple(
        SSAValue(index, "bool")
        for index in range(2, 7)
    )
    operations = (
        "equal", "less", "less_equal", "greater", "greater_equal",
    )
    function = Function("scalar_relations", [left, right], {
        "entry": BasicBlock("entry", [
            *(
                Instr(operation, [left, right], result)
                for operation, result in zip(operations, comparisons)
            ),
            Instr("Ret", list(comparisons), None),
        ]),
    })

    artifact = emit_ssa_function_to_llvm(
        IRModule({function.name: function}), function.name,
    )

    assert artifact.shortfalls == ()
    assert "fcmp oeq double" in artifact.llvm_ir
    assert "fcmp olt double" in artifact.llvm_ir
    assert "fcmp ole double" in artifact.llvm_ir
    assert "fcmp ogt double" in artifact.llvm_ir
    assert "fcmp oge double" in artifact.llvm_ir
    native = compile_artifact(artifact, directory=tmp_path / "scalar_relations")

    for first, second in ((2.0, 2.0), (-1.0, 0.0), (1.0, -2.0),
                          (math.nan, math.nan)):
        execution = prepare_artifact_execution(
            native, {left.id: first, right.id: second},
        )
        execution.run()
        observed = tuple(bool(execution.buffers[value.id]) for value in comparisons)
        expected = (
            first == second,
            first < second,
            first <= second,
            first > second,
            first >= second,
        )
        assert observed == expected
