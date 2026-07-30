from __future__ import annotations

import copy

import pytest

from src.common.tensors.fused_ir import FusedProgram, Meta, OpStep
from src.compiler.precompile_ssa_validator import (
    PrecompileSSAValidationError,
    require_precompile_ssa_compatible,
    validate_precompile_ssa_compatibility,
)


def _meta(*value_ids):
    return {
        value_id: Meta(shape=(8,), dtype="float32", device="glsl")
        for value_id in value_ids
    }


def test_valid_precompile_reports_each_existing_ssa_operation_by_name():
    program = FusedProgram(
        version=1,
        feeds={0, 1},
        steps=[
            OpStep(0, "add", [0, 1], {}, 2),
            OpStep(1, "mul", [2], {"right_scalar": 3.0}, 3),
            OpStep(2, "less", [3, 1], {}, 4),
        ],
        outputs={"mask": 4},
        meta=_meta(0, 1, 2, 3, 4),
    )
    original = copy.deepcopy(program)

    result = validate_precompile_ssa_compatibility(program)

    assert result.valid_precompile
    assert result.ssa_compatible
    assert result.compatibility_shortfalls == ()
    assert {
        (entry.precompile_name, entry.ssa_name)
        for entry in result.compatible_operations
    } == {("add", "Add"), ("mul", "Mul"), ("less", "Lt")}
    assert program == original
    assert require_precompile_ssa_compatible(program) is program


def test_scan_groups_missing_ssa_operations_by_their_precompile_names():
    program = FusedProgram(
        version=1,
        feeds={0, 1, 2},
        steps=[
            OpStep(0, "sin", [0], {}, 3),
            OpStep(1, "stack", [1, 2], {"dim": 0}, 4),
            OpStep(2, "sin", [4], {}, 5),
        ],
        outputs={"result": 5},
        meta=_meta(0, 1, 2, 3, 4, 5),
        extras={"kernel_kind": "stack"},
    )

    result = validate_precompile_ssa_compatibility(program)

    assert result.valid_precompile
    assert not result.ssa_compatible
    assert [
        (item.operation_name, item.count)
        for item in result.compatibility_shortfalls
    ] == [
        ("kernel_kind:stack", 1),
        ("sin", 2),
        ("stack", 1),
    ]
    report = result.compatibility_shortfall_report()
    assert "sin: 2 occurrence(s)" in report
    assert "stack: 1 occurrence(s)" in report
    with pytest.raises(
        PrecompileSSAValidationError,
        match="precompile SSA compatibility shortfalls",
    ):
        require_precompile_ssa_compatible(program)


def test_format_scan_finds_unproduced_values_duplicate_writers_and_metadata():
    program = FusedProgram(
        version=1,
        feeds={0},
        steps=[
            OpStep(0, "add", [0, 99], {}, 1),
            OpStep(1, "mul", [1, 0], {}, 1),
        ],
        outputs={"result": 7},
        meta=_meta(0),
    )

    result = validate_precompile_ssa_compatibility(program)
    codes = {issue.code for issue in result.format_issues}

    assert not result.valid_precompile
    assert "PRECOMPILE_UNPRODUCED_INPUT" in codes
    assert "PRECOMPILE_DUPLICATE_PRODUCER" in codes
    assert "PRECOMPILE_UNPRODUCED_OUTPUT" in codes
    assert "PRECOMPILE_METADATA_MISSING" in codes


def test_native_matmul_is_recognized_but_unrepresented_reduction_is_named():
    matmul = FusedProgram(
        version=1,
        feeds={0, 1},
        steps=[OpStep(0, "matmul", [0, 1], {}, 2)],
        outputs={"result": 2},
        meta=_meta(0, 1, 2),
        extras={"kernel_kind": "matmul"},
    )
    reduction = FusedProgram(
        version=1,
        feeds={0},
        steps=[OpStep(0, "sum", [0], {"axis": 0}, 1)],
        outputs={"result": 1},
        meta=_meta(0, 1),
        extras={"kernel_kind": "reduce"},
    )

    matmul_result = validate_precompile_ssa_compatibility(matmul)
    reduction_result = validate_precompile_ssa_compatibility(reduction)

    assert matmul_result.ssa_compatible
    assert [
        item.operation_name
        for item in reduction_result.compatibility_shortfalls
    ] == ["kernel_kind:reduce", "sum"]


def test_metadata_requirement_can_be_relaxed_for_legacy_precompile_packets():
    program = FusedProgram(
        version=1,
        feeds={0},
        steps=[OpStep(0, "neg", [0], {}, 1)],
        outputs={"result": 1},
    )

    strict = validate_precompile_ssa_compatibility(program)
    legacy = validate_precompile_ssa_compatibility(
        program,
        require_typed_metadata=False,
    )

    assert not strict.valid_precompile
    assert legacy.ssa_compatible
