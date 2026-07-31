from __future__ import annotations

from src.common.tensors.accelerator_backends.backend_torture_runner import (
    format_torture_matrix,
    run_torture_matrix,
)


def test_torture_runner_uses_raw_numpy_and_abstract_eager_backends():
    rows = run_torture_matrix(
        backends=("raw_numpy", "numpy", "torch"),
        case_names=("add", "operator_grab_bag", "advanced_tensor_topology"),
    )

    assert len(rows) == 9
    assert all(row.status == "passed" for row in rows)
    assert all(row.max_abs_error is not None for row in rows)
    rendered = format_torture_matrix(rows)
    assert "raw_numpy" in rendered
    assert "operator_grab_bag" in rendered
    assert "advanced_tensor_topology" in rendered


def test_torture_runner_rejects_unknown_case_selection():
    try:
        run_torture_matrix(
            backends=("raw_numpy",),
            case_names=("not-a-case",),
        )
    except ValueError as error:
        assert "not-a-case" in str(error)
    else:
        raise AssertionError("unknown torture case was accepted")


def test_torture_runner_compares_boolean_outputs_without_subtraction():
    rows = run_torture_matrix(
        backends=("raw_numpy", "numpy", "torch"),
        case_names=("less", "greater_equal"),
    )

    assert len(rows) == 6
    assert all(row.status == "passed" for row in rows)
    assert all(row.max_abs_error == 0.0 for row in rows)
