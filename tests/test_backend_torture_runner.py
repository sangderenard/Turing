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


def test_torture_runner_covers_the_nodus_eager_backend():
    rows = run_torture_matrix(
        backends=("nodus",),
        case_names=("add", "sqrt", "operator_grab_bag", "advanced_tensor_topology"),
    )

    assert len(rows) == 4
    assert all(row.status == "passed" for row in rows)
    assert all(row.max_abs_error is not None for row in rows)


def test_torture_runner_covers_the_python_ast_backends_on_elementwise_cases():
    """python_numpy/python_torch/python_nodus lower the recorded tape's
    FusedProgram to a real ast.Module and run it -- elementwise-only, so
    they only cover the pure-elementwise isolated cases."""

    rows = run_torture_matrix(
        backends=("python_numpy", "python_torch", "python_nodus"),
        case_names=("add", "sqrt", "divide"),
    )

    assert len(rows) == 9
    assert all(row.status == "passed" for row in rows)
    assert all(row.max_abs_error is not None for row in rows)


def test_torture_runner_reports_python_ast_backends_as_failed_off_elementwise():
    """Not silently skipped -- a case outside the elementwise FusedProgram
    vocabulary (here, ``where``) must show up as a named failure."""

    rows = run_torture_matrix(
        backends=("python_numpy",),
        case_names=("operator_grab_bag",),
    )

    assert len(rows) == 1
    assert rows[0].status == "failed"
    assert "where" in rows[0].error
