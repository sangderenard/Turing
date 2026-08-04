from pathlib import Path

import pytest

from src.compiler.native_affine_learner import (
    compile_learning_visualizer,
    emit_learning_fortran,
    load_learning_problem,
)
from src.compiler.ssa_fortran_backend import fortran_compiler


EXAMPLE = Path(__file__).parents[1] / "examples" / "learnable_sort.py"


def test_python_problem_loads_as_exact_build_time_dataset():
    problem = load_learning_problem(
        EXAMPLE, seed=3, train_samples=12, validation_samples=7,
    )

    assert problem.train_inputs.shape == (12, 8)
    assert problem.train_targets.shape == (12, 8)
    assert problem.validation_inputs.shape == (7, 8)
    assert (problem.train_targets[:, 1:] >= problem.train_targets[:, :-1]).all()


def test_emitted_program_contains_native_learning_verification_and_visualization():
    problem = load_learning_problem(
        EXAMPLE, seed=3, train_samples=12, validation_samples=7,
    )
    source = emit_learning_fortran(problem, epochs=10, display_every=2)

    assert "matmul(weight, train_x(:,sample))" in source
    assert "call verify(weight, bias" in source
    assert "TURING NATIVE AFFINE REDUCTION" in source
    assert "best-affine-model.txt" in source
    assert "use iso_fortran_env" in source


@pytest.mark.skipif(fortran_compiler() is None, reason="no Fortran compiler installed")
def test_compiled_native_learner_runs_without_python_runtime(tmp_path):
    artifact = compile_learning_visualizer(
        EXAMPLE,
        tmp_path,
        seed=5,
        train_samples=24,
        validation_samples=12,
        epochs=4,
        display_every=4,
    )
    completed = artifact.run(capture_output=True)

    assert completed.returncode == 0
    assert "native learning complete" in completed.stdout
    assert "CHEAPER CANDIDATE" in completed.stdout
    assert (tmp_path / "best-affine-model.txt").is_file()
