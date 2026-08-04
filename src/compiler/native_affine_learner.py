"""Compile a Python-described learning problem into a native Fortran visualizer.

The Python file is a build-time oracle.  It supplies exact training and
validation pairs; the emitted executable contains those pairs and performs its
own affine fitting, pruning, verification, visualization, and model export in
Fortran.  Python and pygame are not required when the executable runs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import re
import subprocess
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np

from .ssa_fortran_backend import FortranEmissionError, fortran_compiler


def _identifier(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_]", "_", value).strip("_").lower()
    if not result or result[0].isdigit():
        result = "learn_" + result
    return result


def _load_problem(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"turing_learning_problem_{abs(hash(path.resolve()))}", path,
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load learning problem {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _matrix(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or min(result.shape) < 1:
        raise ValueError(f"{name} must be a non-empty samples-by-features matrix")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains a non-finite value")
    return np.ascontiguousarray(result)


@dataclass(frozen=True)
class LearningProblem:
    name: str
    train_inputs: np.ndarray
    train_targets: np.ndarray
    validation_inputs: np.ndarray
    validation_targets: np.ndarray
    reference_operations: int

    @property
    def input_dimension(self) -> int:
        return int(self.train_inputs.shape[1])

    @property
    def output_dimension(self) -> int:
        return int(self.train_targets.shape[1])


def load_learning_problem(
    python_file: str | Path,
    *,
    seed: int = 7,
    train_samples: int = 512,
    validation_samples: int = 256,
) -> LearningProblem:
    """Load the trusted build-time ``build_benchmark`` protocol."""

    path = Path(python_file).resolve()
    module = _load_problem(path)
    factory = getattr(module, "build_benchmark", None)
    if not callable(factory):
        raise ValueError(f"{path} must define build_benchmark(...) ")
    raw = factory(
        seed=int(seed),
        train_samples=int(train_samples),
        validation_samples=int(validation_samples),
    )
    if not isinstance(raw, Mapping):
        raise ValueError("build_benchmark must return a mapping")
    train_x = _matrix(raw.get("train_inputs"), "train_inputs")
    train_y = _matrix(raw.get("train_targets"), "train_targets")
    valid_x = _matrix(raw.get("validation_inputs"), "validation_inputs")
    valid_y = _matrix(raw.get("validation_targets"), "validation_targets")
    if train_x.shape[0] != train_y.shape[0]:
        raise ValueError("training input and target sample counts differ")
    if valid_x.shape[0] != valid_y.shape[0]:
        raise ValueError("validation input and target sample counts differ")
    if train_x.shape[1] != valid_x.shape[1]:
        raise ValueError("training and validation input widths differ")
    if train_y.shape[1] != valid_y.shape[1]:
        raise ValueError("training and validation target widths differ")
    reference_operations = int(raw.get("reference_operations", 1))
    if reference_operations < 1:
        raise ValueError("reference_operations must be positive")
    return LearningProblem(
        name=str(raw.get("name", path.stem)),
        train_inputs=train_x,
        train_targets=train_y,
        validation_inputs=valid_x,
        validation_targets=valid_y,
        reference_operations=reference_operations,
    )


def _fortran_values(array: np.ndarray) -> str:
    # Fortran fills the first dimension first.  Transposing samples-by-features
    # produces the desired x(feature, sample) storage before column-major flatting.
    values = np.asarray(array.T, dtype=np.float64).reshape(-1, order="F")
    tokens = [f"{float(value):.17e}_real64" for value in values]
    lines = []
    for offset in range(0, len(tokens), 3):
        suffix = ", &" if offset + 3 < len(tokens) else ""
        lines.append("    " + ", ".join(tokens[offset:offset + 3]) + suffix)
    return "\n".join(lines)


def emit_learning_fortran(
    problem: LearningProblem,
    *,
    epochs: int = 1600,
    display_every: int = 20,
    learning_rate: float = 0.12,
    l1: float = 2.5e-5,
    prune_threshold: float = 1.0e-3,
    exact_tolerance: float = 1.0e-8,
) -> str:
    """Emit a standalone Fortran 2008 optimizer and ANSI visualization."""

    if epochs < 1 or display_every < 1:
        raise ValueError("epochs and display_every must be positive")
    n_in = problem.input_dimension
    n_out = problem.output_dimension
    return f'''program {_identifier(problem.name)}_affine_learner
  use iso_fortran_env, only: real64, output_unit
  implicit none
  integer, parameter :: n_in={n_in}, n_out={n_out}
  integer, parameter :: n_train={len(problem.train_inputs)}, n_valid={len(problem.validation_inputs)}
  integer, parameter :: default_epochs={int(epochs)}, display_every={int(display_every)}
  integer, parameter :: reference_ops={problem.reference_operations}
  real(real64), parameter :: base_rate={learning_rate:.17e}_real64
  real(real64), parameter :: l1={l1:.17e}_real64
  real(real64), parameter :: prune_threshold={prune_threshold:.17e}_real64
  real(real64), parameter :: exact_tolerance={exact_tolerance:.17e}_real64
  real(real64) :: train_x(n_in,n_train), train_y(n_out,n_train)
  real(real64) :: valid_x(n_in,n_valid), valid_y(n_out,n_valid)
  real(real64) :: weight(n_out,n_in), bias(n_out), best_weight(n_out,n_in), best_bias(n_out)
  real(real64) :: dw(n_out,n_in), db(n_out), prediction(n_out), error(n_out)
  real(real64) :: train_loss, valid_loss, best_loss, score, best_score, rate, scale
  integer :: epoch, sample, row, col, active, best_active, exact, epochs_to_run, ios, budget
  character(len=32) :: argument

  train_x = reshape([ &
{_fortran_values(problem.train_inputs)} &
  ], [n_in,n_train])
  train_y = reshape([ &
{_fortran_values(problem.train_targets)} &
  ], [n_out,n_train])
  valid_x = reshape([ &
{_fortran_values(problem.validation_inputs)} &
  ], [n_in,n_valid])
  valid_y = reshape([ &
{_fortran_values(problem.validation_targets)} &
  ], [n_out,n_valid])

  epochs_to_run = default_epochs
  call get_command_argument(1, argument)
  if (len_trim(argument) > 0) then
    read(argument, *, iostat=ios) epochs_to_run
    if (ios /= 0 .or. epochs_to_run < 1) error stop "epoch argument must be positive"
  end if
  weight = 0.0_real64
  bias = sum(train_y, dim=2) / real(n_train, real64)
  best_weight = weight
  best_bias = bias
  best_loss = huge(1.0_real64)
  best_score = huge(1.0_real64)
  best_active = 0

  do epoch = 1, epochs_to_run
    dw = 0.0_real64
    db = 0.0_real64
    train_loss = 0.0_real64
    do sample = 1, n_train
      prediction = matmul(weight, train_x(:,sample)) + bias
      error = prediction - train_y(:,sample)
      train_loss = train_loss + sum(error * error)
      db = db + error
      do row = 1, n_out
        do col = 1, n_in
          dw(row,col) = dw(row,col) + error(row) * train_x(col,sample)
        end do
      end do
    end do
    scale = 2.0_real64 / real(n_train * n_out, real64)
    rate = base_rate / sqrt(1.0_real64 + 0.002_real64 * real(epoch - 1, real64))
    weight = weight - rate * scale * dw
    bias = bias - rate * scale * db
    weight = sign(max(abs(weight) - rate*l1, 0.0_real64), weight)
    if (mod(epoch, 40) == 0) then
      where (abs(weight) < prune_threshold * max(1.0_real64, maxval(abs(weight))))
        weight = 0.0_real64
      end where
    end if
    if (mod(epoch, display_every) == 0 .or. epoch == epochs_to_run) then
      budget = max(0, reference_ops - n_out - 1)
      budget = max(budget, n_out*n_in - ((n_out*n_in-budget)*epoch)/epochs_to_run)
      call prune_to_budget(weight, budget)
    end if
    train_loss = train_loss / real(n_train * n_out, real64)
    call verify(weight, bias, valid_loss, exact)
    active = count(abs(weight) > 0.0_real64)
    score = valid_loss + 1.0e-5_real64 * real(active + n_out, real64) / real(reference_ops, real64)
    if (active + n_out >= reference_ops) score = score + 1.0_real64
    if (score < best_score) then
      best_score = score
      best_loss = valid_loss
      best_weight = weight
      best_bias = bias
      best_active = active
    end if
    if (epoch == 1 .or. mod(epoch, display_every) == 0 .or. epoch == epochs_to_run) then
      call draw(epoch, epochs_to_run, train_loss, valid_loss, exact, active, weight, bias)
      call pause_frame(0.025_real64)
    end if
  end do
  call save_model(best_weight, best_bias, best_loss, best_active)
  write(*,'(a)') "native learning complete; model: best-affine-model.txt"

contains
  subroutine verify(w, b, loss, exact_count)
    real(real64), intent(in) :: w(n_out,n_in), b(n_out)
    real(real64), intent(out) :: loss
    integer, intent(out) :: exact_count
    integer :: j
    real(real64) :: p(n_out), e(n_out)
    loss = 0.0_real64
    exact_count = 0
    do j = 1, n_valid
      p = matmul(w, valid_x(:,j)) + b
      e = p - valid_y(:,j)
      loss = loss + sum(e*e)
      if (maxval(abs(e)) <= exact_tolerance) exact_count = exact_count + 1
    end do
    loss = loss / real(n_valid*n_out, real64)
  end subroutine verify

  subroutine prune_to_budget(w, budget)
    real(real64), intent(inout) :: w(n_out,n_in)
    integer, intent(in) :: budget
    real(real64) :: smallest
    do while (count(abs(w) > 0.0_real64) > budget)
      smallest = minval(abs(w), mask=abs(w) > 0.0_real64)
      where (abs(w) <= smallest) w = 0.0_real64
    end do
  end subroutine prune_to_budget

  subroutine draw(step, total, training, validation, exact_count, terms, w, b)
    integer, intent(in) :: step, total, exact_count, terms
    real(real64), intent(in) :: training, validation, w(n_out,n_in), b(n_out)
    integer :: i, j, filled, candidate_ops
    real(real64) :: ceiling, p(n_out)
    character(len=1), parameter :: esc=achar(27)
    character(len=10), parameter :: shades=" .:-=+*#%@"
    character(len=72) :: bar
    ceiling = max(maxval(abs(w)), tiny(1.0_real64))
    filled = min(60, int(60.0_real64*real(step,real64)/real(total,real64)))
    bar = repeat("#",filled)//repeat("-",60-filled)
    candidate_ops = terms + n_out
    write(*,'(a)',advance='no') esc//"[2J"//esc//"[H"
    write(*,'(a)') "TURING NATIVE AFFINE REDUCTION :: {_identifier(problem.name)}"
    write(*,'(a)') "["//bar(1:60)//"]"
    write(*,'(a,i0,a,i0,a,es11.3,a,es11.3)') "epoch ",step,"/",total,"  train ",training,"  verify ",validation
    write(*,'(a,i0,a,i0,a,i0,a,i0)') "exact ",exact_count,"/",n_valid,"  active ops ",candidate_ops,"  reference ops ",reference_ops
    if (candidate_ops < reference_ops) then
      write(*,'(a)') "cost verdict: CHEAPER CANDIDATE (correctness remains independently visible)"
    else
      write(*,'(a)') "cost verdict: candidate has not crossed the reference cost"
    end if
    write(*,'(a)') "weight matrix: sign is case, magnitude is density"
    do i = 1, n_out
      write(*,'(2x)',advance='no')
      do j = 1, n_in
        filled = 1 + min(9,int(9.0_real64*abs(w(i,j))/ceiling))
        if (w(i,j) < 0.0_real64) then
          write(*,'(a)',advance='no') achar(iachar(shades(filled:filled))-32)
        else
          write(*,'(a)',advance='no') shades(filled:filled)
        end if
        write(*,'(a)',advance='no') " "
      end do
      write(*,*)
    end do
    p = matmul(w, valid_x(:,1)) + b
    write(*,'(a,*(f7.3,1x))') "input:  ", valid_x(:,1)
    write(*,'(a,*(f7.3,1x))') "target: ", valid_y(:,1)
    write(*,'(a,*(f7.3,1x))') "guess:  ", p
    flush(output_unit)
  end subroutine draw

  subroutine pause_frame(seconds)
    real(real64), intent(in) :: seconds
    integer :: start, now, rate_count
    call system_clock(start, rate_count)
    do
      call system_clock(now)
      if (real(now-start,real64)/real(rate_count,real64) >= seconds) exit
    end do
  end subroutine pause_frame

  subroutine save_model(w, b, loss, terms)
    real(real64), intent(in) :: w(n_out,n_in), b(n_out), loss
    integer, intent(in) :: terms
    integer :: unit, i
    open(newunit=unit, file="best-affine-model.txt", status="replace", action="write")
    write(unit,'(a)') "turing.affine-model.v1"
    write(unit,'(a,es24.16)') "validation_mse=", loss
    write(unit,'(a,i0)') "active_weights=", terms
    write(unit,'(a)') "bias"
    write(unit,'(*(es24.16,1x))') b
    write(unit,'(a)') "matrix_rows"
    do i = 1, n_out
      write(unit,'(*(es24.16,1x))') w(i,:)
    end do
    close(unit)
  end subroutine save_model
end program {_identifier(problem.name)}_affine_learner
'''


@dataclass(frozen=True)
class NativeAffineLearner:
    problem: LearningProblem
    directory: Path
    source_path: Path
    executable_path: Path

    def run(
        self,
        *,
        epochs: int | None = None,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """Run the native visualizer; output is live unless explicitly captured."""

        command = [str(self.executable_path)]
        if epochs is not None:
            if epochs < 1:
                raise ValueError("epochs must be positive")
            command.append(str(int(epochs)))
        environment = dict(os.environ)
        compiler = fortran_compiler()
        if compiler:
            environment["PATH"] = str(Path(compiler).parent) + os.pathsep + environment.get("PATH", "")
        return subprocess.run(
            command,
            cwd=self.directory,
            env=environment,
            text=True,
            capture_output=capture_output,
            check=True,
        )

    __call__ = run


def compile_learning_visualizer(
    python_file: str | Path,
    output_directory: str | Path,
    *,
    seed: int = 7,
    train_samples: int = 512,
    validation_samples: int = 256,
    epochs: int = 1600,
    display_every: int = 20,
) -> NativeAffineLearner:
    """Build a native Fortran learner from a trusted Python benchmark file."""

    compiler = fortran_compiler()
    if compiler is None:
        raise FortranEmissionError("no Fortran compiler found")
    problem = load_learning_problem(
        python_file,
        seed=seed,
        train_samples=train_samples,
        validation_samples=validation_samples,
    )
    directory = Path(output_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    stem = _identifier(problem.name) + "_affine_learner"
    source_path = directory / f"{stem}.f90"
    executable_path = directory / (stem + (".exe" if os.name == "nt" else ""))
    source_path.write_text(
        emit_learning_fortran(problem, epochs=epochs, display_every=display_every),
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PATH"] = str(Path(compiler).parent) + os.pathsep + environment.get("PATH", "")
    completed = subprocess.run(
        [compiler, "-O3", "-std=f2008", str(source_path), "-o", str(executable_path)],
        cwd=directory,
        env=environment,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise FortranEmissionError(f"native learner compilation failed:\n{completed.stderr}")
    return NativeAffineLearner(problem, directory, source_path, executable_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("python_file", type=Path)
    parser.add_argument("--output", type=Path, default=Path("build/native-affine-learner"))
    parser.add_argument("--epochs", type=int, default=1600)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--validation-samples", type=int, default=256)
    parser.add_argument("--display-every", type=int, default=20)
    parser.add_argument("--compile-only", action="store_true")
    arguments = parser.parse_args(argv)
    artifact = compile_learning_visualizer(
        arguments.python_file,
        arguments.output,
        epochs=arguments.epochs,
        train_samples=arguments.train_samples,
        validation_samples=arguments.validation_samples,
        display_every=arguments.display_every,
    )
    print(artifact.executable_path)
    if not arguments.compile_only:
        artifact.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LearningProblem",
    "NativeAffineLearner",
    "compile_learning_visualizer",
    "emit_learning_fortran",
    "load_learning_problem",
    "main",
]
