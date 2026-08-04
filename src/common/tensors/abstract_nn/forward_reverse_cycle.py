"""Runnable forward/reverse parameter-solving cycles.

The cycle is captured from a real AbstractTensor tape.  Its unpruned forward
graph is retained, every terminal output becomes a desired-value input to the
reverse graph, and the two graphs are fused into one ``FusedProgram``.  The
same program can run through ``ProgramRunner`` or the repository's Fortran
target.

Run the included demonstration with::

    python -m src.common.tensors.abstract_nn.forward_reverse_cycle --iterations 12

Pass ``--emit-fortran DIRECTORY`` to write the fused ``.f90`` and ABI YAML;
add ``--compile-fortran`` to compile and execute the final cycle natively.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Protocol

import numpy as np

from ..abstraction import AbstractTensor as AT
from ..autograd import GradTape, autograd
from ..fused_ir import FusedProgram, ordered_feed_ids
from .fused_program import ProgramRunner, build_fused_program
from .reverse_program import (
    ReverseProgramCapture,
    capture_reverse_fused_program,
    retain_uncaptured_outputs,
)


def _tensor_like(value: Any, like: AT) -> AT:
    if isinstance(value, AT):
        return value
    return AT.get_tensor(value, like=like, tape=getattr(like, "_tape", None))


def _live_tensors(tape: GradTape) -> Dict[int, AT]:
    values: Dict[int, AT] = dict(getattr(tape, "_tensor_refs", {}))
    for node in getattr(tape, "_nodes", {}).values():
        for value in node.ctx.get("inputs", ()):
            if isinstance(value, AT):
                values[id(value)] = value
        value = node.ctx.get("result")
        if isinstance(value, AT):
            values[id(value)] = value
    return values


class TargetStrategy(Protocol):
    """Choose a desired value for every retained forward output."""

    def __call__(
        self, iteration: int, current_outputs: Mapping[str, AT]
    ) -> Mapping[str, AT]: ...


@dataclass(frozen=True)
class FixedTargets:
    """Fixed desired values; unmentioned terminal outputs hold their value."""

    values: Mapping[str, Any]
    hold_unmentioned: bool = True

    def __call__(
        self, iteration: int, current_outputs: Mapping[str, AT]
    ) -> Mapping[str, AT]:
        del iteration
        unknown = set(self.values) - set(current_outputs)
        if unknown:
            raise KeyError(f"fixed targets name unknown outputs: {sorted(unknown)}")
        if not self.hold_unmentioned and set(self.values) != set(current_outputs):
            missing = sorted(set(current_outputs) - set(self.values))
            raise KeyError(f"fixed targets omit retained outputs: {missing}")
        return {
            name: _tensor_like(self.values[name], actual)
            if name in self.values else actual.detach()
            for name, actual in current_outputs.items()
        }


@dataclass(frozen=True)
class InterpolatedTargets:
    """Move desired values from the current outputs toward fixed goals."""

    goals: Mapping[str, Any]
    fraction: float | Callable[[int], float]

    def __call__(
        self, iteration: int, current_outputs: Mapping[str, AT]
    ) -> Mapping[str, AT]:
        amount = self.fraction(iteration) if callable(self.fraction) else self.fraction
        if not 0.0 <= float(amount) <= 1.0:
            raise ValueError("target interpolation fraction must be in [0, 1]")
        fixed = FixedTargets(self.goals)(iteration, current_outputs)
        return {
            name: current + (fixed[name] - current) * float(amount)
            for name, current in current_outputs.items()
        }


class CorrectionStrategy(Protocol):
    """Choose capture-time gradient scale and optional host postprocessing."""

    @property
    def fortran_fusible(self) -> bool: ...

    def step_size(self, iteration: int) -> float: ...

    def apply(
        self,
        iteration: int,
        current: Mapping[str, AT],
        proposed: Mapping[str, AT],
    ) -> Mapping[str, AT]: ...


@dataclass(frozen=True)
class GradientCorrection:
    """A fully fusible ``parameter -= scheduled_step * gradient`` update."""

    schedule: float | Callable[[int], float] = 0.1

    @property
    def fortran_fusible(self) -> bool:
        return True

    def step_size(self, iteration: int) -> float:
        value = self.schedule(iteration) if callable(self.schedule) else self.schedule
        if float(value) < 0:
            raise ValueError("gradient correction step must be non-negative")
        return float(value)

    def apply(
        self,
        iteration: int,
        current: Mapping[str, AT],
        proposed: Mapping[str, AT],
    ) -> Mapping[str, AT]:
        del iteration, current
        return dict(proposed)


@dataclass(frozen=True)
class ClippedCorrection:
    """Host hook that clips each proposed parameter displacement."""

    schedule: float | Callable[[int], float] = 0.1
    maximum_change: float = 0.25

    @property
    def fortran_fusible(self) -> bool:
        return False

    def step_size(self, iteration: int) -> float:
        return GradientCorrection(self.schedule).step_size(iteration)

    def apply(
        self,
        iteration: int,
        current: Mapping[str, AT],
        proposed: Mapping[str, AT],
    ) -> Mapping[str, AT]:
        del iteration
        if self.maximum_change < 0:
            raise ValueError("maximum_change must be non-negative")
        corrected: Dict[str, AT] = {}
        for name, value in proposed.items():
            before = np.asarray(current[name].tolist())
            after = np.asarray(value.tolist())
            clipped = before + np.clip(
                after - before, -self.maximum_change, self.maximum_change
            )
            corrected[name] = AT.get_tensor(clipped.tolist(), like=current[name])
        return corrected


@dataclass(frozen=True)
class CallableCorrection:
    """Arbitrary host correction hook following a captured gradient proposal."""

    function: Callable[
        [int, Mapping[str, AT], Mapping[str, AT]], Mapping[str, AT]
    ]
    capture_step: float = 1.0

    @property
    def fortran_fusible(self) -> bool:
        return False

    def step_size(self, iteration: int) -> float:
        del iteration
        return float(self.capture_step)

    def apply(
        self,
        iteration: int,
        current: Mapping[str, AT],
        proposed: Mapping[str, AT],
    ) -> Mapping[str, AT]:
        return self.function(iteration, current, proposed)


def fuse_forward_reverse_program(
    forward: FusedProgram,
    reverse: ReverseProgramCapture,
) -> FusedProgram:
    """Fuse one retained forward evaluation and its reverse proposal."""

    forward = retain_uncaptured_outputs(forward)
    forward_results = {step.result_id for step in forward.steps}
    reverse_results = {step.result_id for step in reverse.program.steps}
    collisions = forward_results & reverse_results
    if collisions:
        raise ValueError(f"forward/reverse result id collision: {sorted(collisions)}")
    steps = list(forward.steps) + list(reverse.program.steps)
    steps = [replace(step, step_id=index) for index, step in enumerate(steps)]
    produced = {step.result_id for step in steps}
    feeds = (set(forward.feeds) | set(reverse.program.feeds)) - produced
    outputs = {f"forward_{name}": value_id for name, value_id in forward.outputs.items()}
    for name, value_id in reverse.program.outputs.items():
        candidate = name
        suffix = 2
        while candidate in outputs:
            candidate = f"{name}_{suffix}"
            suffix += 1
        outputs[candidate] = value_id
    metadata = dict(forward.meta or {})
    metadata.update(reverse.program.meta or {})
    extras = dict(forward.extras or {})
    reverse_extras = dict(reverse.program.extras or {})
    origins = dict(extras.get("capture_feed_origins", {}) or {})
    origins.update(reverse_extras.get("capture_feed_origins", {}) or {})
    extras.update(reverse_extras)
    extras["capture_feed_origins"] = origins
    return FusedProgram(
        version=max(forward.version, reverse.program.version),
        feeds=feeds,
        steps=steps,
        outputs=outputs,
        state_in=(set(forward.state_in or ()) | set(reverse.program.state_in or ())) or None,
        meta=metadata,
        extras=extras,
    )


@dataclass(frozen=True)
class ForwardReverseCycleResult:
    forward_outputs: Dict[str, AT]
    proposed_parameters: Dict[str, AT]
    incidentals: Dict[int, AT]


@dataclass(frozen=True)
class ForwardReverseCycleCapture:
    iteration: int
    forward_program: FusedProgram
    reverse_capture: ReverseProgramCapture
    fused_program: FusedProgram
    feed_values: Dict[int, AT]
    parameter_ids: Dict[str, int]
    target_ids: Dict[str, int]
    correction: CorrectionStrategy
    tape: GradTape

    def run_python(
        self, feeds: Mapping[int, AT] | None = None
    ) -> ForwardReverseCycleResult:
        values = dict(self.feed_values)
        if feeds:
            values.update(feeds)
        previous_tape = autograd.tape
        autograd.tape = self.tape
        try:
            outputs = ProgramRunner(self.fused_program)(values)
        finally:
            autograd.tape = previous_tape
        forward_outputs = {
            name.removeprefix("forward_"): value
            for name, value in outputs.items()
            if name.startswith("forward_")
        }
        proposed = {
            name: outputs[f"proposed_{name}"]
            for name in self.parameter_ids
        }
        incidentals = {
            feed_id: values[feed_id]
            for feed_id in self.reverse_capture.incidental_feed_ids
            if feed_id in values
        }
        return ForwardReverseCycleResult(forward_outputs, proposed, incidentals)

    def emit_fortran(self, *, name: str = "forward_reverse_cycle") -> "FortranCycleArtifact":
        if not self.correction.fortran_fusible:
            raise ValueError(
                f"{type(self.correction).__name__} is a host correction and cannot "
                "be represented by the fused Fortran cycle"
            )
        from ....compiler.machine_targets import emit

        artifact = emit(self.fused_program, "fortran", name=name)
        if not artifact.complete:
            raise RuntimeError(
                "Fortran cannot express the fused cycle: " + "; ".join(artifact.shortfalls)
            )
        # A Fortran module and a contained subroutine may not have the same
        # global name. The generic target historically gives them the same
        # spelling for source-only artifacts, so disambiguate the runnable
        # package while preserving the bind(C) entry symbol.
        module = artifact.module
        if module.name == name:
            module_name = f"{name}_module"
            source = module.source.replace(
                f"module {name}\n", f"module {module_name}\n", 1
            ).replace(
                f"end module {name}\n", f"end module {module_name}\n", 1
            )
            module = replace(
                module,
                name=module_name,
                source=source,
                api=replace(module.api, module=module_name),
            )
        return FortranCycleArtifact(
            program=self.fused_program,
            module=module,
            feed_values=dict(self.feed_values),
            entrypoint=name,
            parameter_ids=dict(self.parameter_ids),
            target_ids=dict(self.target_ids),
        )


def capture_forward_reverse_cycle(
    forward: Callable[[Mapping[str, AT]], Mapping[str, AT] | AT],
    feeds: Mapping[str, AT],
    *,
    solve_for: Iterable[str],
    targets: TargetStrategy,
    correction: CorrectionStrategy = GradientCorrection(),
    iteration: int = 0,
) -> ForwardReverseCycleCapture:
    """Capture one targetable, correctable forward/back parameter cycle."""

    solve_names = tuple(dict.fromkeys(solve_for))
    unknown = set(solve_names) - set(feeds)
    if unknown:
        raise KeyError(f"solve_for names unknown feeds: {sorted(unknown)}")
    tape = GradTape()
    previous_tape = autograd.tape
    autograd.tape = tape
    try:
        bound = dict(feeds)
        for name, value in bound.items():
            value._tape = tape
            value.requires_grad_(name in solve_names)
            tape.create_tensor_node(value)
        raw_outputs = forward(bound)
        declared = (
            {"output": raw_outputs}
            if isinstance(raw_outputs, AT)
            else dict(raw_outputs)
        )
        if not declared or not all(isinstance(value, AT) for value in declared.values()):
            raise TypeError("forward must return an AbstractTensor or a non-empty mapping of them")
        program = build_fused_program(
            tape.graph.copy(),
            outputs={name: id(value) for name, value in declared.items()},
        )
        extras = dict(program.extras or {})
        origins = dict(extras.get("capture_feed_origins", {}) or {})
        for name, value in bound.items():
            origins[id(value)] = {"binding_name": name}
        extras["capture_feed_origins"] = origins
        program.extras = extras
        retained = retain_uncaptured_outputs(program)
        live_values = _live_tensors(tape)
        current_outputs = {
            name: live_values[value_id]
            for name, value_id in retained.outputs.items()
        }
        desired = dict(targets(iteration, current_outputs))
        reverse = capture_reverse_fused_program(
            retained,
            live_values,
            desired,
            step_size=correction.step_size(iteration),
            wrt_feed_ids=(id(bound[name]) for name in solve_names),
        )
        reverse_origins = dict((reverse.program.extras or {}).get("capture_feed_origins", {}) or {})
        reverse_origins.update(origins)
        for name, value_id in reverse.output_parameters.items():
            reverse_origins[value_id] = {"binding_name": f"target_{name}"}
        reverse.program.extras = dict(reverse.program.extras or {})
        reverse.program.extras["capture_feed_origins"] = reverse_origins
        fused = fuse_forward_reverse_program(retained, reverse)
        values = dict(reverse.feed_values)
        values.update(live_values)
        fused_values = {feed_id: values[feed_id] for feed_id in fused.feeds}
        return ForwardReverseCycleCapture(
            iteration=iteration,
            forward_program=retained,
            reverse_capture=reverse,
            fused_program=fused,
            feed_values=fused_values,
            parameter_ids={name: id(bound[name]) for name in solve_names},
            target_ids=dict(reverse.output_parameters),
            correction=correction,
            tape=tape,
        )
    finally:
        autograd.tape = previous_tape


@dataclass(frozen=True)
class SolverStep:
    iteration: int
    forward_outputs: Dict[str, AT]
    parameters: Dict[str, AT]
    capture: ForwardReverseCycleCapture


class ForwardReverseSolver:
    """Recapture and run successive local inverse proposals."""

    def __init__(
        self,
        forward: Callable[[Mapping[str, AT]], Mapping[str, AT] | AT],
        feeds: Mapping[str, AT],
        *,
        solve_for: Iterable[str],
        targets: TargetStrategy,
        correction: CorrectionStrategy = GradientCorrection(),
    ) -> None:
        self.forward = forward
        self.feeds = dict(feeds)
        self.solve_for = tuple(dict.fromkeys(solve_for))
        self.targets = targets
        self.correction = correction
        self.iteration = 0
        self.history: list[Dict[str, AT]] = []

    def capture(self) -> ForwardReverseCycleCapture:
        return capture_forward_reverse_cycle(
            self.forward,
            self.feeds,
            solve_for=self.solve_for,
            targets=self.targets,
            correction=self.correction,
            iteration=self.iteration,
        )

    def step(self) -> SolverStep:
        capture = self.capture()
        result = capture.run_python()
        current = {name: self.feeds[name] for name in self.solve_for}
        corrected = dict(self.correction.apply(
            self.iteration, current, result.proposed_parameters
        ))
        if set(corrected) != set(self.solve_for):
            raise ValueError("correction hook must return every solve_for parameter exactly once")
        self.history.append(dict(current))
        self.feeds.update(corrected)
        completed = SolverStep(
            self.iteration, result.forward_outputs, dict(corrected), capture
        )
        self.iteration += 1
        return completed

    def solve(self, iterations: int) -> tuple[SolverStep, ...]:
        if iterations < 0:
            raise ValueError("iterations must be non-negative")
        return tuple(self.step() for _ in range(iterations))


_CTYPES = {
    "bool": ctypes.c_bool,
    "logical": ctypes.c_bool,
    "float": ctypes.c_float,
    "float32": ctypes.c_float,
    "f32": ctypes.c_float,
    "double": ctypes.c_double,
    "float64": ctypes.c_double,
    "f64": ctypes.c_double,
    "int": ctypes.c_int32,
    "int32": ctypes.c_int32,
    "i32": ctypes.c_int32,
    "int64": ctypes.c_int64,
    "i64": ctypes.c_int64,
}

_NUMPY_DTYPES = {
    "bool": "bool", "logical": "bool", "float": "float32",
    "float32": "float32", "f32": "float32", "double": "float64",
    "float64": "float64", "f64": "float64", "int": "int32",
    "int32": "int32", "i32": "int32", "int64": "int64", "i64": "int64",
}


@dataclass(frozen=True)
class FortranCycleArtifact:
    """Emitted fused cycle with its feed boundary and native compile hook."""

    program: FusedProgram
    module: Any
    feed_values: Dict[int, AT]
    entrypoint: str
    parameter_ids: Dict[str, int]
    target_ids: Dict[str, int]

    def write(self, directory: str | Path) -> Path:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        return self.module.write(output)

    def compile(self, directory: str | Path) -> "FortranCycleExecutable":
        from ....compiler.ssa_fortran_backend import compile_module

        path = compile_module(self.module, directory=Path(directory).resolve())
        return FortranCycleExecutable(self, path)


@dataclass(frozen=True)
class FortranCycleExecutable:
    artifact: FortranCycleArtifact
    library_path: Path

    def __call__(self, feeds: Mapping[int, Any] | None = None) -> Dict[str, np.ndarray]:
        values = {
            feed_id: np.asarray(value.tolist() if isinstance(value, AT) else value)
            for feed_id, value in self.artifact.feed_values.items()
        }
        if feeds:
            values.update({key: np.asarray(value) for key, value in feeds.items()})
        missing = set(self.artifact.program.feeds) - set(values)
        if missing:
            raise KeyError(f"native cycle is missing feeds: {sorted(missing)}")
        compiler_directory = None
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            from ....compiler.ssa_fortran_backend import fortran_compiler

            compiler = fortran_compiler()
            compiler_directory = os.add_dll_directory(str(Path(compiler).parent)) if compiler else None
        library = ctypes.CDLL(str(self.library_path))
        try:
            api = self.artifact.module.api.entry_point(self.artifact.entrypoint)
            native = getattr(library, api.symbol)
            arguments: list[Any] = []
            argument_types: list[Any] = []
            output_arrays: Dict[int, np.ndarray] = {}
            for parameter in api.parameters:
                dtype = str(parameter.dtype).casefold()
                ctype = _CTYPES.get(dtype, ctypes.c_double)
                if parameter.role == "extent":
                    extent = int(parameter.name.rsplit("_", 1)[-1])
                    arguments.append(ctype(extent))
                    argument_types.append(ctype)
                    continue
                value_id = int(parameter.name[1:])
                if parameter.role == "input":
                    array = np.asfortranarray(values[value_id], dtype=_NUMPY_DTYPES.get(dtype, "float64"))
                else:
                    array = np.asfortranarray(np.empty(
                        tuple(parameter.shape) or (), dtype=_NUMPY_DTYPES.get(dtype, "float64")
                    ))
                    output_arrays[value_id] = array
                if parameter.passing == "value":
                    arguments.append(ctype(array.item()))
                    argument_types.append(ctype)
                else:
                    pointer = ctypes.POINTER(ctype)
                    arguments.append(array.ctypes.data_as(pointer))
                    argument_types.append(pointer)
            native.argtypes = argument_types
            native.restype = None
            native(*arguments)
            return {
                name: np.asarray(output_arrays[value_id]).copy()
                for name, value_id in self.artifact.program.outputs.items()
            }
        finally:
            import _ctypes

            if os.name == "nt":
                _ctypes.FreeLibrary(library._handle)
            else:  # pragma: no cover - exercised on POSIX builders
                _ctypes.dlclose(library._handle)
            if compiler_directory is not None:
                compiler_directory.close()

    def cycle(
        self,
        iterations: int,
        *,
        feeds: Mapping[int, Any] | None = None,
        target_hook: Callable[
            [int, Mapping[str, np.ndarray] | None], Mapping[str, Any]
        ] | None = None,
    ) -> tuple[Dict[str, np.ndarray], ...]:
        """Run repeated native forward/back cycles, feeding proposals onward.

        ``target_hook`` can replace desired output feeds between native calls.
        It receives the cycle index and the previous fused outputs.
        """

        if iterations < 0:
            raise ValueError("iterations must be non-negative")
        boundary = dict(feeds or {})
        history: list[Dict[str, np.ndarray]] = []
        previous: Mapping[str, np.ndarray] | None = None
        for iteration in range(iterations):
            if target_hook is not None:
                targets = dict(target_hook(iteration, previous))
                unknown = set(targets) - set(self.artifact.target_ids)
                if unknown:
                    raise KeyError(f"native target hook names unknown outputs: {sorted(unknown)}")
                boundary.update({
                    self.artifact.target_ids[name]: value
                    for name, value in targets.items()
                })
            outputs = self(boundary)
            history.append(outputs)
            for name, value_id in self.artifact.parameter_ids.items():
                boundary[value_id] = outputs[f"proposed_{name}"]
            previous = outputs
        return tuple(history)


def _demo_forward(values: Mapping[str, AT]) -> Mapping[str, AT]:
    prediction = values["parameter"] * 2.0 + 1.0
    observation = values["reference"] * values["reference"]
    return {"prediction": prediction, "observation": observation}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--step-size", type=float, default=0.2)
    parser.add_argument("--emit-fortran", type=Path)
    parser.add_argument("--compile-fortran", action="store_true")
    args = parser.parse_args(argv)
    solver = ForwardReverseSolver(
        _demo_forward,
        {
            "parameter": AT.tensor((-2.0, 0.0, 3.0)),
            "reference": AT.tensor((1.0, 2.0, 3.0)),
        },
        solve_for=("parameter",),
        targets=FixedTargets({"prediction": (5.0, -3.0, 1.0)}),
        correction=GradientCorrection(args.step_size),
    )
    steps = solver.solve(args.iterations)
    final_capture = solver.capture()
    final_result = final_capture.run_python()
    record: Dict[str, Any] = {
        "iterations": args.iterations,
        "parameters": {
            name: solver.feeds[name].tolist() for name in solver.solve_for
        },
        "outputs": {name: value.tolist() for name, value in final_result.forward_outputs.items()},
        "retained_outputs": list(final_capture.forward_program.outputs),
        "fused_steps": len(final_capture.fused_program.steps),
        "history_length": len(steps),
    }
    if args.emit_fortran is not None:
        artifact = final_capture.emit_fortran()
        record["fortran_source"] = str(artifact.write(args.emit_fortran))
        if args.compile_fortran:
            executable = artifact.compile(args.emit_fortran)
            native_cycles = executable.cycle(2)
            record["fortran_outputs"] = {
                name: value.tolist() for name, value in native_cycles[-1].items()
            }
            record["fortran_cycles"] = len(native_cycles)
            record["fortran_library"] = str(executable.library_path)
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a runnable
    raise SystemExit(main())


__all__ = [
    "CallableCorrection", "ClippedCorrection", "CorrectionStrategy",
    "FixedTargets", "FortranCycleArtifact", "FortranCycleExecutable",
    "ForwardReverseCycleCapture", "ForwardReverseCycleResult",
    "ForwardReverseSolver", "GradientCorrection", "InterpolatedTargets",
    "SolverStep", "TargetStrategy", "capture_forward_reverse_cycle",
    "fuse_forward_reverse_program", "main",
]
