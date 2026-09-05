"""Compile a sorting forward/reverse learner through the shared compiler process.

The application contributes no hand-written Fortran.  The captured learning
cycle and a span/index based renderer are ordinary ``FusedProgram`` regions;
the existing control-to-SSA compiler joins them, the Fortran backend emits the
module, and the common C shell supplies caller-owned arenas and presentation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import networkx as nx
import numpy as np

from ..common.tensors.abstraction import AbstractTensor as AT
from ..common.tensors.abstract_nn.forward_reverse_cycle import (
    FixedTargets,
    ForwardReverseCycleCapture,
    GradientCorrection,
    capture_forward_reverse_cycle,
)
from ..common.tensors.fused_ir import FusedProgram, Meta, OpStep
from ..transmogrifier.ssa import IRModule, SSAValue
from .control_source import ControlProgram, SequenceBlock, StatementBlock
from .fortran_c_shell import FortranCShellExecutable, compile_fortran_module_c_shell
from .precompile_to_ssa import lower_precompile_and_control_to_ssa
from .process_graph_fusion import fused_program_to_process_graph
from .shell_io import (
    ShellIOBinding,
    ShellIOCapability,
    ShellIOManifest,
    ShellIORequest,
    attach_shell_io,
)
from .ssa_fortran_backend import FortranModule, emit_module


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"turing_sorting_problem_{abs(hash(path.resolve()))}", path,
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load sorting problem {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class SortingProcessProblem:
    name: str
    input_names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    forward: Any
    reference: Any
    comparators: tuple[tuple[int, int], ...]


def load_sorting_process_problem(python_file: str | Path) -> SortingProcessProblem:
    path = Path(python_file).resolve()
    factory = getattr(_load_module(path), "build_process_problem", None)
    if not callable(factory):
        raise ValueError(f"{path} must define build_process_problem()")
    raw = factory()
    if not isinstance(raw, Mapping):
        raise TypeError("build_process_problem() must return a mapping")
    forward, reference = raw.get("forward"), raw.get("reference")
    inputs = tuple(map(str, raw.get("input_names", ())))
    parameters = tuple(map(str, raw.get("parameter_names", ())))
    comparators = tuple(tuple(map(int, pair)) for pair in raw.get("comparators", ()))
    if not callable(forward) or not callable(reference):
        raise TypeError("sorting problem needs callable forward and reference entries")
    if not inputs or not parameters or len(parameters) != len(comparators):
        raise ValueError("sorting problem needs one parameter per comparator")
    return SortingProcessProblem(
        str(raw.get("name", path.stem)), inputs, parameters,
        forward, reference, comparators,
    )


@dataclass(frozen=True)
class CapturedSortingProcess:
    problem: SortingProcessProblem
    cycle: ForwardReverseCycleCapture
    rows: np.ndarray
    targets: np.ndarray


def capture_sorting_process(
    python_file: str | Path,
    *,
    batch_size: int = 64,
    seed: int = 7,
    step_size: float = 0.02,
    initial_gate: float = 0.05,
) -> CapturedSortingProcess:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    problem = load_sorting_process_problem(python_file)
    rows = np.random.default_rng(seed).uniform(
        -1.0, 1.0, size=(batch_size, len(problem.input_names)),
    )
    targets = np.asarray(problem.reference(rows), dtype=np.float64)
    if targets.shape != rows.shape:
        raise ValueError("sorting reference must preserve the input shape")
    feeds = {
        name: AT.tensor(rows[:, index].tolist())
        for index, name in enumerate(problem.input_names)
    }
    feeds.update({name: AT.tensor(float(initial_gate)) for name in problem.parameter_names})
    desired = {
        f"sorted_{index}": targets[:, index].tolist()
        for index in range(targets.shape[1])
    }
    cycle = capture_forward_reverse_cycle(
        problem.forward,
        feeds,
        solve_for=problem.parameter_names,
        targets=FixedTargets(desired),
        correction=GradientCorrection(step_size),
    )
    return CapturedSortingProcess(problem, cycle, rows, targets)


def _bounded_cycle(captured: CapturedSortingProcess) -> FusedProgram:
    """Keep the feedback policy in IR so the shell only aliases addresses."""

    original = captured.cycle.fused_program
    steps = list(original.steps)
    outputs = dict(original.outputs)
    metadata = dict(original.meta or {})
    next_id = max(
        [*original.feeds, *outputs.values(), *(step.result_id for step in steps)],
        default=-1,
    ) + 1
    for name in captured.problem.parameter_names:
        source_id = outputs[f"proposed_{name}"]
        source_meta = metadata[source_id]
        low_id, bounded_id = next_id, next_id + 1
        next_id += 2
        steps.append(OpStep(len(steps), "maximum", [source_id], {"right_scalar": 0.0}, low_id))
        steps.append(OpStep(len(steps), "minimum", [low_id], {"right_scalar": 1.0}, bounded_id))
        metadata[low_id] = source_meta
        metadata[bounded_id] = source_meta
        outputs[f"proposed_{name}"] = bounded_id
    return FusedProgram(
        original.version, set(original.feeds), steps, outputs,
        state_in=original.state_in, meta=metadata, extras=original.extras,
    )


def _graph_layout(program: FusedProgram, bounds: tuple[int, int, int, int]):
    graph = fused_program_to_process_graph(program)
    dag = nx.DiGraph(graph.G)
    generations = list(nx.topological_generations(dag))
    left, top, right, bottom = bounds
    positions: dict[int, tuple[int, int]] = {}
    for column, generation in enumerate(generations):
        x = int(left + (right-left) * column / max(1, len(generations)-1))
        ordered = sorted(generation)
        for row, node_id in enumerate(ordered):
            y = int(top + (bottom-top) * (row+1) / (len(ordered)+1))
            positions[int(node_id)] = (x, y)
    return graph, positions


def _line(plane: np.ndarray, width: int, start, end, color):
    x0, y0 = start
    x1, y1 = end
    count = max(abs(x1-x0), abs(y1-y0), 1) + 1
    xs = np.rint(np.linspace(x0, x1, count)).astype(int)
    ys = np.rint(np.linspace(y0, y1, count)).astype(int)
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < plane.shape[1] // width)
    plane[:, ys[valid]*width + xs[valid]] = np.asarray(color)[:, None]


def _disk_indices(width: int, height: int, center, radius: int = 3) -> list[int]:
    cx, cy = center
    return [
        y*width+x
        for y in range(max(0, cy-radius), min(height, cy+radius+1))
        for x in range(max(0, cx-radius), min(width, cx+radius+1))
        if (x-cx)**2 + (y-cy)**2 <= radius**2
    ]


@dataclass(frozen=True)
class RenderRegion:
    program: FusedProgram
    feeds: dict[int, np.ndarray]
    manifest: dict[str, Any]


def _render_region(
    captured: CapturedSortingProcess,
    cycle: FusedProgram,
    *,
    width: int,
    height: int,
) -> RenderRegion:
    forward_graph, forward_pos = _graph_layout(
        captured.cycle.forward_program, (18, 24, width-18, height//2-18),
    )
    reverse_graph, reverse_pos = _graph_layout(
        captured.cycle.reverse_capture.program,
        (18, height//2+18, width-18, height-24),
    )
    background_fill = np.asarray((0.025, 0.035, 0.06), dtype=np.float64)
    background = np.empty((3, width*height), dtype=np.float64)
    background[:] = background_fill[:, None]
    for graph, positions, edge_color, node_color in (
        (forward_graph, forward_pos, (0.08, 0.19, 0.27), (0.18, 0.48, 0.65)),
        (reverse_graph, reverse_pos, (0.22, 0.10, 0.26), (0.55, 0.25, 0.62)),
    ):
        for source, target in graph.G.edges:
            _line(background, width, positions[int(source)], positions[int(target)], edge_color)
        for point in positions.values():
            for pixel in _disk_indices(width, height, point, 1):
                background[:, pixel] = node_color

    segment_ids = [
        captured.cycle.parameter_ids[name]
        for name in captured.problem.parameter_names
    ]
    segment_points = [forward_pos[value_id] for value_id in segment_ids]
    proposal_ids = [cycle.outputs[f"proposed_{name}"] for name in captured.problem.parameter_names]
    # The reverse ProcessGraph contains the raw proposal values. The bounded
    # feedback values are two compiler ops later, so display them at the raw
    # proposal node while consuming the bounded value that actually feeds back.
    raw_proposals = [
        captured.cycle.reverse_capture.proposed_inputs[f"proposed_{name}"]
        for name in captured.problem.parameter_names
    ]
    segment_points.extend(reverse_pos[value_id] for value_id in raw_proposals)
    segment_ids.extend(proposal_ids)

    overlay_indices: list[int] = []
    overlay_value_ids: list[int] = []
    side_values: list[float] = []
    for index, (value_id, point) in enumerate(zip(segment_ids, segment_points)):
        pixels = _disk_indices(width, height, point, 4)
        overlay_indices.extend(pixels)
        overlay_value_ids.extend([value_id] * len(pixels))
        side_values.extend([0.0 if index < len(segment_ids)//2 else 1.0] * len(pixels))

    used = [*cycle.feeds, *cycle.outputs.values(), *(step.result_id for step in cycle.steps)]
    next_id = max(used, default=-1) + 1
    (
        static_indices_id, static_red_id, static_green_id, static_blue_id,
        indices_id, side_id,
    ) = range(next_id, next_id+6)
    next_id += 6
    static_mask = np.any(background != background_fill[:, None], axis=0)
    static_indices = np.flatnonzero(static_mask).astype(np.int64)
    metadata = {
        static_indices_id: Meta((len(static_indices),), "int64"),
        static_red_id: Meta((len(static_indices),), "float64"),
        static_green_id: Meta((len(static_indices),), "float64"),
        static_blue_id: Meta((len(static_indices),), "float64"),
        indices_id: Meta((len(overlay_indices),), "int64"),
        side_id: Meta((len(overlay_indices),), "float64"),
    }
    # These values are produced by the numerical cycle and consumed by this
    # separately compiled renderer region.  Retain their arena shapes at the
    # region boundary; omitting them makes the generic lowering conservatively
    # invent rank-zero values and turns one-element spans into scalars.
    for value_id in segment_ids:
        metadata[value_id] = cycle.meta[value_id]
    # The image planes are compiler spans: Fill initializes their already
    # allocated arenas with one scalar assignment, then generic indexed
    # scatter writes the sparse graph and live node overlays.  Pixel data is
    # never expanded into a Fortran array constructor.
    red_base, green_base, blue_base = range(next_id, next_id+3)
    next_id += 3
    red_graph, green_graph, blue_graph = range(next_id, next_id+3)
    next_id += 3
    steps: list[OpStep] = [
        OpStep(0, "full", [], {"fill_value": float(background_fill[0])}, red_base),
        OpStep(1, "full", [], {"fill_value": float(background_fill[1])}, green_base),
        OpStep(2, "full", [], {"fill_value": float(background_fill[2])}, blue_base),
        OpStep(3, "scatter", [red_base, static_indices_id, static_red_id], {"dim": 0}, red_graph),
        OpStep(4, "scatter", [green_base, static_indices_id, static_green_id], {"dim": 0}, green_graph),
        OpStep(5, "scatter", [blue_base, static_indices_id, static_blue_id], {"dim": 0}, blue_graph),
    ]
    for value_id in (red_base, green_base, blue_base, red_graph, green_graph, blue_graph):
        metadata[value_id] = Meta((width*height,), "float64")

    values_id = next_id
    next_id += 1
    steps.append(OpStep(0, "concat", overlay_value_ids, {"dim": 0}, values_id))
    metadata[values_id] = Meta((len(overlay_value_ids),), "float64")

    inverse_id, red_values, green_scaled, green_values, blue_scaled, blue_values = range(next_id, next_id+6)
    next_id += 6
    steps.extend((
        OpStep(len(steps), "sub", [values_id], {"right_scalar": 1.0, "reverse": True}, inverse_id),
        OpStep(len(steps)+1, "mul", [inverse_id], {"right_scalar": 0.75}, red_values),
        OpStep(len(steps)+2, "mul", [values_id], {"right_scalar": 0.75}, green_scaled),
        OpStep(len(steps)+3, "add", [green_scaled], {"right_scalar": 0.20}, green_values),
        OpStep(len(steps)+4, "mul", [side_id], {"right_scalar": 0.55}, blue_scaled),
        OpStep(len(steps)+5, "add", [blue_scaled], {"right_scalar": 0.30}, blue_values),
    ))
    for value_id in (inverse_id, red_values, green_scaled, green_values, blue_scaled, blue_values):
        metadata[value_id] = Meta((len(overlay_indices),), "float64")

    red_id, green_id, blue_id = range(next_id, next_id+3)
    steps.extend((
        OpStep(len(steps), "scatter", [red_graph, indices_id, red_values], {"dim": 0}, red_id),
        OpStep(len(steps)+1, "scatter", [green_graph, indices_id, green_values], {"dim": 0}, green_id),
        OpStep(len(steps)+2, "scatter", [blue_graph, indices_id, blue_values], {"dim": 0}, blue_id),
    ))
    for value_id in (red_id, green_id, blue_id):
        metadata[value_id] = Meta((width*height,), "float64")

    origins = {
        static_indices_id: {"binding_name": "graph_static_indices"},
        static_red_id: {"binding_name": "graph_static_red"},
        static_green_id: {"binding_name": "graph_static_green"},
        static_blue_id: {"binding_name": "graph_static_blue"},
        indices_id: {"binding_name": "graph_overlay_indices"},
        side_id: {"binding_name": "graph_overlay_side"},
    }
    program = FusedProgram(
        1,
        {
            static_indices_id, static_red_id, static_green_id, static_blue_id,
            indices_id, side_id, *segment_ids,
        },
        steps,
        {"red": red_id, "green": green_id, "blue": blue_id},
        meta=metadata,
        extras={"capture_feed_origins": origins},
    )
    feeds = {
        static_indices_id: static_indices,
        static_red_id: background[0, static_mask],
        static_green_id: background[1, static_mask],
        static_blue_id: background[2, static_mask],
        indices_id: np.asarray(overlay_indices, dtype=np.int64),
        side_id: np.asarray(side_values, dtype=np.float64),
    }
    manifest = {
        "schema": "turing.sorting-process-graphs.v2",
        "forward": {
            "nodes": len(forward_graph.G.nodes), "edges": len(forward_graph.G.edges),
            "positions": {str(key): value for key, value in forward_pos.items()},
        },
        "reverse": {
            "nodes": len(reverse_graph.G.nodes), "edges": len(reverse_graph.G.edges),
            "positions": {str(key): value for key, value in reverse_pos.items()},
        },
        "segments": [
            {"value_id": int(value_id), "x": point[0], "y": point[1]}
            for value_id, point in zip(segment_ids, segment_points)
        ],
    }
    return RenderRegion(program, feeds, manifest)


def _value(function, value_id: int) -> SSAValue:
    for value in function.args:
        if value.id == value_id:
            return value
    for block in function.blocks.values():
        for instruction in block.instrs:
            if instruction.res is not None and instruction.res.id == value_id:
                return instruction.res
            for argument in instruction.args:
                if argument.id == value_id:
                    return argument
    raise KeyError(f"SSA function {function.name} has no value {value_id}")


@dataclass(frozen=True)
class NativeSortingProcessWindow:
    captured: CapturedSortingProcess
    executable: FortranCShellExecutable
    graph_manifest_path: Path

    @property
    def executable_path(self) -> Path:
        return self.executable.executable_path

    def run(self, *, frames: int = 0, capture_output: bool = False):
        return self.executable.run(frames=frames, capture_output=capture_output)

    __call__ = run


def compile_sorting_process_window(
    python_file: str | Path,
    output_directory: str | Path,
    *,
    batch_size: int = 64,
    seed: int = 7,
    step_size: float = 0.02,
    initial_gate: float = 0.05,
    width: int = 900,
    height: int = 640,
) -> NativeSortingProcessWindow:
    captured = capture_sorting_process(
        python_file, batch_size=batch_size, seed=seed,
        step_size=step_size, initial_gate=initial_gate,
    )
    cycle = _bounded_cycle(captured)
    renderer = _render_region(captured, cycle, width=width, height=height)
    control_name = "sorting_process_learning_window"
    control = ControlProgram(
        SequenceBlock((
            StatementBlock(("__scheduled_region_0__",)),
            StatementBlock(("__scheduled_region_1__",)),
        )),
        region_indices=(0, 1),
    )

    origins: dict[int, str] = {}
    for program in (cycle, renderer.program):
        for value_id, record in dict((program.extras or {}).get("capture_feed_origins", {})).items():
            name = record.get("binding_name") if isinstance(record, Mapping) else None
            if name:
                origins[int(value_id)] = str(name)
    origins.update({value_id: name for name, value_id in renderer.program.outputs.items()})
    origins.update({
        cycle.outputs[f"proposed_{name}"]: f"proposed_{name}"
        for name in captured.problem.parameter_names
    })
    identity_table = {name: (value_id,) for value_id, name in origins.items()}
    public_outputs = (
        "red", "green", "blue",
        *(f"proposed_{name}" for name in captured.problem.parameter_names),
    )
    public_inputs = tuple(
        name for value_id, name in origins.items()
        if value_id in (cycle.feeds | renderer.program.feeds)
        and value_id not in cycle.outputs.values()
    )
    lowering = lower_precompile_and_control_to_ssa(
        cycle,
        control,
        numerical_name="sorting_cycle_validation",
        control_name=control_name,
        region_programs={0: cycle, 1: renderer.program},
        identity_table=identity_table,
        function_outputs=public_outputs,
        function_parameters=public_inputs,
    )
    if not lowering.complete:
        raise RuntimeError(lowering.shortfall_report())
    functions = {
        name: lowering.module.functions[name]
        for name in ("numerical_region_0", "numerical_region_1", control_name)
    }
    control_function = functions[control_name]
    output_ids = [
        renderer.program.outputs["red"],
        renderer.program.outputs["green"],
        renderer.program.outputs["blue"],
        *(cycle.outputs[f"proposed_{name}"] for name in captured.problem.parameter_names),
    ]
    outputs = {
        "numerical_region_0": [
            _value(functions["numerical_region_0"], value_id)
            for value_id in cycle.outputs.values()
        ],
        "numerical_region_1": [
            _value(functions["numerical_region_1"], value_id)
            for value_id in renderer.program.outputs.values()
        ],
        control_name: [_value(control_function, value_id) for value_id in output_ids],
    }
    emitted = emit_module(
        IRModule(functions), name="sorting_process_compiled", outputs=outputs,
    )
    if not emitted.complete:
        raise RuntimeError("Fortran emission failed: " + "; ".join(
            item.format() for item in emitted.shortfalls
        ))
    manifest = ShellIOManifest(
        requests=(ShellIORequest.create(
            ShellIOCapability.DISPLAY,
            attributes={
                "pixel_format": "rgb_f64_planar", "width": width, "height": height,
                "title": "Sorting forward / derived reverse process",
            },
        ),),
        bindings=tuple(
            ShellIOBinding(f"display.{channel}", control_name, channel)
            for channel in ("red", "green", "blue")
        ),
    )
    module = FortranModule(
        emitted.name, emitted.source, emitted.subroutines,
        api=attach_shell_io(emitted.api, manifest),
    )

    values_by_id: dict[int, np.ndarray] = {
        value_id: np.asarray(value.tolist() if isinstance(value, AT) else value)
        for value_id, value in captured.cycle.feed_values.items()
        if value_id in cycle.feeds
    }
    values_by_id.update(renderer.feeds)
    entry = module.api.entry_point(control_name)
    inputs = {}
    for parameter in entry.parameters:
        if parameter.role != "input":
            continue
        value_id = int(parameter.name.removeprefix("t"))
        inputs[str(parameter.source_name or parameter.name)] = values_by_id[value_id]
    feedback = {
        name: f"proposed_{name}" for name in captured.problem.parameter_names
    }
    output = Path(output_directory).resolve()
    executable = compile_fortran_module_c_shell(
        module, inputs, output, entrypoint=control_name,
        state_feedback=feedback, name=control_name,
    )
    graph_manifest = output / "sorting-process-graphs.json"
    graph_manifest.write_text(
        json.dumps(renderer.manifest, indent=2, sort_keys=True), encoding="utf-8",
    )
    return NativeSortingProcessWindow(captured, executable, graph_manifest)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("python_file", type=Path)
    parser.add_argument("--output", type=Path, default=Path("build/native-sorting-process"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--step-size", type=float, default=0.02)
    parser.add_argument("--initial-gate", type=float, default=0.05)
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args(argv)
    artifact = compile_sorting_process_window(
        args.python_file, args.output, batch_size=args.batch_size,
        seed=args.seed, step_size=args.step_size, initial_gate=args.initial_gate,
        width=args.width, height=args.height,
    )
    print(artifact.executable_path)
    if not args.compile_only:
        artifact.run(frames=args.frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CapturedSortingProcess", "NativeSortingProcessWindow", "SortingProcessProblem",
    "capture_sorting_process", "compile_sorting_process_window",
    "load_sorting_process_problem", "main",
]
