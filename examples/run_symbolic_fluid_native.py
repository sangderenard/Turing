"""Build and run the whole managed-time symbolic fluid program natively.

This is a coordination demo, not a second fluid implementation.  It loads the
repository SSA produced by ``symbolic_fluid_direct_control``, emits the real
Fortran program and its C shell, initializes the authored ABI objects, and
runs the resident state for as many outer frames as requested.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import pickle
from typing import Any

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from src.common.dt_system.dt_controller import STController, Targets
from src.common.dt_system.dt_scaler import Metrics
from src.compiler.fortran_c_shell import compile_fortran_module_c_shell
from src.compiler.ssa_fortran_backend import emit_module
from src.compiler.symbolic_fluid_direct_control import bounded_compile
from src.compiler.symbolic_fluid_dt import SymbolicFluidGridState


ENTRY_SUFFIX = "symbolic_fluid_frame"


def _entry(module: Any) -> Any:
    return next(
        point
        for point in module.api.entry_points
        if point.name.endswith(ENTRY_SUFFIX)
    )


def _extent_manifest(
    emitted: Any,
    repository_module: Any,
    *,
    grid_size: int,
    collection_capacity: int,
) -> dict[str, int]:
    """Resolve emitted extents from their function-local SSA provenance."""

    uses: dict[str, set[str]] = defaultdict(set)
    for point in emitted.api.entry_points:
        function = repository_module.functions.get(point.name)
        values = {}
        if function is not None:
            values.update((int(value.id), value) for value in function.args)
            values.update(
                (int(instruction.res.id), instruction.res)
                for block in function.blocks.values()
                for instruction in block.instrs
                if instruction.res is not None
            )
        for parameter in point.parameters:
            if not parameter.extents:
                continue
            value = (
                values.get(int(parameter.name[1:]))
                if parameter.name.startswith("t")
                else None
            )
            accounting = dict(value.accounting or {}) if value is not None else {}
            kind = (
                "grid"
                if len(parameter.extents) == 2
                else "row"
                if accounting.get("projected_row_source_id") is not None
                else "collection"
            )
            for name in parameter.extents:
                uses[str(name)].add(kind)

    return {
        parameter.name: (
            grid_size
            if uses[parameter.name] & {"grid", "row"}
            else collection_capacity
        )
        for parameter in _entry(emitted).parameters
        if parameter.role == "extent"
        and parameter.name.startswith("extent_dynamic")
    }


def _inputs(
    entry: Any,
    *,
    state: SymbolicFluidGridState,
    targets: Targets,
    controller: STController,
    frame_duration: float,
    dt_initial: float,
) -> dict[str, Any]:
    metrics = Metrics(0.0, 0.0, 0.0, 0.0)
    roots = {
        "state": state,
        "targets": targets,
        "controller": controller,
        "ctrl": controller,
        "self": controller,
        "value": metrics,
    }
    result: dict[str, Any] = {
        "frame_duration": float(frame_duration),
        "dt_initial": float(dt_initial),
    }
    for parameter in entry.parameters:
        if parameter.role not in {"input", "inout"}:
            continue
        name = str(parameter.source_name or parameter.name)
        if name in result:
            continue
        root, separator, field = name.partition(".")
        if not separator or root not in roots:
            raise KeyError(f"no authored native input source for {name!r}")
        value = getattr(roots[root], field)
        # Optional/reference-valued metrics need tagged optionals in the final
        # ABI.  The current numerical contract represents their absent/default
        # state as zero; keep that stopgap explicit here.
        if value is None or isinstance(value, dict):
            value = 0.0
        result[name] = value
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("build/symbolic-fluid-native-live"))
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--max-iters", type=int, default=256)
    parser.add_argument("--frame-duration", type=float, default=1.0 / 30.0)
    parser.add_argument("--dt-initial", type=float, default=1.0e-3)
    parser.add_argument("--reingest", action="store_true")
    parser.add_argument(
        "--stream-frames", action="store_true",
        help="emit one flushed JSON event after every resident native frame",
    )
    args = parser.parse_args()
    if args.frames < 0 or args.size < 4 or args.max_iters < 1:
        parser.error("frames must be nonnegative, size >= 4, and max-iters positive")

    output = args.output.resolve()
    repository_path = output / "control_repository_ssa.pkl"
    if args.reingest or not repository_path.is_file():
        report = bounded_compile(output, max_memory_gb=0.0)
        if not report["completed"]:
            raise RuntimeError(json.dumps(report, indent=2))

    with repository_path.open("rb") as stream:
        repository_module, outputs, exports = pickle.load(stream)
    emitted = emit_module(
        repository_module,
        name="symbolic_fluid_control",
        outputs=outputs,
        extra_roots=exports,
    )
    entry = _entry(emitted)
    state = SymbolicFluidGridState.initial(args.size, args.size)
    targets = Targets(
        cfl=0.45,
        div_max=1.0,
        mass_max=1.0e-8,
        error_limits={"height_positivity": 0.0, "tracer_bounds": 0.0},
    )
    controller = STController(dt_min=1.0e-8, dt_max=args.frame_duration)
    artifact = compile_fortran_module_c_shell(
        emitted,
        _inputs(
            entry,
            state=state,
            targets=targets,
            controller=controller,
            frame_duration=args.frame_duration,
            dt_initial=args.dt_initial,
        ),
        output / "native",
        entrypoint=entry.name,
        extent_overrides=_extent_manifest(
            emitted,
            repository_module,
            grid_size=args.size,
            collection_capacity=args.max_iters,
        ),
        name="symbolic_fluid_live",
        standalone=True,
    )
    completed = artifact.run(
        frames=args.frames,
        stream_frames=args.stream_frames,
        capture_output=not args.stream_frames,
    )
    if completed.stdout is not None:
        print(completed.stdout, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
