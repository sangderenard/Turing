"""Optimize the complete Mandelbrot recording precompile and audit/lower SSA."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ....compiler.precompile_to_ssa import (
    PrecompileSSALoweringResult,
    lower_precompile_and_control_to_ssa,
    ssa_module_dictionary,
)
from .demo_mandelbrot_fusion import (
    build_parametric_mandelbrot_glsl_deployment,
    normalized_plane,
)
from .gl_context import require_gl_context


@dataclass(frozen=True)
class MandelbrotSSAAudit:
    source_nodes: int
    scheduled_nodes: int
    dispatch_regions: int
    precompile_steps: int
    lowering: PrecompileSSALoweringResult


def mandelbrot_recording_feeds(
    width: int,
    height: int,
    iterations: int,
    frame_count: int,
) -> dict[str, Any]:
    """Build ordinary inputs for the existing complete recording entrypoint."""

    if width < 1 or height < 1:
        raise ValueError("width and height must be positive")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    unit_x, unit_y = normalized_plane(width, height)
    phase = np.linspace(
        0.0,
        1.0,
        frame_count,
        endpoint=False,
        dtype=np.float32,
    )
    return {
        "unit_x": unit_x,
        "unit_y": unit_y,
        "center_x": np.full(
            frame_count, -0.743643887, dtype=np.float32
        ),
        "center_y": np.full(
            frame_count, 0.131825904, dtype=np.float32
        ),
        "span": np.full(frame_count, 0.004, dtype=np.float32),
        "family_mix": np.zeros(frame_count, dtype=np.float32),
        "julia_x": np.full(frame_count, -0.72, dtype=np.float32),
        "julia_y": np.full(frame_count, 0.24, dtype=np.float32),
        "palette_phase": phase,
        "color_drive": np.full(
            frame_count, 0.52, dtype=np.float32
        ),
        "width": int(width),
        "height": int(height),
        "iterations": int(iterations),
    }


def _planned_shells(root: Any) -> tuple[Any, ...]:
    pending = [root, getattr(root, "module_shell", None)]
    seen: set[int] = set()
    shells = []
    while pending:
        shell = pending.pop()
        if shell is None or id(shell) in seen:
            continue
        seen.add(id(shell))
        shells.append(shell)
        pending.extend(
            getattr(shell, "function_shells", {}).values()
        )
        pending.extend(
            getattr(shell, "callsite_function_shells", {}).values()
        )
    return tuple(shells)


def _compilation_owner(deployment: Any) -> Any:
    candidates = [
        shell
        for shell in _planned_shells(deployment)
        if getattr(shell, "compiled_shell_program", None) is not None
        and getattr(shell, "shell_control_program", None) is not None
    ]
    if not candidates:
        raise RuntimeError(
            "optimized Mandelbrot deployment produced no numerical "
            "precompile/control pair"
        )
    return max(
        candidates,
        key=lambda shell: len(
            shell.compiled_shell_program.program.steps
        ),
    )


def audit_complete_mandelbrot_ssa(
    *,
    width: int = 16,
    height: int = 16,
    iterations: int = 8,
    frame_count: int = 1,
    verbose_profile: bool = False,
) -> MandelbrotSSAAudit:
    """Run the existing optimizer/precompile, then validate and lower SSA."""

    require_gl_context()
    deployment, graph = build_parametric_mandelbrot_glsl_deployment(
        iterations,
        profiling=verbose_profile,
        verbose_profile=verbose_profile,
        entrypoint="mandelbrot_recording_program",
    )
    try:
        feeds = mandelbrot_recording_feeds(
            width,
            height,
            iterations,
            frame_count,
        )
        deployment.compile_process_graph()
        deployment.capture_fused_programs(feeds)
        owner = _compilation_owner(deployment)
        precompile = owner.compiled_shell_program
        control = owner.shell_control_program
        lowering = lower_precompile_and_control_to_ssa(
            precompile,
            control,
            numerical_name="mandelbrot_recording_numerical",
            control_name="mandelbrot_recording_control",
            region_programs={
                int(region_index): captured
                for region_index, captured
                in getattr(
                    owner,
                    "captured_region_programs",
                    {},
                ).items()
                if isinstance(region_index, int)
            },
        )
        return MandelbrotSSAAudit(
            source_nodes=int(owner.source_node_count),
            scheduled_nodes=int(owner.primitive_count),
            dispatch_regions=int(owner.dispatch_count),
            precompile_steps=len(precompile.program.steps),
            lowering=lowering,
        )
    finally:
        deployment.release()


def _print_audit(audit: MandelbrotSSAAudit) -> None:
    result = audit.lowering
    print(
        "optimized Mandelbrot recording: "
        f"{audit.source_nodes} source nodes -> "
        f"{audit.scheduled_nodes} scheduled nodes -> "
        f"{audit.dispatch_regions} regions"
    )
    print(f"precompile steps: {audit.precompile_steps}")
    print(
        "precompile format: "
        + ("valid" if result.validation.valid_precompile else "invalid")
    )
    print(
        "SSA compatibility: "
        + ("complete" if result.validation.ssa_compatible else "shortfalls")
    )
    print(result.shortfall_report())
    if result.cycles:
        print("SSA cycles:")
        for cycle in result.cycles:
            print(
                f"- {cycle.function}: blocks={cycle.blocks}; "
                f"edges={cycle.back_edges}; "
                f"phi_blocks={cycle.phi_blocks}"
            )
    else:
        print("SSA cycles: none")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--profile-verbose", action="store_true")
    parser.add_argument(
        "--ssa-json",
        type=Path,
        help="write the partially or completely lowered SSA dictionary",
    )
    args = parser.parse_args(argv)
    audit = audit_complete_mandelbrot_ssa(
        width=args.width,
        height=args.height,
        iterations=args.iterations,
        frame_count=args.frames,
        verbose_profile=args.profile_verbose,
    )
    _print_audit(audit)
    if args.ssa_json is not None:
        args.ssa_json.write_text(
            json.dumps(
                ssa_module_dictionary(audit.lowering.module),
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"SSA dictionary: {args.ssa_json}")
    return 0 if audit.lowering.complete else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MandelbrotSSAAudit",
    "audit_complete_mandelbrot_ssa",
    "main",
    "mandelbrot_recording_feeds",
]
