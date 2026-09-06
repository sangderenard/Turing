"""Publish the compiled symbolic-fluid repository SSA to the root gallery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.compiler.site_bundle import (
    DEFAULT_PUBLISH_ROOT,
    publish_prebuilt_program_bundle,
)
from src.compiler.ssa_javascript_html import (
    emit_repository_ssa_javascript_page,
)
from src.compiler.symbolic_fluid_dt import SYMBOLIC_FLUID_DT_SOURCE


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("javascript", type=Path)
    parser.add_argument("--compile", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=DEFAULT_PUBLISH_ROOT)
    args = parser.parse_args()
    side = 8
    page = {
        "title": "Compiled managed-time shallow fluid",
        "grid": {"width": side, "height": side},
        "renderFps": 8,
        "scratchCapacity": max(4096, side * side * 4),
        "ownerAliases": {"ctrl": "controller", "self": "controller"},
        "inputs": {"1": 1.0e-3, "2": 1.0 / 60.0},
        "fields": {
            "state.height": {
                "kind": "grid", "fill": 1.0,
                "bumps": [{"x": 0.5, "y": 0.5, "radius": 2.2, "amplitude": 0.18}],
            },
            "state.momentum_x": {"kind": "grid", "fill": 0.0},
            "state.momentum_y": {"kind": "grid", "fill": 0.0},
            "state.tracer": {
                "kind": "grid", "fill": 0.0,
                "bumps": [{"x": 0.34, "y": 0.5, "radius": 1.8, "amplitude": 1.0}],
            },
            "state.next_height": {"kind": "grid", "fill": 0.0},
            "state.next_momentum_x": {"kind": "grid", "fill": 0.0},
            "state.next_momentum_y": {"kind": "grid", "fill": 0.0},
            "state.next_tracer": {"kind": "grid", "fill": 0.0},
            "state.dx": 1.0 / side,
            "state.gravity": 1.0,
            "state.viscosity": 2.0e-4,
            "state.tracer_diffusivity": 1.0e-4,
            "state.linear_drag": 2.0e-3,
            "state.coriolis": 0.08,
            "state.minimum_height": 1.0e-4,
            "state.height_count": side,
            "state.width_count": side,
            "state.last_wave_speed": 0.0,
            "state.last_height_violation": 0.0,
            "state.last_tracer_violation": 0.0,
            "controller.Kp": 0.4,
            "controller.Ki": 0.05,
            "controller.A": 1.5,
            "controller.shrink": 0.5,
            "controller.dt_min": 1.0e-8,
            "controller.dt_max": 1.0 / 30.0,
            "controller.acc": 0.0,
            "controller.max_vel_ever": 1.0e-30,
            "targets.cfl": 0.45,
            "targets.div_max": 1.0,
            "targets.mass_max": 1.0e-8,
            "targets.error_limits.length": 0,
            "targets.error_limits.keys": [],
            "targets.error_limits.values": [],
        },
        "display": [
            {"label": "height", "field": "state.height"},
            {"label": "momentum x", "field": "state.momentum_x"},
            {"label": "momentum y", "field": "state.momentum_y"},
            {"label": "tracer", "field": "state.tracer"},
        ],
        "feedback": {"inputValueId": 1, "outputIndex": 4},
        "diagnosticOutputIndex": 5,
        "interaction": {"field": "state.height", "amplitude": 0.012, "radius": 1.6},
    }
    html, host, page_json = emit_repository_ssa_javascript_page(page)
    compile_root = args.compile.resolve()
    javascript = args.javascript.resolve()
    artifacts = {
        "program/program.js": (javascript / "program.mjs").read_bytes(),
        "program/program.json": (javascript / "program.json").read_bytes(),
        "program/page-config.json": page_json,
        "runtime/host.js": host,
        "build/control_summary.json": (compile_root / "control_summary.json").read_bytes(),
        "build/control_repository_ssa.contract.json": (
            compile_root / "control_repository_ssa.contract.json"
        ).read_bytes(),
    }
    bundle = publish_prebuilt_program_bundle(
        destination=args.destination,
        slug="compiled-managed-time-shallow-fluid",
        title=str(page["title"]),
        entrypoint="symbolic_fluid_control__symbolic_fluid_frame",
        html=html,
        source_filename="symbolic_fluid_dt_program.py",
        source=SYMBOLIC_FLUID_DT_SOURCE,
        artifacts=artifacts,
        runtime={
            "backend": "repository-ssa-javascript",
            "manifest": "program/program.json",
            "host": "generic-manifest-grid-v1",
            "compiled_function_count": 55,
        },
        refresh_gallery=True,
    )
    print(json.dumps({
        "directory": str(bundle.directory),
        "url": bundle.url,
        "version": bundle.manifest["version"]["id"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
