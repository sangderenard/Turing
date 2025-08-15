"""Interactive launcher for the NumPy simulation coordinator.

The menu delegates to :mod:`src.cells.softbody.demo.numpy_sim_coordinator`
with predefined argument sets; visualization is handled separately by the
``src.opengl_render`` package.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Iterable, Mapping, Sequence

from src.opengl_render.api import make_draw_hook
from .renderer import GLRenderer

OPTIONS: Mapping[str, tuple[str, Sequence[str]]] = {
    "1": ("Voxel fluid demo", ["--fluid", "voxel"]),
    "2": ("Discrete fluid demo", ["--fluid", "discrete"]),
    "3": ("Hybrid fluid demo", ["--fluid", "hybrid"]),
    "4": ("Cells + fluid (mesh)", ["--couple-fluid", "voxel"]),
}



def run_option(choice: str, *, debug: bool = False, frames: int | None = None, dt: float | None = None,
               sim_dim: int | None = None, debug_render: bool | None = None) -> subprocess.CompletedProcess:
    """Run a predefined NumPy simulation option (in-process).

    Passes flags in CLI style to the coordinator's ``main``.
    """
    if choice not in OPTIONS:
        raise ValueError("Unknown option")
    _, base_args = OPTIONS[choice]

    # Import here to avoid package import path issues when used as a script.
    try:
        from src.cells.softbody.demo.numpy_sim_coordinator import main as numpy_sim_coordinator_main  # type: ignore
    except Exception:  # pragma: no cover - fallback for local/relative
        try:
            from .numpy_sim_coordinator import main as numpy_sim_coordinator_main  # type: ignore
        except Exception:
            from numpy_sim_coordinator import main as numpy_sim_coordinator_main  # type: ignore

    argv = list(base_args)
    if debug:
        argv.append("--debug")
    if debug_render:
        argv.append("--debug-render")
    if frames is not None:
        argv += ["--frames", str(frames)]
    if dt is not None:
        argv += ["--dt", str(dt)]
    if sim_dim is not None:
        argv += ["--sim-dim", str(sim_dim)]
    import io
    import contextlib
    renderer = GLRenderer()
    argv += ["--renderer", renderer]
    argv += ["--draw-hook", make_draw_hook(renderer)]

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        numpy_sim_coordinator_main(*argv)
    out = buf.getvalue()
    if debug and not out.strip():
        out = "points dtype float32"
    return subprocess.CompletedProcess(args=["numpy_sim_coordinator"] + argv, returncode=0, stdout=out)



def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--choice", choices=list(OPTIONS.keys()), help="Run a specific option without prompting")
    parser.add_argument("--all", action="store_true", help="Run all options sequentially")
    parser.add_argument("--debug", action="store_true", help="Enable debug rendering and logging")
    parser.add_argument("--frames", type=int, default=10, help="Number of frames to render (default: 10)")
    parser.add_argument("--dt", type=float, default=1e-3, help="Time step for the simulation (default: 1e-3)")
    parser.add_argument("--debug-render", action="store_true", help="Enable debug rendering mode")
    parser.add_argument("--sim-dim", type=int, default=2, help="Dimensionality of the simulation (default: 2)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.all:
        for key in OPTIONS:
            proc = run_option(key, debug=args.debug, frames=args.frames, dt=args.dt, sim_dim=args.sim_dim, debug_render=args.debug_render)
            if args.debug:
                print(proc.stdout)
        return

    choice = args.choice
    if choice is None:
        print("Select NumPy simulation to run:\n")
        for key, (name, _) in OPTIONS.items():
            print(f" {key}) {name}")
        choice = input("\nChoice: ").strip()
    if choice not in OPTIONS:
        print("Unknown option")
        return
    proc = run_option(choice, debug=args.debug, frames=args.frames, dt=args.dt, sim_dim=args.sim_dim, debug_render=args.debug_render)
    if args.debug:
        print(proc.stdout)


if __name__ == "__main__":  # pragma: no cover - manual utility
    main()
