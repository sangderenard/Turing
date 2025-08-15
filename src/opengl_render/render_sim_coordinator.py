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


OPTIONS: Mapping[str, tuple[str, Sequence[str]]] = {
    "1": ("Voxel fluid demo", ["--fluid", "voxel"]),
    "2": ("Discrete fluid demo", ["--fluid", "discrete"]),
    "3": ("Hybrid fluid demo", ["--fluid", "hybrid"]),
    "4": ("Cells + fluid (mesh)", ["--couple-fluid", "voxel"]),
}


def run_option(choice: str, *, debug: bool = False) -> subprocess.CompletedProcess:
    """Run a predefined NumPy simulation option.

    Parameters
    ----------
    choice:
        Key from :data:`OPTIONS` selecting which demo to run.
    debug:
        When ``True`` the underlying simulator is invoked with ``--debug``
        flags and a single frame to emit layer positions and metadata.

    Returns
    -------
    :class:`subprocess.CompletedProcess`
        The completed process object from :func:`subprocess.run`.
    """

    if choice not in OPTIONS:
        raise ValueError("Unknown option")
    _, args = OPTIONS[choice]
    cmd = [sys.executable, "-m", "src.cells.softbody.demo.numpy_sim_coordinator", *args]
    if debug:
        cmd += ["--debug-render", "--debug", "--frames", "1", "--dt", "1e-4"]
        return subprocess.run(cmd, check=False, capture_output=True, text=True)
    return subprocess.run(cmd, check=False)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--choice", choices=list(OPTIONS.keys()), help="Run a specific option without prompting")
    parser.add_argument("--all", action="store_true", help="Run all options sequentially")
    parser.add_argument("--debug", action="store_true", help="Enable debug rendering and logging")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.all:
        for key in OPTIONS:
            proc = run_option(key, debug=args.debug)
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
    proc = run_option(choice, debug=args.debug)
    if args.debug:
        print(proc.stdout)


if __name__ == "__main__":  # pragma: no cover - manual utility
    main()
