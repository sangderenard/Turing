"""Interactive launcher for numpy softbody demo and OpenGL renderer wiring.

The menu delegates to :mod:`src.cells.softbody.demo.run_numpy_demo` with
predefined argument sets and now includes an option to stream live geometry
to the minimal GLRenderer (``--stream opengl-renderer``).
"""
from __future__ import annotations

import subprocess
import sys


OPTIONS = {
    "1": ("Voxel fluid demo", ["--fluid", "voxel"]),
    "2": ("Discrete fluid demo", ["--fluid", "discrete"]),
    "3": ("Hybrid fluid demo", ["--fluid", "hybrid"]),
    "4": ("Cells + fluid (mesh)", ["--couple-fluid", "voxel"]),
    "5": ("Point cloud export", ["--export-kind", "opengl-points"]),
}


def main() -> None:
    print("Select numpy demo to run:\n")
    for key, (name, _) in OPTIONS.items():
        print(f" {key}) {name}")
    choice = input("\nChoice: ").strip()
    if choice not in OPTIONS:
        print("Unknown option")
        return
    name, args = OPTIONS[choice]
    # Force all runs to use the GLRenderer streaming path unless exporting only
    if "--export-kind" not in args:
        args = [*args, "--stream", "opengl-renderer"]
    cmd = [sys.executable, "-m", "src.cells.softbody.demo.run_numpy_demo", *args]
    subprocess.run(cmd, check=False)


if __name__ == "__main__":  # pragma: no cover - manual utility
    main()
