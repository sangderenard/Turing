"""Interactive launcher for the numpy-based softbody demo.

The menu delegates to :mod:`src.cells.softbody.demo.run_numpy_demo` with
predefined argument sets for common scenarios (voxel fluid, discrete fluid,
mesh‑coupled cellsim and a point cloud export).  It is a thin wrapper around
``subprocess.run`` so the existing CLI of the numpy demo remains the source of
truth.
"""
from __future__ import annotations

import subprocess


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
    cmd = ["python", "-m", "src.cells.softbody.demo.run_numpy_demo", *args]
    subprocess.run(cmd, check=False)


if __name__ == "__main__":  # pragma: no cover - manual utility
    main()
