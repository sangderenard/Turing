"""Interactive launcher for the NumPy simulation coordinator.

The menu delegates to :mod:`src.cells.softbody.demo.numpy_sim_coordinator`
with predefined argument sets; visualization is handled separately by the
``src.opengl_render`` package.
"""
from __future__ import annotations

import subprocess
import sys


OPTIONS = {
    "1": ("Voxel fluid demo", ["--fluid", "voxel"]),
    "2": ("Discrete fluid demo", ["--fluid", "discrete"]),
    "3": ("Hybrid fluid demo", ["--fluid", "hybrid"]),
    "4": ("Cells + fluid (mesh)", ["--couple-fluid", "voxel"]),
}


def main() -> None:
    print("Select NumPy simulation to run:\n")
    for key, (name, _) in OPTIONS.items():
        print(f" {key}) {name}")
    choice = input("\nChoice: ").strip()
    if choice not in OPTIONS:
        print("Unknown option")
        return
    name, args = OPTIONS[choice]
    # Run the NumPy simulation coordinator; rendering is expected to be handled externally
    cmd = [sys.executable, "-m", "src.cells.softbody.demo.numpy_sim_coordinator", *args]
    subprocess.run(cmd, check=False)


if __name__ == "__main__":  # pragma: no cover - manual utility
    main()
