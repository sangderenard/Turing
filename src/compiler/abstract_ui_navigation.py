"""Hot-swappable assembly navigation kernels for AbstractUI entities.

The navigation host owns geometry rasterization and pose presentation.  Route
search is an interchangeable WebAssembly kernel behind a small pointer ABI, so
an entity can change algorithms without changing its controller or body.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Iterable


MAX_GRID_CELLS = 128 * 128


ASTAR_C_SOURCE = r"""
#define MAX_CELLS 16384
#define INF 0x3fffffff

static int g_score[MAX_CELLS];
static int came_from[MAX_CELLS];
static unsigned char open_set[MAX_CELLS];
static unsigned char closed_set[MAX_CELLS];

static int abs_i(int value) { return value < 0 ? -value : value; }

static int heuristic(int cell, int goal, int width) {
    int x0 = cell % width, z0 = cell / width;
    int x1 = goal % width, z1 = goal / width;
    int dx = abs_i(x1 - x0), dz = abs_i(z1 - z0);
    int diagonal = dx < dz ? dx : dz;
    return diagonal * 14 + (dx + dz - 2 * diagonal) * 10;
}

__attribute__((export_name("navigation_pathfind")))
int navigation_pathfind(const int *blocked, int width, int height,
                        int start, int goal, int *path, int path_capacity) {
    int count = width * height;
    if (width <= 0 || height <= 0 || count > MAX_CELLS) return -1;
    if (start < 0 || goal < 0 || start >= count || goal >= count) return -1;
    if (blocked[start] || blocked[goal]) return -2;
    for (int i = 0; i < count; ++i) {
        g_score[i] = INF; came_from[i] = -1;
        open_set[i] = 0; closed_set[i] = 0;
    }
    g_score[start] = 0; open_set[start] = 1;
    while (1) {
        int current = -1, best = INF;
        for (int i = 0; i < count; ++i) {
            if (open_set[i] && !closed_set[i]) {
                int score = g_score[i] + heuristic(i, goal, width);
                if (score < best) { best = score; current = i; }
            }
        }
        if (current < 0) return 0;
        if (current == goal) break;
        open_set[current] = 0; closed_set[current] = 1;
        int cx = current % width, cz = current / width;
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dz == 0) continue;
                int nx = cx + dx, nz = cz + dz;
                if (nx < 0 || nz < 0 || nx >= width || nz >= height) continue;
                int neighbor = nz * width + nx;
                if (blocked[neighbor] || closed_set[neighbor]) continue;
                if (dx && dz && (blocked[cz * width + nx] || blocked[nz * width + cx])) continue;
                int candidate = g_score[current] + (dx && dz ? 14 : 10);
                if (!open_set[neighbor] || candidate < g_score[neighbor]) {
                    came_from[neighbor] = current;
                    g_score[neighbor] = candidate;
                    open_set[neighbor] = 1;
                }
            }
        }
    }
    int length = 0;
    for (int cell = goal; cell >= 0; cell = came_from[cell]) {
        if (length >= path_capacity) return -3;
        path[length++] = cell;
        if (cell == start) break;
    }
    for (int left = 0, right = length - 1; left < right; ++left, --right) {
        int temporary = path[left]; path[left] = path[right]; path[right] = temporary;
    }
    return length;
}
"""


@dataclass(frozen=True, slots=True)
class NavigationAssemblyKernel:
    identity: str
    label: str
    binary: bytes
    entrypoint: str = "navigation_pathfind"

    def to_data(self) -> dict[str, Any]:
        return {
            "schema": "abstract-ui-navigation-assembly-v0",
            "identity": self.identity,
            "label": self.label,
            "format": "webassembly",
            "binary_base64": base64.b64encode(self.binary).decode("ascii"),
            "binary_bytes": len(self.binary),
            "entrypoint": self.entrypoint,
            "abi": {
                "arguments": [
                    "blocked:i32*", "width:i32", "height:i32", "start:i32",
                    "goal:i32", "path:i32*", "path_capacity:i32",
                ],
                "result": "path_length:i32",
                "path_order": "start-to-goal",
                "maximum_grid_cells": MAX_GRID_CELLS,
                "ownership": "caller-owned-linear-memory",
            },
            "algorithm": {
                "family": "a-star",
                "connectivity": 8,
                "heuristic": "octile",
                "diagonal_corner_cutting": False,
            },
        }


@lru_cache(maxsize=1)
def compile_astar_navigation_kernel() -> NavigationAssemblyKernel:
    """Compile the complete grid search to freestanding WebAssembly."""

    with tempfile.TemporaryDirectory(prefix="abstract-ui-navigation-") as temporary:
        directory = Path(temporary)
        source = directory / "navigation_astar.c"
        output = directory / "navigation_astar.wasm"
        source.write_text(ASTAR_C_SOURCE, encoding="utf-8")
        command = [
            sys.executable, "-m", "ziglang", "cc", "--target=wasm32-freestanding",
            "-nostdlib", "-O3", "-Wl,--no-entry",
            "-Wl,--export=navigation_pathfind", "-Wl,--export=__heap_base",
            "-Wl,--export-memory", "-Wl,--initial-memory=2097152",
            "-o", str(output), str(source),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0 or not output.is_file():
            raise RuntimeError("navigation WASM compilation failed: " + completed.stderr[-2000:])
        binary = output.read_bytes()
    return NavigationAssemblyKernel(
        "navigation-kernel:astar-octile-v0", "A* · octile / 8-way", binary,
    )


def navigation_model(entity_identities: Iterable[str]) -> dict[str, Any]:
    """Publish the kernel registry and mutable per-entity assignment table."""

    kernel = compile_astar_navigation_kernel()
    identities = tuple(entity_identities)
    return {
        "schema": "abstract-ui-navigation-v0",
        "identity": "navigation:living-data-map",
        "kernel_abi": "abstract-ui-navigation-assembly-v0",
        "default_kernel": kernel.identity,
        "kernels": [kernel.to_data()],
        "assignments": {identity: kernel.identity for identity in identities},
        "execution": {
            "planning_host": "dedicated-worker",
            "graphics_host": "animation-frame-main-thread",
            "handoff": "structured-clone-certified-route-samples",
            "main_thread_work": "route-installation-and-interpolation-only",
        },
        "grid": {
            "width": 64, "height": 64, "clearance": 0.14,
            "domain_geometry_kinds": ["courtyard", "building", "room"],
        },
        "traversal_chart": {
            "source": "linear-topology-grid",
            "projection": "piecewise-linear-to-nonlinear-hierarchy-space",
            "axis_interval": "identity-bounds-and-opening-landmarks",
            "gap_transform": "d<=2 ? d : 2+log1p(d-2)",
            "inverse": "piecewise-linear",
        },
        "traversal": {
            "speed": 5.2,
            "position_spline": "obstacle-validated-catmull-rom",
            "orientation_spline": "shortest-arc-quaternion-slerp",
            "collision_validation": "continuous-swept-clearance",
            "manual_input": "interrupt",
        },
        "waypoints": {
            "click_policy": "append-per-entity",
            "presence_pause_seconds": 0.85,
            "presence_event": "abstract-ui:navigation-presence",
            "presence_hooks": "minimum-pause-and-registered-async-holds",
            "planning": "sequential-from-certified-arrival-pose",
        },
        "hot_swap": {
            "scope": "per-entity",
            "operation": "assign-navigation-kernel",
            "preserves": ["entity-identity", "controller", "pose", "archetype"],
        },
    }


__all__ = [
    "ASTAR_C_SOURCE", "MAX_GRID_CELLS", "NavigationAssemblyKernel",
    "compile_astar_navigation_kernel", "navigation_model",
]
