"""Measure the compiled fluid step's cost per cell, separating native from marshalling.

Why this exists: "the sim is slow" is not a measurement, and the two candidate
causes -- the native kernel versus the Python-side array feeding across the ABI
-- have completely different fixes. This times them apart, at several grid
sizes, so a flat ns/cell says "the kernel is the cost" and a curved one says
"fixed overhead per call".

Run:  PYTHONPATH=. python tools/bench_native_step.py [build-directory]

Baseline recorded 2026-08-19 (before any identity pass), zig cc -shared -O2,
no target triple, no noalias, no fast-math:

    24^2       576       0.970 ms   marshal   0.075 ms   1684 ns/cell
    48^2      2304       3.852 ms   marshal  ~0          1672 ns/cell
    96^2      9216      15.486 ms   marshal  ~0          1680 ns/cell
   192^2     36864      62.138 ms   marshal  ~0          1686 ns/cell
   256^2     65536     112.044 ms   marshal  ~0          1710 ns/cell

Flat ns/cell with zero marshalling: the cost is entirely inside the kernel.
The per-cell kernel is 290 instructions -- 140 Mul, 91 Add, 24 Pow, 16 Max,
10 Abs, 9 Const -- and every Pow exponent is a compile-time constant
(10x 2, 6x -1, 6x 0.5, 2x -2) lowered to @llvm.pow.f64. That is the number.

2026-08-19, after ir_identities.reduce_constant_exponent_pow:

    exact set only (default: x**2 -> x*x, x**-1 -> 1/x, 16 of 24 Pow):
        ~900 ns/cell across all grids            -- 1.85x
    TURING_POW_INEXACT=1 (adds x**0.5 -> sqrt, x**-2 -> 1/(x*x)):
        ~480 ns/cell across all grids            -- 3.5x
        scorecard still 10/10 at 0.0e+00; fluid mass_err <= 1e-15 held

The remaining ~480 ns is 266 scalar ops with no vectorization: no target
triple, no noalias, no FMA (see catalogue sections 2.5 and P2).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.compiler.symbolic_fluid_dt import SymbolicFluidGridState
from src.compiler.symbolic_fluid_native_runtime import (
    load_symbolic_fluid_managed_functions,
)

GRIDS = (24, 48, 96, 192, 256)


def main() -> int:
    build_directory = sys.argv[1] if len(sys.argv) > 1 else "build/bench-native-step"
    started = time.perf_counter()
    advance = load_symbolic_fluid_managed_functions(build_directory)[
        "symbolic_fluid_advance"
    ]
    print(f"load/compile: {time.perf_counter() - started:.2f}s")
    print(f"{'grid':>8} {'cells':>9} {'run() ms':>11} {'marshal ms':>12} {'ns/cell':>10}")
    for extent in GRIDS:
        # One artifact, one binding per grid: `prepare_artifact_execution`
        # sizes the public buffers from the feeds, so a new extent needs a new
        # binding. Clearing `execution` is how the runtime is asked for one --
        # reloading the module instead would try to rewrite a DLL this process
        # already has open.
        advance.execution = None
        state = SymbolicFluidGridState.initial(extent, extent)
        advance(state, 1.0e-4)
        repetitions = 10 if extent >= 192 else 30
        clock = time.perf_counter()
        for _ in range(repetitions):
            advance(state, 1.0e-4)
        whole = (time.perf_counter() - clock) / repetitions
        clock = time.perf_counter()
        for _ in range(repetitions):
            advance.execution.run()
        native = (time.perf_counter() - clock) / repetitions
        cells = extent * extent
        print(
            f"{extent:6d}^2 {cells:9d} {native * 1e3:11.3f} "
            f"{(whole - native) * 1e3:12.3f} {native / cells * 1e9:10.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
