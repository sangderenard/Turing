"""Prove gemm runs on the expected tiling with a commensurate gain.

The whole chain, each link the real machinery, none of it a demo-only
shim:

1. The KERNEL BANK builds and admits the parametric gemm and a
   size-specialized core, AUTO-PROFILING each at build time -- the launch
   and compute averages the deployment strategy is otherwise starved of
   (``bank.performance_chart``).
2. The DEPLOYMENT STRATEGY (``select_deployment_strategy``) chooses
   workers and CHUNK for a thread-workers pool from that evidence: stated
   cores, the task's lane count, barrier join -- the same call every
   bundle build makes, reasons attached.
3. The kernel's own SOURCE says how the data partitions
   (``KernelSpec.item_data``): one m-axis item owns its A rows and C rows
   and shares B whole -- derived from the authored index arithmetic, not
   declared by hand.
4. The HOST DEPLOYMENT POOL (``deploy_span``) executes the C-block lanes
   over that partition with the strategy's chunk. Each lane owns a
   PRIVATE prepared execution of the admitted core artifact (the native
   call releases the GIL across ctypes, so lanes overlap); ``workers=0``
   runs the identical code path serially -- the pool's own design -- so
   the serial baseline cannot drift from the parallel one.

MEASUREMENT INSTRUMENT, NOT A PRODUCT PATTERN. The host pool here stands
in for the native pool with the same frame semantics (turing_pool.c:
persistent workers, atomic chunk claiming, barrier join) so the
strategy's workers/chunk numbers can be validated today. A compiled
finished product does NOT acquire HostDeploymentPool -- or Python -- as
a dependency: the strategy's choices belong inside the emitted artifact,
lowered onto the native pool runtime, and this demo is the evidence that
the numbers the plan bakes in are worth baking.

Run:

    python tools/demo_gemm_tiled_deployment.py
    python tools/demo_gemm_tiled_deployment.py --size 384 --tile 64
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.compiler.deployment_host_pool import HostDeploymentPool
from src.compiler.deployment_lowering import select_deployment_strategy
from src.compiler.kernel_bank import open_blas_bank
from src.compiler.ssa_llvm_backend import prepare_artifact_execution


class TiledGemmLanes:
    """The C-block lanes of one tiled gemm, each with a private execution.

    A lane covers one (row-block, column-block) of C and runs its k-steps
    serially -- accumulation orders them -- while lanes are mutually
    independent (disjoint C blocks). This is the lane shape the deployment
    vocabulary already recognizes for barrier joins.
    """

    def __init__(self, core, size: int, tile: int):
        self.size = int(size)
        self.tile = int(tile)
        if self.size % self.tile:
            raise ValueError("demo sizes are exact multiples of the tile")
        self.blocks = self.size // self.tile
        entry = next(
            name for name in core.module.functions
            if name.endswith(f"__{core.spec.function_name}")
        )
        function = core.module.functions[entry]
        self._parameters = dict(function.metadata["parameter_names"])
        self._function = function
        self._core = core
        self._lane_state: list[dict] = []

    def bind(self, a2, b2, out, alpha: float, beta: float):
        """One private execution per lane, bound to its own tile buffers."""

        tile = self.tile
        self._lane_state = []
        for bi in range(self.blocks):
            for bj in range(self.blocks):
                a_buf = np.zeros(tile * tile)
                b_buf = np.zeros(tile * tile)
                c_buf = np.ascontiguousarray(
                    out[bi * tile:(bi + 1) * tile,
                        bj * tile:(bj + 1) * tile]
                ).reshape(-1).copy()
                feeds = {
                    self._parameters["A"]: a_buf,
                    self._parameters["B"]: b_buf,
                    self._parameters["C"]: c_buf,
                    self._parameters["alpha"]: np.array([alpha]),
                    self._parameters["beta"]: np.array([beta]),
                }
                for formal in self._function.args:
                    if int(formal.id) not in feeds:
                        feeds[int(formal.id)] = (
                            np.array([0]) if formal.dtype == "int"
                            else np.zeros(tile * tile)
                        )
                execution = prepare_artifact_execution(
                    self._core.native, feeds
                )
                self._lane_state.append({
                    "bi": bi, "bj": bj,
                    "a": a_buf, "b": b_buf,
                    "c": np.asarray(
                        execution.buffers[self._parameters["C"]]
                    ),
                    "beta": np.asarray(
                        execution.buffers[self._parameters["beta"]]
                    ),
                    "execution": execution,
                })
        self._a2, self._b2, self._out = a2, b2, out
        self._beta0 = float(beta)

    def run_lane(self, index: int) -> None:
        state = self._lane_state[index]
        tile, blocks = self.tile, self.blocks
        bi, bj = state["bi"], state["bj"]
        a_view = np.asarray(
            state["execution"].buffers[self._parameters["A"]]
        )
        b_view = np.asarray(
            state["execution"].buffers[self._parameters["B"]]
        )
        for bp in range(blocks):
            a_view[:] = np.ascontiguousarray(
                self._a2[bi * tile:(bi + 1) * tile,
                         bp * tile:(bp + 1) * tile]
            ).reshape(-1)
            b_view[:] = np.ascontiguousarray(
                self._b2[bp * tile:(bp + 1) * tile,
                         bj * tile:(bj + 1) * tile]
            ).reshape(-1)
            state["beta"][...] = self._beta0 if bp == 0 else 1.0
            state["execution"].run()
        self._out[bi * tile:(bi + 1) * tile,
                  bj * tile:(bj + 1) * tile] = (
            state["c"].reshape(tile, tile)
        )

    @property
    def lane_count(self) -> int:
        return len(self._lane_state)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--tile", type=int, default=64)
    parser.add_argument("--root", type=Path,
                        default=ROOT / "build" / "kernel_bank")
    args = parser.parse_args()
    size, tile = args.size, args.tile

    bank = open_blas_bank(args.root)
    print("== 1. bank: build + auto-profile ==")
    parametric = bank.get("gemm")
    core = bank.get(
        "gemm", specialized={"m": tile, "n": tile, "k": tile},
    )
    for row in bank.performance_chart("gemm")[-4:]:
        label = str(row["specialized"] or "parametric")
        print(f"  {label:<30} "
              f"first_launch={row['first_launch_seconds']*1e6:8.1f}us "
              f"relaunch={row['relaunch_avg_seconds']*1e6:8.1f}us "
              f"compute={row['compute_avg_seconds']*1e6:8.1f}us "
              f"sizes={row['sizes']}")

    blocks = size // tile
    lane_total = blocks * blocks
    print("\n== 2. deployment strategy chooses workers + chunk ==")
    choice = select_deployment_strategy(
        backend="llvm", execution_class="thread-workers",
        join_mode="barrier", work=lane_total,
        cores=os.cpu_count(),
    )
    for reason in choice.reasons:
        print("  -", reason)
    print(f"  => strategy={choice.strategy} workers={choice.workers} "
          f"chunk={choice.chunk}")

    print("\n== 3. the source says how data partitions per m-item ==")
    partition = core.spec.item_data(
        "m", {"m": size, "n": size, "k": size},
    )
    print(f"  split : {partition['split']} elements per item")
    print(f"  shared: {partition['shared']} elements, whole")

    print(f"\n== 4. execute {size}^3 as {lane_total} C-block lanes ==")
    rng = np.random.default_rng(11)
    a2 = rng.standard_normal((size, size))
    b2 = rng.standard_normal((size, size))
    c2 = rng.standard_normal((size, size))
    alpha, beta = 1.25, 0.5
    expected = alpha * (a2 @ b2) + beta * c2

    lanes = TiledGemmLanes(core, size, tile)

    def timed(workers: int, repeats: int = 3) -> tuple[float, np.ndarray]:
        pool = HostDeploymentPool(workers=workers)
        samples, executed, out = [], 0, None
        try:
            for _ in range(repeats):
                out = c2.copy()
                lanes.bind(a2, b2, out, alpha, beta)
                started = time.perf_counter()
                executed = pool.deploy_span(
                    lambda start, stop: [
                        lanes.run_lane(i) for i in range(start, stop)
                    ],
                    lanes.lane_count,
                    chunk=choice.chunk,
                )
                samples.append(time.perf_counter() - started)
        finally:
            pool.close()
        elapsed = float(np.median(samples))
        worst = float(np.max(np.abs(out - expected)))
        print(f"  workers={workers}: {elapsed*1000:8.2f} ms over "
              f"{executed} chunk(s), worst |err| {worst:.2e}")
        assert worst < 1e-9, "tiled result diverged from the oracle"
        return elapsed, out

    serial_seconds, _ = timed(0)
    workers = choice.workers or max(1, (os.cpu_count() or 2) - 1)
    pool_seconds, _ = timed(workers)

    # Steady-state medians for the reference rows, matching
    # docs/BLAS_VS_NUMPY_PROFILE.md's methodology -- a cold single shot
    # penalizes numpy's thread-pool spin-up and our first dispatch alike,
    # and a comparison against a handicapped reference is not a gain.
    def steady(operation, repeats: int = 5) -> float:
        operation()  # warm
        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            operation()
            samples.append(time.perf_counter() - started)
        return float(np.median(samples))

    parametric_seconds = steady(lambda: parametric.run({
        "A": a2.reshape(-1), "B": b2.reshape(-1), "C": c2.reshape(-1),
        "alpha": alpha, "beta": beta, "m": size, "n": size, "k": size,
    }))
    numpy_seconds = steady(lambda: alpha * (a2 @ b2) + beta * c2)

    flops = 2.0 * size ** 3
    print("\n== summary ==")
    for label, seconds in (
        ("parametric single call", parametric_seconds),
        ("tiled serial (workers=0)", serial_seconds),
        (f"tiled pool  (workers={workers})", pool_seconds),
        ("numpy", numpy_seconds),
    ):
        print(f"  {label:<26} {seconds*1000:8.2f} ms  "
              f"{flops/seconds/1e9:6.2f} GF/s")
    print(f"  pool vs serial : {serial_seconds/pool_seconds:5.2f}x")
    print(f"  pool vs single : {parametric_seconds/pool_seconds:5.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
