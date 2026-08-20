"""Strategic tiling as a DEPLOYMENT decision, made from evidence.

This module is the deployment layer's answer to one question: given a call
with concrete sizes, should it be served by a TILED COMPOSITION of prebaked
cores -- and with which tile, under how many workers? It deliberately
mirrors ``deployment_classification.py``'s contract: every choice AND every
veto carries a reason string, and the full candidate set survives alongside
the decision so a dispatcher with different constraints can fall back
without re-deriving anything. Classification answers "what KIND of site
serves a scheduled region"; this answers "should this call become many
prebaked-core calls" -- the two compose: a tiled plan's lanes are exactly
the independent-lane barrier shape classification already recognizes for
``thread-workers``.

Evidence consulted, in order:

* **The bank's admitted variants** (existence + admission are the only
  proof a core runs correctly at a size; a refused or stale variant is
  never a candidate).
* **Profiling numbers** from admission manifests. Steady-state compute time
  is projected through the candidate's number of k-steps, independent output
  lanes, and available worker slots. This ranks the composed critical path,
  rather than blindly selecting the fastest isolated square core and then
  discovering that it produced too few lanes to occupy the machine.
* **The task's shape**: a core larger than the task on any axis cannot
  tile it; a task not meaningfully larger than the best core gains nothing
  from composition overhead.
* **Cores available and NESTED deployment depth**: a tiled plan inside an
  already-parallel deployment must not multiply worlds. The worker budget
  is tempered by nesting (``cores // (1 + nested_parallelism)``). Without
  at least two workers, composition is vetoed unless a future calibration
  supplies positive serial-tiling evidence.

The plan object (:class:`TiledDeploymentPlan`) is shaped to mirror
``ControlDeploymentRegion``: ``kind``/``schedule``/``lanes``/barrier join,
``origin="kernel_bank"``. Each lane is one C-block -- lanes are mutually
independent (disjoint output tiles), which is the same independence proof
``_barrier_lane_memberships`` demands; the k-steps WITHIN a lane are
ordered (each accumulates into the block), which is why a lane is a
sequence of steps and not itself split. Executing a lane on a worker is
therefore safe by construction; executing steps of one lane concurrently
is not, and the plan's shape makes that distinction unrepresentable.

There is deliberately no executor in this decision module. The product
consumer is ``native_gemm_product``: it compiles the plan's packing, admitted
core calls and worker budget into one native artifact over ``turing_pool.c``.
The host-pool path in ``tools/demo_gemm_tiled_deployment.py`` remains a
measurement instrument and comparison baseline, not a product dependency.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping

TILE_COMPOSITION_KIND = "tile_composition"


@dataclass(frozen=True)
class TileStep:
    """One k-step of one lane: a single prebaked-kernel call."""

    p0: int
    kp: int
    beta: float          # original beta on the first step, 1.0 after
    uses_core: bool      # exact-tile core, else the parametric edge kernel


@dataclass(frozen=True)
class TileLane:
    """One C-block: independent of every other lane (disjoint output)."""

    index: int
    i0: int
    j0: int
    mi: int
    nj: int
    steps: tuple[TileStep, ...]


@dataclass(frozen=True)
class TiledDeploymentPlan:
    """A tiled composition, in the deployment vocabulary.

    Mirrors ``ControlDeploymentRegion``'s shape (kind / schedule / lanes /
    barrier join) so downstream deployment machinery reads it the way it
    reads any independent-lane region. ``worker_budget`` is the tempered
    concurrency ceiling the executor must respect; 1 means serial.
    """

    kind: str
    schedule: str
    origin: str
    tile: int
    m: int
    n: int
    k: int
    lanes: tuple[TileLane, ...]
    join_mode: str
    worker_budget: int
    nested_parallelism: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class TilingDecision:
    """The decision and its complete evidence, veto reasons included."""

    tiled: bool
    tile: int | None
    worker_budget: int
    candidates: tuple[tuple[int, float | None], ...]  # (size, probe_seconds)
    reasons: tuple[str, ...] = field(default=())


def _admitted_square_core_sizes(
    bank: Any, name: str, contract: str | None
) -> list[tuple[int, float | None]]:
    """Admitted, CURRENT (non-stale) square specialized sizes with probe
    timings. Staleness is decided the only honest way: by asking the bank
    (``compile_missing=False``), never by trusting a manifest row whose
    compiler fingerprint may no longer match."""

    from .kernel_bank import BankRefusal

    seen: dict[int, float | None] = {}
    for row in bank.inventory():
        if row.get("kernel") != name:
            continue
        specialized = row.get("specialized") or {}
        sizes = {int(v) for v in specialized.values()}
        if len(specialized) < 2 or len(sizes) != 1:
            continue
        verification = row.get("verification") or {}
        if not verification.get("admitted"):
            continue
        size = sizes.pop()
        # The build-time profile's steady-state compute average is the
        # honest throughput evidence; the admission probe's cold single
        # call (launch conflated with compute) is only the fallback for
        # manifests written before profiling existed.
        profile = row.get("profile") or {}
        probe = (
            profile.get("compute_avg_seconds")
            or verification.get("probe_call_seconds")
        )
        if size not in seen or (
            probe is not None
            and (seen[size] is None or probe < seen[size])
        ):
            seen[size] = probe
    live: list[tuple[int, float | None]] = []
    for size, probe in sorted(seen.items()):
        try:
            bank.get(
                name, contract=contract,
                specialized={p: size for p in bank.specs[name].size_parameters},
                compile_missing=False,
            )
        except BankRefusal:
            continue
        live.append((size, probe))
    return live


def decide_tiling(
    bank: Any,
    name: str,
    sizes: Mapping[str, int],
    *,
    contract: str | None = None,
    cores: int | None = None,
    nested_parallelism: int = 0,
    must_divide: bool = False,
) -> TilingDecision:
    """Should this call be tiled, with which core, under how many workers?

    Evidence-based and honest about vetoes; see the module docstring for
    the sources. Only ``gemm``-shaped calls (m, n, k all present) are
    currently decidable -- everything else returns an explained refusal.
    """

    reasons: list[str] = []
    cores = int(cores) if cores else (os.cpu_count() or 1)
    execution_slots = max(
        1, cores // (1 + max(0, int(nested_parallelism)))
    )
    # Both CPU pools enlist the caller. Plans state parked background
    # workers; projected critical paths use all active execution slots.
    worker_budget = max(0, execution_slots - 1)
    if nested_parallelism > 0:
        reasons.append(
            f"nested inside {nested_parallelism} parallel deployment "
            f"level(s): execution slots tempered to {execution_slots} of "
            f"{cores} cores ({worker_budget} background worker(s) plus caller)"
        )

    if not all(axis in sizes for axis in ("m", "n", "k")):
        reasons.append(
            "not a gemm-shaped call (m, n, k not all present); no tiling "
            "strategy is defined for this shape yet"
        )
        return TilingDecision(False, None, worker_budget, (), tuple(reasons))

    m, n, k = (int(sizes[axis]) for axis in ("m", "n", "k"))
    candidates = _admitted_square_core_sizes(bank, name, contract)
    if not candidates:
        reasons.append(
            "no admitted, current square specialized core exists in the "
            "bank; nothing prebaked to compose with"
        )
        return TilingDecision(False, None, worker_budget, (), tuple(reasons))

    fitting = [
        (size, probe) for size, probe in candidates
        if size <= min(m, n, k)
        and (
            not must_divide
            or (m % size == 0 and n % size == 0 and k % size == 0)
        )
    ]
    if must_divide:
        reasons.append(
            "must_divide: only cores dividing every axis are candidates "
            "(the executor at hand has no parametric edge path)"
        )
    if not fitting:
        within_axes = [
            size for size, _probe in candidates
            if size <= min(m, n, k)
        ]
        if must_divide and within_axes:
            reasons.append(
                f"admitted cores {within_axes} fit within the task but none "
                f"divide every axis of {m}x{n}x{k}"
            )
        else:
            reasons.append(
                f"every admitted core ({[s for s, _ in candidates]}) "
                f"exceeds the task's smallest axis ({min(m, n, k)})"
            )
        return TilingDecision(
            False, None, worker_budget, tuple(candidates), tuple(reasons)
        )

    def composed_seconds(entry: tuple[int, float | None]) -> float:
        size, probe = entry
        if probe is None or probe <= 0:
            return float("inf")
        lanes = ((m + size - 1) // size) * ((n + size - 1) // size)
        steps_per_lane = (k + size - 1) // size
        active_slots = max(1, min(lanes, worker_budget + 1))
        lane_waves = (lanes + active_slots - 1) // active_slots
        return float(probe) * steps_per_lane * lane_waves

    measured = [entry for entry in fitting if composed_seconds(entry) < float("inf")]
    best_size, best_probe = (
        min(measured, key=lambda entry: (composed_seconds(entry), -entry[0]))
        if measured else max(fitting, key=lambda entry: entry[0])
    )
    estimates = ", ".join(
        f"{size}:{composed_seconds((size, probe)) * 1e3:.3f}ms"
        for size, probe in fitting if probe is not None and probe > 0
    )
    reasons.append(
        f"core {best_size}^3 chosen from admitted candidates "
        f"{[s for s, _ in fitting]} by projected composed critical path "
        f"over {worker_budget + 1} execution slot(s) "
        f"({worker_budget} background worker(s) plus caller; "
        f"{estimates or 'no timings'})"
    )

    if (m, n, k) == (best_size,) * 3:
        reasons.append(
            "task is exactly the core's own size; the exact-size "
            "specialized route already serves it with zero composition "
            "overhead"
        )
        return TilingDecision(
            False, None, worker_budget, tuple(candidates), tuple(reasons)
        )

    if worker_budget < 1:
        reasons.append(
            "tiled composition refused with no background workers: current "
            "packed serial composition has no positive calibration evidence"
        )
        return TilingDecision(
            False, None, worker_budget, tuple(candidates), tuple(reasons)
        )

    reasons.append(
        f"tiled composition chosen: {m}x{n}x{k} covered by {best_size}^3 "
        "cores plus padded edges, with independent output lanes and "
        f"worker budget {worker_budget} available to the executor"
    )
    return TilingDecision(
        True, best_size, worker_budget, tuple(candidates), tuple(reasons)
    )


def build_gemm_tile_plan(
    m: int, n: int, k: int, tile: int, *,
    worker_budget: int = 1,
    nested_parallelism: int = 0,
    reasons: tuple[str, ...] = (),
) -> TiledDeploymentPlan:
    """The full decomposition, stated before anything runs."""

    lanes: list[TileLane] = []
    for i0 in range(0, m, tile):
        mi = min(tile, m - i0)
        for j0 in range(0, n, tile):
            nj = min(tile, n - j0)
            steps = tuple(
                TileStep(
                    p0=p0,
                    kp=min(tile, k - p0),
                    beta=1.0 if index_p else float("nan"),  # nan = caller's beta
                    uses_core=(
                        mi == tile and nj == tile
                        and min(tile, k - p0) == tile
                    ),
                )
                for index_p, p0 in enumerate(range(0, k, tile))
            )
            lanes.append(TileLane(
                index=len(lanes), i0=i0, j0=j0, mi=mi, nj=nj, steps=steps,
            ))
    return TiledDeploymentPlan(
        kind=TILE_COMPOSITION_KIND,
        schedule="independent_lanes",
        origin="kernel_bank",
        tile=int(tile), m=int(m), n=int(n), k=int(k),
        lanes=tuple(lanes),
        join_mode="barrier",
        worker_budget=int(worker_budget),
        nested_parallelism=int(nested_parallelism),
        reasons=tuple(reasons),
    )


def prebake_gemm_launch_matrix(
    plan: TiledDeploymentPlan,
    *,
    variant_key: str,
    parameter_ids: Mapping[str, int],
    total_layout: Mapping[str, Any],
    core_layout: Mapping[str, Any],
    chunk_size: int,
) -> dict[str, Any]:
    """Encode every packing permutation and pool claim before execution.

    This is the compile-complementary bridge between a tiled decision and
    hyperspecific native modules. Source offsets/strides address the total
    matrices; packed offsets/strides address the specialized core ABI. No
    runtime component has to rediscover either layout or repartition lanes.
    Partial edges use the same square core with zero-filled packed margins;
    only the valid C window is published. Thus arbitrary positive m/n/k can
    be prebaked without manufacturing a family of tiny edge modules.
    """

    chunk_size = max(1, int(chunk_size))
    tile = int(plan.tile)
    calls = []
    for lane in plan.lanes:
        lane_calls = []
        for step_index, step in enumerate(lane.steps):
            lane_calls.append({
                "step": step_index,
                "module_key": str(variant_key),
                "parameters_by_name": {
                    "A": {
                        "source_offset": lane.i0 * plan.k + step.p0,
                        "source_shape": [lane.mi, step.kp],
                        "source_strides": [plan.k, 1],
                        "packed_shape": [tile, tile],
                        "packed_strides": [tile, 1],
                        "zero_fill_packed_margin": not (
                            lane.mi == tile and step.kp == tile
                        ),
                    },
                    "B": {
                        "source_offset": step.p0 * plan.n + lane.j0,
                        "source_shape": [step.kp, lane.nj],
                        "source_strides": [plan.n, 1],
                        "packed_shape": [tile, tile],
                        "packed_strides": [tile, 1],
                        "zero_fill_packed_margin": not (
                            step.kp == tile and lane.nj == tile
                        ),
                    },
                    "C": {
                        "source_offset": lane.i0 * plan.n + lane.j0,
                        "source_shape": [lane.mi, lane.nj],
                        "source_strides": [plan.n, 1],
                        "packed_shape": [tile, tile],
                        "packed_strides": [tile, 1],
                        "zero_fill_packed_margin": not (
                            lane.mi == tile and lane.nj == tile
                        ),
                        "publish_after_last_step": True,
                    },
                    "alpha": "caller_alpha",
                    "beta": "caller_beta" if step_index == 0 else 1.0,
                },
            })
        calls.append({
            "lane": lane.index,
            "output_origin": [lane.i0, lane.j0],
            "calls": lane_calls,
        })
    return {
        "schema": "turing.prebaked-gemm-launch-matrix.v1",
        "module_key": str(variant_key),
        "module_binding_by_name": {
            str(name): int(identifier)
            for name, identifier in sorted(parameter_ids.items())
        },
        "problem_shape": [plan.m, plan.n, plan.k],
        "tile_shape": [tile, tile, tile],
        "total_parameter_layout": dict(total_layout),
        "core_parameter_layout": dict(core_layout),
        "launch": {
            "join": plan.join_mode,
            "workers": plan.worker_budget,
            "chunk_size": chunk_size,
            "lane_count": len(plan.lanes),
            "spans": [
                [start, min(start + chunk_size, len(plan.lanes))]
                for start in range(0, len(plan.lanes), chunk_size)
            ],
        },
        "lanes": calls,
    }


__all__ = [
    "TILE_COMPOSITION_KIND",
    "TileStep",
    "TileLane",
    "TiledDeploymentPlan",
    "TilingDecision",
    "decide_tiling",
    "build_gemm_tile_plan",
    "prebake_gemm_launch_matrix",
]
