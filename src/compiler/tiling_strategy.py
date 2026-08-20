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
* **Profiling numbers** from admission manifests
  (``verification.probe_call_seconds``). These are COLD-call timings --
  they include first-call preparation, so their absolute GF/s understates
  steady state badly -- but the bias applies to every candidate alike, so
  they rank candidates fairly. When steady-state numbers exist (a future
  routing-log aggregation), they should replace this ranking.
* **The task's shape**: a core larger than the task on any axis cannot
  tile it; a task not meaningfully larger than the best core gains nothing
  from composition overhead.
* **Cores available and NESTED deployment depth**: a tiled plan inside an
  already-parallel deployment must not multiply worlds. The worker budget
  is tempered by nesting (``cores // (1 + nested_parallelism)``), and at
  budget 1 the plan is still emitted -- tiling wins serially through cache
  locality (measured 1.34x at 256^3, ``docs/KERNEL_BANK_DESIGN.md``
  section 4.5) -- but the budget and its tempering are recorded so the
  executor never over-subscribes.

The plan object (:class:`TiledDeploymentPlan`) is shaped to mirror
``ControlDeploymentRegion``: ``kind``/``schedule``/``lanes``/barrier join,
``origin="kernel_bank"``. Each lane is one C-block -- lanes are mutually
independent (disjoint output tiles), which is the same independence proof
``_barrier_lane_memberships`` demands; the k-steps WITHIN a lane are
ordered (each accumulates into the block), which is why a lane is a
sequence of steps and not itself split. Executing a lane on a worker is
therefore safe by construction; executing steps of one lane concurrently
is not, and the plan's shape makes that distinction unrepresentable.

There is deliberately NO executor in this module and no runtime path
anywhere else: tiling is a COMPILER choice made by the deployment layer.
The decision and the plan are compile-time data for the deployment pass to
consume when it lowers a recognized region -- emitting the tile loop, the
packing, and the prebaked-core calls natively, with the plan's
worker-budget bound feeding the same pool machinery (``turing_pool.c``)
every other independent-lane region uses. Host-side composition of kernel
calls at runtime was tried, measured, and REMOVED (owner's direction:
this is not outer-code work); the numbers it produced survive in
``docs/KERNEL_BANK_DESIGN.md`` as evidence that the lowering is worth
building -- serial tiled composition alone was 1.34x over a single
parametric call at 256^3.
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
    worker_budget = max(1, cores // (1 + max(0, int(nested_parallelism))))
    if nested_parallelism > 0:
        reasons.append(
            f"nested inside {nested_parallelism} parallel deployment "
            f"level(s): worker budget tempered to {worker_budget} of "
            f"{cores} cores"
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
        reasons.append(
            f"every admitted core ({[s for s, _ in candidates]}) exceeds "
            f"the task's smallest axis ({min(m, n, k)})"
        )
        return TilingDecision(
            False, None, worker_budget, tuple(candidates), tuple(reasons)
        )

    def rank(entry: tuple[int, float | None]) -> float:
        size, probe = entry
        if probe is None or probe <= 0:
            return 0.0
        return 2.0 * size ** 3 / probe  # cold-call GF/s: fair RANKING only

    best_size, best_probe = max(fitting, key=rank)
    reasons.append(
        f"core {best_size}^3 chosen from admitted candidates "
        f"{[s for s, _ in fitting]} by admission-probe throughput "
        "(cold-call timing: biased low but uniformly so)"
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

    reasons.append(
        f"tiled composition chosen: {m}x{n}x{k} covered by {best_size}^3 "
        "cores plus parametric edges; serial tiling already measured "
        "faster than one parametric call (cache locality), worker budget "
        f"{worker_budget} available to the executor"
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


__all__ = [
    "TILE_COMPOSITION_KIND",
    "TileStep",
    "TileLane",
    "TiledDeploymentPlan",
    "TilingDecision",
    "decide_tiling",
    "build_gemm_tile_plan",
]
