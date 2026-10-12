"""Stage-boundary memory regulation for one compile.

A compile keeps in RAM what its stages need.  This module is the place where
that is *regulated* rather than hoped for:

* ``stage_boundary(label)`` is called where one stage of the lowering ends and
  the next begins.  It reads the process resident set
  (``shell_telemetry.process_rss_bytes``, the one shared helper) and, when the
  work contract declares a ``memory_budget_bytes`` and the resident set is
  above it, releases RECOMPUTABLE items in the declared order of
  ``RELEASE_ORDER`` until it is under the budget.  Each release is a
  ``memory_release_receipt`` row on the compile's book, DERIVED from the
  ``compile_policy`` ``memory_budget_bytes`` cell (the policy is declared on
  the book when the first release reads it, so a compile that never exceeds
  its budget posts nothing and leaves the book the unregulated compile
  leaves).  Nothing is killed or refused.

* ``spill_cold_pages`` (the work contract's policy, declared beside the
  budget) and a budget breach both move the identity book's COLD pages -- the
  pages no later stage touches (``page_lifecycle``) -- out of RAM into the
  book's spill file (``identity_spill``).  Under the policy a boundary
  spills whatever its completed stages made cold, whatever the resident
  set; over budget, spilling is one of the releases, after the memos and
  before the collection.  A spilled page is read back, transparently, by
  whatever touches it, so the compile's output does not depend on either;
  each spill and reload is a ``page_spill_receipt`` row derived from the
  policy cell.

* ``end_regulation`` runs at the end of the compile.  What it lets go of is
  not a decision but a retention bug fixed at its source -- state keyed to a
  book that no later read can reach (``release_compile_planning_state``) and
  the cyclic garbage the planning stages leave -- so it is unconditional and
  posts no receipt.

A release is admissible here only if dropping the item cannot change the
compile's output: a pure memo that is recomputed on its next miss
(dependency levels, the polymorphic-formal scan) or garbage nothing can
reach (``gc.collect``).  Items whose recomputation would post rows again or
re-plan a callsite (the callsite shell-type cache, the child-signature
results) are NOT releasable mid-compile; they are released when their book's
compile ends.
"""
from __future__ import annotations

import contextvars
import gc
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .shell_telemetry import process_rss_bytes


def _release_dependency_levels() -> int:
    from .region_scheduling import release_dependency_levels

    return release_dependency_levels()


def _release_polymorphic_formal_scans() -> int:
    from .glsl_deployment_strategy import release_polymorphic_formal_scans

    return release_polymorphic_formal_scans()


def _collect_cyclic_garbage() -> int:
    return gc.collect()


def _spill_cold_pages() -> int:
    regulator = _ACTIVE.get()
    return 0 if regulator is None else regulator.spill_cold("budget")


#: The recomputable items a budget breach releases, in this order (cheapest
#: to recompute first; the collection last, so it also frees what the memo
#: releases above it dropped).
RELEASE_ORDER: tuple[tuple[str, Callable[[], int]], ...] = (
    ("dependency_level_memo", _release_dependency_levels),
    ("polymorphic_formal_scan_memo", _release_polymorphic_formal_scans),
    ("cold_pages", _spill_cold_pages),
    ("cyclic_garbage", _collect_cyclic_garbage),
)


@dataclass
class MemoryRegulator:
    """One compile's regulation: its book, its budget and its readings."""

    book: Any
    budget_bytes: int | None = None
    #: The work contract's ``spill_cold_pages``: spill the book's cold pages
    #: at every boundary, over budget or not.
    spill_policy: bool = False
    compress_policy: bool = False
    tile_disk_policy: bool = False
    _last_pressure_read: float = 0.0
    #: ``(stage label, resident set)`` at every boundary, in order.
    history: list[tuple[str, int]] = field(default_factory=list)
    releases: int = 0
    #: The last stage known to have completed (a ``page_lifecycle`` stage
    #: key), and the label of the boundary that said so.
    completed: str | None = None
    label: str = ""
    #: Whether this regulator may spill its book: it opened the book itself.
    #: A compile resuming its caller's book does not know which stages the
    #: caller has finished with.
    owns_book: bool = True
    # One shared dispatch cache across every specialization/shell of this
    # compile; per-shell limits would multiply with the expanded call graph.
    graph_pool: Any = None
    live_arena: Any = None
    live_token: Any = None
    _report: bool = field(
        default_factory=lambda: bool(os.environ.get("TURING_COMPILE_RSS")),
    )

    def boundary(self, label: str) -> int:
        """Read the resident set at a stage boundary; above the budget,
        release.  Returns the reading (after any release)."""

        resident = process_rss_bytes()
        self.history.append((str(label), resident))
        self.label = str(label)
        if self.live_arena is not None:
            self.live_arena.flush()
            if self.live_arena.disk_mode:
                self._live_receipt(str(label))
        from .page_lifecycle import BOUNDARY_COMPLETES

        self.completed = BOUNDARY_COMPLETES.get(self.label, self.completed)
        if self.compress_policy and self.owns_book and self.completed is not None:
            from .page_lifecycle import cold_pages
            from .identity_spill import PARTITIONED

            names = cold_pages(tuple(dict.keys(self.book.pages)), self.completed)
            if self.label == "compile: end":
                names.extend(PARTITIONED)
            self.book.compress_pages(names, budget_bytes=self.budget_bytes,
                                     boundary=self.label)
            resident = process_rss_bytes()
        if self.spill_policy:
            self.spill_cold("policy")
            resident = process_rss_bytes()
        if self._report:
            print(
                f"[compiler-rss] {label}: {resident / 2**20:.0f} MiB",
                file=sys.stderr, flush=True,
            )
        budget = self.budget_bytes
        if budget is None or resident <= budget:
            return resident
        if self.compress_policy and self.tile_disk_policy:
            self.offload_tiles(str(label), resident)
            resident = process_rss_bytes()
            if resident <= budget:
                return resident
        if self.compress_policy and label == "compile: end":
            # Caches/garbage were released by end_regulation before this call.
            # Do not thaw frozen private tables merely to journal no-op cleanup.
            return resident
        for item, release in RELEASE_ORDER:
            before = process_rss_bytes()
            released = int(release())
            resident = process_rss_bytes()
            self._post(str(label), item, before, resident, budget, released)
            if resident <= budget:
                break
        return resident

    def offload_tiles(self, label: str, resident: int):
        if not (self.compress_policy and self.tile_disk_policy and self.owns_book
                and self.budget_bytes is not None and resident > self.budget_bytes):
            return None
        from .identity_compression import offload_book_tiles

        result = offload_book_tiles(self.book, budget_bytes=self.budget_bytes,
                                   resident_bytes=resident, boundary=label)
        if self.live_arena is not None and not self.live_arena.disk_mode:
            stats = self.live_arena.offload_to_disk()
            self._live_receipt(label)
            print(f"[compiler-memory] {label}: LIVE graph/control/deployment/SSA "
                  f"disk tiles={stats.disk_tiles} disk={stats.disk_bytes / 2**20:.2f} MiB "
                  f"decoded limit={self.live_arena.hot_limit}", file=sys.stderr, flush=True)
        if result.released_bytes:
            print(f"[compiler-memory] {label}: disk tiles={result.disk_tiles} "
                  f"disk={result.disk_bytes / 2**20:.2f} MiB "
                  f"released tile buffers/indexes={result.released_bytes / 2**20:.2f} MiB "
                  f"tile_target_met={result.target_met}", file=sys.stderr, flush=True)
        return result

    def _live_receipt(self, label):
        stats = self.live_arena.stats()
        self.book.__dict__["_live_tile_report"] = {
            "boundary": label, "budget_bytes": self.budget_bytes,
            "disk_tiles": stats.disk_tiles, "disk_bytes": stats.disk_bytes,
            "hot_limit": self.live_arena.hot_limit, "tile_rows": self.live_arena.tile_rows,
            "bucket_rows": self.live_arena.bucket_rows,
            "decoded_budget_bytes": self.live_arena.decoded_budget_bytes,
            "decoded_bytes": stats.decoded_bytes,
            "policy_sources": getattr(self, "live_policy_sources", ()),
        }

    def pressure_checkpoint(self, label: str):
        """Throttled pressure check INSIDE instantiation, independent of UI."""
        if not (self.compress_policy and self.tile_disk_policy and self.owns_book
                and self.budget_bytes is not None):
            return None
        now = time.monotonic()
        if now - self._last_pressure_read < 0.5:
            return None
        self._last_pressure_read = now
        return self.offload_tiles(label, process_rss_bytes())

    def spill_cold(self, trigger: str) -> int:
        """Spill the book's pages that the completed stages made cold; the
        number of cells dropped.  ``trigger``: ``"policy"`` or ``"budget"``."""

        if self.compress_policy or not self.owns_book or self.completed is None:
            return 0
        from .page_lifecycle import spill_cold_pages

        result = spill_cold_pages(
            self.book, self.completed, trigger=trigger,
            budget_bytes=self.budget_bytes, boundary=self.label,
        )
        return int(result.cells_dropped)

    def _post(
        self, label: str, item: str, before: int, after: int, budget: int,
        released: int,
    ) -> None:
        from .concordance_declarations import (
            COMPILE_POLICY, MEMORY_RELEASE_RECEIPT, MEMORY_REGULATION,
            MemoryRelease, POLICY_DECLARATION,
        )
        from .identity_concordance import Derived, Mode, Novel

        from .identity_spill import ensure_declared

        book = self.book
        ensure_declared(book)       # (a book with a registry of its own)
        book.post(
            COMPILE_POLICY, ("memory_budget_bytes",), str(budget),
            stage=MEMORY_REGULATION,
            provenance=Novel(POLICY_DECLARATION, ()), mode=Mode.CONCORD,
        )
        policy_cell = book.latest_ref(COMPILE_POLICY, ("memory_budget_bytes",))
        book.post(
            MEMORY_RELEASE_RECEIPT, (label, self.releases),
            MemoryRelease(item, before, after, budget, released),
            stage=MEMORY_REGULATION, provenance=Derived((policy_cell,)),
            mode=Mode.CONCORD,
        )
        self.releases += 1


_ACTIVE: contextvars.ContextVar[MemoryRegulator | None] = (
    contextvars.ContextVar("memory_regulator", default=None)
)


def begin_regulation(
    book: Any, *, owned_book: bool = True,
) -> tuple[MemoryRegulator, contextvars.Token, bool]:
    """Open the regulation of a compile on ``book``.

    The budget is the active work contract's ``memory_budget_bytes``.  The
    third result says whether this compile is the OUTERMOST (no regulation
    was already open), which ``end_regulation`` needs: a nested compile
    must not collect or release on behalf of the compile that is still
    running."""

    from .work_contract import active_contract

    contract = active_contract()
    regulator = MemoryRegulator(
        book, contract.memory_budget_bytes,
        spill_policy=bool(contract.spill_cold_pages), owns_book=owned_book,
        compress_policy=bool(contract.compress_cold_pages),
        tile_disk_policy=bool(contract.compressed_tile_disk),
    )
    outermost = _ACTIVE.get() is None
    token = _ACTIVE.set(regulator)
    try:
        _initialize_regulation(regulator, contract)
        regulator.boundary("compile: begin")
    except BaseException:
        # A failing initial disk write/table conversion occurs before the
        # caller receives the token. Restore BOTH contexts here; otherwise
        # the next source compile silently inherits the failed arena.
        if regulator.live_token is not None:
            from .live_tiles import _ACTIVE_ARENA
            regulator.live_arena.pressure_probe = None
            _ACTIVE_ARENA.reset(regulator.live_token)
        _ACTIVE.reset(token)
        raise
    return regulator, token, outermost and owned_book


def _initialize_regulation(regulator, contract):
    book = regulator.book
    if regulator.compress_policy:
        from .live_tiles import LiveTileArena, _ACTIVE_ARENA
        from .concordance_declarations import COMPILE_POLICY, MEMORY_REGULATION, POLICY_DECLARATION
        from .identity_concordance import Mode, Novel

        policies = {
            "compressed_tile_disk": "true" if regulator.tile_disk_policy else "false",
            "live_tile_rows": str(contract.live_tile_rows),
            "live_resident_tiles": str(contract.live_resident_tiles),
            "live_bucket_rows": str(contract.live_bucket_rows),
            "live_decoded_budget_bytes": str(contract.live_decoded_budget_bytes),
        }
        if regulator.budget_bytes is not None:
            policies["memory_budget_bytes"] = str(regulator.budget_bytes)
        for key, value in policies.items():
            book.post(COMPILE_POLICY, (key,), value, stage=MEMORY_REGULATION,
                      provenance=Novel(POLICY_DECLARATION, ()), mode=Mode.CONCORD)
        regulator.live_policy_sources = tuple(book.latest_ref(COMPILE_POLICY, (key,)) for key in policies)
        regulator.live_arena = LiveTileArena(tile_rows=contract.live_tile_rows,
                                           hot_tiles=contract.live_resident_tiles,
                                           bucket_rows=contract.live_bucket_rows,
                                           decoded_budget_bytes=contract.live_decoded_budget_bytes)
        regulator.live_token = _ACTIVE_ARENA.set(regulator.live_arena)
        from .live_tiles import tile_identity_page
        for page in tuple(dict.values(book.pages)):
            if not page.__dict__.get("spill") and not page.__dict__.get("_compressed_ram"):
                tile_identity_page(page, regulator.live_arena)
        regulator.live_arena.pressure_probe = regulator.pressure_checkpoint


def stage_boundary(label: str) -> None:
    """Mark the end of a lowering stage (a no-op outside a regulated compile)."""

    regulator = _ACTIVE.get()
    if regulator is not None:
        regulator.boundary(label)


def pressure_checkpoint(label: str) -> None:
    regulator = _ACTIVE.get()
    if regulator is not None:
        regulator.pressure_checkpoint(label)


def region_topology_pool():
    """The active compile's dispatch cache (legacy non-compile planners stay eager)."""
    regulator = _ACTIVE.get()
    if regulator is None:
        return None
    from .graph_work import RegionTopologyPool, graph_work_settings

    _workers, limit, _admission = graph_work_settings()
    if limit is None:
        return None
    if regulator.graph_pool is None:
        regulator.graph_pool = RegionTopologyPool(regulator.book, limit)
    return regulator.graph_pool


def stage_end(label: str) -> None:
    """Mark the end of a stage whose working set is DEAD here (the stage's
    objects are unreachable by the next one): collect the cyclic garbage it
    leaves -- planned shell classes and closures are reference cycles that
    only a full collection frees -- then take the boundary reading."""

    regulator = _ACTIVE.get()
    if regulator is not None:
        if regulator.live_arena is not None:
            # Python collection alone cannot see dead records in compressed
            # tiles. The SAME arena traces its physical handles after dropping
            # decoded cache roots; it does not mint new concordance identities.
            regulator.live_arena.collect_unused()
        else:
            gc.collect()
    stage_boundary(label)


def end_regulation(
    regulator: MemoryRegulator, token: contextvars.Token, *,
    releases_book_state: bool,
) -> None:
    """Close the compile's regulation.

    ``releases_book_state`` (the compile owned its book and was the
    outermost): let go of the planning state keyed to that book, which no
    later read can reach, then collect the cyclic garbage the stages leave.
    A compile that resumed its caller's book keeps that state -- the caller's
    next lowering on the same book may hit it."""

    try:
        if releases_book_state:
            from .glsl_deployment_strategy import release_compile_planning_state

            release_compile_planning_state(regulator.book)
            _release_dependency_levels()
            _release_polymorphic_formal_scans()
            if regulator.live_arena is not None:
                regulator.live_arena.collect_unused()
            else:
                gc.collect()
        regulator.boundary("compile: end")
    finally:
        try:
            if regulator.graph_pool is not None:
                regulator.graph_pool.close()
        finally:
            if regulator.live_token is not None:
                from .live_tiles import _ACTIVE_ARENA
                # Returned IR keeps its tile store, not a closed compiler's
                # regulator callback/book retention cycle or stale policies.
                regulator.live_arena.pressure_probe = None
                _ACTIVE_ARENA.reset(regulator.live_token)
            _ACTIVE.reset(token)
