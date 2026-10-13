"""Stage-boundary memory regulation for one compile.

A compile keeps in RAM what its stages need.  This module is the place where
that is *regulated* rather than hoped for:

* ``stage_boundary(label)`` is called where one stage of the lowering ends and
  the next begins.  It reads the process resident set
  (``shell_telemetry.process_rss_bytes``, the one shared helper) and, when the
  work contract declares a ``memory_budget_bytes`` and the resident set is
  above the signed elastic ceiling, releases RECOMPUTABLE items in the declared order of
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
import weakref
from dataclasses import dataclass, field
from typing import Any, Callable

from .shell_telemetry import process_rss_bytes, system_available_memory_bytes


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
    elastic_available_percent: int = 25
    system_reserve_bytes: int = 2 << 30
    _system_available_bytes: int | None = None
    _elastic_adjustment_bytes: int = 0
    _effective_budget_bytes: int | None = None
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
    live_registrar_token: Any = None
    live_roots: dict[int, weakref.ReferenceType] = field(default_factory=dict)
    live_settings: tuple[int, int, int, int] | None = None
    live_policy_values: dict[str, str] = field(default_factory=dict)
    _report: bool = field(
        default_factory=lambda: bool(os.environ.get("TURING_COMPILE_RSS")),
    )

    def boundary(self, label: str) -> int:
        """Read the resident set at a stage boundary; above the budget,
        release.  Returns the reading (after any release)."""

        resident = process_rss_bytes()
        self.history.append((str(label), resident))
        self.label = str(label)
        from .page_lifecycle import BOUNDARY_COMPLETES

        self.completed = BOUNDARY_COMPLETES.get(self.label, self.completed)
        if self.spill_policy:
            self.spill_cold("policy")
            resident = process_rss_bytes()
        if self._report:
            print(
                f"[compiler-rss] {label}: {resident / 2**20:.0f} MiB",
                file=sys.stderr, flush=True,
            )
        budget = self.budget_bytes
        if budget is None:
            return resident
        effective_budget, _available, _adjustment = (
            self._effective_memory_ceiling()
        )
        if resident <= effective_budget:
            return resident
        if self.compress_policy and self.tile_disk_policy:
            self._activate_live_storage()
            resident = self._climb_storage_ladder(
                str(label), resident, ceiling=effective_budget,
            )
            if resident <= effective_budget:
                return resident
        if self.compress_policy and label == "compile: end":
            # Caches/garbage were released by end_regulation before this call.
            # Do not thaw frozen private tables merely to journal no-op cleanup.
            return resident
        for item, release in RELEASE_ORDER:
            before = process_rss_bytes()
            released = int(release())
            resident = process_rss_bytes()
            self._post(
                str(label), item, before, resident,
                int(self.budget_bytes), released,
            )
            if resident <= effective_budget:
                break
        return resident

    def _effective_memory_ceiling(self):
        """Return the signed elastic process ceiling and its system sample."""

        if self.budget_bytes is None:
            return None, None, 0
        available = system_available_memory_bytes()
        adjustment = (
            0 if available is None else
            (int(available) - int(self.system_reserve_bytes))
            * int(self.elastic_available_percent) // 100
        )
        if adjustment > 0 and (
            self.live_arena is None or self.live_arena.pressure_level == 0
        ):
            # Surplus RAM enlarges only the disposable decoded cache.  Before
            # compression activates, the ordinary graph heap is not elastic.
            adjustment = 0
        effective = max(1, int(self.budget_bytes) + adjustment)
        self._system_available_bytes = available
        self._elastic_adjustment_bytes = adjustment
        self._effective_budget_bytes = effective
        return effective, available, adjustment

    def register_live_root(self, value):
        """Remember a native fast-path root weakly for possible late adoption."""
        identity = id(value)
        try:
            self.live_roots[identity] = weakref.ref(
                value, lambda _ref, key=identity, roots=self.live_roots: roots.pop(key, None))
        except TypeError:
            pass
        return value

    def _activate_live_storage(self):
        """Adopt live roots only after the process actually crosses its limit."""
        if self.live_arena is not None or self.live_settings is None:
            return self.live_arena
        from .live_tiles import (
            LiveTileArena, TiledObject, _ACTIVE_ARENA, adopt_digraph, tile_identity_page,
        )

        from .concordance_declarations import COMPILE_POLICY, MEMORY_REGULATION, POLICY_DECLARATION
        from .identity_concordance import Mode, Novel

        for key, value in self.live_policy_values.items():
            self.book.post(COMPILE_POLICY, (key,), value, stage=MEMORY_REGULATION,
                           provenance=Novel(POLICY_DECLARATION, ()), mode=Mode.CONCORD)
        self.live_policy_sources = tuple(
            self.book.latest_ref(COMPILE_POLICY, (key,)) for key in self.live_policy_values)

        rows, hot, buckets, decoded = self.live_settings
        arena = LiveTileArena(tile_rows=rows, hot_tiles=hot, bucket_rows=buckets,
                              decoded_budget_bytes=decoded)
        self.live_arena = arena
        self.live_token = _ACTIVE_ARENA.set(arena)
        try:
            for page in tuple(dict.values(self.book.pages)):
                if not page.__dict__.get("spill") and not page.__dict__.get("_compressed_ram"):
                    tile_identity_page(page, arena)
            for reference in tuple(self.live_roots.values()):
                value = reference()
                if value is None:
                    continue
                adopt_digraph(value, arena)
                if isinstance(value, TiledObject):
                    arena.attach(value)
            arena.pressure_probe = self.pressure_checkpoint
            # Persist only after adoption succeeds: the returned native module
            # may build after regulation closes, but a failed setup must not
            # masquerade as an activated compressed book.
            self.book.__dict__["_storage_pressure_activated"] = True
        except BaseException:
            _ACTIVE_ARENA.reset(self.live_token)
            self.live_token = None
            self.live_arena = None
            raise
        return arena

    def _bind_inherited_storage(self, arena):
        """Continue the SAME arena when source lowering resumes a dual-IR book."""
        if self.live_arena is arena and self.live_token is not None:
            return arena
        from .live_tiles import _ACTIVE_ARENA

        self.live_arena = arena
        self.live_token = _ACTIVE_ARENA.set(arena)
        arena.pressure_probe = self.pressure_checkpoint
        self.book.__dict__["_storage_pressure_activated"] = True
        self._live_receipt("compile: inherited storage")
        stats = arena.stats()
        print(
            f"[compiler-memory] compile: inherited storage level={stats.pressure_level} "
            f"decoded={stats.decoded_bytes / 2**20:.2f} MiB "
            f"payload_hot={stats.payload_hot_tiles} disk_tiles={stats.disk_tiles} "
            f"route_hits={stats.route_hits} route_misses={stats.route_misses} "
            f"payload_hits={stats.payload_cache_hits} "
            f"payload_reloads={stats.payload_reloads}",
            file=sys.stderr, flush=True,
        )
        return arena

    def _cold_page_names(self):
        if not (self.owns_book and self.completed is not None):
            return ()
        from .page_lifecycle import cold_pages
        from .identity_spill import PARTITIONED

        names = cold_pages(tuple(dict.keys(self.book.pages)), self.completed)
        if self.label == "compile: end":
            names.extend(PARTITIONED)
        return tuple(names)

    def _climb_storage_ladder(
        self, label: str, resident: int, *, ceiling: int | None = None,
    ) -> int:
        """Escalate only while the process remains over its effective ceiling.

        0 -> disk-back the cold decoded tail; 1 -> compress the remaining RAM
        contents; 2 -> put all backing on disk and keep one decoded payload
        tile plus the compact search metadata required to address it.
        This ladder is monotonic for one compile and never runs below budget.
        The maximum tier's storage pass is linear for the configured fixed
        tile/codec windows; LiveTileArena.collect_unused documents the stricter
        parameterized bound of its one-tile traversal cache.  ``ceiling`` is
        the signed elastic ceiling sampled by the caller; the declared work-
        contract budget remains unchanged in concordance receipts.
        """
        ceiling = self.budget_bytes if ceiling is None else int(ceiling)
        if not (self.compress_policy and self.tile_disk_policy
                and ceiling is not None and resident > ceiling
                and self.live_arena is not None):
            return resident
        arena = self.live_arena
        contracted_limit = (
            1 if arena.pressure_level >= 3 else arena.base_payload_limit
        )
        if (
            arena.decoded_budget_bytes > arena.base_decoded_budget_bytes
            or arena.payload_limit > contracted_limit
        ):
            before = resident
            started = time.perf_counter()
            stats = arena.contract_decoded_front()
            resident = process_rss_bytes()
            print(
                f"[compiler-memory] {label}: contract-decoded-front "
                f"level={stats.pressure_level} "
                f"elapsed={time.perf_counter() - started:.3f}s "
                f"rss={before / 2**20:.0f}->{resident / 2**20:.0f} MiB "
                f"decoded={stats.decoded_bytes / 2**20:.2f} MiB "
                f"payload_hot={stats.payload_hot_tiles}",
                file=sys.stderr, flush=True,
            )
            self._live_receipt(label)
            if resident <= ceiling:
                return resident
        while resident > ceiling and self.live_arena.pressure_level < 3:
            before = resident
            started = time.perf_counter()
            level = self.live_arena.pressure_level
            if level == 0:
                stats = self.live_arena.offload_cold_tail()
                action = "cold-tail-disk"
            elif level == 1:
                names = self._cold_page_names()
                if names:
                    self.book.compress_pages(names, budget_bytes=ceiling,
                                             boundary=label)
                stats = self.live_arena.compress_resident_ram()
                # Mark/sweep is deliberately deferred until compression is
                # already necessary; the fully decoded fast path never pays it.
                self.live_arena.collect_unused()
                action = "compressed-ram"
            else:
                self.offload_tiles(label, resident, budget_bytes=ceiling)
                stats = self.live_arena.stats()
                action = "minimum-working-set"
            self._live_receipt(label)
            resident = process_rss_bytes()
            print(f"[compiler-memory] {label}: {action} level={stats.pressure_level} "
                  f"elapsed={time.perf_counter() - started:.3f}s "
                  f"rss={before / 2**20:.0f}->{resident / 2**20:.0f} MiB "
                  f"decoded={stats.decoded_bytes / 2**20:.2f} MiB "
                  f"payload_hot={stats.payload_hot_tiles} "
                  f"disk_tiles={stats.disk_tiles} disk={stats.disk_bytes / 2**20:.2f} MiB "
                  f"route_hits={stats.route_hits} route_misses={stats.route_misses} "
                  f"payload_hits={stats.payload_cache_hits} "
                  f"payload_reloads={stats.payload_reloads} "
                  f"payload_evictions={stats.payload_evictions} "
                  f"compression_jobs={stats.parallel_compression_jobs} "
                  f"compression_workers={stats.compression_workers}",
                  file=sys.stderr, flush=True)
        return resident

    def offload_tiles(
        self, label: str, resident: int, *, budget_bytes: int | None = None,
    ):
        budget = self.budget_bytes if budget_bytes is None else int(budget_bytes)
        if not (self.compress_policy and self.tile_disk_policy
                and budget is not None and resident > budget):
            return None
        from .identity_compression import offload_book_tiles

        result = offload_book_tiles(self.book, budget_bytes=budget,
                                   resident_bytes=resident, boundary=label)
        if self.live_arena is not None and self.live_arena.pressure_level < 3:
            stats = self.live_arena.offload_to_disk()
            self._live_receipt(label)
            print(f"[compiler-memory] {label}: LIVE graph/control/deployment/SSA "
                  f"disk tiles={stats.disk_tiles} disk={stats.disk_bytes / 2**20:.2f} MiB "
                  f"decoded limit={self.live_arena.payload_limit}", file=sys.stderr, flush=True)
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
            "working_hot_limit": self.live_arena.payload_limit,
            "bucket_rows": self.live_arena.bucket_rows,
            "decoded_budget_bytes": self.live_arena.base_decoded_budget_bytes,
            "working_decoded_budget_bytes": self.live_arena.decoded_budget_bytes,
            "elastic_available_percent": self.elastic_available_percent,
            "system_reserve_bytes": self.system_reserve_bytes,
            "system_available_bytes": self._system_available_bytes,
            "elastic_adjustment_bytes": self._elastic_adjustment_bytes,
            "effective_budget_bytes": self._effective_budget_bytes,
            "decoded_bytes": stats.decoded_bytes,
            "pressure_level": stats.pressure_level,
            # These cumulative counters expose the A*S locality term: a high
            # reload/compression count at level three distinguishes cache
            # thrash from the one-time O(B+T) disk transition.
            "tile_compressions": stats.compressions,
            "tile_reloads": stats.reloads,
            "routing_reloads": stats.routing_reloads,
            "payload_reloads": stats.payload_reloads,
            "route_hits": stats.route_hits,
            "route_misses": stats.route_misses,
            "retained_index_bytes": stats.retained_index_bytes,
            "hot_peak": stats.hot_peak,
            "payload_cache_hits": stats.payload_cache_hits,
            "payload_evictions": stats.payload_evictions,
            "compression_jobs": stats.parallel_compression_jobs,
            "compression_seconds": stats.compression_seconds,
            "compression_workers": stats.compression_workers,
            "source_and_ssa_paged": any(
                category == "ProcessGraph.NetworkX"
                or category.endswith(".ControlProgram")
                or category.endswith(".Instr")
                for category in self.live_arena.categories
            ),
            "policy_sources": getattr(self, "live_policy_sources", ()),
        }

    def pressure_checkpoint(self, label: str):
        """Throttled pressure check INSIDE instantiation, independent of UI."""
        if not (self.compress_policy and self.tile_disk_policy
                and self.budget_bytes is not None):
            return None
        now = time.monotonic()
        if now - self._last_pressure_read < 0.5:
            return None
        self._last_pressure_read = now
        resident = process_rss_bytes()
        elastic_ceiling, available, adjustment = (
            self._effective_memory_ceiling()
        )
        if resident <= elastic_ceiling:
            if (
                self.live_arena is not None
                and self.live_arena.pressure_level > 0
            ):
                # Keep a reserve for non-arena allocations.  The remainder is
                # cache capacity, not an instruction to inflate memory: tiles
                # enter it only when the active stage actually reads them.
                reserve = max(64 << 20, self.budget_bytes // 20)
                headroom = max(0, elastic_ceiling - resident - reserve)
                target = (
                    self.live_arena.base_decoded_budget_bytes
                    + headroom
                )
                before = self.live_arena.decoded_budget_bytes
                # Ignore small availability jitter; a material reduction is
                # still honored and releases decoded payloads immediately.
                shrink = target < before - (64 << 20)
                if target > before or shrink:
                    stats = self.live_arena.resize_decoded_front(target)
                    resident = process_rss_bytes()
                    self._live_receipt(label)
                    if self._report:
                        print(
                            f"[compiler-memory] {label}: resize-decoded-front "
                            f"level={stats.pressure_level} "
                            f"limit={before / 2**20:.0f}->"
                            f"{self.live_arena.decoded_budget_bytes / 2**20:.0f} MiB "
                            f"available={0 if available is None else available / 2**20:.0f} MiB "
                            f"elastic_adjustment={adjustment / 2**20:.0f} MiB "
                            f"rss={resident / 2**20:.0f} MiB",
                            file=sys.stderr, flush=True,
                        )
            return resident
        self._activate_live_storage()
        return self._climb_storage_ladder(
            label, resident, ceiling=elastic_ceiling,
        )

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
        ordinal = int(book.__dict__.get("_memory_release_ordinal", self.releases))
        book.post(
            COMPILE_POLICY, ("memory_budget_bytes",), str(budget),
            stage=MEMORY_REGULATION,
            provenance=Novel(POLICY_DECLARATION, ()), mode=Mode.CONCORD,
        )
        policy_cell = book.latest_ref(COMPILE_POLICY, ("memory_budget_bytes",))
        book.post(
            MEMORY_RELEASE_RECEIPT, (label, ordinal),
            MemoryRelease(item, before, after, budget, released),
            stage=MEMORY_REGULATION, provenance=Derived((policy_cell,)),
            mode=Mode.CONCORD,
        )
        self.releases = ordinal + 1
        book.__dict__["_memory_release_ordinal"] = self.releases


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
        elastic_available_percent=int(
            contract.memory_elastic_available_percent
        ),
        system_reserve_bytes=int(contract.memory_system_reserve_bytes),
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
        if regulator.live_registrar_token is not None:
            from .live_tiles import _ACTIVE_REGISTRAR
            _ACTIVE_REGISTRAR.reset(regulator.live_registrar_token)
        _ACTIVE.reset(token)
        raise
    return regulator, token, outermost and owned_book


def _initialize_regulation(regulator, contract):
    book = regulator.book
    if regulator.compress_policy:
        from .live_tiles import _ACTIVE_REGISTRAR, arena_for_book
        regulator.live_settings = (
            contract.live_tile_rows, contract.live_resident_tiles,
            contract.live_bucket_rows, contract.live_decoded_budget_bytes,
        )
        regulator.live_registrar_token = _ACTIVE_REGISTRAR.set(regulator.register_live_root)

        inherited = arena_for_book(book)
        if inherited is not None:
            expected = regulator.live_settings
            actual = (inherited.tile_rows, inherited.hot_limit,
                      inherited.bucket_rows,
                      inherited.base_decoded_budget_bytes)
            if actual != expected:
                raise ValueError(
                    f"resumed live storage settings {actual!r} disagree with "
                    f"active work contract {expected!r}")
            regulator._bind_inherited_storage(inherited)

        regulator.live_policy_values = {
            "compressed_tile_disk": "true" if regulator.tile_disk_policy else "false",
            "live_tile_rows": str(contract.live_tile_rows),
            "live_resident_tiles": str(contract.live_resident_tiles),
            "live_bucket_rows": str(contract.live_bucket_rows),
            "live_decoded_budget_bytes": str(contract.live_decoded_budget_bytes),
            "memory_elastic_available_percent": str(
                contract.memory_elastic_available_percent
            ),
            "memory_system_reserve_bytes": str(
                contract.memory_system_reserve_bytes
            ),
        }
        if regulator.budget_bytes is not None:
            regulator.live_policy_values["memory_budget_bytes"] = str(regulator.budget_bytes)


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
        if (regulator.live_arena is not None
                and regulator.live_arena.pressure_level > 0):
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
            if (regulator.live_arena is not None
                    and regulator.live_arena.pressure_level > 0):
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
            if regulator.live_registrar_token is not None:
                from .live_tiles import _ACTIVE_REGISTRAR
                _ACTIVE_REGISTRAR.reset(regulator.live_registrar_token)
            _ACTIVE.reset(token)
