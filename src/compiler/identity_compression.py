"""Compressed IN-RAM storage behind the existing identity-page accessors.

Unlike identity_spill, this never removes table attributes or installs a spill
marker. They become compressed mappings. ``cell/latest/history/stamp_at`` read
only matching tiles; scans stream tiles. The canonical writer explicitly
promotes its page before mutation. The same mappings can escalate independent
tiles/index chunks to stronger disk compression; there is no second concordance.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Iterable

from .compressed_tiles import CompressedTable, DiskTileStore
from .identity_spill import SPILLABLE_ATTRS, _dispatch_table, _LOAD_REGISTRY
from .tile_row_codec import decode_rows

TABLES = tuple(name for name in SPILLABLE_ATTRS if name != "columns")


@dataclass(frozen=True)
class CompressionResult:
    compressed: tuple[str, ...]
    refused: tuple[tuple[str, str], ...]
    cells_dropped: int
    raw_bytes: int
    resident_bytes: int
    budget_met: bool | None


def _loader(registry):
    def load(data):
        token = _LOAD_REGISTRY.set(registry)
        try:
            return decode_rows(data)
        finally:
            _LOAD_REGISTRY.reset(token)
    return load


def thaw_page(page: Any) -> None:
    """Only a writer/copy requests a mutable page; ordinary reads never do."""
    state = page.__dict__
    if not state.get("_compressed_ram"):
        return
    # Prepare all tables before changing the page, so a failed decode is atomic.
    restored = {name: state[name].mutable() for name in TABLES}
    restored["columns"] = list(state["columns"])
    state.update(restored)
    state.pop("_compressed_ram", None)


def compression_stats(book: Any) -> tuple[int, int]:
    raw = resident = 0
    for page in dict.values(book.pages):
        if page.__dict__.get("_compressed_ram"):
            for name in TABLES:
                stats = page.__dict__[name].stats()
                raw += stats.raw_bytes
                resident += stats.resident_bytes
            resident += sys.getsizeof(page.__dict__["columns"])
    return raw, resident


def disk_tile_stats(book: Any) -> tuple[int, int]:
    tiles = stored = 0
    for page in dict.values(book.pages):
        if page.__dict__.get("_compressed_ram"):
            for name in TABLES:
                stats = page.__dict__[name].stats()
                tiles += stats.disk_tiles
                stored += stats.disk_bytes
    return tiles, stored


@dataclass(frozen=True)
class TileOffloadResult:
    released_bytes: int
    resident_bytes: int
    disk_bytes: int
    disk_tiles: int
    target_met: bool
    policy_sources: tuple = ()
    boundary: str = ""
    budget_bytes: int = 0
    resident_before: int = 0


def offload_book_tiles(book, *, budget_bytes, resident_bytes, boundary=""):
    """Escalate already-compressed tiles, NOT whole-page disk spill.

    Indexes and tile-local access stay intact. Existing policy refs license the
    movement; its sourced runtime receipt does not thaw private provenance
    tables merely to record storage statistics. Actual sizes are returned and
    retained on the book, with the budget and pressure boundary.
    """
    _raw, before = compression_stats(book)
    reserved = max(0, resident_bytes - before)
    target = max(0, budget_bytes - reserved)
    pages = [page for page in dict.values(book.pages)
             if page.__dict__.get("_compressed_ram")]
    if before <= target or not pages:
        tiles, disk = disk_tile_stats(book)
        return TileOffloadResult(0, before, disk, tiles, before <= target)
    # Storage movement is not a value transformation. Its runtime receipt is
    # derived from policies already on the SAME book. Posting new provenance
    # edges here would promote a compressed private page and undo the release.
    disk_policy = book.registry.pages.get("compile_policy")
    policy_sources = () if disk_policy is None else tuple(
        ref for row in (("compressed_tile_disk",), ("memory_budget_bytes",))
        if (ref := book.latest_ref(disk_policy, row)) is not None
    )
    if not policy_sources or book.page(disk_policy).latest(("compressed_tile_disk",)) != "true":
        raise ValueError("disk tile policy must be declared before compressing the book")
    if book.page(disk_policy).latest(("memory_budget_bytes",)) != str(budget_bytes):
        raise ValueError("disk tile threshold disagrees with the book's declared memory budget")
    for ref in policy_sources:
        book._source_stamp(ref)
    tables = [page.__dict__[name] for page in pages
              if page.__dict__.get("_compressed_ram") for name in TABLES]
    tables.sort(key=lambda table: table.stats().resident_bytes, reverse=True)
    store = book.__dict__.get("_tile_store")
    if store is None and tables:
        store = book.__dict__["_tile_store"] = DiskTileStore()
    current = sum(table.stats().resident_bytes for table in tables)
    for table in tables:
        if current <= target:
            break
        previous = table.stats().resident_bytes
        after = table.offload_to_disk(store=store, target_bytes=max(0, target - (current - previous)))
        current += after.resident_bytes - previous
    _raw, after = compression_stats(book)
    tiles, disk = disk_tile_stats(book)
    result = TileOffloadResult(max(0, before - after), after, disk, tiles,
                              reserved + after <= budget_bytes, policy_sources,
                              boundary, budget_bytes, resident_bytes)
    book.__dict__["_tile_offload_report"] = result
    return result


def fit_book_budget(book: Any, budget_bytes: int, resident_bytes: int) -> bool:
    """Allocate actual remaining headroom proportionally, not a promised ratio.

    RSS minus owned compressed buffers is the observed non-compressible part.
    Allocator slack, live compiler graphs and held Python objects remain there.
    If it already exhausts the limit, compression explicitly reports unmet.
    """
    _raw, stored = compression_stats(book)
    reserved = max(0, resident_bytes - stored)
    available = max(0, budget_bytes - reserved)
    tables = [page.__dict__[name]
              for page in dict.values(book.pages)
              if page.__dict__.get("_compressed_ram") for name in TABLES]
    total = sum(table.stats().resident_bytes for table in tables)
    for table in tables:
        size = table.stats().resident_bytes
        table.fit_budget(available * size // max(1, total))
    _raw, after = compression_stats(book)
    return reserved + after <= budget_bytes


def _receipt(book, owner, action, raw, stored, cells, rows, boundary, *, storage="ram", budget_bytes=None):
    from .concordance_declarations import (
        COMPILE_POLICY, MEMORY_REGULATION, PAGE_SPILL_RECEIPT,
        POLICY_DECLARATION, PageSpill,
    )
    from .identity_concordance import Derived, Mode, Novel
    from .identity_spill import ensure_declared

    ensure_declared(book)
    book.post(COMPILE_POLICY, ("compress_cold_pages",), "ram-tiles",
              stage=MEMORY_REGULATION, provenance=Novel(POLICY_DECLARATION, ()),
              mode=Mode.CONCORD)
    policy = book.latest_ref(COMPILE_POLICY, ("compress_cold_pages",))
    causes = [policy]
    if budget_bytes is not None:
        book.post(COMPILE_POLICY, ("memory_budget_bytes",), str(budget_bytes),
                  stage=MEMORY_REGULATION, provenance=Novel(POLICY_DECLARATION, ()),
                  mode=Mode.CONCORD)
        causes.append(book.latest_ref(COMPILE_POLICY, ("memory_budget_bytes",)))
    ordinal = book.__dict__.get("_storage_receipts", 0)
    book.post(PAGE_SPILL_RECEIPT, (owner, ordinal),
              PageSpill(action, owner, TABLES, -1, (), stored, raw, cells, rows,
                        "ram-tiles", boundary, storage=storage),
              stage=MEMORY_REGULATION, provenance=Derived(tuple(causes)),
              mode=Mode.CONCORD)
    book.__dict__["_storage_receipts"] = ordinal + 1


def compress_pages(
    book: Any, names: Iterable[str], *, tile_rows: int = 128,
    budget_bytes: int | None = None, boundary: str = "",
) -> CompressionResult:
    """Compress eligible pages atomically; never reload them merely to read.

    Externally held table dictionaries and unserialisable facts are refused.
    Policy/scope pages remain live. At compile-end private provenance tables
    can also be compressed: unlike disk partitions they stay directly readable.
    """
    from .identity_spill import PARTITIONED

    # Declare the disk-tier capability BEFORE freezing private edge tables.
    # The later representation change derives its runtime receipt from these
    # existing refs instead of thawing a whole provenance page under pressure.
    from .concordance_declarations import COMPILE_POLICY, MEMORY_REGULATION, POLICY_DECLARATION
    from .identity_concordance import Mode, Novel
    from .identity_spill import ensure_declared
    from .work_contract import active_contract

    ensure_declared(book)
    policies = [(("compressed_tile_disk",),
                 "true" if active_contract().compressed_tile_disk else "false")]
    if budget_bytes is not None:
        policies.append((("memory_budget_bytes",), str(budget_bytes)))
    for row, value in policies:
        if book.page(COMPILE_POLICY).latest(row) is None:
            book.post(COMPILE_POLICY, row, value, stage=MEMORY_REGULATION,
                      provenance=Novel(POLICY_DECLARATION, ()), mode=Mode.CONCORD)

    compressed, refused = [], []
    dropped = 0
    dispatch = _dispatch_table(book.registry)
    load = _loader(book.registry)
    names = tuple(dict.fromkeys(names))
    # Receipts themselves write edge/mint rows. Authorize the private batch
    # BEFORE packing it; posting after packing would immediately promote it.
    # Admission records carry no guessed byte count. Live buffer stats carry
    # the measured sizes. Ordinary pages retain measured per-page receipts.
    ordinary = tuple(name for name in names if name not in PARTITIONED)
    for name in ordinary:
        result = _compress_one(book, name, tile_rows, dispatch, load, boundary,
                               record=True)
        if result[0]:
            compressed.append(name)
            dropped += result[1]
        if result[2]:
            refused.append((name, result[2]))
    private = tuple(
        name for name in names if name in PARTITIONED
        and (page := dict.get(book.pages, name)) is not None
        and not page.__dict__.get("_compressed_ram")
        and not page.__dict__.get("_live_tile_owner")
        and not page.__dict__.get("_detached")
        and page.cell_count()
    )
    for name in private:
        _receipt(book, name, "compress-ram-admit", 0, 0, 0, 0, boundary)
    for name in private:
        result = _compress_one(book, name, tile_rows, dispatch, load, boundary,
                               record=False)
        if result[0]:
            compressed.append(name)
            dropped += result[1]
        if result[2]:
            refused.append((name, result[2]))
    met = None
    if budget_bytes is not None:
        from .shell_telemetry import process_rss_bytes

        reading = process_rss_bytes()
        met = None if reading == 0 else fit_book_budget(book, budget_bytes, reading)
        book.__dict__["_compression_budget_met"] = met
    raw, resident = compression_stats(book)
    return CompressionResult(tuple(compressed), tuple(refused), dropped,
                             raw, resident, met)


def _compress_one(book, name, tile_rows, dispatch, load, boundary, *, record):
    from .identity_spill import NEVER_SPILLED, PARTITIONED

    if name in NEVER_SPILLED and name not in PARTITIONED:
        return False, 0, None
    page = dict.get(book.pages, name)
    if page is None:
        return False, 0, None
    state = page.__dict__
    if state.get("_live_tile_owner") is not None:
        # Already writable tile-local storage in the compile-wide arena.
        # Taking another immutable snapshot would duplicate it and reintroduce
        # whole-page thaw on its next accepted transformation.
        return False, 0, None
    if state.get("spill") or state.get("_detached") or state.get("_compressed_ram"):
        return False, 0, None
    if not state["cells"]:
        return False, 0, None
    held = [key for key in SPILLABLE_ATTRS if sys.getrefcount(state[key]) > 2]
    if held:
        reason = "held: " + ", ".join(held)
        if record:
            _receipt(book, name, "refuse-ram", 0, 0, 0, 0, reason)
        return False, 0, reason
    try:
        tables = {key: CompressedTable(state[key].items(), tile_rows=tile_rows,
                                      dispatch_table=dispatch, loader=load)
                  for key in TABLES}
        columns = tuple(state["columns"])
    except (TypeError, AttributeError, ValueError) as error:
        reason = f"{type(error).__name__}: {error}"
        if record:
            _receipt(book, name, "refuse-ram", 0, 0, 0, 0, reason)
        return False, 0, reason
    raw = sum(table.stats().raw_bytes for table in tables.values())
    stored = sum(table.stats().resident_bytes for table in tables.values())
    cells, rows = len(state["cells"]), len(state["row_columns"])
    state.update(tables)
    state["columns"] = columns
    state["_compressed_ram"] = True
    if record:
        _receipt(book, name, "compress-ram", raw, stored, cells, rows, boundary)
    return True, cells, None
