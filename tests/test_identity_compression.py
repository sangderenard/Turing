from __future__ import annotations

import copy
import pickle
from dataclasses import replace

import pytest

from src.compiler.compressed_tiles import CompressedTable
from src.compiler.identity_compression import compression_stats
from src.compiler.identity_concordance import Derived, IdentityBook, Mode, Novel, Ref, Registry, RowField, RowFieldKind
from src.compiler import memory_regulation, page_lifecycle
from src.compiler.work_contract import PRESETS, active_contract, set_active_contract


def populated(count=1024):
    registry = Registry()
    page = registry.declare_page("compressed_probe", (
        RowField("scope", RowFieldKind.SCOPE), RowField("id", RowFieldKind.VALUE_ID)), object)
    stage = registry.declare_stage("compression_probe")
    root = registry.declare_transform("compression_root", 0)
    book = IdentityBook(registry=registry)
    for i in range(count):
        book.post(page, ("scope", i), ("stable repeated text " * 20, i), stage=stage,
                  provenance=Novel(root, ()), mode=Mode.CONCORD)
    return book, page, stage, root


def test_book_accessors_remain_compressed_and_never_touch_disk(monkeypatch):
    book, declared, stage, root = populated()
    page = book.page(declared)
    expected = (page.latest(("scope", 700)), page.history(("scope", 700)),
                page.stamp_at(("scope", 700), 0), page.scope_rows("scope"))
    ref = Ref(declared, ("scope", 700), 0)
    mint = book.mint_of(ref)
    def forbidden(*args, **kwargs):
        raise AssertionError("compressed RAM must not create a disk spill")
    monkeypatch.setattr("src.compiler.identity_spill.tempfile.mkstemp", forbidden)
    result = book.compress_pages((declared.name,), tile_rows=32)
    assert result.compressed == (declared.name,) and result.cells_dropped == 1024
    assert book._spill is None
    before = page.cells.stats().decoded_tiles
    assert page.latest(("scope", 700)) == expected[0]
    assert page.cells.stats().decoded_tiles - before == 1
    assert page.history(("scope", 700)) == expected[1]
    assert page.stamp_at(("scope", 700), 0) == expected[2]
    assert page.scope_rows("scope") == expected[3]
    assert book.mint_of(ref) == mint and mint[0] is root
    assert book.compressed_pages() == (declared.name,)
    assert isinstance(page.cells, CompressedTable)
    receipts = book.page("page_spill_receipt")
    receipt = list(receipts.cells.values())[-1]
    assert receipt.storage == "ram" and receipt.segment_offsets == ()
    assert book.edges_into(book.latest_ref(book.registry.pages["page_spill_receipt"], (declared.name, 0)))


def test_canonical_writer_promotes_page_and_preserves_history_and_registry():
    book, declared, stage, root = populated(128)
    page = book.page(declared)
    original = page.latest(("scope", 2))
    source = book.latest_ref(declared, ("scope", 2))
    book.compress_pages((declared.name,))
    assert book.stamp_of(source) == page.stamp_at(source.row, source.column)
    book.post(declared, ("scope", 2), "revision", stage=stage,
              provenance=Derived((source,)), mode=Mode.REVISE)
    assert page.history(("scope", 2)) == ((0, original), (1, "revision"))
    assert not book.compressed_pages()
    assert book.edges_into(book.latest_ref(declared, ("scope", 2))) == ((source, stage),)
    book.compress_pages((declared.name,))
    restored = pickle.loads(pickle.dumps(book))
    assert restored.page(declared).history(("scope", 2)) == page.history(("scope", 2))


def test_external_table_holder_and_unserializable_fact_are_refused():
    book, declared, stage, root = populated(4)
    held = book.page(declared).cells
    result = book.compress_pages((declared.name,))
    assert result.refused and "held: cells" in result.refused[0][1]
    assert book.page(declared).cells is held
    del held
    book.post(declared, ("scope", 10), lambda: None, stage=stage,
              provenance=Novel(root, ()), mode=Mode.CONCORD)
    result = book.compress_pages((declared.name,))
    assert result.refused and not result.compressed
    assert isinstance(book.page(declared).cells, dict)


def test_budget_and_work_contract_wiring(monkeypatch):
    book, declared, stage, root = populated()
    monkeypatch.setitem(page_lifecycle.PAGE_LIFECYCLE, declared.name,
                        page_lifecycle.PageLifecycle("source-closure", None, "source-closure"))
    monkeypatch.setattr(memory_regulation, "process_rss_bytes", lambda: 100)
    regulator = memory_regulation.MemoryRegulator(book, compress_policy=True)
    regulator.boundary("ssa-source: topology reduced, program ABI reattached")
    assert declared.name in book.compressed_pages()
    assert compression_stats(book)[0] > 0 and compression_stats(book)[1] > 0
    monkeypatch.setenv("TURING_COMPRESS_COLD_PAGES", "1")
    monkeypatch.setenv("TURING_SPILL_COLD_PAGES", "0")
    assert active_contract().compress_cold_pages
    with pytest.raises(ValueError, match="not both"):
        replace(PRESETS["develop"], compress_cold_pages=True, spill_cold_pages=True)


def test_private_provenance_remains_compressed_and_registry_refs_are_exact():
    from src.compiler.identity_spill import PARTITIONED

    book, declared, stage, root = populated(512)
    source = book.latest_ref(declared, ("scope", 4))
    target = book.post(declared, ("scope", 900), "derived", stage=stage,
                       provenance=Derived((source,)), mode=Mode.CONCORD)
    incoming = book.edges_into(target)
    minted = book.mint_of(source)
    book.compress_pages((declared.name, *PARTITIONED), tile_rows=32)
    compressed = book.compressed_pages()
    assert "concordance_edge" in compressed
    assert "concordance_dependents" in compressed
    assert "concordance_mint" in compressed
    assert book.edges_into(target) == incoming
    assert book.mint_of(source) == minted
    assert book.mint_of(source)[0] is root
    assert book.edges_into(target)[0][0].page is declared
    assert book.compressed_pages() == compressed
    # A subsequent accepted transformation still posts through the same writer.
    book.post(declared, ("scope", 901), "next", stage=stage,
              provenance=Derived((target,)), mode=Mode.CONCORD)
    assert book.page(declared).latest(("scope", 901)) == "next"


def test_ram_policy_does_not_open_disk_when_over_budget(monkeypatch):
    book, declared, stage, root = populated(4)
    monkeypatch.setitem(page_lifecycle.PAGE_LIFECYCLE, declared.name,
                        page_lifecycle.PageLifecycle("source-closure", None, "source-closure"))
    monkeypatch.setattr(memory_regulation, "process_rss_bytes", lambda: 10**9)
    monkeypatch.setattr("src.compiler.shell_telemetry.process_rss_bytes", lambda: 10**9)
    monkeypatch.setattr(memory_regulation, "RELEASE_ORDER", (("cold_pages", memory_regulation._spill_cold_pages),))
    def forbidden(*args, **kwargs):
        raise AssertionError("RAM regulation opened a disk file")
    monkeypatch.setattr("src.compiler.identity_spill.tempfile.mkstemp", forbidden)
    regulator = memory_regulation.MemoryRegulator(book, 100, compress_policy=True)
    token = memory_regulation._ACTIVE.set(regulator)
    try:
        regulator.boundary("ssa-source: topology reduced, program ABI reattached")
    finally:
        memory_regulation._ACTIVE.reset(token)
    assert book._spill is None
    assert book.__dict__["_compression_budget_met"] is False


def test_pressure_offloads_book_without_thawing_private_pages(tmp_path, monkeypatch):
    from src.compiler.identity_compression import offload_book_tiles, disk_tile_stats
    from src.compiler.identity_spill import PARTITIONED

    monkeypatch.setenv("TURING_IDENTITY_SPILL_DIR", str(tmp_path))
    book, declared, stage, root = populated(4096)
    book.compress_pages((declared.name, *PARTITIONED), budget_bytes=1)
    before = compression_stats(book)[1]
    compressed = book.compressed_pages()
    clock = book.clock[0]
    result = offload_book_tiles(book, budget_bytes=1, resident_bytes=10**9)
    assert result.released_bytes > before / 2
    assert result.disk_tiles > 0 and result.disk_bytes > 0
    assert result.policy_sources and book.clock[0] == clock
    assert book.compressed_pages() == compressed
    assert book.page(declared).latest(("scope", 2300)) == ("stable repeated text " * 20,2300)
    source = book.latest_ref(declared, ("scope",2300))
    assert book.mint_of(source)[0] is root
    assert book._spill is None
    assert disk_tile_stats(book)[0] > 0
    restored = pickle.loads(pickle.dumps(book))
    assert restored.page(declared).latest(("scope", 2300)) == book.page(declared).latest(("scope",2300))


def test_in_stage_pressure_checkpoint_uses_declared_disk_tier(tmp_path, monkeypatch):
    from src.compiler.identity_compression import disk_tile_stats

    monkeypatch.setenv("TURING_IDENTITY_SPILL_DIR", str(tmp_path))
    book, declared, stage, root = populated(1024)
    book.compress_pages((declared.name,), budget_bytes=1)
    monkeypatch.setattr(memory_regulation, "process_rss_bytes", lambda: 10**9)
    regulator = memory_regulation.MemoryRegulator(book, 1, compress_policy=True, tile_disk_policy=True)
    regulator.pressure_checkpoint("deployment: callsite specialization")
    assert disk_tile_stats(book)[0] > 0
    previous = book.__dict__["_tile_offload_report"]
    regulator.pressure_checkpoint("deployment: callsite specialization")
    assert book.__dict__["_tile_offload_report"] is previous


def test_storage_audit_reports_missing_policy_cause(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from src.compiler.identity_compression import offload_book_tiles
    from src.compiler.identity_concordance import CorrelationTable

    monkeypatch.setenv("TURING_IDENTITY_SPILL_DIR", str(tmp_path))
    book, declared, stage, root = populated(256)
    book.compress_pages((declared.name,), budget_bytes=1)
    offload_book_tiles(book, budget_bytes=1, resident_bytes=10**9)
    module = SimpleNamespace(metadata={"identity_book": book})
    assert CorrelationTable._tile_storage_findings(module) == []
    book.__dict__["_tile_offload_report"] = replace(book.__dict__["_tile_offload_report"], policy_sources=())
    assert CorrelationTable._tile_storage_findings(module)[0].kind == "tile-storage-policy"
