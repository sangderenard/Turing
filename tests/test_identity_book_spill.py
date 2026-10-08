"""Cold-page spill of the identity book, proven on hand-posted rows.

Pure data-structure tests on real ``IdentityBook`` objects.  Nothing here
lowers, compiles or runs a probe.  The unit of every claim is a page of a book
whose rows were posted by hand, spilled to the book's own spill file, and read
back through the ordinary api.
"""
from __future__ import annotations

import copy
import os
import pickle

import pytest

from src.compiler import memory_regulation as regulation
from src.compiler import page_lifecycle
from src.compiler.concordance_declarations import (
    COMPILE_POLICY, MEMORY_RELEASE_RECEIPT, PAGE_SPILL_RECEIPT, PageSpill,
)
from src.compiler.identity_concordance import (
    NEW, Derived, IdentityBook, IdentityLogLevel, Mode, Novel, Ref, Registry,
    RowField, RowFieldKind, iter_identity_book_lines, write_identity_log,
)
from src.compiler.identity_spill import PAGE, PARTITIONED, never_spilled
from src.compiler.page_lifecycle import PageLifecycle

FIELDS = (
    RowField("scope", RowFieldKind.SCOPE),
    RowField("id", RowFieldKind.VALUE_ID),
)
COLD, HOT, COLD2 = "cold_a", "hot_b", "cold_c"


def make_book():
    registry = Registry()
    pages = {
        name: registry.declare_page(name, FIELDS, object)
        for name in (COLD, HOT, COLD2)
    }
    stage = registry.declare_stage("spill_test")
    root = registry.declare_transform("spill_root", 0)
    return IdentityBook(registry=registry), pages, stage, root


class Rows:
    """Posting helpers for one book."""

    def __init__(self):
        self.book, self.pages, self.stage, self.root = make_book()

    def novel(self, name, row, fact):
        return self.book.post(
            self.pages[name], row, fact, stage=self.stage,
            provenance=Novel(self.root, ()), mode=Mode.CONCORD,
        )

    def derive(self, name, row, fact, *sources):
        return self.book.post(
            self.pages[name], row, fact, stage=self.stage,
            provenance=Derived(tuple(sources)), mode=Mode.CONCORD,
        )

    def revise(self, name, row, fact, *sources):
        return self.book.post(
            self.pages[name], row, fact, stage=self.stage,
            provenance=Derived(tuple(sources)), mode=Mode.REVISE,
        )


def populated(rounds=6):
    """cold_a: roots (some minted, some raw-revised); hot_b derived from a;
    cold_c derived from b and the next a."""
    rows = Rows()
    book = rows.book
    a, b, c = {}, {}, {}
    for i in range(rounds):
        a[i] = rows.novel(COLD, ("s", i), f"a{i}")
    minted = book.post(
        rows.pages[COLD], ("m", NEW), "minted", stage=rows.stage,
        provenance=Novel(rows.root, ()), mode=Mode.CONCORD,
    )
    for i in range(rounds):
        b[i] = rows.derive(HOT, ("s", i), f"b{i}", a[i])
        c[i] = rows.derive(COLD2, ("s", i), f"c{i}", b[i], a[(i + 1) % rounds])
    # A raw revision tags the unsourced page: a cold page's unsourced rows
    # are part of what it owns.
    book.page(COLD).revise(("s", 0), "a0-revised")
    book.page(COLD).revise(("s", 0), "a0-revised-again")
    # A fact holding a Ref (a Page inside a fact must come back as the
    # registry's own object).
    rows.novel(COLD, ("ref", 0), (b[0], "points at b"))
    rows.refs = {"a": a, "b": b, "c": c, "minted": minted}
    return rows


def raw_state(page):
    """A page's six tables, exactly (order included), as plain data."""
    d = page.__dict__
    return (
        list(d["cells"].items()), list(d["stamps"].items()),
        list(d["columns"]), [(k, list(v)) for k, v in d["scopes"].items()],
        [(k, list(v)) for k, v in d["row_columns"].items()],
        list(d["column_positions"].items()),
    )


def all_state(book):
    book.reload_all("test")
    return {name: raw_state(page) for name, page in dict.items(book.pages)}


def edge_signature(book, refs):
    return {
        ref.key: (
            tuple((s.key, st.name) for s, st in book.edges_into(ref)),
            tuple((t.key, st.name) for t, st in book.edges_out_of(ref)),
        )
        for ref in refs
    }


def is_receipt_row(row):
    return "page_spill_receipt" in repr(row) or "compile_policy" in repr(row)


# ------------------------------------------------------- spill and reload
def test_spill_drops_the_cold_page_and_the_edges_it_owns_from_ram():
    rows = populated()
    book = rows.book
    before = book.resident_cell_count()
    owned = (
        book.page(COLD).cell_count()
        + sum(1 for key in book.pages["concordance_dependents"].cells
              if key[0][0][0] == COLD)
        + sum(1 for key in book.pages["concordance_mint"].cells
              if key[0][0][0] == COLD)
        + sum(1 for key in book.pages["concordance_unsourced"].cells
              if key[0][0] == COLD)
    )
    assert owned > book.page(COLD).cell_count()      # it owns edge rows too

    result = book.spill_pages([COLD], trigger="explicit")

    assert result.spilled == (COLD,) and not result.refused
    page = dict.get(book.pages, COLD)
    assert page.spilled and not any(
        attribute in page.__dict__
        for attribute in ("cells", "stamps", "columns", "scopes",
                          "row_columns", "column_positions")
    )
    assert book.spilled_pages() == (COLD,)
    assert result.cells_dropped == owned
    # The RSS proxy dropped by what the page owned (less the receipt's own
    # few cells, which are posted after the drop).
    receipts = sum(1 for _ in book.pages[PAGE_SPILL_RECEIPT.name].rows())
    assert receipts == 1
    assert before - book.resident_cell_count() >= owned - 12
    assert book.resident_cell_count() < before
    # Counted from the segment, without reading it back.
    assert page.cell_count() == page.__dict__["spill"].cells > 0
    assert page.row_count() == page.__dict__["spill"].rows > 0
    assert page.spilled and book.spilled_pages() == (COLD,)


def test_reads_reload_byte_identical_facts_stamps_and_edges():
    rows = populated()
    book = rows.book
    refs = [*rows.refs["a"].values(), *rows.refs["b"].values(),
            *rows.refs["c"].values(), rows.refs["minted"]]
    state0 = all_state(book)
    edges0 = edge_signature(book, refs)
    history0 = {
        row: book.pages[COLD].history(row) for row in book.pages[COLD].rows()
    }

    book.spill_pages([COLD], trigger="explicit")
    assert book.spilled_pages() == (COLD,)

    # Every read api reloads what it needs and answers as before.
    page = dict.get(book.pages, COLD)
    assert page.latest(("s", 0)) == "a0-revised-again"
    assert not page.spilled
    for row, history in history0.items():
        assert page.history(row) == history
    assert page.cell(("s", 3), 0) == "a3"
    assert page.stamp_at(("s", 3), 0) == state0[COLD][1][
        [key for key, _ in state0[COLD][1]].index((("s", 3), 0))
    ][1]
    assert set(page.rows()) == set(history0)
    assert page.spans(("s", 0))[-1][2] == "a0-revised-again"
    assert edge_signature(book, refs) == edges0
    # the Ref inside a fact is the registry's own Page object
    fact = page.latest(("ref", 0))
    assert fact[0].page is book.registry.pages[HOT]

    # Spill again and compare the complete state, order included.
    book.spill_pages([COLD], trigger="explicit")
    state1 = all_state(book)
    for name in (COLD, HOT, COLD2):
        assert state1[name] == state0[name]
    for name in PARTITIONED:
        old, new = state0.get(name), state1[name]
        if old is None:
            continue
        old_cells, new_cells = old[0], new[0]
        # the old rows come back in the order they had; the receipts follow
        kept = [item for item in new_cells if not is_receipt_row(item[0])]
        assert kept == old_cells
        assert old[1] == [item for item in new[1] if not is_receipt_row(item[0])]


def test_private_edge_page_order_survives_posts_between_spills():
    rows = populated()
    book = rows.book
    refs = rows.refs
    book.spill_pages([COLD], trigger="explicit")
    # New posts that derive from a spilled cell reload it; posts that do not
    # touch it append beside the hole the spill left.
    extra = rows.derive(HOT, ("s", 100), "b-new", refs["b"][1])
    book.spill_pages([COLD2], trigger="explicit")
    book.spill_pages([COLD], trigger="explicit")
    later = rows.derive(HOT, ("s", 101), "b-newer", extra)
    book.reload_all("test")
    edge = book.pages["concordance_edge"]
    keys = [key[0] for key in edge.cells if not is_receipt_row(key[0])]
    # The unspilled book's order: for i in 0..5, hot_b[i] <- a[i], then
    # cold_c[i] <- b[i], cold_c[i] <- a[i+1]; the two new edges come last.
    expected = []
    for i in range(6):
        expected.append((HOT, ("s", i)))
    # (the populate loop posts b[i] then c[i] for each i, in turn)
    interleaved = []
    for i in range(6):
        interleaved.append((HOT, ("s", i)))
        interleaved.append((COLD2, ("s", i)))
        interleaved.append((COLD2, ("s", i)))
    interleaved += [(HOT, ("s", 100)), (HOT, ("s", 101))]
    assert [(row[0][0], row[0][1]) for row in keys] == interleaved
    assert extra.row == ("s", 100) and later.row == ("s", 101)


def test_a_write_to_a_spilled_page_reloads_it_first():
    rows = populated()
    book = rows.book
    a0 = rows.refs["a"][0]
    book.spill_pages([COLD], trigger="explicit")
    assert book.spilled_pages() == (COLD,)

    # a CONCORD repeat of the recorded fact is not a disagreement ...
    rows.novel(COLD, ("s", 4), "a4")
    assert not dict.get(book.pages, COLD).spilled
    # ... and a different fact is: the old row is there to disagree with.
    book.spill_pages([COLD], trigger="explicit")
    with pytest.raises(ValueError, match="disagreement"):
        rows.novel(COLD, ("s", 4), "different")
    book.spill_pages([COLD], trigger="explicit")
    # a raw write reloads the page AND its unsourced rows before tagging
    book.page(COLD).revise(("s", 1), "a1-raw")
    assert book.page(COLD).history(("s", 1))[-1] == (1, "a1-raw")
    tags = [r for r in book.pages["concordance_unsourced"].rows() if r[0] == COLD]
    assert len({(r[1], r[2]) for r in tags}) == len(tags)
    # the cell survives with its stamp and its edges
    assert book.page(COLD).cell(("s", 0), 0) == "a0"
    assert a0.key == (COLD, ("s", 0), 0)


def test_a_post_sourced_from_a_spilled_cell_reloads_the_source_and_its_dependents():
    rows = populated()
    book = rows.book
    a2 = rows.refs["a"][2]
    before = [(t.key, st.name) for t, st in book.edges_out_of(a2)]
    book.spill_pages([COLD], trigger="explicit")
    new = rows.derive(HOT, ("s", 200), "from-cold", a2)
    assert not dict.get(book.pages, COLD).spilled
    after = [(t.key, st.name) for t, st in book.edges_out_of(a2)]
    assert after == before + [(new.key, "spill_test")]


def test_edge_reads_load_only_the_partition_they_need():
    rows = populated()
    book = rows.book
    b1, a1 = rows.refs["b"][1], rows.refs["a"][1]
    book.spill_pages([COLD], trigger="explicit")
    spill = book._spill
    reloads = spill.reloads
    # b's edges are b's: nothing of cold_a's comes back for them
    assert [s.key for s, _ in book.edges_into(b1)] == [a1.key]
    assert dict.get(book.pages, COLD).spilled
    assert spill.reloads == reloads
    # a's dependents are a's: that partition comes back, the page does not
    assert {t.page.name for t, _ in book.edges_out_of(a1)} == {HOT, COLD2}
    assert dict.get(book.pages, COLD).spilled
    assert PAGE in spill.detached[COLD]
    assert "concordance_dependents" not in spill.detached[COLD]
    assert spill.reloads == reloads + 1


# ------------------------------------------------------------------- logs
def sections(lines):
    out, current = {}, None
    for line in lines:
        if line.startswith("["):
            current = line[1:line.index("]")]
            out[current] = {"header": line, "rows": []}
        elif line.startswith("unsourced:"):
            current = "<unsourced tally>"
            out[current] = {"header": line, "rows": []}
        elif current is not None and line.startswith("  "):
            out[current]["rows"].append(line)
    return out


@pytest.mark.parametrize("level", [
    IdentityLogLevel.FULL, IdentityLogLevel.FACTS, IdentityLogLevel.SUMMARY,
])
def test_the_log_of_a_spilled_book_is_the_log_of_the_unspilled_book(level):
    rows = populated()
    book = rows.book
    before = sections(iter_identity_book_lines(book, level))

    book.spill_pages([COLD, COLD2], trigger="explicit")
    assert set(book.spilled_pages()) == {COLD, COLD2}
    # (the summary counts come from the segments: nothing is read back)
    resident = book.resident_cell_count()
    after_lines = list(iter_identity_book_lines(book, level))
    after = sections(after_lines)
    if level is IdentityLogLevel.SUMMARY:
        assert set(book.spilled_pages()) == {COLD, COLD2}
        assert book.resident_cell_count() >= resident        # unsourced tally only
    else:
        # borrowed for the log, let go after it
        assert set(book.spilled_pages()) == {COLD, COLD2}

    receipt_pages = {COMPILE_POLICY.name, PAGE_SPILL_RECEIPT.name}
    for name, section in before.items():
        assert name in after
        mine = after[name]
        if name in PARTITIONED:
            def rows_only(lines):
                # (group headers count rows; the receipts change the counts)
                return [
                    line for line in lines
                    if not line.startswith("  (") and "page_spill_receipt" not in line
                    and "compile_policy" not in line
                ]

            assert rows_only(mine["rows"]) == rows_only(section["rows"]), name
            continue
        assert mine["rows"] == section["rows"], name
        assert mine["header"] == section["header"], name
    assert receipt_pages <= set(after)


def test_the_xz_log_written_from_a_spilled_book_matches_the_unspilled_one(tmp_path):
    import lzma

    rows = populated()
    book = rows.book
    plain = write_identity_log(book, str(tmp_path / "plain"), level="full")
    book.spill_pages([COLD, COLD2], trigger="explicit")
    spilled = write_identity_log(book, str(tmp_path / "spilled"), level="full")
    with lzma.open(plain, "rt", encoding="utf-8") as handle:
        first = sections(handle.read().split("\n"))
    with lzma.open(spilled, "rt", encoding="utf-8") as handle:
        second = sections(handle.read().split("\n"))
    for name in (COLD, HOT, COLD2):
        assert first[name] == second[name]


# --------------------------------------------------------------- receipts
def test_every_spill_and_reload_is_a_receipt_derived_from_the_policy_cell():
    rows = populated()
    book = rows.book
    book.spill_pages([COLD], trigger="explicit")
    page = dict.get(book.pages, COLD)
    segment = page.__dict__["spill"]
    assert segment.kind == PAGE and segment.owner == COLD and segment.offset == 0
    assert book._spill.index[(PAGE, COLD)] == segment

    receipts = book.pages[PAGE_SPILL_RECEIPT.name]
    (row,) = receipts.rows()
    spill = receipts.latest(row)
    assert isinstance(spill, PageSpill)
    assert (spill.action, spill.page, spill.trigger) == ("spill", COLD, "explicit")
    assert spill.segment_offset == segment.offset == spill.segment_offsets[0]
    assert spill.parts[0] == PAGE and set(spill.parts[1:]) <= set(PARTITIONED)
    assert spill.stored_bytes > 0 and spill.raw_bytes > 0
    assert spill.cells > segment.cells               # + the edge rows it owns
    policy = book.latest_ref(COMPILE_POLICY, ("spill_explicit",))
    ref = book.latest_ref(PAGE_SPILL_RECEIPT, row)
    assert [source for source, _ in book.edges_into(ref)] == [policy]

    page.latest(("s", 0))                            # read it back
    reload_rows = [r for r in receipts.rows() if r != row]
    reloads = [receipts.latest(r) for r in reload_rows]
    assert [item.action for item in reloads] == ["reload"]
    assert reloads[0].parts == (PAGE,) and reloads[0].segment_offset == 0
    assert reloads[0].trigger == "access"
    reload_ref = book.latest_ref(PAGE_SPILL_RECEIPT, reload_rows[0])
    # a reload derives from the policy cell AND the spill it undoes
    assert {source.key for source, _ in book.edges_into(reload_ref)} == {
        policy.key, ref.key,
    }


# ---------------------------------------------- what is never spilled, or held
def test_only_cold_pages_spill_and_a_read_of_a_hot_page_never_does(monkeypatch):
    rows = populated()
    book = rows.book
    monkeypatch.setitem(page_lifecycle.PAGE_LIFECYCLE, COLD,
                        PageLifecycle(None, None, "source-closure"))
    monkeypatch.setitem(page_lifecycle.PAGE_LIFECYCLE, COLD2,
                        PageLifecycle(None, None, "ssa-lowering"))
    monkeypatch.setitem(page_lifecycle.PAGE_LIFECYCLE, HOT,
                        PageLifecycle(None, None, "emission"))
    for completed, expected in (
        ("topology-reduction", {COLD}),
        ("ssa-lowering:functions", {COLD}),
        ("ssa-lowering", {COLD, COLD2}),
        ("pre-native-repairs", {COLD, COLD2}),
        ("build", {COLD, COLD2, HOT}),
    ):
        assert set(page_lifecycle.cold_pages(
            tuple(dict.keys(book.pages)), completed,
        )) == expected
    page_lifecycle.spill_cold_pages(book, "pre-native-repairs")
    assert set(book.spilled_pages()) == {COLD, COLD2}
    for _ in range(3):                               # reads of a hot page
        assert book.page(HOT).latest(("s", 2)) == "b2"
        assert book.edges_into(rows.refs["b"][2])
    assert not dict.get(book.pages, HOT).spilled
    assert set(book.spilled_pages()) == {COLD, COLD2}
    # an unclassified page is never spilled
    book.page("unclassified").set(("x", 1), 0, "kept")
    page_lifecycle.spill_cold_pages(book, "build")
    assert not dict.get(book.pages, "unclassified").spilled


def test_the_pages_the_machinery_reads_are_never_spilled():
    rows = populated()
    book = rows.book
    book.spill_pages([COLD], trigger="explicit")
    names = [PAGE_SPILL_RECEIPT.name, COMPILE_POLICY.name,
             *PARTITIONED, "scope_registry"]
    assert all(never_spilled(book, name) for name in names)
    result = book.spill_pages(names, trigger="explicit")
    assert result.spilled == () and result.refused == ()
    assert not any(
        dict.get(book.pages, name).spilled for name in names
        if dict.get(book.pages, name) is not None
    )


def test_a_page_whose_tables_a_reader_holds_is_refused_not_spilled():
    rows = populated()
    book = rows.book
    held = book.pages[COLD].cells                    # a stale-table risk
    result = book.spill_pages([COLD], trigger="explicit")
    assert result.spilled == () and [name for name, _ in result.refused] == [COLD]
    assert "held" in result.refused[0][1]
    assert not dict.get(book.pages, COLD).spilled
    receipts = book.pages[PAGE_SPILL_RECEIPT.name]
    (row,) = receipts.rows()
    assert receipts.latest(row).action == "refuse"
    # asked again while still held: one receipt per holding, not per ask
    book.spill_pages([COLD], trigger="explicit")
    assert len(receipts.rows()) == 1
    del held
    result = book.spill_pages([COLD], trigger="explicit")
    assert result.spilled == (COLD,)


def test_facts_that_do_not_serialise_leave_the_page_in_ram_and_the_file_clean():
    rows = populated()
    book = rows.book
    rows.novel(COLD2, ("lambda", 0), lambda: None)   # cannot be pickled
    book.spill_pages([COLD], trigger="explicit")
    size = os.path.getsize(book._spill.path)
    result = book.spill_pages([COLD2], trigger="explicit")
    assert result.spilled == () and "unserialisable" in result.refused[0][1]
    assert not dict.get(book.pages, COLD2).spilled
    assert os.path.getsize(book._spill.path) == size
    assert book.page(COLD2).latest(("s", 1)) == "c1"
    # it is not asked again
    assert book.spill_pages([COLD2], trigger="explicit").refused == ()
    # and the page that did spill is intact
    assert book.page(COLD).latest(("s", 1)) == "a1"


# ----------------------------------------------------- borrow, copy, pickle
def test_a_borrowed_page_is_read_and_let_go_without_a_receipt():
    rows = populated()
    book = rows.book
    book.spill_pages([COLD], trigger="explicit")
    receipts = len(book.pages[PAGE_SPILL_RECEIPT.name].rows())
    clock = book.clock[0]
    with book.borrowed(COLD) as page:
        assert page.latest(("s", 5)) == "a5"
    assert dict.get(book.pages, COLD).spilled
    assert book.clock[0] == clock
    assert len(book.pages[PAGE_SPILL_RECEIPT.name].rows()) == receipts
    # a write while it is on loan keeps the page
    with book.borrowed(COLD) as page:
        page.set(("s", 9), 0, "written on loan")
    assert not dict.get(book.pages, COLD).spilled
    assert book.page(COLD).latest(("s", 9)) == "written on loan"
    assert book.page(COLD).latest(("s", 2)) == "a2"


def test_a_copy_or_pickle_of_a_spilled_book_is_the_whole_book():
    rows = populated()
    book = rows.book
    expected = {
        name: raw_state(page)
        for name, page in dict.items(book.pages)
    }
    book.spill_pages([COLD, COLD2], trigger="explicit")
    for clone in (pickle.loads(pickle.dumps(book)), copy.deepcopy(book)):
        assert clone.spilled_pages() == ()
        assert clone._spill is None
        for name in (COLD, HOT, COLD2):
            assert raw_state(dict.get(clone.pages, name)) == expected[name]
        assert clone.pages.book is clone


def test_the_spill_file_goes_with_the_book():
    import gc

    rows = populated()
    book = rows.book
    book.spill_pages([COLD], trigger="explicit")
    path = book._spill.path
    assert os.path.exists(path) and os.path.exists(path + ".index")
    index = open(path + ".index", encoding="utf-8").read().split("\n")
    assert index[0].split("\t")[:3] == [PAGE, COLD, "0"]
    del book, rows
    gc.collect()
    assert not os.path.exists(path) and not os.path.exists(path + ".index")


# ---------------------------------------------------------- the regulator
def regulator_book(monkeypatch):
    rows = populated()
    monkeypatch.setitem(page_lifecycle.PAGE_LIFECYCLE, COLD,
                        PageLifecycle(None, None, "source-closure"))
    monkeypatch.setitem(page_lifecycle.PAGE_LIFECYCLE, COLD2,
                        PageLifecycle(None, None, "ssa-lowering"))
    monkeypatch.setitem(page_lifecycle.PAGE_LIFECYCLE, HOT,
                        PageLifecycle(None, None, "emission"))
    return rows


def test_the_policy_spills_what_each_completed_stage_made_cold(monkeypatch):
    rows = regulator_book(monkeypatch)
    book = rows.book
    regulator = regulation.MemoryRegulator(book, None, spill_policy=True)
    regulator.boundary("compile: begin")
    assert book.spilled_pages() == ()                 # nothing completed yet
    regulator.boundary("ssa-source: topology reduced, program ABI reattached")
    assert book.spilled_pages() == (COLD,)
    regulator.boundary("ssa-source: repository SSA lowered")
    assert set(book.spilled_pages()) == {COLD, COLD2}
    regulator.boundary("compile: end")
    assert set(book.spilled_pages()) == {COLD, COLD2}   # hot_b waits for 'build'
    receipts = book.pages[PAGE_SPILL_RECEIPT.name]
    triggers = {receipts.latest(row).trigger for row in receipts.rows()}
    assert triggers == {"policy"}
    policy = book.latest_ref(COMPILE_POLICY, ("spill_cold_pages",))
    assert book.pages[COMPILE_POLICY.name].latest(("spill_cold_pages",)) == "true"
    for row in receipts.rows():
        ref = book.latest_ref(PAGE_SPILL_RECEIPT, row)
        assert policy in [source for source, _ in book.edges_into(ref)]
    # no budget was declared, so no memory_release_receipt exists
    assert MEMORY_RELEASE_RECEIPT.name not in book.pages


def test_a_boundary_over_budget_spills_cold_pages_as_a_release(monkeypatch):
    rows = regulator_book(monkeypatch)
    book = rows.book
    def read():     # over budget until the cold pages are out of RAM
        return 10**9 if not book.spilled_pages() else 50

    monkeypatch.setattr(regulation, "process_rss_bytes", read)
    monkeypatch.setattr(regulation, "RELEASE_ORDER", (
        ("dependency_level_memo", lambda: 0),
        ("cold_pages", regulation._spill_cold_pages),
        ("cyclic_garbage", lambda: 0),
    ))
    regulator = regulation.MemoryRegulator(book, 100)
    token = regulation._ACTIVE.set(regulator)
    try:
        regulator.boundary("ssa-source: topology reduced, program ABI reattached")
    finally:
        regulation._ACTIVE.reset(token)
    assert book.spilled_pages() == (COLD,)
    release = book.pages[MEMORY_RELEASE_RECEIPT.name]
    rows_ = release.rows()
    assert [release.latest(row).item for row in rows_] == [
        "dependency_level_memo", "cold_pages",
    ]
    assert release.latest(rows_[1]).released > 0       # cells dropped
    (spill_row,) = book.pages[PAGE_SPILL_RECEIPT.name].rows()
    spill = book.pages[PAGE_SPILL_RECEIPT.name].latest(spill_row)
    assert spill.trigger == "budget" and spill.boundary.startswith("ssa-source")
    # derived from the memory_budget_bytes cell, which states the budget
    budget = book.latest_ref(COMPILE_POLICY, ("memory_budget_bytes",))
    ref = book.latest_ref(PAGE_SPILL_RECEIPT, spill_row)
    assert [source for source, _ in book.edges_into(ref)] == [budget]
    assert book.pages[COMPILE_POLICY.name].latest(("memory_budget_bytes",)) == "100"


def test_under_budget_and_without_the_policy_nothing_spills(monkeypatch):
    rows = regulator_book(monkeypatch)
    book = rows.book
    monkeypatch.setattr(regulation, "process_rss_bytes", lambda: 10)
    regulator = regulation.MemoryRegulator(book, 10**12)
    regulator.boundary("ssa-source: topology reduced, program ABI reattached")
    regulator.boundary("compile: end")
    assert book.spilled_pages() == () and book._spill is None
    assert PAGE_SPILL_RECEIPT.name not in book.pages


def test_a_compile_resuming_a_callers_book_does_not_spill_it(monkeypatch):
    rows = regulator_book(monkeypatch)
    book = rows.book
    regulator = regulation.MemoryRegulator(
        book, None, spill_policy=True, owns_book=False,
    )
    regulator.boundary("compile: end")
    assert book.spilled_pages() == ()


def test_begin_regulation_reads_the_contract_policy():
    from src.compiler.work_contract import (
        WorkContract, active_contract, set_active_contract,
    )
    import dataclasses

    base = active_contract()
    assert base.spill_cold_pages is False
    set_active_contract(dataclasses.replace(base, spill_cold_pages=True))
    try:
        book = IdentityBook()
        regulator, token, _ = regulation.begin_regulation(book)
        try:
            assert regulator.spill_policy is True
        finally:
            regulation._ACTIVE.reset(token)
    finally:
        set_active_contract(None)
    for bad in (1, "yes", None):
        with pytest.raises(ValueError):
            WorkContract(
                "t", register_reuse=False, inexact_identities=False,
                contract_multiply_add=False, spill_cold_pages=bad,
            )


def test_environment_policy_overrides_without_renaming_the_contract(monkeypatch):
    from src.compiler.work_contract import PRESETS, active_contract, set_active_contract

    set_active_contract(None)
    for variable in ("TURING_WORK_CONTRACT", "TURING_POW_INEXACT",
                     "TURING_FMA_CONTRACT", "TURING_MEMORY_BUDGET_BYTES",
                     "TURING_SPILL_COLD_PAGES"):
        monkeypatch.delenv(variable, raising=False)
    assert active_contract() is PRESETS["develop"]
    monkeypatch.setenv("TURING_SPILL_COLD_PAGES", "1")
    contract = active_contract()
    assert contract.spill_cold_pages is True and contract.name == "develop"
    monkeypatch.setenv("TURING_SPILL_COLD_PAGES", "0")
    assert active_contract().spill_cold_pages is False


# ------------------------------------------------- the declared classification
def test_every_declared_page_is_classified_and_the_table_is_consistent():
    import src.compiler.concordance_declarations  # noqa: F401  (declares)
    from src.compiler.identity_concordance import REGISTRY
    from src.compiler.shell_telemetry import COMPILE_STAGES

    assert page_lifecycle.SPILL_STAGES == tuple(
        item for stage in COMPILE_STAGES
        for item in (("ssa-lowering:functions", stage.key)
                     if stage.key == "ssa-lowering" else (stage.key,))
    )
    missing = sorted(set(REGISTRY.pages) - set(page_lifecycle.PAGE_LIFECYCLE))
    assert missing == []
    index = {key: i for i, key in enumerate(page_lifecycle.SPILL_STAGES)}
    for name, entry in page_lifecycle.PAGE_LIFECYCLE.items():
        assert name in REGISTRY.pages, name
        for stage in (entry.written_in, entry.read_in, entry.cold_after):
            assert stage is None or stage in index, (name, stage)
        if entry.cold_after is not None:
            touched = [s for s in (entry.written_in, entry.read_in) if s]
            assert all(index[s] <= index[entry.cold_after] for s in touched), name
        else:
            assert never_spilled(IdentityBook(), name), name
    # the pages the spill machinery itself needs are the ones never spilled
    never = {n for n, e in page_lifecycle.PAGE_LIFECYCLE.items() if e.cold_after is None}
    assert never == {
        "compile_policy", "memory_release_receipt", "page_spill_receipt",
        "scope_registry", "scope_origin", "shape_scope_function", *PARTITIONED,
    }
    # every boundary the lowering declares completes a known stage
    for label, stage in page_lifecycle.BOUNDARY_COMPLETES.items():
        assert stage in index, label


def test_the_boundary_labels_are_the_ones_the_lowering_emits():
    import re

    text = open(
        os.path.join(os.path.dirname(__file__), "..", "src", "compiler",
                     "fortran_c_shell.py"), encoding="utf-8",
    ).read()
    emitted = set(re.findall(r'stage_(?:boundary|end)\(\s*"([^"]+)"', text))
    declared = set(page_lifecycle.BOUNDARY_COMPLETES) - {"compile: end"}
    assert declared == emitted


# ------------------------------------------------ forks reading a spilled origin
def test_a_copy_on_read_fork_of_a_spilled_origin_reads_and_materialises():
    import networkx as nx

    from src.common.tensors.topological_reducer import fork_read_scope
    from src.compiler.concordance_declarations import (
        LEXICAL_READ_BINDING, READ_SCOPE_FORK, SYNTHESIZED_NO_SOURCE,
    )
    from src.compiler.identity_concordance import (
        Unsourced, begin_identity_book, end_identity_book,
    )

    class Graph:
        def __init__(self, nodes, scope):
            self.G = nx.DiGraph()
            self.G.add_nodes_from(nodes)
            self.G.graph["lexical_read_scope"] = scope

    book, token = begin_identity_book()
    try:
        source = book.mint_scope("fn", READ_SCOPE_FORK)
        for consumer, fact in ((1, "a"), (2, "b"), (3, "c")):
            book.post(
                LEXICAL_READ_BINDING, (source, consumer, "arg0", 0), fact,
                stage=READ_SCOPE_FORK,
                provenance=Unsourced(SYNTHESIZED_NO_SOURCE), mode=Mode.CONCORD,
            )
        graph = Graph((1, 2, 3), source)
        fork_read_scope(graph, "test", source_graph=None)
        forked = tuple(graph.G.graph["lexical_read_scope"])
        page = book.page(LEXICAL_READ_BINDING)
        row = (forked, 2, "arg0", 0)
        assert page.latest(row) == "b"
        before = list(page.scope_rows(forked))

        book.spill_pages([LEXICAL_READ_BINDING.name], trigger="explicit")
        assert dict.get(book.pages, LEXICAL_READ_BINDING.name).spilled
        # a read through the chain reaches the spilled origin page
        assert page.latest(row) == "b"
        assert list(page.scope_rows(forked)) == before
        book.spill_pages([LEXICAL_READ_BINDING.name], trigger="explicit")
        # a write under the fork materialises the row from the origin
        page.revise(row, "b2")
        assert page.history(row)[0] == (0, "b") and page.latest(row) == "b2"
        assert book.page(LEXICAL_READ_BINDING).latest((source, 2, "arg0", 0)) == "b"
    finally:
        end_identity_book(token)


# ---------------------------------------------------- the guarded page table
def test_a_partial_edge_page_is_never_handed_out_incomplete():
    rows = populated()
    book = rows.book

    def owned(page):                # edge rows whose target cell is on cold_c
        return sum(1 for key in page.cells if key[0][0][0] == COLD2)

    live = dict.get(book.pages, "concordance_edge")
    held = owned(live)
    assert held == 12 and type(book.pages).__name__ == "_PageTable"
    # every route a caller has to the page hands it whole
    for route in (
        lambda: book.pages["concordance_edge"],
        lambda: book.pages.get("concordance_edge"),
        lambda: book.page("concordance_edge"),
        lambda: dict(book.pages.items())["concordance_edge"],
        lambda: {p.name: p for p in book.pages.values()}["concordance_edge"],
    ):
        book.spill_pages([COLD2], trigger="explicit")
        assert type(book.pages).__name__ == "_PartialPageTable"
        assert owned(dict.get(book.pages, "concordance_edge")) == 0
        handed = route()
        assert handed is live and owned(handed) == held
        assert type(book.pages).__name__ == "_PageTable"      # whole again
        book.page(COLD2).latest(("s", 0))                     # (and the page)


def test_the_build_hook_spills_what_no_stage_reads_after_it(monkeypatch):
    from types import SimpleNamespace

    from src.compiler import fortran_c_shell
    from src.compiler.work_contract import (
        active_contract, set_active_contract,
    )
    import dataclasses

    rows = regulator_book(monkeypatch)
    book = rows.book
    module = SimpleNamespace(metadata={"identity_book": book})
    fortran_c_shell._spill_book_after_build(module)           # no policy
    assert book.spilled_pages() == ()
    set_active_contract(
        dataclasses.replace(active_contract(), spill_cold_pages=True))
    try:
        fortran_c_shell._spill_book_after_build(module)
        assert set(book.spilled_pages()) == {COLD, COLD2, HOT}
        fortran_c_shell._spill_book_after_build(SimpleNamespace(metadata={}))
        fortran_c_shell._spill_book_after_build(SimpleNamespace())
    finally:
        set_active_contract(None)


# ------------------------------------------------ randomised twin comparison
PAGE_NAMES = (COLD, HOT, COLD2)


def _apply(rows, refs, op):
    """Apply one hand-posted operation to a book; returns what it read."""
    kind = op[0]
    if kind == "novel":
        _, name, index, fact = op
        refs.append(rows.novel(name, ("r", index), fact))
    elif kind == "derive":
        _, name, index, fact, picks = op
        sources = [refs[pick % len(refs)] for pick in picks]
        refs.append(rows.derive(name, ("r", index), fact, *dict.fromkeys(sources)))
    elif kind == "raw":
        _, name, pick, fact = op
        row = refs[pick % len(refs)]
        rows.book.page(row.page.name).revise(row.row, fact)
    elif kind == "latest":
        ref = refs[op[1] % len(refs)]
        return rows.book.page(ref.page.name).latest(ref.row)
    elif kind == "history":
        ref = refs[op[1] % len(refs)]
        return rows.book.page(ref.page.name).history(ref.row)
    elif kind == "into":
        ref = refs[op[1] % len(refs)]
        return tuple((s.key, st.name) for s, st in rows.book.edges_into(ref))
    elif kind == "out":
        ref = refs[op[1] % len(refs)]
        return tuple((t.key, st.name) for t, st in rows.book.edges_out_of(ref))
    return None


def _random_ops(seed, count):
    import random

    rng = random.Random(seed)
    ops, index = [], 0
    for _ in range(count):
        roll = rng.random()
        name = rng.choice(PAGE_NAMES)
        if roll < 0.15 or index < 3:
            ops.append(("novel", name, index, f"n{index}"))
            index += 1
        elif roll < 0.5:
            picks = [rng.randrange(10**6) for _ in range(rng.randint(1, 3))]
            ops.append(("derive", name, index, f"d{index}", picks))
            index += 1
        elif roll < 0.6:
            ops.append(("raw", name, rng.randrange(10**6), f"v{rng.randrange(99)}"))
        else:
            ops.append((rng.choice(("latest", "history", "into", "out")),
                        rng.randrange(10**6)))
    return ops


def _spill_ops(seed, count):
    import random

    rng = random.Random(seed * 7 + 1)
    out = []
    for _ in range(count):
        roll = rng.random()
        if roll < 0.45:
            out.append(("spill", rng.sample(PAGE_NAMES, rng.randint(1, 3))))
        elif roll < 0.6:
            out.append(("reload_page", rng.choice(PAGE_NAMES)))
        elif roll < 0.7:
            out.append(("reload_part", rng.choice(PAGE_NAMES),
                        rng.choice(PARTITIONED)))
        elif roll < 0.75:
            out.append(("reload_all",))
        elif roll < 0.85:
            out.append(("borrow", rng.choice(PAGE_NAMES)))
        else:
            out.append(("none",))
    return out


def _canonical(book, keep):
    """Per page: the rows in order, with each cell's fact and the dense rank of
    its stamp (the receipts the spilled book posted tick its clock, so the
    absolute readings differ and their ORDER is what must agree)."""
    book.reload_all("test")
    out = {}
    for name, page in dict.items(book.pages):
        if name in (COMPILE_POLICY.name, PAGE_SPILL_RECEIPT.name):
            continue
        d = page.__dict__
        items = [
            (key, fact, d["stamps"][key]) for key, fact in d["cells"].items()
            if keep(key)
        ]
        ranks = {
            stamp: rank
            for rank, stamp in enumerate(sorted({item[2] for item in items}))
        }
        out[name] = [(key, fact, ranks[stamp]) for key, fact, stamp in items]
    return out


@pytest.mark.parametrize("seed", range(4))
def test_a_book_spilled_and_reloaded_at_random_is_the_book_never_spilled(seed):
    plain, spilled = Rows(), Rows()
    plain_refs, spilled_refs = [], []
    ops = _random_ops(seed, 80)
    spill_ops = _spill_ops(seed, 80)
    for op, extra in zip(ops, spill_ops):
        # (the same operation on both books, and what it read must agree)
        assert _apply(spilled, spilled_refs, op) == _apply(plain, plain_refs, op), op
        book = spilled.book
        kind = extra[0]
        if kind == "spill":
            book.spill_pages(extra[1], trigger="explicit")
        elif kind == "reload_page":
            book._reload(extra[1], (PAGE,), "test")
        elif kind == "reload_part":
            book._reload(extra[1], (extra[2],), "test")
        elif kind == "reload_all":
            book.reload_all("test")
        elif kind == "borrow":
            with book.borrowed(extra[1]) as page:
                assert page is None or page.rows() is not None

    # (the run really did spill, and was really read back)
    assert spilled.book._spill.spills >= 3 and spilled.book._spill.reloads >= 3

    def keep(key):
        return not is_receipt_row(key[0])

    assert _canonical(spilled.book, keep) == _canonical(plain.book, keep)
    # the logs agree too, receipts aside
    for level in (IdentityLogLevel.FULL, IdentityLogLevel.FACTS):
        a = sections(iter_identity_book_lines(plain.book, level))
        b = sections(iter_identity_book_lines(spilled.book, level))
        for name, section in a.items():
            mine = b[name]
            if name in PARTITIONED:
                def kept(lines):
                    return [l for l in lines if not l.startswith("  (")
                            and "page_spill_receipt" not in l
                            and "compile_policy" not in l]
                assert kept(mine["rows"]) == kept(section["rows"]), name
            elif name != "<unsourced tally>":
                assert mine["rows"] == section["rows"], name
