"""Copy-on-read scopes: ``fork_read_scope`` posts no copied rows.

Pure data-structure tests on a real ``IdentityBook`` with hand-posted rows.
Nothing here lowers, compiles, or runs a probe: the forks are made by
``fork_read_scope`` over a bare networkx graph (it reads only the graph's
metadata and node set), and the eager behaviour it replaced is restated in
``_eager_fork`` (the old copy loop, row for row) so the edges a row written in
a fork has can be compared with the edges the copy used to post.
"""

import networkx as nx
import pytest

from src.common.tensors.topological_reducer import fork_read_scope
from src.compiler.concordance_declarations import (
    CONSUMER_OPERAND, LEXICAL_READ_BINDING, OPERAND_POSITION,
    READ_SCOPE_FORK, ROW_PROJECTED_OUT_OF_SCOPE, SCOPE_ORIGIN, SCOPE_REGISTRY,
    SYNTHESIZED_NO_SOURCE, ScopeFork,
)
from src.compiler.identity_concordance import (
    RAW_PRIMITIVE, Derived, Mode, ProjectedScopeRead, Ref, Unresolved,
    Unsourced, begin_identity_book, end_identity_book,
)

LRB = LEXICAL_READ_BINDING
OPS = CONSUMER_OPERAND


class _Graph:
    """What ``fork_read_scope`` reads of a graph: ``G.graph`` and ``G``'s
    nodes."""

    def __init__(self, nodes, scope):
        self.G = nx.DiGraph()
        self.G.add_nodes_from(nodes)
        self.G.graph["lexical_read_scope"] = scope


@pytest.fixture
def book():
    book, token = begin_identity_book()
    try:
        yield book
    finally:
        end_identity_book(token)


def _root(book, page, row, fact):
    return book.post(
        page, row, fact, stage=READ_SCOPE_FORK,
        provenance=Unsourced(SYNTHESIZED_NO_SOURCE), mode=Mode.CONCORD,
    )


def _source_rows(book, scope):
    """Four binding rows, one operand-structure row and a raw (undeclared)
    page's row, all under ``scope``."""
    for consumer, fact in ((1, "a"), (2, "b"), (3, "c"), ("return", "r")):
        _root(book, LRB, (scope, consumer, "arg0", 0), fact)
    _root(book, OPS, (scope, 2, 1), ((1, "arg0", 0),))
    book.page("raw_rows").set((scope, "k"), 0, 5)


def _fork(book, source, nodes=(1, 2, 3), source_graph=None, cause="test"):
    graph = _Graph(nodes, source)
    fork_read_scope(graph, cause, source_graph=source_graph)
    return tuple(graph.G.graph["lexical_read_scope"])


def _eager_fork(book, source, cause="test"):
    """The copy loop ``fork_read_scope`` used to run, for a whole-graph copy:
    every page's rows under ``source`` re-posted under a minted scope, each
    DERIVED from the cell it copies (stage ``read_scope_fork``)."""
    forked = book.mint_scope(f"{source[0]}|fork", READ_SCOPE_FORK)
    skip = {
        "planner_specialization", "planner_tensor_descriptor", "scope_origin",
        *book.registry.private_pages,
    }
    for page in tuple(book.pages.values()):
        if page.name in skip:
            continue
        declared = book.registry.pages.get(page.name)
        for row in page.materialised_scope_rows(source):
            fact = page.latest(row)
            if fact is None:
                continue
            if declared is None:
                page.set((forked, *row[1:]), 0, fact)
                continue
            book.post(
                declared, (forked, *row[1:]), fact, stage=READ_SCOPE_FORK,
                provenance=Derived((book.latest_ref(declared, row),)),
                mode=Mode.CONCORD,
            )
    book.post(
        SCOPE_ORIGIN, (forked,), ScopeFork(source, cause),
        stage=READ_SCOPE_FORK,
        provenance=Derived((book.latest_ref(SCOPE_REGISTRY, source),)),
        mode=Mode.CONCORD,
    )
    return forked


def _edges(book, page, row, column):
    return {
        (source.key, stage.name)
        for source, stage in book.edges_into(Ref(page, row, column))
    }


def _cell_count(book, name):
    page = book.pages.get(name)
    return 0 if page is None else len(page.cells)


# ---------------------------------------------------------------- reads
def test_fork_posts_no_rows_and_reads_resolve_through_the_origin(book):
    source = book.mint_scope("fn", READ_SCOPE_FORK)
    _source_rows(book, source)
    page = book.page(LRB)
    before = {name: _cell_count(book, name) for name in book.pages}

    forked = _fork(book, source)

    # No row of any page was posted under the fork: only the scope's own
    # registry / origin cells (and their edges) exist.
    for name in ("lexical_read_binding", "consumer_operand", "raw_rows"):
        assert _cell_count(book, name) == before[name]
        assert book.pages[name].materialised_scope_rows(forked) == ()
    origin = book.page(SCOPE_ORIGIN).latest((forked,))
    assert origin.copy_on_read is True and origin.source_scope == source
    # Reads answer as the source's rows did, and are not counted as writes.
    assert page.latest((forked, 2, "arg0", 0)) == "b"
    assert page.latest_column((forked, 2, "arg0", 0)) == 0
    assert page.history((forked, 2, "arg0", 0)) == ((0, "b"),)
    assert page.spans((forked, 2, "arg0", 0)) == ((0, 0, "b"),)
    assert book.page(OPS).latest((forked, 2, 1)) == ((1, "arg0", 0),)
    assert book.page("raw_rows").latest((forked, "k")) == 5
    assert page.cell((forked, 2, "arg0", 0), 0) == "b"
    assert page.latest((forked, 99, "arg0", 0)) is None
    for name in ("lexical_read_binding", "consumer_operand", "raw_rows"):
        assert _cell_count(book, name) == before[name]
    # The source is untouched, and an unmaterialised row is not a row.
    assert len(page.rows()) == 4
    assert all(row[0] == source for row in page.rows())


def test_pages_that_are_per_copy_or_private_are_not_read_through(book):
    source = book.mint_scope("fn", READ_SCOPE_FORK)
    book.page("planner_specialization").set((source, "p"), 0, "literal")
    forked = _fork(book, source)
    # The planner's specialization rows are per COPY: never inherited.
    assert book.page("planner_specialization").latest((forked, "p")) is None
    assert book.page("planner_specialization").scope_rows(forked) == ()
    # The scope's origin is its own fact, not the source's.
    assert book.page(SCOPE_ORIGIN).latest((forked,)).source_scope == source


def test_a_withdrawn_row_is_not_a_row_of_the_fork(book):
    source = book.mint_scope("fn", READ_SCOPE_FORK)
    _source_rows(book, source)
    page = book.page(LRB)
    page.revise((source, 3, "arg0", 0), None)   # the source withdrew it
    forked = _fork(book, source)
    assert page.latest((forked, 3, "arg0", 0)) is None
    assert page.history((forked, 3, "arg0", 0)) == ()
    assert (forked, 3, "arg0", 0) not in page.scope_rows(forked)
    assert page.scope_row_count(forked) == 3


# ------------------------------------------------------------ the shadow
def test_a_write_in_the_fork_shadows_the_origin_for_that_row_only(book):
    source = book.mint_scope("fn", READ_SCOPE_FORK)
    _source_rows(book, source)
    page = book.page(LRB)
    forked = _fork(book, source)
    row = (forked, 2, "arg0", 0)

    page.revise(row, "B")

    assert page.history(row) == ((0, "b"), (1, "B"))
    assert page.latest(row) == "B"
    assert page.latest((source, 2, "arg0", 0)) == "b"
    # Only the written row has cells; the rest still read through.
    assert page.materialised_scope_rows(forked) == (row,)
    assert page.latest((forked, 1, "arg0", 0)) == "a"
    assert (forked, 1, "arg0", 0) not in page.materialised_scope_rows(forked)
    # A withdrawal is a write too.
    page.revise((forked, 1, "arg0", 0), None)
    assert page.latest((forked, 1, "arg0", 0)) is None
    assert page.latest((source, 1, "arg0", 0)) == "a"
    assert page.history((forked, 1, "arg0", 0)) == ((0, "a"), (1, None))


def test_enumeration_is_origin_rows_union_fork_rows_minus_shadowed(book):
    source = book.mint_scope("fn", READ_SCOPE_FORK)
    _source_rows(book, source)
    page = book.page(LRB)
    forked = _fork(book, source)
    page.revise((forked, 2, "arg0", 0), "B")           # shadows an origin row
    page.revise((forked, 7, "arg0", 0), "new")          # a row only the fork has
    # A row the ORIGIN gains after the fork is not the fork's.
    page.revise((source, 8, "arg0", 0), "late")

    resolved = page.scope_rows(forked)
    expected = [
        (forked, 1, "arg0", 0), (forked, 2, "arg0", 0),
        (forked, 3, "arg0", 0), (forked, "return", "arg0", 0),
        (forked, 7, "arg0", 0),
    ]
    assert list(resolved) == expected          # origin order, then fork-only
    assert page.scope_row_count(forked) == 5
    assert len(set(resolved)) == len(resolved)
    assert {row: page.latest(row) for row in resolved} == {
        (forked, 1, "arg0", 0): "a", (forked, 2, "arg0", 0): "B",
        (forked, 3, "arg0", 0): "c", (forked, "return", "arg0", 0): "r",
        (forked, 7, "arg0", 0): "new",
    }
    # The book's own cells are only what was written.
    assert page.materialised_scope_rows(forked) == (
        (forked, 2, "arg0", 0), (forked, 7, "arg0", 0),
    )
    # The mapping view (a pass keeping its working state on a page) agrees.
    mapping = page.mapping(forked)
    assert sorted(map(str, mapping)) == sorted(map(str, (
        1, 2, 3, 7, "return",
    )))
    # ``rows()`` is the materialised book.
    assert {row for row in page.rows() if row[0] == forked} == {
        (forked, 2, "arg0", 0), (forked, 7, "arg0", 0),
    }


def test_alias_bindings_see_a_forks_inherited_rows(book):
    source = book.mint_scope("fn", READ_SCOPE_FORK)
    page = book.page("planning_alias")
    page.bind_alias(source, 5, 6)
    page.bind_alias(source, 7, 8)
    forked = _fork(book, source)
    assert page.alias_bindings(forked) == {5: 6, 7: 8}
    assert page.resolve_alias(forked, 5) == 6
    page.bind_alias(forked, 5, 9)
    assert page.alias_bindings(forked) == {5: 9, 7: 8}
    assert page.alias_bindings(source) == {5: 6, 7: 8}


# ------------------------------------------------------------- snapshots
def test_the_fork_is_a_snapshot_of_its_source(book):
    source = book.mint_scope("fn", READ_SCOPE_FORK)
    _source_rows(book, source)
    page = book.page(LRB)
    forked = _fork(book, source)
    # The source is rewritten after the fork: a revision, a withdrawal ...
    page.revise((source, 1, "arg0", 0), "A2")
    page.revise((source, 2, "arg0", 0), None)
    # ... and an in-place overwrite of a cell the fork reads through to.
    book.page("raw_rows").set((source, "k"), 0, 99)

    assert page.latest((forked, 1, "arg0", 0)) == "a"
    assert page.latest((forked, 2, "arg0", 0)) == "b"
    assert book.page("raw_rows").latest((forked, "k")) == 5
    assert book.page("raw_rows").latest((source, "k")) == 99
    assert page.latest((source, 1, "arg0", 0)) == "A2"
    # The overwrite took the fork's snapshot of that row (and only it).
    assert book.page("raw_rows").materialised_scope_rows(forked) == (
        (forked, "k"),
    )
    assert page.materialised_scope_rows(forked) == ()


def test_a_fork_of_a_fork_chains_and_each_hop_is_its_own_snapshot(book):
    root = book.mint_scope("fn", READ_SCOPE_FORK)
    _source_rows(book, root)
    page = book.page(LRB)
    parent = _fork(book, root)
    page.revise((parent, 2, "arg0", 0), "B")             # the parent's write
    child = _fork(book, parent)                          # fork of a fork
    # After the child: the root and the parent move on.
    page.revise((root, 1, "arg0", 0), "A2")
    page.revise((parent, 3, "arg0", 0), "C2")
    page.revise((parent, 1, "arg0", 0), "A3")

    assert page.latest((child, 1, "arg0", 0)) == "a"      # root, as of parent's fork
    assert page.latest((child, 2, "arg0", 0)) == "B"      # parent's own cell
    assert page.latest((child, 3, "arg0", 0)) == "c"      # root, parent not yet written
    assert page.latest((child, "return", "arg0", 0)) == "r"
    assert page.history((child, 2, "arg0", 0)) == ((0, "B"),)
    assert list(page.scope_rows(child)) == [
        (child, 1, "arg0", 0), (child, 2, "arg0", 0), (child, 3, "arg0", 0),
        (child, "return", "arg0", 0),
    ]
    assert page.materialised_scope_rows(child) == ()
    # The parent's own view is its snapshot of the root plus its writes.
    assert page.latest((parent, 1, "arg0", 0)) == "A3"
    assert page.latest((parent, "return", "arg0", 0)) == "r"
    assert page.latest((root, 3, "arg0", 0)) == "c"


def test_writing_a_fork_of_a_fork_materialises_the_parent_row_first(book):
    root = book.mint_scope("fn", READ_SCOPE_FORK)
    _source_rows(book, root)
    page = book.page(LRB)
    parent = _fork(book, root)
    child = _fork(book, parent)
    row = (child, 1, "arg0", 0)
    assert page.materialised_scope_rows(parent) == ()

    page.revise(row, "X")

    # child cell 0 derives from the PARENT's cell, which derives from the
    # root's: the chain the eager copies made, one hop each.
    assert page.history(row) == ((0, "a"), (1, "X"))
    assert page.materialised_scope_rows(parent) == ((parent, 1, "arg0", 0),)
    assert _edges(book, LRB, row, 0) == {
        (("lexical_read_binding", (parent, 1, "arg0", 0), 0),
         READ_SCOPE_FORK.name),
    }
    assert _edges(book, LRB, (parent, 1, "arg0", 0), 0) == {
        (("lexical_read_binding", (root, 1, "arg0", 0), 0),
         READ_SCOPE_FORK.name),
    }
    # Other rows of both forks are still unwritten.
    assert page.materialised_scope_rows(child) == (row,)


# ----------------------------------------------------- edges, as eager
def _scenario(book, fork):
    """The same rows, fork, and writes against either kind of fork."""
    source = book.mint_scope("fn", READ_SCOPE_FORK)
    _source_rows(book, source)
    forked = fork(book, source)
    page = book.page(LRB)
    cause = _root(book, LRB, (source, "cause", "arg0", 0), "cause")
    # A REVISE derived from a cell newer than the fork (what ``_set_operands``
    # does on a fold), a CONCORD repeating the fact with a further source,
    # a withdrawal, and a row only the fork has.
    book.post(
        LRB, (forked, 1, "arg0", 0), "a2", stage=OPERAND_POSITION,
        provenance=Derived((cause,)), mode=Mode.REVISE,
    )
    book.post(
        LRB, (forked, 2, "arg0", 0), "b", stage=OPERAND_POSITION,
        provenance=Derived((cause,)), mode=Mode.CONCORD,
    )
    page.revise((forked, 3, "arg0", 0), None)
    book.post(
        LRB, (forked, 9, "arg0", 0), "n", stage=OPERAND_POSITION,
        provenance=Derived((cause,)), mode=Mode.REVISE,
    )
    book.page("raw_rows").revise((forked, "k"), 6)
    return source, forked


def test_edges_of_rows_written_in_the_fork_are_the_edges_the_copy_posted():
    lazy_book, token = begin_identity_book()
    try:
        _src_a, lazy = _scenario(lazy_book, lambda b, s: _fork(b, s))
    finally:
        end_identity_book(token)
    eager_book, token = begin_identity_book()
    try:
        _src_b, eager = _scenario(eager_book, _eager_fork)
    finally:
        end_identity_book(token)

    assert lazy == eager       # both minted the same scope: comparable keys
    for name, declared in (("lexical_read_binding", LRB),
                           ("consumer_operand", OPS),
                           ("raw_rows", None)):
        lazy_page, eager_page = lazy_book.pages[name], eager_book.pages[name]
        for row in eager_page.materialised_scope_rows(eager):
            # Same history, whether the row was copied or read through.
            assert lazy_page.history(row) == eager_page.history(row), row
            assert lazy_page.latest(row) == eager_page.latest(row), row
            if declared is None:
                continue
            # The edges of EVERY cell of the row, written or not.
            for column, _fact in eager_page.history(row):
                assert _edges(lazy_book, declared, row, column) == _edges(
                    eager_book, declared, row, column), (row, column)
        # The fork reads the same rows the eager copy listed.
        assert set(lazy_page.scope_rows(lazy)) == set(
            eager_page.scope_rows(eager)
        )
        assert lazy_page.scope_row_count(lazy) == eager_page.scope_row_count(
            eager
        )
    # What the lazy fork never touched is not a row of the book at all.
    assert len(lazy_book.page(OPS).materialised_scope_rows(lazy)) == 0
    assert len(eager_book.page(OPS).materialised_scope_rows(eager)) == 1
    # The raw page's copy is tagged exactly as the eager raw copy was.
    def tags(book, scope):
        return sorted(
            (name, row, reason.name, stage.name)
            for name, row, reason, stage in book.unsourced_rows()
            if row[0] == scope and name == "raw_rows"
        )

    # (The row's copy and its revision are both raw writes: tagged alike.)
    assert tags(lazy_book, lazy) == tags(eager_book, eager) == [(
        "raw_rows", (lazy, "k"), RAW_PRIMITIVE.name, "raw_primitive",
    )]


def test_unmaterialised_cells_report_the_edge_they_will_have(book):
    source = book.mint_scope("fn", READ_SCOPE_FORK)
    _source_rows(book, source)
    page = book.page(LRB)
    forked = _fork(book, source)
    row = (forked, 2, "arg0", 0)
    cell = book.latest_ref(LRB, row)
    assert cell == Ref(LRB, row, 0)
    before = _cell_count(book, "lexical_read_binding")

    # Asking for the cell, its edges and its stamp writes nothing ...
    assert _edges(book, LRB, row, 0) == {
        (("lexical_read_binding", (source, 2, "arg0", 0), 0),
         READ_SCOPE_FORK.name),
    }
    assert book.stamp_of(cell) == page.stamp_at(row, 0)
    origin_cell = book.latest_ref(SCOPE_ORIGIN, (forked,))
    assert book.stamp_of(cell) == book.stamp_of(origin_cell)
    assert _cell_count(book, "lexical_read_binding") == before
    # ... naming it as a source writes it (it must exist to be derived from),
    # with the edge the copy had.
    derived = book.post(
        LRB, (forked, 50, "arg0", 0), "from-fork", stage=OPERAND_POSITION,
        provenance=Derived((cell,)), mode=Mode.CONCORD,
    )
    assert page.materialised_scope_rows(forked) == (row, (forked, 50, "arg0", 0))
    assert _cell_count(book, "lexical_read_binding") == before + 2
    assert (cell.key, OPERAND_POSITION.name) in {
        (source_ref.key, stage.name)
        for source_ref, stage in book.edges_into(derived)
    }
    assert _edges(book, LRB, row, 0) == {
        (("lexical_read_binding", (source, 2, "arg0", 0), 0),
         READ_SCOPE_FORK.name),
    }


def test_materialising_a_row_ticks_nothing_and_keeps_the_forks_stamp(book):
    source = book.mint_scope("fn", READ_SCOPE_FORK)
    _source_rows(book, source)
    page = book.page(LRB)
    forked = _fork(book, source)
    fork_stamp = book.stamp_of(book.latest_ref(SCOPE_ORIGIN, (forked,)))
    row = (forked, 2, "arg0", 0)
    # Writes elsewhere move the clock on.
    _root(book, LRB, (source, 70, "arg0", 0), "later")
    clock = book.clock[0]

    page.revise(row, "B")

    assert page.stamps[(row, 0)] == fork_stamp          # the copy, at the fork
    assert page.stamps[(row, 1)] > page.stamps[(row, 0)]
    assert page.stamps[(row, 1)] == clock               # the write, now
    assert book.clock[0] == clock + 1                   # the write's one tick


def test_concord_of_the_same_fact_records_edges_and_a_different_one_disagrees(
    book,
):
    source = book.mint_scope("fn", READ_SCOPE_FORK)
    _source_rows(book, source)
    forked = _fork(book, source)
    row = (forked, 2, "arg0", 0)
    cause = _root(book, LRB, (source, "cause", "arg0", 0), "cause")
    ref = book.post(
        LRB, row, "b", stage=OPERAND_POSITION,
        provenance=Derived((cause,)), mode=Mode.CONCORD,
    )
    assert ref == Ref(LRB, row, 0)
    assert book.page(LRB).history(row) == ((0, "b"),)
    assert _edges(book, LRB, row, 0) == {
        (("lexical_read_binding", (source, 2, "arg0", 0), 0),
         READ_SCOPE_FORK.name),
        (cause.key, OPERAND_POSITION.name),
    }
    with pytest.raises(ValueError, match="disagreement"):
        book.post(
            LRB, (forked, 1, "arg0", 0), "other", stage=OPERAND_POSITION,
            provenance=Derived((cause,)), mode=Mode.CONCORD,
        )
    # The refused post still took the copy first (the incumbent it disagreed
    # with): the eager copy was there to disagree with.
    assert book.page(LRB).history((forked, 1, "arg0", 0)) == ((0, "a"),)


def test_a_revise_with_no_changed_source_is_refused_as_before(book):
    source = book.mint_scope("fn", READ_SCOPE_FORK)
    _source_rows(book, source)
    forked = _fork(book, source)
    row = (forked, 1, "arg0", 0)
    origin_cell = book.latest_ref(LRB, (source, 1, "arg0", 0))
    from src.compiler.identity_concordance import ConcordanceRefusal
    # Deriving from exactly the cell the copy derives from, older than it:
    # no cause, as with the eager copy.
    with pytest.raises(ConcordanceRefusal, match="REVISE without a changed"):
        book.post(
            LRB, row, "a2", stage=OPERAND_POSITION,
            provenance=Derived((origin_cell,)), mode=Mode.REVISE,
        )


# ------------------------------------------------------------ projection
def _region_scenario(book):
    source = book.mint_scope("fn", READ_SCOPE_FORK)
    for consumer in (1, 2, 3, 4, "return"):
        _root(book, LRB, (source, consumer, "arg0", 0), f"v{consumer}")
    _root(book, LRB, (source, 9, "arg0", 0), "retired")   # a node of neither
    return source


def test_a_dispatch_region_fork_composes_projection_with_copy_on_read(book):
    source = _region_scenario(book)
    page = book.page(LRB)
    whole = _Graph((1, 2, 3, 4), source)
    region = _Graph((2, 3), source)
    fork_read_scope(region, "region", source_graph=whole)
    forked = tuple(region.G.graph["lexical_read_scope"])

    assert page.materialised_scope_rows(forked) == ()
    # Held nodes (and the non-node ``return`` consumer) read through.
    assert page.latest((forked, 2, "arg0", 0)) == "v2"
    assert page.latest((forked, "return", "arg0", 0)) == "vreturn"
    assert {row[1] for row in page.scope_rows(forked)} == {2, 3, "return"}
    assert page.scope_row_count(forked) == 3
    # The projection is declared on the fact, with the counts the copy had.
    projection = book.page(SCOPE_ORIGIN).latest((forked,)).projection
    assert (projection.node_count, projection.source_node_count) == (2, 4)
    assert (projection.rows_carried, projection.rows_excluded,
            projection.rows_stale) == (2, 2, 1)
    # A source node the region lacks is refused, with its Unresolved receipt.
    with pytest.raises(ProjectedScopeRead):
        page.latest((forked, 4, "arg0", 0))
    receipt = page.latest((forked, 4, "arg0", 0))
    assert isinstance(receipt, Unresolved)
    assert receipt.reason is ROW_PROJECTED_OUT_OF_SCOPE
    # An id of neither graph is a plain miss (it is not held, not refused) ...
    assert page.latest((forked, 9, "arg0", 0)) is None
    assert (forked, 9, "arg0", 0) not in page.scope_rows(forked)
    # ... and so is a node nobody ever had.
    assert page.latest((forked, 99, "arg0", 0)) is None

    # A write in the region still shadows only its row.
    page.revise((forked, 2, "arg0", 0), "w2")
    assert page.history((forked, 2, "arg0", 0)) == ((0, "v2"), (1, "w2"))
    assert page.latest((source, 2, "arg0", 0)) == "v2"
    assert page.latest((forked, 3, "arg0", 0)) == "v3"


def test_a_copy_of_a_region_reads_through_it_and_inherits_its_refusal(book):
    source = _region_scenario(book)
    page = book.page(LRB)
    region = _Graph((2, 3), source)
    fork_read_scope(region, "region", source_graph=_Graph((1, 2, 3, 4), source))
    regional = tuple(region.G.graph["lexical_read_scope"])
    # The region retires node 2's binding (an operand left the region).
    page.revise((regional, 2, "arg0", 0), None)

    copy = _Graph((2, 3), regional)
    fork_read_scope(copy, "copy of region")        # a whole-graph copy
    again = tuple(copy.G.graph["lexical_read_scope"])

    assert page.materialised_scope_rows(again) == ()
    assert page.latest((again, 3, "arg0", 0)) == "v3"           # two hops
    assert page.latest((again, 2, "arg0", 0)) is None           # withdrawn
    assert {row[1] for row in page.scope_rows(again)} == {3, "return"}
    with pytest.raises(ProjectedScopeRead):
        page.latest((again, 1, "arg0", 0))
    with pytest.raises(ProjectedScopeRead):
        page.latest((again, 4, "arg0", 0))
    assert page.latest((again, 9, "arg0", 0)) is None
    # The projection objects are the ones the eager forks registered.
    projections = book.scope_projections
    assert projections[again].inherited is projections[regional]
    assert projections[regional].held == frozenset({2, 3})
    assert projections[regional].excluded == frozenset()
    # A write in the copy materialises through the region to the source.
    page.revise((again, 3, "arg0", 0), "w3")
    assert page.history((again, 3, "arg0", 0)) == ((0, "v3"), (1, "w3"))
    assert page.materialised_scope_rows(regional) == (
        (regional, 2, "arg0", 0), (regional, 3, "arg0", 0),
    )


def test_a_whole_graph_fork_declares_no_projection_and_reads_everything(book):
    source = _region_scenario(book)
    forked = _fork(book, source, nodes=(1, 2, 3, 4))
    origin = book.page(SCOPE_ORIGIN).latest((forked,))
    assert origin.projection is None and origin.copy_on_read
    assert forked not in book.scope_projections
    page = book.page(LRB)
    assert page.scope_row_count(forked) == 6
    assert page.latest((forked, 9, "arg0", 0)) == "retired"


# ------------------------------------------------ the book's relation
def test_the_origin_relation_and_materialised_rows_are_all_the_viewer_sees(book):
    source = book.mint_scope("fn", READ_SCOPE_FORK)
    _source_rows(book, source)
    forked = _fork(book, source)
    page = book.page(LRB)
    # The relation: scope_origin carries the origin and the flag, derived
    # from the source's registry cell.
    origin_cell = book.latest_ref(SCOPE_ORIGIN, (forked,))
    assert (book.latest_ref(SCOPE_REGISTRY, source).key, READ_SCOPE_FORK.name) in {
        (s.key, st.name) for s, st in book.edges_into(origin_cell)
    }
    assert book.read_through[forked][0] == source
    # Unmaterialised rows are not rows: no cell, no edge, not unsourced.
    assert page.rows() == tuple(
        row for row in page.rows() if row[0] == source
    )
    unsourced_before = book.unsourced_rows()
    page.revise((forked, 1, "arg0", 0), "A")
    page.latest((forked, 2, "arg0", 0))
    # A materialised row is a sourced row: it has its edge, so the
    # unsourced listing grows only by the (raw) revision, never by the copy.
    new = [
        entry for entry in book.unsourced_rows()
        if entry not in unsourced_before
    ]
    assert [
        (name.name, row, reason, stage.name) for name, row, reason, stage in new
    ] == [(
        "lexical_read_binding", (forked, 1, "arg0", 0), RAW_PRIMITIVE,
        "raw_primitive",
    )]
    assert (forked, 2, "arg0", 0) not in page.rows()
    assert (forked, 1, "arg0", 0) in page.rows()
    # ``holds`` is the book's cell; ``history`` is the resolved view.
    assert not page.holds((forked, 2, "arg0", 0))
    assert page.holds((forked, 1, "arg0", 0))
    assert page.history((forked, 2, "arg0", 0)) == ((0, "b"),)


def test_an_old_pickle_of_a_book_has_no_forks_and_still_works(book):
    import pickle

    source = book.mint_scope("fn", READ_SCOPE_FORK)
    _source_rows(book, source)
    forked = _fork(book, source)
    book.page(LRB).revise((forked, 1, "arg0", 0), "A")
    clone = pickle.loads(pickle.dumps(book))
    page = clone.page(LRB)
    # Cells survive; the chain is restated by the relation on the book.
    assert page.history((forked, 1, "arg0", 0)) == ((0, "a"), (1, "A"))
    assert clone.read_through == book.read_through
    assert page.latest((forked, 2, "arg0", 0)) == "b"
    state = dict(book.__dict__)
    for key in ("read_through", "read_through_origins", "_stamp_override",
                "_materialising"):
        state.pop(key)
    legacy = object.__new__(type(book))
    legacy.__setstate__(state)
    assert legacy.read_through == {} and legacy._stamp_override is None


# ------------------------------------------------------------- siblings
def test_sibling_forks_of_one_origin_are_independent_and_counts_hold(book):
    source = book.mint_scope("fn", READ_SCOPE_FORK)
    _source_rows(book, source)
    page = book.page(LRB)
    first = _fork(book, source)
    second = _fork(book, source)
    assert first != second
    assert page.scope_row_count(first) == page.scope_row_count(second) == 4

    page.revise((first, 1, "arg0", 0), "first")
    page.revise((second, 1, "arg0", 0), None)
    page.revise((second, 5, "arg0", 0), "second-only")

    assert page.latest((first, 1, "arg0", 0)) == "first"
    assert page.latest((second, 1, "arg0", 0)) is None
    assert page.latest((source, 1, "arg0", 0)) == "a"
    assert page.latest((first, 5, "arg0", 0)) is None
    # A count is the same number a listing has, before and after writes.
    for scope in (first, second):
        assert page.scope_row_count(scope) == len(page.scope_rows(scope))
    assert page.scope_row_count(first) == 4
    assert page.scope_row_count(second) == 4 + 1          # row 1 withdrawn but own
    assert len(set(page.scope_rows(second))) == page.scope_row_count(second)


def test_an_unresolved_fact_and_a_raw_fact_on_a_declared_page_are_copied_as_they_are(
    book,
):
    source = book.mint_scope("fn", READ_SCOPE_FORK)
    receipt = Unresolved(ROW_PROJECTED_OUT_OF_SCOPE)
    book.post(
        LRB, (source, 1, "arg0", 0), receipt, stage=READ_SCOPE_FORK,
        provenance=Unsourced(SYNTHESIZED_NO_SOURCE), mode=Mode.CONCORD,
    )
    # A declared page whose writer still writes a fact of another shape.
    book.page(LRB).set((source, 2, "arg0", 0), 0, ("not", "a", "str"))
    forked = _fork(book, source)
    page = book.page(LRB)
    assert page.latest((forked, 1, "arg0", 0)) == receipt
    assert page.latest((forked, 2, "arg0", 0)) == ("not", "a", "str")

    page.revise((forked, 1, "arg0", 0), "resolved")
    page.revise((forked, 2, "arg0", 0), "now-a-str")

    # The Unresolved copy is a sourced cell; the odd-shaped one a tagged raw.
    assert _edges(book, LRB, (forked, 1, "arg0", 0), 0) == {
        (("lexical_read_binding", (source, 1, "arg0", 0), 0),
         READ_SCOPE_FORK.name),
    }
    assert _edges(book, LRB, (forked, 2, "arg0", 0), 0) == set()
    assert any(
        row == (forked, 2, "arg0", 0) and name.name == "lexical_read_binding"
        for name, row, _reason, _stage in book.unsourced_rows()
    )
    assert page.history((forked, 2, "arg0", 0)) == (
        (0, ("not", "a", "str")), (1, "now-a-str"),
    )
