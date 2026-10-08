"""Copy-on-read function-subgraph scopes: ``fork_operand_position_scope``
posts no copied rows.

Pure data-structure tests on a real ``IdentityBook`` with hand-posted rows.
Nothing here lowers, compiles, or runs a probe: the forks are made over a
bare networkx graph (``fork_operand_position_scope`` reads only the graph's
metadata and node set) and the operand rewrites are the reducer's own
``_set_operands`` over that graph.  The eager behaviour the fork replaced is
restated in ``_eager_function_fork`` (the old copy loop, row for row), so the
rows and edges of a row written in the fork can be compared with the copy
the eager fork made.
"""

import networkx as nx
import pytest

from src.common.tensors.topological_reducer import (
    _set_operands, fork_operand_position_scope, node_identity_cell,
)
from src.compiler.concordance_declarations import (
    CONSUMER_OPERAND, FUNCTION_SUBGRAPH, FUNCTION_SUBGRAPH_FILTER,
    IDENTITY_TRANSITION, INGESTION_VALUE, LEXICAL_READ_BINDING,
    OPERAND_POSITION, READ_SCOPE_FORK, ROW_PROJECTED_OUT_OF_SCOPE,
    SCOPE_ORIGIN, SCOPE_REGISTRY, SYNTHESIZED_NO_SOURCE, NodeFact,
    OperandAppend, OperandMove, OperandRetire, ScopeFork,
)
from src.compiler.identity_concordance import (
    Derived, Mode, ProjectedScopeRead, Ref, Unresolved, Unsourced,
    begin_identity_book, end_identity_book,
)

LRB = LEXICAL_READ_BINDING
OPS = CONSUMER_OPERAND
IT = IDENTITY_TRANSITION
OPERAND_PAGES = (IT, LRB, OPS)


class _Graph:
    """What ``fork_operand_position_scope`` and ``_set_operands`` read of a
    graph: ``G`` (a DiGraph whose nodes carry ``parents`` / ``children``)
    and ``G.graph``'s scope metadata."""

    def __init__(self, parents, scope, ingestion=None):
        self.G = nx.DiGraph()
        for node, operands in parents.items():
            self.G.add_node(node, parents=list(operands), children=[])
        for node, operands in parents.items():
            for parent, role in operands:
                self.G.add_edge(parent, node, role=role)
                self.G.nodes[parent]["children"].append((node, role))
        self.G.graph["operand_position_scope"] = scope
        if ingestion is not None:
            self.G.graph["ingestion_value_scope"] = ingestion

    def subgraph(self, members):
        sub = _Graph({}, None)
        sub.G = self.G.subgraph(members).copy()
        sub.G.graph = dict(self.G.graph)
        return sub


@pytest.fixture
def book():
    book, token = begin_identity_book()
    try:
        yield book
    finally:
        end_identity_book(token)


def _root(book, page, row, fact, mode=Mode.CONCORD):
    return book.post(
        page, row, fact, stage=READ_SCOPE_FORK,
        provenance=Unsourced(SYNTHESIZED_NO_SOURCE), mode=mode,
    )


#: node -> operands.  The function is {1, 2, 3}; 4 and 5 are outside it and
#: node 2 reads node 5 (an operand the function-subgraph filter drops).
PARENTS = {
    1: [],
    2: [(1, "arg"), (5, "arg")],
    3: [(2, "arg")],
    4: [(3, "arg")],
    5: [(1, "arg")],
}
MEMBERS = frozenset({1, 2, 3})


def _source_rows(book, scope, ingestion):
    """The operand-position rows a root graph's reduction left under its
    operand-position ``scope``: one ``identity_transition`` Append, one
    ``lexical_read_binding`` and one ``consumer_operand`` row per operand,
    and (in the build's ``ingestion`` scope, where ``ensure_node`` posts
    them) every node's identity row."""
    for node, operands in PARENTS.items():
        _root(book, INGESTION_VALUE, (ingestion, node),
              NodeFact("Op", "op", f"n{node}"))
        seen = {}
        for parent, role in operands:
            ordinal = seen.get(role, 0)
            seen[role] = ordinal + 1
            _root(book, IT, (scope, node, role, ordinal),
                  OperandAppend("build", parent), Mode.REVISE)
            _root(book, LRB, (scope, node, role, ordinal), f"read{parent}")
            _root(book, OPS, (scope, node, parent), ((role, ordinal),))


def _world(book):
    ingestion = book.mint_scope("ingestion", READ_SCOPE_FORK)
    scope = book.mint_scope("lexical_reads", READ_SCOPE_FORK)
    _source_rows(book, scope, ingestion)
    return ingestion, scope


def _eager_function_fork(book, source, members, cause="function_subgraph"):
    """The copy loop ``fork_operand_position_scope`` used to run, row for
    row: every operand-position row of a member consumer re-posted under a
    minted scope, DERIVED from the cell it copies (stage
    ``function_subgraph``)."""
    forked = book.mint_scope(f"{source[0]}|operands", FUNCTION_SUBGRAPH)
    registered = book.registry.pages
    for name in (IT.name, "lexical_read_binding", "consumer_operand"):
        page = book.pages.get(name)
        if page is None:
            continue
        declared = registered.get(name)
        for row in tuple(page.scope_rows(source)):
            if len(row) < 2 or row[1] not in members:
                continue
            fact = page.latest(row)
            if fact is None:
                continue
            book.post(
                declared, (forked, *row[1:]), fact, stage=FUNCTION_SUBGRAPH,
                provenance=Derived((book.latest_ref(declared, row),)),
                mode=Mode.CONCORD,
            )
    book.post(
        SCOPE_ORIGIN, (forked,), ScopeFork(source, cause),
        stage=FUNCTION_SUBGRAPH,
        provenance=Derived((book.latest_ref(SCOPE_REGISTRY, source),)),
        mode=Mode.CONCORD,
    )
    return forked


def _function(book, scope, ingestion, *, lazy=True):
    """The function subgraph of the root graph, forked either way; returns
    ``(function graph, fork scope)``."""
    root = _Graph(PARENTS, scope, ingestion)
    function = root.subgraph(MEMBERS)
    if lazy:
        fork_operand_position_scope(
            function, MEMBERS, "function_subgraph", source_graph=root,
        )
        return function, tuple(function.G.graph["operand_position_scope"])
    forked = _eager_function_fork(book, scope, MEMBERS)
    function.G.graph["operand_position_scope"] = forked
    return function, forked


def _filter(function):
    """The reducer's function-subgraph filter loop (parameter-Input rewrite
    omitted): every member drops the operands that left the subgraph."""
    for member in tuple(function.G):
        _set_operands(function, member, [
            (parent, role)
            for parent, role in function.G.nodes[member].get("parents", ())
            if parent in function.G
        ], cause=FUNCTION_SUBGRAPH_FILTER)


def _edges(book, page, row, column):
    return {
        (source.key, stage.name)
        for source, stage in book.edges_into(Ref(page, row, column))
    }


def _cell_count(book, name):
    page = book.pages.get(name)
    return 0 if page is None else len(page.cells)


# ---------------------------------------------------------------- no rows
def test_the_fork_posts_no_rows_and_names_itself_the_operand_scope(book):
    ingestion, scope = _world(book)
    before = {name: _cell_count(book, name) for name in book.pages}
    root = _Graph(PARENTS, scope, ingestion)
    root.G.graph["lexical_read_scope"] = scope
    function = root.subgraph(MEMBERS)

    fork_operand_position_scope(
        function, MEMBERS, "function_subgraph", source_graph=root,
    )

    forked = tuple(function.G.graph["operand_position_scope"])
    assert forked != scope
    # The read scope is the fork_read_scope lane's, not this one's.
    assert function.G.graph["lexical_read_scope"] == scope
    # Not one operand-position row was posted: only the fork's origin cell
    # (and its registry / edge bookkeeping) exists.
    for page in OPERAND_PAGES:
        assert _cell_count(book, page.name) == before[page.name]
        assert book.pages[page.name].materialised_scope_rows(forked) == ()
        assert not book.pages[page.name].holds((forked, 2, "arg", 0))
    origin = book.page(SCOPE_ORIGIN).latest((forked,))
    assert origin.copy_on_read is True
    assert (origin.source_scope, origin.cause) == (scope, "function_subgraph")
    projection = origin.projection
    assert (projection.node_count, projection.source_node_count) == (3, 5)
    # Counted, not copied: the three operand pages hold 3 rows each of
    # members 2 and 3 (9 carried) and 1 each of source nodes 4 and 5 (6
    # excluded); no id was retired.
    assert (projection.rows_carried, projection.rows_excluded,
            projection.rows_stale) == (9, 6, 0)
    # The origin is derived from the source scope's registry cell at the
    # function_subgraph stage, as the eager fork's was.
    origin_cell = book.latest_ref(SCOPE_ORIGIN, (forked,))
    assert (book.latest_ref(SCOPE_REGISTRY, scope).key,
            FUNCTION_SUBGRAPH.name) in {
        (s.key, st.name) for s, st in book.edges_into(origin_cell)
    }
    assert book.read_through[forked][0] == scope


def test_a_graph_with_no_operand_scope_is_not_forked(book):
    graph = _Graph(PARENTS, None)
    graph.G.graph.pop("operand_position_scope")
    fork_operand_position_scope(graph, MEMBERS, "function_subgraph")
    assert "operand_position_scope" not in graph.G.graph
    assert book.read_through == {}


# ------------------------------------------------------------------ reads
def test_member_rows_read_through_as_the_source_did(book):
    ingestion, scope = _world(book)
    _function_graph, forked = _function(book, scope, ingestion)
    lrb, ops, it = book.page(LRB), book.page(OPS), book.page(IT)

    assert lrb.latest((forked, 2, "arg", 1)) == "read5"
    assert lrb.latest_column((forked, 2, "arg", 1)) == 0
    assert lrb.history((forked, 2, "arg", 1)) == ((0, "read5"),)
    assert ops.latest((forked, 2, 5)) == (("arg", 1),)
    assert it.latest((forked, 3, "arg", 0)) == OperandAppend("build", 2)
    assert it.cell((forked, 3, "arg", 0), 0) == OperandAppend("build", 2)
    # The reads wrote nothing.
    for page in OPERAND_PAGES:
        assert book.pages[page.name].materialised_scope_rows(forked) == ()
    # An operand-less member has no row; the source is untouched.
    assert lrb.latest((forked, 1, "arg", 0)) is None
    assert len(lrb.rows()) == 5
    assert all(row[0] == scope for row in lrb.rows())


def test_a_node_outside_the_subgraph_is_refused_and_an_unknown_id_is_a_miss(book):
    ingestion, scope = _world(book)
    _function_graph, forked = _function(book, scope, ingestion)
    lrb = book.page(LRB)
    # The subgraph lists its own nodes' rows only.
    assert {row[1] for row in lrb.scope_rows(forked)} == {2, 3}

    # Node 5 is the source graph's, not the subgraph's: refused, after an
    # Unresolved receipt is posted at the key.
    with pytest.raises(ProjectedScopeRead):
        lrb.latest((forked, 5, "arg", 0))
    receipt = lrb.latest((forked, 5, "arg", 0))
    assert isinstance(receipt, Unresolved)
    assert receipt.reason is ROW_PROJECTED_OUT_OF_SCOPE
    # ... on each node-keyed page.
    with pytest.raises(ProjectedScopeRead):
        book.page(IT).latest((forked, 4, "arg", 0))
    with pytest.raises(ProjectedScopeRead):
        book.page(OPS).latest((forked, 4, 3))
    # An id that was a node of neither graph is a plain miss.
    assert lrb.latest((forked, 99, "arg", 0)) is None
    # The source's rows for the refused nodes are untouched.
    assert lrb.latest((scope, 5, "arg", 0)) == "read1"
    assert lrb.latest((scope, 4, "arg", 0)) == "read3"


def test_without_the_source_graph_the_subgraph_still_holds_only_its_nodes(book):
    ingestion, scope = _world(book)
    root = _Graph(PARENTS, scope, ingestion)
    function = root.subgraph(MEMBERS)
    fork_operand_position_scope(function, MEMBERS, "function_subgraph")
    forked = tuple(function.G.graph["operand_position_scope"])
    lrb = book.page(LRB)
    assert lrb.latest((forked, 2, "arg", 0)) == "read1"
    # No universe is stated, so nothing is *refused*; a non-member simply
    # holds no row here (the eager copy held none either).
    assert lrb.latest((forked, 5, "arg", 0)) is None
    assert {row[1] for row in lrb.scope_rows(forked)} == {2, 3}


# --------------------------------------------------- ordinals and counts
def test_scope_rows_and_row_counts_are_what_the_eager_copy_held():
    lazy_book, token = begin_identity_book()
    try:
        ingestion, scope = _world(lazy_book)
        _g, lazy = _function(lazy_book, scope, ingestion)
        lazy_counts = {
            page.name: lazy_book.page(page).scope_row_count(lazy)
            for page in OPERAND_PAGES
        }
        lazy_rows = {
            page.name: lazy_book.page(page).scope_rows(lazy)
            for page in OPERAND_PAGES
        }
    finally:
        end_identity_book(token)
    eager_book, token = begin_identity_book()
    try:
        ingestion, scope = _world(eager_book)
        _g, eager = _function(eager_book, scope, ingestion, lazy=False)
        eager_counts = {
            page.name: eager_book.page(page).scope_row_count(eager)
            for page in OPERAND_PAGES
        }
        eager_rows = {
            page.name: eager_book.page(page).scope_rows(eager)
            for page in OPERAND_PAGES
        }
    finally:
        end_identity_book(token)

    assert lazy == eager       # both minted the same scope: comparable keys
    assert lazy_counts == eager_counts == {
        "identity_transition": 3, "lexical_read_binding": 3,
        "consumer_operand": 3,
    }
    # The same rows in the same order (the source's order, then the fork's).
    assert lazy_rows == eager_rows


def test_a_row_only_the_fork_holds_takes_the_next_ordinal(book):
    ingestion, scope = _world(book)
    _g, forked = _function(book, scope, ingestion)
    lrb = book.page(LRB)
    assert lrb.scope_row_count(forked) == 3
    # An append at a fresh position: a row the source never had.
    _root(book, LRB, (forked, 3, "arg", 1), "new", Mode.CONCORD)
    assert lrb.scope_row_count(forked) == 4
    assert lrb.scope_rows(forked)[-1] == (forked, 3, "arg", 1)
    # A write to a row it reads through to adds none.
    lrb.revise((forked, 3, "arg", 0), "w")
    assert lrb.scope_row_count(forked) == 4
    assert len(lrb.scope_rows(forked)) == lrb.scope_row_count(forked)


# --------------------------------------------- writes: the operand rewrite
def test_the_function_subgraph_filter_writes_as_it_did_over_the_copy():
    runs = {}
    for lazy in (True, False):
        run_book, token = begin_identity_book()
        try:
            ingestion, scope = _world(run_book)
            function, forked = _function(run_book, scope, ingestion, lazy=lazy)
            _filter(function)
            runs[lazy] = (run_book, function, forked, scope)
        finally:
            end_identity_book(token)
    (lazy_book, lazy_graph, lazy, lazy_source) = runs[True]
    (eager_book, eager_graph, eager, eager_source) = runs[False]

    assert lazy == eager
    # The graph's own operands agree.
    assert lazy_graph.G.nodes[2]["parents"] == [(1, "arg")]
    assert lazy_graph.G.nodes[2]["parents"] == eager_graph.G.nodes[2]["parents"]
    assert set(lazy_graph.G.edges) == set(eager_graph.G.edges)

    # What the rewrite touched is exactly node 2's dropped position; every
    # other member row was never written -- and is not a row of the book.
    touched = {
        "identity_transition": {(lazy, 2, "arg", 1)},
        "lexical_read_binding": {(lazy, 2, "arg", 1)},
        "consumer_operand": {(lazy, 2, 5)},
    }
    for page in OPERAND_PAGES:
        lazy_page = lazy_book.pages[page.name]
        eager_page = eager_book.pages[page.name]
        assert set(lazy_page.materialised_scope_rows(lazy)) == touched[page.name]
        assert len(eager_page.materialised_scope_rows(eager)) == 3
        # Every row either fork holds reads the same: history, latest, edges
        # of every cell -- the copy's (column 0) and the rewrite's.
        for row in eager_page.scope_rows(eager):
            assert lazy_page.history(row) == eager_page.history(row), row
            assert lazy_page.latest(row) == eager_page.latest(row), row
        assert set(lazy_page.scope_rows(lazy)) == set(
            eager_page.scope_rows(eager)
        )
        assert lazy_page.scope_row_count(lazy) == eager_page.scope_row_count(
            eager
        )
        for row in touched[page.name]:
            for column, _fact in eager_page.history(row):
                lazy_edges = _edges(lazy_book, page, row, column)
                eager_edges = _edges(eager_book, page, row, column)
                if column == 0:
                    # The materialised copy: the same source cell, the same
                    # fact, the same column -- labelled with the stage the
                    # book's materialisation uses (``read_scope_fork``) where
                    # the eager copy said ``function_subgraph``.
                    assert {s for s, _ in lazy_edges} == {
                        s for s, _ in eager_edges}, (row, column)
                    assert {st for _, st in lazy_edges} == {
                        READ_SCOPE_FORK.name}
                    assert {st for _, st in eager_edges} == {
                        FUNCTION_SUBGRAPH.name}
                else:
                    assert lazy_edges == eager_edges, (row, column)
    # The retire is the operator the eager copy had, caused by the filter.
    retire = lazy_book.page(IT).latest((lazy, 2, "arg", 1))
    assert retire == OperandRetire("function_subgraph_filter")
    # The source's rows are the root graph's and were never written.
    for page in OPERAND_PAGES:
        assert all(
            lazy_book.pages[page.name].history(row)
            == eager_book.pages[page.name].history(row)
            for row in lazy_book.pages[page.name].scope_rows(lazy_source)
        )
    assert lazy_book.page(LRB).latest((lazy_source, 2, "arg", 1)) == "read5"
    assert lazy_book.page(LRB).history((lazy_source, 2, "arg", 1)) == (
        (0, "read5"),
    )
    # The retire's operand is the position's own copy cell (column 0 of the
    # fork's row), which is derived from the source's: the chain the eager
    # copy made.
    mint = lazy_book.mint_of(Ref(IT, (lazy, 2, "arg", 1), 1))
    assert mint == eager_book.mint_of(Ref(IT, (lazy, 2, "arg", 1), 1))
    assert mint[1] == (Ref(IT, (lazy, 2, "arg", 1), 0),)


def test_unsourced_tags_of_the_rewrite_match_the_eager_fork():
    tags = {}
    for lazy in (True, False):
        run_book, token = begin_identity_book()
        try:
            ingestion, scope = _world(run_book)
            function, forked = _function(run_book, scope, ingestion, lazy=lazy)
            _filter(function)
            tags[lazy] = sorted(
                (name.name if hasattr(name, "name") else name, row,
                 reason.name, stage.name)
                for name, row, reason, stage in run_book.unsourced_rows()
                if isinstance(row, tuple) and row and row[0] == forked
            )
        finally:
            end_identity_book(token)
    assert tags[True] == tags[False]


def test_a_move_in_the_subgraph_continues_the_position_row(book):
    """A position that moves (a call rebuilt from ``args`` to ``arg``): the
    fork's rows for the OLD position are the source's, read through; the new
    position's row is the fork's own."""
    ingestion, scope = _world(book)
    function, forked = _function(book, scope, ingestion)
    _set_operands(
        function, 3, [(2, "renamed")], cause=FUNCTION_SUBGRAPH_FILTER,
    )
    lrb, it = book.page(LRB), book.page(IT)
    # The vacated position: copy then withdrawal; the arrival: the binding
    # moved to the new position.
    assert lrb.history((forked, 3, "arg", 0)) == ((0, "read2"), (1, None))
    assert lrb.latest((forked, 3, "renamed", 0)) == "read2"
    assert it.latest((forked, 3, "arg", 0)) == OperandMove(
        "function_subgraph_filter", 3, "renamed", 0,
    )
    # Only node 3's rows were written; node 2's never were.
    for page in (LRB, IT):
        assert {
            row[1] for row in book.pages[page.name].materialised_scope_rows(
                forked,
            )
        } == {3}
    assert lrb.latest((forked, 2, "arg", 1)) == "read5"
    assert lrb.latest((scope, 3, "arg", 0)) == "read2"


# ------------------------------------------- readers that name a source ref
def test_the_canonical_relabel_reads_the_scope_and_materialises_what_it_cites(
    book,
):
    """``_normalize_lexical_values``' tail enumerates the entry operand
    scope's ``identity_transition`` rows and posts each canonical row DERIVED
    from ``book.latest_ref`` of the row it continues.  That is a reader
    through ``scope_rows`` (chain-aware) and a *source reference* (it
    materialises the row it names, with the edge the copy had)."""
    ingestion, scope = _world(book)
    _function_graph, forked = _function(book, scope, ingestion)
    it = book.page(IT)
    canonical = book.mint_scope("canonical", READ_SCOPE_FORK)

    rows = tuple(it.scope_rows(forked))
    assert {row[1] for row in rows} == {2, 3}
    assert it.materialised_scope_rows(forked) == ()
    for row in rows:
        fact = it.latest(row)
        book.post(
            IT, (canonical, *row[1:]), fact, stage=OPERAND_POSITION,
            provenance=Derived((book.latest_ref(IT, row),)),
            mode=Mode.REVISE,
        )

    # Every cited row now exists in the fork, derived from the source's cell.
    assert set(it.materialised_scope_rows(forked)) == set(rows)
    for row in rows:
        source_row = (scope, *row[1:])
        assert _edges(book, IT, row, 0) == {
            (Ref(IT, source_row, 0).key, READ_SCOPE_FORK.name),
        }
        assert (Ref(IT, row, 0).key, OPERAND_POSITION.name) in {
            (s.key, st.name)
            for s, st in book.edges_into(
                book.latest_ref(IT, (canonical, *row[1:]))
            )
        }
    # The binding and operand-structure rows nobody cited stay unwritten.
    assert book.page(LRB).materialised_scope_rows(forked) == ()
    assert book.page(OPS).materialised_scope_rows(forked) == ()


def test_node_identity_cell_finds_the_build_scope_row_not_a_fork_row(book):
    """``node_identity_cell`` reads ``ingestion_value`` under the operand
    scope first, then the build's.  The source scope holds no such row, so
    the subgraph reads the build's, exactly as it did over the copy."""
    ingestion, scope = _world(book)
    function, forked = _function(book, scope, ingestion)
    cell = node_identity_cell(function, 2)
    assert cell == Ref(INGESTION_VALUE, (ingestion, 2), 0)
    assert book.page("ingestion_value").materialised_scope_rows(forked) == ()


# ------------------------------------------------- documented read widening
def test_what_the_copy_never_held_is_read_through_and_is_listed_here(book):
    """The eager copy carried three pages' rows keyed by a MEMBER node.  The
    book's fork reads *every* page (but the planner's per-copy ones) through,
    with the node-keyed pages restricted to the held nodes -- exactly the
    dispatch-region fork.  These are the differences, pinned so the first
    compile's reader audit can find them:

    * a row keyed by something that is not a node id (``return``) is read
      through, and counted by ``scope_row_count``;
    * a page the eager copy did not carry (here a raw one) reads through.
    """
    ingestion, scope = _world(book)
    _root(book, LRB, (scope, "return", "arg", 0), "ret")
    book.page("raw_rows").set((scope, "k"), 0, 5)
    function, forked = _function(book, scope, ingestion)
    lrb = book.page(LRB)

    assert lrb.latest((forked, "return", "arg", 0)) == "ret"
    assert lrb.scope_row_count(forked) == 4              # 3 + the return row
    assert book.page("raw_rows").latest((forked, "k")) == 5
