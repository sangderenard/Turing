"""``_set_operands`` batching: the same graphs, edge-at-a-time and batched.

Authoring a consumer's N operands used to rewrite its operand list N times
(``connect`` once per edge).  ``build_graph`` now stages a consumer's edges
and commits them with one ``_set_operands``.  Appending moves no earlier
position, so the book must hold the same rows, and the graph the same
``parents`` / ``children`` / edges, as the edge-at-a-time writes left.
"""

from __future__ import annotations

import ast
import contextlib
import io
import re

from src.compiler.identity_concordance import (
    begin_identity_book,
    end_identity_book,
)
from src.transmogrifier.graph.graph_express2 import ProcessGraph


SOURCES = (
    # Wide call, repeated constants (shared non-AST field nodes), keywords.
    "def f(a, b, c, d):\n"
    "    return g(a, b, c, d, 'dim', 'dim', k=a, m=b.attr, n=(a, b, a))\n",
    # An object read many times: one hub, many attribute consumers.
    "def advance(s):\n"
    "    return h(s.x, s.y, s.z, s.x, s.w)\n",
    # Nested calls, a loop, subscripts, a dict and a conditional.
    "def k(xs, y):\n"
    "    total = 0\n"
    "    for i in range(3):\n"
    "        total = total + xs[i] * y\n"
    "    return {'a': total, 'b': max(total, y)} if y else None\n",
)


def _normalized_rows(book, node_ids):
    """Every page's cells as text, process ids replaced by build ordinals."""

    names = {int(node): f"n{ordinal}" for ordinal, node in enumerate(node_ids)}
    pattern = re.compile(
        "|".join(
            str(node)
            for node in sorted(names, key=lambda node: -len(str(node)))
        )
    ) if names else None

    def text(value):
        rendered = re.sub(r"0x[0-9A-Fa-f]+", "0x?", repr(value))
        if pattern is not None:
            rendered = pattern.sub(lambda m: names[int(m.group(0))], rendered)
        # An id of a node no longer in the graph: still a process address.
        return re.sub(r"\d{11,}", "ID", rendered)

    return {
        name: [
            (text(row), column, text(fact))
            for (row, column), fact in page.cells.items()
        ]
        for name, page in book.pages.items()
    }


def _normalized_graph(graph):
    names = {
        int(node): f"n{ordinal}"
        for ordinal, node in enumerate(graph.G.nodes)
    }

    def name(node):
        return names.get(int(node), f"x{node}")

    return {
        name(node): {
            "parents": [(name(p), role) for p, role in data["parents"]],
            "children": [(name(c), role) for c, role in data["children"]],
        }
        for node, data in graph.G.nodes(data=True)
    }, sorted(
        (
            name(u), name(v), data.get("role"),
            sorted(
                (name(e.source), name(e.target), e.id[2], e.id[3])
                for e in data.get("extra", ())
            ),
        )
        for u, v, data in graph.G.edges(data=True)
    )


def _build(source, *, edge_at_a_time):
    graph = ProcessGraph(materialize_memory=False)
    commit = ProcessGraph._commit_edges
    if edge_at_a_time:
        def one_by_one(self, tgt, edges):
            for edge in edges:
                commit(self, tgt, [edge])

        ProcessGraph._commit_edges = one_by_one
    book, token = begin_identity_book()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(ast.parse(source))
        return (
            _normalized_rows(book, list(graph.G.nodes)),
            _normalized_graph(graph),
        )
    finally:
        ProcessGraph._commit_edges = commit
        end_identity_book(token)


def test_build_graph_batched_commit_matches_edge_at_a_time():
    for source in SOURCES:
        batched_rows, batched_graph = _build(source, edge_at_a_time=False)
        single_rows, single_graph = _build(source, edge_at_a_time=True)

        assert batched_graph == single_graph
        assert batched_rows.keys() == single_rows.keys()
        for page in single_rows:
            if page in ("concordance_edge", "concordance_dependents"):
                # The global post logs: a run of leaf children posts each
                # child's own rows before the run's appends, so the same
                # posts arrive in a different interleave.
                assert sorted(batched_rows[page]) == sorted(single_rows[page])
            else:
                # Every page: the same rows, the same order.
                assert batched_rows[page] == single_rows[page], page


def test_build_graph_commits_a_run_of_edges_with_one_set_operands():
    import src.common.tensors.topological_reducer as reducer

    source = "def f(a, b, c, d, e):\n    return g(a, b, c, d, e)\n"
    counts = {}
    for mode in (False, True):
        calls = []
        original = reducer._set_operands

        def counting(graph, node_id, parents, **kwargs):
            calls.append(node_id)
            return original(graph, node_id, parents, **kwargs)

        reducer._set_operands = counting
        try:
            _build(source, edge_at_a_time=mode)
        finally:
            reducer._set_operands = original
        counts[mode] = len(calls)
    # The five arguments of ``g(...)`` are leaves: one write instead of five.
    assert counts[True] - counts[False] >= 4


def test_commit_edges_writes_the_rows_connect_wrote_in_order():
    """One consumer, edges handed over together vs one ``connect`` each:
    identical parents/children/edges and every page's cells in identical
    order (no field-value row interleaves: all endpoints are AST nodes)."""

    roles = (
        ("output", "args"), ("output", "args"), ("output", "keywords"),
        ("output", "args"), ("output", "args"), ("output", "keywords"),
    )

    def build(batched):
        sources = [ast.Name(id=f"v{i}", ctx=ast.Load()) for i in range(6)]
        target = ast.Call(func=ast.Name(id="f", ctx=ast.Load()), args=[], keywords=[])
        graph = ProcessGraph(materialize_memory=False)
        book, token = begin_identity_book()
        try:
            graph.G.graph["ingestion_value_scope"] = book.mint_scope("ingestion:t")
            for node in (*sources, target):
                graph.ensure_node(node)
            # source 1 is authored twice (the second adds only an Edge).
            edges = [
                (id(sources[i]), id(target), producer, consumer, None)
                for i, (producer, consumer) in enumerate(roles)
            ]
            edges.insert(3, (id(sources[1]), id(target), "output", "args", None))
            if batched:
                for src, tgt, _p, consumer, _s in edges:
                    graph._stage_edge(src, tgt, consumer)
                graph._commit_edges(
                    id(target), [(s, p, c, st) for s, _t, p, c, st in edges],
                )
            else:
                for src, tgt, producer, consumer, store in edges:
                    graph.connect(src, tgt, producer, consumer, store)
            return (
                _normalized_rows(book, list(graph.G.nodes)),
                _normalized_graph(graph),
            )
        finally:
            end_identity_book(token)

    batched_rows, batched_graph = build(True)
    single_rows, single_graph = build(False)
    assert batched_graph == single_graph
    assert batched_rows == single_rows
    assert any(
        row.startswith("(('") or "'args'" in row
        for row, _column, _fact in single_rows["identity_transition"]
    )


def test_append_only_writes_what_the_general_path_writes():
    """``append_only`` is a shortcut for lists that are old-plus-tail: the
    rows, ``parents``, ``children`` and edges must be the general path's,
    including when a tail operand repeats an existing one (the shortcut
    declines) and when earlier positions carry ``lexical_read_binding``
    facts that must stay put."""

    from src.common.tensors.topological_reducer import _set_operands
    from src.compiler.concordance_declarations import INGEST_EDGE

    steps = (
        [("a", "args")],
        [("b", "args"), ("c", "kw")],
        [("d", "args")],
        [("a", "kw")],                       # repeats an existing operand
        [("e", "args"), ("f", "args")],
    )

    def build(append_only):
        names = "abcdef"
        sources = {n: ast.Name(id=n, ctx=ast.Load()) for n in names}
        target = ast.Call(func=ast.Name(id="f", ctx=ast.Load()), args=[], keywords=[])
        graph = ProcessGraph(materialize_memory=False)
        book, token = begin_identity_book()
        try:
            scope = book.mint_scope("ingestion:t")
            graph.G.graph["ingestion_value_scope"] = scope
            for node in (*sources.values(), target):
                graph.ensure_node(node)
                graph.G.nodes[id(node)].setdefault("children", [])
            tgt = id(target)
            for step in steps:
                parents = list(graph.G.nodes[tgt]["parents"])
                parents += [(id(sources[n]), role) for n, role in step]
                # A binding fact on a position that must survive the append.
                book.page("lexical_read_binding").revise(
                    (scope, tgt, "args", 0), "bound-first",
                )
                _set_operands(
                    graph, tgt, parents, cause=INGEST_EDGE,
                    edge_payload={"extra": set()}, append_only=append_only,
                )
            return _normalized_rows(book, list(graph.G.nodes)), _normalized_graph(graph)
        finally:
            end_identity_book(token)

    assert build(True) == build(False)
