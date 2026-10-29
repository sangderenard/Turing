"""CALL_RESULT_PROJECTION rows are scoped to the calling function.

Program: ``env, s, t = v.batch(...)`` in ``root`` and again in ``one``
(string-core ``intern_batch`` called beside ``intern``).  Wrong record: rows
were keyed by (call node, projection node) alone.  Node ids are local to one
function's graph, so two functions whose call and projection nodes share ids
but project different positions shared a row, and the second post was refused
as a ``call_result_projection_concordance`` disagreement.
"""

from __future__ import annotations

import networkx as nx


class _Graph:
    def __init__(self, name, positions):
        self.G = nx.DiGraph(function_name=name)
        self.G.add_node(5, op="call", type="call", label="batch", parents=[])
        index_node = 100
        for projection, position in positions.items():
            self.G.add_node(
                index_node, op="const", type="Constant", label=str(position),
                constant=position, parents=[],
            )
            self.G.add_node(
                projection, op="indexed", type="Indexed", label="r[i]",
                parents=[(5, "base"), (index_node, "index")],
            )
            self.G.add_edge(5, projection)
            self.G.add_edge(index_node, projection)
            index_node += 1


def _ingest(graph, book):
    """Enter the synthetic nodes through the identity API, as a build does."""

    from src.compiler.concordance_declarations import (
        INGEST_SOURCE, INGESTION_VALUE, NodeFact, READ_SCOPE_FORK,
    )
    from src.compiler.identity_concordance import Mode, Novel

    scope = book.mint_scope(
        f"projection_fixture_{graph.G.graph['function_name']}", READ_SCOPE_FORK,
    )
    graph.G.graph.update(
        ingestion_value_scope=scope, lexical_read_scope=scope,
        operand_position_scope=scope,
    )
    for node_id, data in graph.G.nodes(data=True):
        book.post(
            INGESTION_VALUE, (scope, node_id),
            NodeFact(data.get("type"), data.get("op"), data.get("label")),
            stage=READ_SCOPE_FORK, provenance=Novel(INGEST_SOURCE, ()),
            mode=Mode.CONCORD,
        )


def test_two_functions_with_equal_node_ids_keep_their_own_paths():
    from src.compiler.concordance_declarations import CALL_RESULT_PROJECTION
    from src.compiler.glsl_deployment_strategy import (
        _post_call_result_projections, _projection_result_bindings,
    )
    from src.compiler.identity_concordance import (
        begin_identity_book, end_identity_book,
    )

    root = _Graph("root", {6: 0, 7: 1})
    one = _Graph("one", {6: 1, 7: 0})
    book, token = begin_identity_book()
    try:
        _ingest(root, book)
        _ingest(one, book)
        assert _post_call_result_projections(root, 5) == {(0,): 6, (1,): 7}
        assert _post_call_result_projections(one, 5) == {(1,): 6, (0,): 7}
        page = book.pages[CALL_RESULT_PROJECTION.name]
        assert page.latest(("root", 5, 6)) == (0,)
        assert page.latest(("one", 5, 6)) == (1,)
        assert _projection_result_bindings(root, 5, (40, 41)) == (
            (40, 6), (41, 7),
        )
        assert _projection_result_bindings(one, 5, (40, 41)) == (
            (40, 7), (41, 6),
        )
    finally:
        end_identity_book(token)
