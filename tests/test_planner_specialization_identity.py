"""Callsite specialization facts have traversal-independent identity."""

from types import SimpleNamespace

import networkx as nx

from src.compiler.concordance_declarations import (
    CANONICAL_VALUE,
    SPECIALIZATION_DYNAMIC_ARGUMENT,
)
from src.compiler.glsl_deployment_strategy import (
    _canonical_planner_specialization_cells,
)
from src.compiler.identity_concordance import (
    Unresolved,
    begin_identity_book,
    current_identity_book,
    end_identity_book,
)
from src.compiler.planner_concordance_posts import (
    _post_planner_specialization,
)


def test_dynamic_specialization_cell_order_is_not_a_revision():
    graph = SimpleNamespace(G=nx.DiGraph())
    graph.G.graph["lexical_read_scope"] = ("callee", 0)

    _book, token = begin_identity_book()
    try:
        page = current_identity_book().page(CANONICAL_VALUE)
        page.set((("caller", 0), 1), 0, ("first",))
        page.set((("caller", 0), 2), 0, ("second",))
        first = current_identity_book().latest_ref(
            CANONICAL_VALUE, (("caller", 0), 1),
        )
        second = current_identity_book().latest_ref(
            CANONICAL_VALUE, (("caller", 0), 2),
        )

        ordered = _canonical_planner_specialization_cells((
            (object(), (first, second), False),
        ))
        reordered = _canonical_planner_specialization_cells((
            (object(), (second, first), False),
        ))
        assert ordered == reordered

        _post_planner_specialization(
            graph, "args",
            Unresolved(SPECIALIZATION_DYNAMIC_ARGUMENT, read=ordered),
            ordered,
        )
        _post_planner_specialization(
            graph, "args",
            Unresolved(SPECIALIZATION_DYNAMIC_ARGUMENT, read=reordered),
            reordered,
        )
        row = (("callee", 0), "args")
        assert current_identity_book().page(
            "planner_specialization"
        ).history(row) == ((0, Unresolved(
            SPECIALIZATION_DYNAMIC_ARGUMENT, read=ordered,
        )),)
    finally:
        end_identity_book(token)
