"""Eager comprehension activity follows iterator, clause and body order."""

import ast

import pytest

from src.compiler.concordance_declarations import (
    SOURCE_ACTIVITY_EDGE, SOURCE_ACTIVITY_NODE,
)
from src.compiler.identity_concordance import (
    begin_identity_book, end_identity_book,
)
from src.compiler.source_control_ingestion import ingest_control_activity
from src.transmogrifier.graph.graph_express2 import (
    _annotate_visual_source_owners, post_source_span,
)


@pytest.fixture
def activity():
    book, token = begin_identity_book()
    try:
        def build(source):
            tree = ast.parse(source)
            _annotate_visual_source_owners(tree, module_name="test.comprehension")
            definition = tree.body[0]
            scope = ("test.comprehension", definition.name)
            ingest_control_activity(None, definition, scope)

            def point(node, phase):
                occurrence = post_source_span(None, node)
                return book.latest_ref(
                    SOURCE_ACTIVITY_NODE, (scope, occurrence, phase),
                )

            def has_edge(before, after, relation="flow"):
                return book.latest_ref(
                    SOURCE_ACTIVITY_EDGE, (scope, before, after, relation),
                ) is not None

            return tree, point, has_edge

        yield book, build
    finally:
        end_identity_book(token)


@pytest.mark.parametrize(
    ("display", "element_path"),
    [
        ("[item for item in items]", "elt"),
        ("{item for item in items}", "elt"),
        ("{item: item + 1 for item in items}", "value"),
    ],
)
def test_eager_clause_has_select_and_body_owned_element(
    activity, display, element_path,
):
    book, build = activity
    tree, point, has_edge = build(
        f"def root(items):\n    return {display}\n"
    )
    expression = tree.body[0].body[0].value
    clause = expression.generators[0]
    element = getattr(expression, element_path)
    assert has_edge(point(clause.iter, "exit"), point(clause, "select"))
    assert has_edge(
        point(clause, "select"), point(clause, "body_enter"), "body",
    )
    assert has_edge(point(clause, "body_enter"), point(clause.target, "enter"))
    assert has_edge(
        point(clause, "body_exit"), point(clause, "select"), "repeat",
    )
    assert has_edge(
        point(clause, "select"), point(clause, "exit"), "exhausted",
    )
    element_fact = book.page(SOURCE_ACTIVITY_NODE).cell(
        point(element, "enter").row, point(element, "enter").column,
    )
    assert (post_source_span(None, clause), "body") in element_fact[1]


def test_nested_filters_reject_before_inner_clause_and_element(activity):
    book, build = activity
    tree, point, has_edge = build(
        "def root(rows):\n"
        "    return [(left, right) for left in rows if left "
        "for right in left if right]\n"
    )
    display = tree.body[0].body[0].value
    outer, inner = display.generators
    outer_guard = point(outer, "filter_0_select")
    inner_guard = point(inner, "filter_0_select")
    assert has_edge(point(outer.ifs[0], "exit"), outer_guard)
    assert has_edge(outer_guard, point(outer, "select"), "repeat")
    assert has_edge(outer_guard, point(inner, "enter"), "body")
    assert has_edge(point(inner.ifs[0], "exit"), inner_guard)
    assert has_edge(inner_guard, point(inner, "select"), "repeat")
    assert has_edge(inner_guard, point(display.elt, "enter"), "body")
    elt_owners = book.page(SOURCE_ACTIVITY_NODE).cell(
        point(display.elt, "enter").row,
        point(display.elt, "enter").column,
    )[1]
    assert elt_owners[-2:] == (
        (post_source_span(None, outer), "body"),
        (post_source_span(None, inner), "body"),
    )


def test_generator_expression_keeps_existing_lazy_boundary(activity):
    _book, build = activity
    tree, point, _has_edge = build(
        "def root(items):\n    return sum(item for item in items)\n"
    )
    generator = tree.body[0].body[0].value.args[0].generators[0]
    assert point(generator, "select") is None
