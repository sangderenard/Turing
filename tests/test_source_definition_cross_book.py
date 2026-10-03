import ast

from src.compiler.concordance_declarations import (
    BACKWARD_RULE_DEFINITION,
    INGESTION,
    SOURCE_SPAN,
)
from src.compiler.identity_concordance import (
    begin_identity_book,
    end_identity_book,
)
from src.compiler.process_graph_autograd import _post_backward_rule_definitions
from src.transmogrifier.graph.graph_express2 import (
    SourceDefinition,
    _annotate_visual_source_owners,
    post_source_span,
)


def _stamp_rule_tree(tree, source_cells):
    """Apply the source-cell stamping used by build_from_ast before owner restamping."""
    tree._turing_source_module = "<abstract-tensor-backward-rules>"
    for statement in tree.body:
        source_cell = source_cells.get(getattr(statement, "name", None))
        if source_cell is None:
            continue
        for descendant in ast.walk(statement):
            descendant._turing_source_cells = (source_cell,)
    tree._turing_source_cells_by_name = dict(source_cells)
    _annotate_visual_source_owners(tree)


def _post_rule_span_in_fresh_book(tree):
    book, token = begin_identity_book()
    try:
        source_cells = _post_backward_rule_definitions((), ())
        _stamp_rule_tree(tree, source_cells)
        definition = tree.body[0]
        operation = next(
            node for node in ast.walk(definition) if isinstance(node, ast.Add)
        )
        stamp = operation._turing_source_cells[0]
        assert isinstance(stamp, SourceDefinition)

        span = post_source_span(definition, operation)
        declaration = book.latest_ref(
            BACKWARD_RULE_DEFINITION, ("BACKWARD_RULES", "add"),
        )
        assert declaration is not None
        assert book.edges_into(span) == ((declaration, INGESTION),)
        assert book.latest_ref(SOURCE_SPAN, span.row) == span
        return declaration, span, stamp
    finally:
        end_identity_book(token)


def _post_saved_stamp_in_fresh_book(tree, saved_stamp):
    book, token = begin_identity_book()
    try:
        _post_backward_rule_definitions((), ())
        operation = next(
            node for node in ast.walk(tree.body[0]) if isinstance(node, ast.Add)
        )
        assert operation._turing_source_cells[0] is saved_stamp

        span = post_source_span(tree.body[0], operation)
        declaration = book.latest_ref(
            BACKWARD_RULE_DEFINITION, ("BACKWARD_RULES", "add"),
        )
        assert declaration is not None
        assert book.edges_into(span) == ((declaration, INGESTION),)
        assert operation._turing_source_cells[0] is saved_stamp
        return declaration, span
    finally:
        end_identity_book(token)


def test_cached_backward_rule_ast_resolves_source_stamps_per_book():
    # ast.Add is a CPython singleton shared across parsed trees. Reusing this
    # exact AST across books reproduces the lifetime boundary that previously
    # let a Ref from the first book reach post_source_span in the second.
    tree = ast.parse("def bw_add(x):\n    return x + x\n")

    first_declaration, first_span, saved_stamp = _post_rule_span_in_fresh_book(tree)
    saved_declaration, saved_span = _post_saved_stamp_in_fresh_book(
        tree, saved_stamp,
    )

    assert first_declaration.row == saved_declaration.row
    assert first_span.row == saved_span.row

    # A separate path exercises the owner visitor's normal re-stamp behavior
    # when the exact AST object is intentionally rebuilt under another book.
    restamped_tree = ast.parse("def bw_add(x):\n    return x + x\n")
    restamped_declaration, restamped_span, _ = _post_rule_span_in_fresh_book(
        restamped_tree,
    )
    next_declaration, next_span, _ = _post_rule_span_in_fresh_book(
        restamped_tree,
    )
    assert restamped_declaration.row == next_declaration.row
    assert restamped_span.row == next_span.row
