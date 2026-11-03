"""A caller field-effect interval needs every normal source edge."""

import ast
from types import SimpleNamespace

import pytest

from src.common.tensors.topological_reducer import reduce_abstract_tensor_topology
from src.compiler.concordance_declarations import (
    REDUCTION, SOURCE_ACTIVITY_EDGE, SOURCE_CALL_ACTIVITY_CORRIDOR,
    SOURCE_CALL_FIELD_STATE_TRANSFER,
)
from src.compiler.exception_graph import MissingResourceControlTopology
from src.compiler.identity_concordance import (
    CorrelationTable, Derived, Mode, begin_identity_book, end_identity_book,
)
from src.transmogrifier.graph.graph_express2 import ProcessGraph
from tests.test_cross_function_mapping_resource_evidence import SOURCE


def _corridors(source):
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        graph.build_from_ast(ast.parse(source))
        with pytest.raises(MissingResourceControlTopology):
            reduce_abstract_tensor_topology(graph)
        rows = tuple(book.page(SOURCE_CALL_ACTIVITY_CORRIDOR).rows())
        assert not CorrelationTable._source_call_activity_corridor_findings(
            SimpleNamespace(metadata={"identity_book": book}),
        )
        return rows, book, graph
    finally:
        end_identity_book(token)


def test_straight_line_calls_have_one_normal_corridor():
    rows, book, graph = _corridors(SOURCE)
    scope = graph.function_table.entry("root").graph.G.graph[
        "authored_function_identity"
    ]
    matching = [row for row in rows if row[0] == scope and row[4] == "metadata"]
    assert len(matching) == 1
    fact = book.page(SOURCE_CALL_ACTIVITY_CORRIDOR).latest(matching[0])
    assert len(fact[6]) == 4  # Call exit, Expr exit, Return enter, Call enter.
    assert len(fact[7]) == 3
    transfers = tuple(book.page(SOURCE_CALL_FIELD_STATE_TRANSFER).rows())
    matching_transfer = [row for row in transfers
                         if row[0] == scope and row[5] == "metadata"]
    assert len(matching_transfer) == 1
    transfer = book.page(SOURCE_CALL_FIELD_STATE_TRANSFER).latest(
        matching_transfer[0],
    )
    assert transfer[1] == book.latest_ref(
        SOURCE_CALL_ACTIVITY_CORRIDOR, matching[0],
    )
    assert not CorrelationTable._source_call_field_state_transfer_findings(
        SimpleNamespace(metadata={"identity_book": book}),
    )


def test_corridor_audit_detects_later_alternate_successor():
    rows, book, _graph = _corridors(SOURCE)
    row = next(row for row in rows if row[4] == "metadata")
    fact = book.page(SOURCE_CALL_ACTIVITY_CORRIDOR).latest(row)
    start, end = fact[4], fact[5]
    book.post(
        SOURCE_ACTIVITY_EDGE, (row[0], start, end, "flow"), True,
        stage=REDUCTION, provenance=Derived((start, end)), mode=Mode.CONCORD,
    )
    assert CorrelationTable._source_call_activity_corridor_findings(
        SimpleNamespace(metadata={"identity_book": book}),
    )


@pytest.mark.parametrize("intervening", [
    '    module.metadata["identity_book"] = IdentityBook()\n',
    '    if name:\n        marker = 1\n',
])
def test_intervening_writer_or_branch_has_no_corridor(intervening):
    source = SOURCE.replace(
        "    publish(module)\n    return consume(module, name)",
        "    publish(module)\n" + intervening + "    return consume(module, name)",
    )
    rows, _book, graph = _corridors(source)
    scope = graph.function_table.entry("root").graph.G.graph[
        "authored_function_identity"
    ]
    assert not [row for row in rows if row[0] == scope]
    assert not [row for row in _book.page(
        SOURCE_CALL_FIELD_STATE_TRANSFER,
    ).rows() if row[0] == scope]
