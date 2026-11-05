"""An authored try's normal successor reads its actual protected value."""

import ast
import contextlib
import io
from types import SimpleNamespace

import pytest

from src.common.tensors.topological_reducer import (
    node_identity_cell,
    reduce_abstract_tensor_topology,
)
from src.compiler.concordance_declarations import (
    LEXICAL_READ_BINDING,
    SOURCE_EXCEPTION_NORMAL_BINDING,
)
from src.compiler.identity_concordance import (
    CorrelationTable,
    begin_identity_book,
    end_identity_book,
)
from src.compiler.source_exception_ingestion import ingest_exception_syntax
from src.transmogrifier.graph.graph_express2 import ProcessGraph


SOURCE = '''
def root(value):
    out: list[int] = []
    try:
        integer = int(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError("invalid") from error
    out.append(integer)
    return out
'''


def _reduce(source):
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(
                ast.parse(source),
                resolve_unresolved_parents=True,
                source_control_normalizers=(
                    lambda tree: ingest_exception_syntax(
                        tree, module_name="exception_normal_probe",
                    ),
                ),
            )
            reduce_abstract_tensor_topology(graph)
        return graph.function_table.entry("root").graph.G, book
    finally:
        end_identity_book(token)


def _findings(book):
    return CorrelationTable._operator_safety_findings(
        SimpleNamespace(metadata={"identity_book": book}, functions={}),
    )


def test_protected_integer_append_reads_exact_normal_continuation():
    graph, book = _reduce(SOURCE)
    append_id, append = next(
        (value_id, data) for value_id, data in graph.nodes(data=True)
        if data.get("op") == "append"
    )
    producer_id, = (
        value_id for value_id, data in graph.nodes(data=True)
        if data.get("op") == "int"
    )
    assert (producer_id, "arg:0") in append["parents"]

    page = book.page(SOURCE_EXCEPTION_NORMAL_BINDING)
    row, = page.rows()
    receipt = book.latest_ref(SOURCE_EXCEPTION_NORMAL_BINDING, row)
    producer_cell = node_identity_cell(graph, producer_id, book=book)
    assert page.latest(row)[2] in {
        source for source, _stage in book.edges_into(producer_cell)
    }
    read = next(
        (book.latest_ref(LEXICAL_READ_BINDING, read_row)
         for read_row in book.page(LEXICAL_READ_BINDING).rows()
         if read_row[1:] == (append_id, "arg:0", 0)),
        None,
    )
    assert read is not None
    visited, pending = set(), [read]
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        pending.extend(source for source, _stage in book.edges_into(current))
    assert receipt in visited
    assert page.latest(row)[1] in visited
    assert not [finding for finding in _findings(book)
                if finding.kind == "exception-normal-binding"]


def test_each_direct_protected_assignment_reaches_its_normal_successor():
    source = SOURCE.replace(
        "integer = int(value)",
        "numeric = float(value)\n        integer = int(numeric)",
    ).replace("out.append(integer)", "out.append(numeric)\n    out.append(integer)")
    graph, book = _reduce(source)
    receipts = tuple(book.page(SOURCE_EXCEPTION_NORMAL_BINDING).rows())
    assert {book.page(SOURCE_EXCEPTION_NORMAL_BINDING).latest(row)[0]
            for row in receipts} == {"numeric", "integer"}
    producers = {
        data.get("op"): value_id
        for value_id, data in graph.nodes(data=True)
        if data.get("op") in {"float", "int"}
    }
    append_inputs = {
        parent for _value_id, data in graph.nodes(data=True)
        if data.get("op") == "append"
        for parent, role in data.get("parents", ()) if role == "arg:0"
    }
    assert append_inputs == set(producers.values())
    assert not [finding for finding in _findings(book)
                if finding.kind == "exception-normal-binding"]


@pytest.mark.parametrize("source", [
    SOURCE.replace(
        'raise RuntimeError("invalid") from error',
        'out.append(0)',
    ),
    SOURCE.replace(
        'integer = int(value)',
        'if value > 0:\n            integer = int(value)',
    ),
    SOURCE.replace(
        '    out.append(integer)',
        '    finally:\n        integer = 9\n    out.append(integer)',
    ),
])
def test_unproved_continuation_does_not_publish_binding(source):
    _graph, book = _reduce(source)
    assert not book.page(SOURCE_EXCEPTION_NORMAL_BINDING).rows()


def test_forged_handler_and_producer_proofs_are_rejected():
    _graph, book = _reduce(SOURCE)
    page = book.page(SOURCE_EXCEPTION_NORMAL_BINDING)
    row, = page.rows()
    receipt = book.latest_ref(SOURCE_EXCEPTION_NORMAL_BINDING, row)
    key = (receipt.row, receipt.column)
    original = page.cells[key]
    try:
        changed = list(original)
        changed[4] = (original[3],)  # A Try is not a terminating Raise.
        page.cells[key] = tuple(changed)
        assert any(f.kind == "exception-normal-binding"
                   for f in _findings(book))
        changed = list(original)
        changed[2] = original[5]  # The barrier is not the value producer.
        page.cells[key] = tuple(changed)
        assert any(f.kind == "exception-normal-binding"
                   for f in _findings(book))
    finally:
        page.cells[key] = original
