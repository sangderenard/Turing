"""A transferred mapping member is read only through an exact keyed get."""

import ast
from types import SimpleNamespace

import pytest

from src.common.tensors.topological_reducer import reduce_abstract_tensor_topology
from src.compiler.concordance_declarations import SOURCE_MAPPING_GET_READ
from src.compiler.exception_graph import MissingResourceControlTopology
from src.compiler.identity_concordance import (
    CorrelationTable, begin_identity_book, end_identity_book,
)
from src.compiler.source_mapping_get import post_mapping_get_reads
from src.transmogrifier.graph.graph_express2 import ProcessGraph
from tests.test_cross_function_mapping_resource_evidence import SOURCE


def _reduced(source):
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        graph.build_from_ast(ast.parse(source))
        with pytest.raises(MissingResourceControlTopology):
            reduce_abstract_tensor_topology(graph)
        return book, graph
    finally:
        end_identity_book(token)


def test_exact_get_read_has_stored_member_and_repeats_in_reverse_order():
    book, graph = _reduced(SOURCE)
    page = book.page(SOURCE_MAPPING_GET_READ)
    rows = tuple(page.rows())
    assert len(rows) == 1
    row = rows[0]
    assert row[0] == graph.function_table.entry('consume').graph.G.graph[
        'authored_function_identity'
    ]
    fact = page.latest(row)
    assert fact[3].row[-1] != fact[4].row[-1]  # Separate authored key nodes.
    assert fact[5].row[-1] == 4  # Exact publish IndexedStore value member.
    assert not CorrelationTable._source_mapping_get_read_findings(
        SimpleNamespace(metadata={'identity_book': book},
                        function_table=graph.function_table),
    )
    before = len(tuple(page.history(row)))
    reversed_entries = tuple(reversed(tuple(graph.function_table)))
    post_mapping_get_reads(reversed_entries, book=book)
    post_mapping_get_reads(reversed_entries, book=book)
    assert tuple(page.rows()) == rows
    assert len(tuple(page.history(row))) == before


@pytest.mark.parametrize('source', [
    SOURCE.replace('metadata.get("identity_book")', 'metadata.get("other")'),
    SOURCE.replace('    book = module.metadata.get("identity_book")',
                   '    module.metadata["identity_book"] = IdentityBook()\n'
                   '    book = module.metadata.get("identity_book")'),
])
def test_key_mismatch_or_intervening_write_has_no_get_read(source):
    book, _graph = _reduced(source)
    assert not tuple(book.page(SOURCE_MAPPING_GET_READ).rows())
