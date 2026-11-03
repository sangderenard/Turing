"""An authored method needs a concorded class for its formal receiver."""

import ast
from types import SimpleNamespace

from src.common.tensors.topological_reducer import reduce_abstract_tensor_topology
from src.compiler.concordance_declarations import (
    CALL_BINDING, CLASS_METHOD_DECLARATION, REDUCTION,
    SOURCE_METHOD_RECEIVER_CLASS_DEMAND, SOURCE_VALUE_CLASS,
)
from src.compiler.identity_concordance import (
    CorrelationTable, Derived, Mode, begin_identity_book, end_identity_book,
)
from src.compiler.receiver_resolution_prepasses import _resolve_grounded_method_references
from src.transmogrifier.graph.graph_express2 import ProcessGraph


SOURCE = '''\
class CodeBuilder:
    def to_body(self):
        return 1

def run(body):
    return body.to_body()
'''


def _reduced(source=SOURCE):
    book, token = begin_identity_book()
    graph = ProcessGraph(materialize_memory=False)
    graph.build_from_ast(ast.parse(source))
    reduce_abstract_tensor_topology(graph)
    return book, token, graph


def test_abi_hint_cannot_bind_method_without_formal_source_class():
    book, token, graph = _reduced()
    try:
        function = graph.function_table.entry('run').graph
        scope = function.G.graph['authored_function_identity']
        assert book.latest_ref(SOURCE_VALUE_CLASS, (scope, 0)) is None
        assert book.latest_ref(CLASS_METHOD_DECLARATION,
                               (scope[0], 'CodeBuilder', 'to_body')) is not None
        function.G.graph['parameter_record_abi'] = {
            'body': {'identity': 'CodeBuilder'}
        }
        function.G.nodes[0]['attributes']['program_abi_record_identity'] = 'CodeBuilder'
        _resolve_grounded_method_references(function)
        assert function.G.nodes[2]['attributes'].get('method_ref') is None
        assert book.latest_ref(CALL_BINDING,
                               (function.G.graph['lexical_read_scope'], 2)) is None
        row = (scope, 2, 0)
        cell = book.latest_ref(SOURCE_METHOD_RECEIVER_CLASS_DEMAND, row)
        assert cell is not None
        fact = book.page(SOURCE_METHOD_RECEIVER_CLASS_DEMAND).cell(
            cell.row, cell.column)
        assert fact[0] == 'missing_source_value_class'
        assert not CorrelationTable._source_method_receiver_class_demand_findings(
            SimpleNamespace(metadata={'identity_book': book},
                            function_table=graph.function_table))
        before = tuple(book.clock)
        _resolve_grounded_method_references(function)
        assert tuple(book.clock) == before
    finally:
        end_identity_book(token)


def test_method_receiver_demand_audit_rejects_wrong_operand():
    book, token, graph = _reduced()
    try:
        function = graph.function_table.entry('run').graph
        _resolve_grounded_method_references(function)
        page = book.page(SOURCE_METHOD_RECEIVER_CLASS_DEMAND)
        row, = tuple(page.rows())
        cell = book.latest_ref(SOURCE_METHOD_RECEIVER_CLASS_DEMAND, row)
        fact = page.cell(cell.row, cell.column)
        wrong = (*fact[:3], fact[1], *fact[4:])
        book.post(SOURCE_METHOD_RECEIVER_CLASS_DEMAND, row, wrong,
                  stage=REDUCTION, provenance=Derived((cell, fact[1])),
                  mode=Mode.REVISE)
        assert CorrelationTable._source_method_receiver_class_demand_findings(
            SimpleNamespace(metadata={'identity_book': book},
                            function_table=graph.function_table))
    finally:
        end_identity_book(token)
