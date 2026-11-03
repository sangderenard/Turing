"""Grounded authored methods read the published source class of a formal."""

import ast

from src.common.tensors.topological_reducer import (
    _source_numeric_scope, reduce_abstract_tensor_topology,
)
from src.compiler.concordance_declarations import (
    CALL_BINDING, CLASS_DECLARATION, CLASS_METHOD_DECLARATION,
    FUNCTION_ADDRESS, SOURCE_VALUE_CLASS,
)
from src.compiler.identity_concordance import begin_identity_book, end_identity_book
from src.compiler.receiver_resolution_prepasses import (
    _resolve_grounded_method_references,
)
from src.compiler.source_evidence_resolver import ClassEvidence, require_class
from src.compiler import source_evidence_resolver
from src.compiler.source_evidence_resolver import UnresolvedSourceDemand
from src.transmogrifier.graph.graph_express2 import ProcessGraph


def _reduced(source):
    book, token = begin_identity_book()
    graph = ProcessGraph(materialize_memory=False)
    graph.build_from_ast(ast.parse(source))
    reduce_abstract_tensor_topology(graph)
    return graph, book, token


def test_annotated_formal_method_uses_exact_class_and_declarations():
    graph, book, token = _reduced('''
class CodeBuilder:
    def to_body(self):
        return 1
def run(body: CodeBuilder):
    return body.to_body()
''')
    try:
        function = graph.function_table.entry('run').graph
        scope = _source_numeric_scope(function)
        receiver = next(int(node) for node, data in function.G.nodes(data=True)
                        if data.get('type') == 'Input'
                        and (data.get('attributes') or {}).get('binding_name') == 'body')
        call = next(int(node) for node, data in function.G.nodes(data=True)
                    if isinstance(data.get('expr_obj'), ast.Call)
                    and isinstance(data['expr_obj'].func, ast.Attribute))
        evidence = require_class(scope, receiver, book=book)
        assert isinstance(evidence, ClassEvidence)
        assert evidence.class_identity == 'CodeBuilder'
        function.G.graph['parameter_record_abi'] = {}
        function.G.graph['planner_specializations'] = {}
        function.G.nodes[receiver]['attributes'].pop('class_ref', None)
        function.G.nodes[call]['attributes'].pop('method_ref', None)
        function.G.nodes[call]['attributes'].pop('callee_ref', None)
        _resolve_grounded_method_references(function)
        assert function.G.nodes[call]['attributes']['method_ref'] == (
            graph.G.graph['class_table']['CodeBuilder']['methods']['to_body'])
        binding = book.latest_ref(CALL_BINDING,
                                  (function.G.graph['lexical_read_scope'], call))
        assert binding is not None
        causes = {cell for cell, _stage in book.edges_into(binding)}
        assert evidence.subject_cell in causes
        assert book.latest_ref(CLASS_DECLARATION,
            (scope[0], 'CodeBuilder')) in causes
        assert book.latest_ref(CLASS_METHOD_DECLARATION,
            (scope[0], 'CodeBuilder', 'to_body')) in causes
        assert any(cell.page == FUNCTION_ADDRESS for cell in causes)
    finally:
        end_identity_book(token)


def test_untyped_formal_does_not_select_unique_same_named_method():
    graph, book, token = _reduced('''
class CodeBuilder:
    def to_body(self):
        return 1
def run(body):
    return body.to_body()
''')
    try:
        function = graph.function_table.entry('run').graph
        scope = _source_numeric_scope(function)
        receiver = next(int(node) for node, data in function.G.nodes(data=True)
                        if data.get('type') == 'Input'
                        and (data.get('attributes') or {}).get('binding_name') == 'body')
        assert book.latest_ref(SOURCE_VALUE_CLASS, (scope, receiver)) is None
        call = next(int(node) for node, data in function.G.nodes(data=True)
                    if isinstance(data.get('expr_obj'), ast.Call)
                    and isinstance(data['expr_obj'].func, ast.Attribute))
        function.G.graph['parameter_record_abi'] = {}
        function.G.graph['planner_specializations'] = {}
        _resolve_grounded_method_references(function)
        assert function.G.nodes[call]['attributes'].get('method_ref') is None
        assert book.latest_ref(CALL_BINDING,
            (function.G.graph['lexical_read_scope'], call)) is None
    finally:
        end_identity_book(token)


def test_constructor_binding_keeps_both_declaration_causes():
    graph, book, token = _reduced('''
class CodeBuilder:
    def __init__(self):
        self.value = 1
def run():
    return CodeBuilder()
''')
    try:
        function = graph.function_table.entry('run').graph
        scope = _source_numeric_scope(function)
        call = next(int(node) for node, data in function.G.nodes(data=True)
                    if isinstance(data.get('expr_obj'), ast.Call)
                    and isinstance(data['expr_obj'].func, ast.Name)
                    and data['expr_obj'].func.id == 'CodeBuilder')
        function.G.nodes[call]['attributes'].pop('constructor_ref', None)
        _resolve_grounded_method_references(function)
        binding = book.latest_ref(CALL_BINDING,
                                  (function.G.graph['lexical_read_scope'], call))
        assert binding is not None
        causes = {cell for cell, _stage in book.edges_into(binding)}
        assert book.latest_ref(CLASS_DECLARATION,
            (scope[0], 'CodeBuilder')) in causes
        assert book.latest_ref(CLASS_METHOD_DECLARATION,
            (scope[0], 'CodeBuilder', '__init__')) in causes
        assert any(cell.page == FUNCTION_ADDRESS for cell in causes)
    finally:
        end_identity_book(token)


def test_constructed_receiver_method_uses_exact_class_and_declarations():
    graph, book, token = _reduced('''
class Builder:
    def __init__(self):
        self.value = 1
    def to_body(self):
        return self.value
def run():
    return Builder().to_body()
''')
    try:
        function = graph.function_table.entry('run').graph
        scope = _source_numeric_scope(function)
        constructor = next(int(node) for node, data in function.G.nodes(data=True)
                           if isinstance(data.get('expr_obj'), ast.Call)
                           and isinstance(data['expr_obj'].func, ast.Name)
                           and data['expr_obj'].func.id == 'Builder')
        method_call = next(int(node) for node, data in function.G.nodes(data=True)
                           if isinstance(data.get('expr_obj'), ast.Call)
                           and isinstance(data['expr_obj'].func, ast.Attribute)
                           and data['expr_obj'].func.attr == 'to_body')
        evidence = require_class(scope, constructor, book=book)
        assert isinstance(evidence, ClassEvidence)
        assert evidence.class_identity == 'Builder'
        function.G.nodes[constructor]['attributes'].pop('class_ref', None)
        function.G.nodes[method_call]['attributes'].pop('method_ref', None)
        function.G.nodes[method_call]['attributes'].pop('callee_ref', None)
        _resolve_grounded_method_references(function)
        assert function.G.nodes[method_call]['attributes']['method_ref'] == (
            graph.G.graph['class_table']['Builder']['methods']['to_body'])
        binding = book.latest_ref(CALL_BINDING,
                                  (function.G.graph['lexical_read_scope'], method_call))
        assert binding is not None
        causes = {cell for cell, _stage in book.edges_into(binding)}
        assert evidence.subject_cell in causes
        assert book.latest_ref(CLASS_DECLARATION,
            (scope[0], 'Builder')) in causes
        assert book.latest_ref(CLASS_METHOD_DECLARATION,
            (scope[0], 'Builder', 'to_body')) in causes
    finally:
        end_identity_book(token)


def test_constructed_receiver_without_class_cell_does_not_use_graph_class_ref(
    monkeypatch,
):
    graph, book, token = _reduced('''
class Builder:
    def __init__(self):
        self.value = 1
    def to_body(self):
        return self.value
def run():
    return Builder().to_body()
''')
    try:
        function = graph.function_table.entry('run').graph
        constructor = next(int(node) for node, data in function.G.nodes(data=True)
                           if isinstance(data.get('expr_obj'), ast.Call)
                           and isinstance(data['expr_obj'].func, ast.Name)
                           and data['expr_obj'].func.id == 'Builder')
        method_call = next(int(node) for node, data in function.G.nodes(data=True)
                           if isinstance(data.get('expr_obj'), ast.Call)
                           and isinstance(data['expr_obj'].func, ast.Attribute)
                           and data['expr_obj'].func.attr == 'to_body')
        scope = _source_numeric_scope(function)
        original = source_evidence_resolver.require_class

        def missing_constructor_class(subject_scope, value_id, *, book):
            if value_id == constructor and subject_scope == scope:
                return UnresolvedSourceDemand(
                    subject_scope, value_id, 'class', 'missing_source_value_class')
            return original(subject_scope, value_id, book=book)

        monkeypatch.setattr(source_evidence_resolver, 'require_class',
                            missing_constructor_class)
        function.G.nodes[method_call]['attributes'].pop('method_ref', None)
        function.G.nodes[method_call]['attributes'].pop('callee_ref', None)
        assert function.G.nodes[constructor]['attributes']['class_ref'] == 'Builder'
        _resolve_grounded_method_references(function)
        assert function.G.nodes[method_call]['attributes'].get('method_ref') is None
        assert book.latest_ref(CALL_BINDING,
            (function.G.graph['lexical_read_scope'], method_call)) is None
    finally:
        end_identity_book(token)
