"""A mapping copy keeps an exact missing state contract on the book."""

import ast
from types import SimpleNamespace

import pytest

from src.common.tensors.topological_reducer import reduce_abstract_tensor_topology
from src.compiler.concordance_declarations import (
    PROCESS_GRAPH_OPERANDS, REDUCTION, SOURCE_MAPPING_GET_COPY_STATE_DEMAND,
    SOURCE_MAPPING_GET_READ, SOURCE_FUNCTION_EXIT_FIELD_STATE,
    SOURCE_CALL_FIELD_STATE_TRANSFER, SOURCE_ACTIVITY_EDGE, SOURCE_SPAN,
    REDUCER_FIELD_STATE, SOURCE_GUARDED_MAPPING_PAYLOAD,
    SOURCE_VALUE_CLASS, SOURCE_AGGREGATE_LAYOUT, SOURCE_PYTHON_IDENTITY,
    CALLABLE_IDENTITY, SOURCE_METHOD_IDENTITY,
    SOURCE_FUNCTION_REACHABILITY, SOURCE_CALL_OBJECT_ALIAS,
    SOURCE_CALL_ACTUAL_FORMAL, SOURCE_PARAMETER_MEMORY_CONTRACT,
    CLASS_FIELD_DECLARATION, SOURCE_ACTIVITY_NODE, SOURCE_FORMAL_ORDER,
    SOURCE_CONSTRUCTOR_FIELD_EFFECT_DEMAND, CLASS_METHOD_DECLARATION,
    FUNCTION_ADDRESS, CLASS_DECLARATION,
    FieldStateKind,
)
from src.compiler.exception_graph import MissingResourceControlTopology
from src.compiler.identity_concordance import (
    CorrelationTable, Derived, Mode, begin_identity_book, end_identity_book,
)
from src.compiler.source_mapping_get import post_mapping_get_reads
from src.compiler.source_constructor_field_demand import post_constructor_field_demands
from src.transmogrifier.graph.graph_express2 import ProcessGraph
from tests.test_cross_function_mapping_resource_evidence import SOURCE


COPY_SOURCE = SOURCE.replace(
    '    book = module.metadata.get("identity_book")',
    '    copied = module.metadata.copy()\n'
    '    book = copied.get("identity_book")',
)
SECOND_PUBLISH_STORE = SOURCE.replace(
    '    module.metadata["identity_book"] = IdentityBook()',
    '    module.metadata["identity_book"] = IdentityBook()\n'
    '    module.metadata["other"] = IdentityBook()',
)


def _reduced(source):
    book, token = begin_identity_book()
    graph = ProcessGraph(materialize_memory=False)
    graph.build_from_ast(ast.parse(source))
    with pytest.raises(MissingResourceControlTopology):
        reduce_abstract_tensor_topology(graph)
    return book, token, graph


def test_mapping_copy_get_names_missing_state_contract_from_exact_operands():
    book, token, graph = _reduced(COPY_SOURCE)
    try:
        page = book.page(SOURCE_MAPPING_GET_COPY_STATE_DEMAND)
        rows = tuple(page.rows())
        assert len(rows) == 1
        row = rows[0]
        fact = page.latest(row)
        assert fact[0] == 'mapping_copy_alias_or_state_contract'
        transfer, observed, field_value, copy_mutation, copy_call, copy_inputs, *_ = fact[1:]
        assert book.page(PROCESS_GRAPH_OPERANDS).cell(
            copy_inputs.row, copy_inputs.column,
        ) == ((field_value, 'operand'),)
        assert copy_call.row[-1] == copy_mutation.row[-1]
        # The exact copy Call is connected to the observed field, but the
        # incumbent method classifier has not published a copy-state result.
        # A .get from that result must therefore remain an audited demand.
        copy_row = (row[0], copy_call.row[-1])
        assert book.latest_ref(SOURCE_VALUE_CLASS,
                               (row[0], field_value.row[-1])) is None
        for family in (SOURCE_VALUE_CLASS, SOURCE_AGGREGATE_LAYOUT,
                       SOURCE_PYTHON_IDENTITY, CALLABLE_IDENTITY,
                       SOURCE_METHOD_IDENTITY):
            assert book.latest_ref(family, copy_row) is None
        assert book.page(SOURCE_MAPPING_GET_READ).latest(row) is None
        assert not CorrelationTable._source_mapping_get_copy_state_demand_findings(
            SimpleNamespace(metadata={'identity_book': book},
                            function_table=graph.function_table),
        )
        before = tuple(book.clock)
        post_mapping_get_reads(tuple(reversed(tuple(graph.function_table))), book=book)
        assert tuple(book.clock) == before
        assert tuple(page.rows()) == rows
    finally:
        end_identity_book(token)


def test_direct_get_does_not_publish_copy_state_demand():
    book, token, _graph = _reduced(SOURCE)
    try:
        assert not tuple(book.page(SOURCE_MAPPING_GET_COPY_STATE_DEMAND).rows())
        assert len(tuple(book.page(SOURCE_MAPPING_GET_READ).rows())) == 1
    finally:
        end_identity_book(token)


def test_direct_attribute_exit_audit_rejects_wrong_final_field_state():
    book, token, graph = _reduced(COPY_SOURCE)
    try:
        init_graph = graph.function_table.entry('Module.__init__').graph.G
        scope = init_graph.graph['authored_function_identity']
        receipt = book.latest_ref(SOURCE_FUNCTION_EXIT_FIELD_STATE,
                                  (scope, 0, 'metadata'))
        fact = book.page(SOURCE_FUNCTION_EXIT_FIELD_STATE).cell(
            receipt.row, receipt.column)
        consume = graph.function_table.entry('consume').graph.G
        wrong = book.latest_ref(REDUCER_FIELD_STATE,
                                (consume.graph['lexical_read_scope'],
                                 0, 'metadata'))
        assert wrong is not None and wrong != fact[1]
        book.post(SOURCE_FUNCTION_EXIT_FIELD_STATE, receipt.row,
                  (fact[0], wrong, *fact[2:]), stage=REDUCTION,
                  provenance=Derived((receipt, wrong)), mode=Mode.REVISE)
        assert CorrelationTable._source_function_exit_field_state_findings(
            SimpleNamespace(metadata={'identity_book': book},
                            function_table=graph.function_table))
    finally:
        end_identity_book(token)


def test_init_dict_class_has_exact_field_exit_but_no_constructor_transfer():
    book, token, graph = _reduced(COPY_SOURCE)
    try:
        init_graph = graph.function_table.entry('Module.__init__').graph.G
        init_scope = init_graph.graph['authored_function_identity']
        lexical_scope = init_graph.graph['lexical_read_scope']
        state_page = book.page(REDUCER_FIELD_STATE)
        written = tuple(
            state_page.latest(row) for row in state_page.rows()
            if row == (lexical_scope, 0, 'metadata')
        )
        assert len(written) == 1
        state = written[0]
        assert state.kind is FieldStateKind.WRITTEN
        assert state.value.row[-1] == 1
        assert book.page(SOURCE_VALUE_CLASS).latest((init_scope, 1))[0] == 'builtins.dict'
        layouts = tuple(
            book.page(SOURCE_AGGREGATE_LAYOUT).latest(row)
            for row in book.page(SOURCE_AGGREGATE_LAYOUT).rows()
            if row == (lexical_scope, 1)
        )
        assert len(layouts) == 1 and layouts[0].aggregate_kind == 'dict'
        exit_cell = book.latest_ref(SOURCE_FUNCTION_EXIT_FIELD_STATE,
                                    (init_scope, 0, 'metadata'))
        assert exit_cell is not None
        formal, final_state, exit_activity, assignment, flow = (
            book.page(SOURCE_FUNCTION_EXIT_FIELD_STATE).cell(
                exit_cell.row, exit_cell.column)
        )
        assert formal.row[-1] == 0
        assert final_state == book.latest_ref(REDUCER_FIELD_STATE,
                                              (lexical_scope, 0, 'metadata'))
        assert book.page(SOURCE_ACTIVITY_NODE).cell(
            exit_activity.row, exit_activity.column) == ('FunctionDef', ())
        assert assignment.row[-1] == (('body', 0),)
        assert flow.row[1].row[-1] == 'exit'
        assert book.latest_ref(SOURCE_FORMAL_ORDER, (init_scope,)) in {
            source for source, _stage in book.edges_into(exit_cell)
        } or any(source.page == SOURCE_FUNCTION_EXIT_FIELD_STATE
                 for source, _stage in book.edges_into(exit_cell))
        original = next(source for source, _stage in book.edges_into(exit_cell)
                        if source.page == SOURCE_FUNCTION_EXIT_FIELD_STATE)
        schema = book.latest_ref(CLASS_FIELD_DECLARATION,
                                 (init_scope[0], 'Module', 'metadata'))
        assert schema in {source for source, _stage in book.edges_into(original)}
        assert not CorrelationTable._source_function_exit_field_state_findings(
            SimpleNamespace(metadata={'identity_book': book},
                            function_table=graph.function_table))
        for family in (SOURCE_FUNCTION_REACHABILITY,
                       SOURCE_CALL_ACTUAL_FORMAL, SOURCE_CALL_OBJECT_ALIAS):
            assert not any(init_scope in row for row in book.page(family).rows())
        assert book.latest_ref(SOURCE_PARAMETER_MEMORY_CONTRACT,
                               (init_scope, 'self')) is None
        demands = tuple(book.page(SOURCE_CONSTRUCTOR_FIELD_EFFECT_DEMAND).rows())
        assert len(demands) == 1
        demand_row = demands[0]
        assert demand_row[1:] == (2, init_scope, 'metadata')
        demand_cell = book.latest_ref(SOURCE_CONSTRUCTOR_FIELD_EFFECT_DEMAND,
                                      demand_row)
        demand = book.page(SOURCE_CONSTRUCTOR_FIELD_EFFECT_DEMAND).cell(
            demand_cell.row, demand_cell.column)
        assert demand[0] == 'constructor_receiver_effect_unproved'
        assert demand[3].page == CLASS_DECLARATION
        assert demand[4].page == CLASS_METHOD_DECLARATION
        assert demand[5].page == FUNCTION_ADDRESS
        assert demand[6] == exit_cell
        assert not CorrelationTable._source_constructor_field_effect_demand_findings(
            SimpleNamespace(metadata={'identity_book': book},
                            function_table=graph.function_table))
        before = tuple(book.clock)
        post_constructor_field_demands(tuple(reversed(tuple(graph.function_table))),
                                       book=book)
        assert tuple(book.clock) == before
    finally:
        end_identity_book(token)


def test_second_publisher_store_preserves_earlier_distinct_key():
    book, token, graph = _reduced(SECOND_PUBLISH_STORE)
    try:
        exit_rows = tuple(row for row in
                          book.page(SOURCE_FUNCTION_EXIT_FIELD_STATE).rows()
                          if row[0][1] == 'publish')
        assert len(exit_rows) == 2  # Ingestion and canonical projection.
        canonical = next(row for row in exit_rows if row[1] == 1)
        ingestion = next(row for row in exit_rows if row != canonical)
        original = book.latest_ref(SOURCE_FUNCTION_EXIT_FIELD_STATE, ingestion)
        original_fact = book.page(SOURCE_FUNCTION_EXIT_FIELD_STATE).cell(
            original.row, original.column,
        )
        original_causes = {source for source, _stage in book.edges_into(original)}
        assert original_fact[1].column == 2
        assert len(tuple(source for source in original_causes
                         if source.page == SOURCE_ACTIVITY_EDGE)) == 2
        assert {source.row[-1] for source in original_causes
                if source.page == SOURCE_SPAN} == {
                    (('body', 0),), (('body', 1),),
                }
        assert any(source.page == REDUCER_FIELD_STATE
                   and source.column == 1 for source in original_causes)
        final_state = book.page(SOURCE_FUNCTION_EXIT_FIELD_STATE).latest(
            canonical,
        )[1]
        assert final_state.column == 2  # OBSERVED, first write, second write.
        assert len(tuple(book.page(SOURCE_CALL_FIELD_STATE_TRANSFER).rows())) == 1
        read_rows = tuple(book.page(SOURCE_MAPPING_GET_READ).rows())
        assert len(read_rows) == 1
        read = book.page(SOURCE_MAPPING_GET_READ).latest(read_rows[0])
        assert read[1].row[-1] == 5  # Earlier identity_book store.
        assert len(read) == 20 and len(read[19][1]) == 1
        assert read[19][1][0][4].row[-1] == 9  # Later other store.
        guard_page = book.page(SOURCE_GUARDED_MAPPING_PAYLOAD)
        guard_row = next(iter(guard_page.rows()))
        assert guard_page.latest(guard_row)[1] == read[5]
        assert not tuple(book.page(SOURCE_MAPPING_GET_COPY_STATE_DEMAND).rows())
        assert not CorrelationTable._source_function_exit_field_state_findings(
            SimpleNamespace(metadata={'identity_book': book},
                            function_table=graph.function_table),
        )
        assert not CorrelationTable._source_mapping_get_read_findings(
            SimpleNamespace(metadata={'identity_book': book},
                            function_table=graph.function_table),
        )
    finally:
        end_identity_book(token)


def test_copy_state_demand_audit_rejects_wrong_get_operand():
    book, token, graph = _reduced(COPY_SOURCE)
    try:
        page = book.page(SOURCE_MAPPING_GET_COPY_STATE_DEMAND)
        row = next(iter(page.rows()))
        original = book.latest_ref(SOURCE_MAPPING_GET_COPY_STATE_DEMAND, row)
        fact = page.cell(original.row, original.column)
        wrong = (*fact[:9], fact[6], *fact[10:])
        book.post(
            SOURCE_MAPPING_GET_COPY_STATE_DEMAND, row, wrong,
            stage=REDUCTION, provenance=Derived((original, fact[6])),
            mode=Mode.REVISE,
        )
        assert CorrelationTable._source_mapping_get_copy_state_demand_findings(
            SimpleNamespace(metadata={'identity_book': book},
                            function_table=graph.function_table),
        )
    finally:
        end_identity_book(token)


def test_constructor_demand_audit_rejects_wrong_method_address():
    book, token, graph = _reduced(COPY_SOURCE)
    try:
        page = book.page(SOURCE_CONSTRUCTOR_FIELD_EFFECT_DEMAND)
        row = next(iter(page.rows()))
        cell = book.latest_ref(SOURCE_CONSTRUCTOR_FIELD_EFFECT_DEMAND, row)
        fact = page.cell(cell.row, cell.column)
        wrong = (*fact[:5], fact[6], *fact[6:])
        book.post(SOURCE_CONSTRUCTOR_FIELD_EFFECT_DEMAND, row, wrong,
                  stage=REDUCTION, provenance=Derived((cell, fact[6])),
                  mode=Mode.REVISE)
        assert CorrelationTable._source_constructor_field_effect_demand_findings(
            SimpleNamespace(metadata={'identity_book': book},
                            function_table=graph.function_table))
    finally:
        end_identity_book(token)
