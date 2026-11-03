"""Exact call-edge refusal for a tuple-shaped mapping key."""

import ast
from types import SimpleNamespace

import pytest

from src.common.tensors.topological_reducer import (
    reduce_abstract_tensor_topology,
)
from src.compiler.concordance_declarations import (
    SOURCE_CALL_ACTUAL_FORMAL,
    SOURCE_CALL_CLOSED_TUPLE_LENGTHS,
    SOURCE_CLOSED_AGGREGATE_LENGTHS,
    SOURCE_AGGREGATE_LAYOUT,
    SOURCE_FORMAL_CLOSED_TUPLE_LENGTHS,
    SOURCE_TUPLE_KEY_NATIVE_GAP,
    SEMANTIC_OUTPUT, CANONICAL_VALUE, CANONICAL_RELABEL,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import _field_slot_ops, lower_ast_source_to_ssa
from src.compiler.identity_concordance import (
    CorrelationTable, begin_identity_book, end_identity_book,
    Derived, Mode,
)
from src.compiler.source_evidence_resolver import (
    UnresolvedSourceDemand, require_length,
)
from src.transmogrifier.graph.graph_express2 import ProcessGraph


SOURCE = '''def key_of(flag: int) -> tuple[int, ...]:
    return (1, 2) if flag else (1, 2, 3)
def use(first: tuple[int, ...], second: tuple[int, ...]) -> int:
    d = {}
    d[first] = 5
    d[second] = 7
    return d.get(first, 0) * 10 + d.get(second, 0) + len(d) * 100
def root(flag: int) -> int:
    return use((1, 2), key_of(flag))
'''


def test_tuple_key_refusal_carries_exact_return_and_formal_cells():
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        graph.build_from_ast(ast.parse(SOURCE))
        reduce_abstract_tensor_topology(graph)
        key_graph = graph.function_table.entry("key_of").graph.G
        root_graph = graph.function_table.entry("root").graph.G
        use_graph = graph.function_table.entry("use").graph.G
        key_scope = tuple(key_graph.graph["authored_function_identity"])
        root_scope = tuple(root_graph.graph["authored_function_identity"])
        use_scope = tuple(use_graph.graph["authored_function_identity"])
        transport_row = (root_scope, 4, key_scope, 8)
        transport_ref = book.latest_ref(
            SOURCE_CALL_CLOSED_TUPLE_LENGTHS, transport_row,
        )
        assert transport_ref is not None
        transported = book.page(SOURCE_CALL_CLOSED_TUPLE_LENGTHS).cell(
            transport_ref.row, transport_ref.column,
        )
        assert transported[0].page == SEMANTIC_OUTPUT
        assert book.page(SEMANTIC_OUTPUT).cell(
            transported[0].row, transported[0].column,
        ) == ("result_0", 8)
        assert transported[1].row[1] == 8
        domain = book.page(SOURCE_CLOSED_AGGREGATE_LENGTHS).cell(
            transported[2].row, transported[2].column,
        )
        assert domain.lengths == (2, 3)
        binding_ref = book.latest_ref(
            SOURCE_CALL_ACTUAL_FORMAL, (root_scope, 5, use_scope, 1),
        )
        assert binding_ref is not None
        binding = book.page(SOURCE_CALL_ACTUAL_FORMAL).cell(
            binding_ref.row, binding_ref.column,
        )
        assert binding[0].row[1] == 4
        formal_scope = book.page(SOURCE_CALL_ACTUAL_FORMAL).cell(
            binding_ref.row, binding_ref.column,
        )[4].row[0]
        formal_ref = book.latest_ref(
            SOURCE_FORMAL_CLOSED_TUPLE_LENGTHS,
            (formal_scope, 1, root_scope, 5),
        )
        assert formal_ref is not None
        formal_length = require_length(formal_scope, 1, book=book)
        assert isinstance(formal_length, UnresolvedSourceDemand)
        assert formal_length.missing == (
            "closed_length_alternatives_require_native_layout"
        )
        assert formal_length.alternatives == (2, 3)
        assert formal_length.subject_cell == formal_ref
        assert {binding_ref, transport_ref, transported[0]} <= set(
            formal_length.proof_cells
        )
        with pytest.raises(
            ValueError, match=r"native_tuple_key_arm_projection_missing: "
                              r"store=6, formal=1, closed_lengths=\(2, 3\), "
                              r"operation=runtime_tuple_to_key_columns",
        ):
            _field_slot_ops(use_graph)
        gap_ref = book.latest_ref(SOURCE_TUPLE_KEY_NATIVE_GAP, (use_scope, 6))
        assert gap_ref is not None
        gap = book.page(SOURCE_TUPLE_KEY_NATIVE_GAP).cell(
            gap_ref.row, gap_ref.column,
        )
        assert gap[2] == formal_ref
        assert gap[5] == "runtime_tuple_to_key_columns"
        assert book.latest_ref(SOURCE_AGGREGATE_LAYOUT,
                               (formal_scope, 1)) is None
        module = SimpleNamespace(
            metadata={"identity_book": book},
            function_table=graph.function_table,
        )
        assert CorrelationTable._source_tuple_key_native_gap_findings(module) == []
        forged_operation = book.post(
            SOURCE_TUPLE_KEY_NATIVE_GAP, gap_ref.row,
            (*gap[:5], "native_tuple_hash"),
            stage=CANONICAL_RELABEL,
            provenance=Derived((gap_ref,)), mode=Mode.REVISE,
        )
        assert any(
            finding.kind == "source-tuple-key-native-gap-disagreement"
            for finding in CorrelationTable._source_tuple_key_native_gap_findings(
                module)
        )
        book.post(
            SOURCE_TUPLE_KEY_NATIVE_GAP, gap_ref.row, gap,
            stage=CANONICAL_RELABEL,
            provenance=Derived((forged_operation,)), mode=Mode.REVISE,
        )
        altered_source = book.latest_ref(
            CANONICAL_VALUE, (transported[0].row[0], 7),
        )
        book.post(
            SEMANTIC_OUTPUT, transported[0].row, ("result_0", 7),
            stage=CANONICAL_RELABEL,
            provenance=Derived((altered_source,)),
            mode=Mode.REVISE,
        )
        findings = CorrelationTable._source_tuple_key_native_gap_findings(
            module
        )
        assert any(
            finding.kind == "source-call-closed-tuple-length-disagreement"
            for finding in findings
        )
    finally:
        end_identity_book(token)


def test_sanctioned_lower_refuses_before_native_scalar_key_emission():
    contract = ExtractionContract(
        "extraction_contracts/program_extraction.yaml"
    ).with_execution_file(
        "extraction_contracts/vehicle_full_native_execution.yaml"
    )
    with pytest.raises(
        ValueError, match="native_tuple_key_arm_projection_missing",
    ):
        lower_ast_source_to_ssa(
            SOURCE, "root", name="tuple_key_native_gap",
            extraction_contract=contract, progress=lambda _message: None,
        )


def test_return_annotation_alone_does_not_publish_closed_length_transport():
    source = SOURCE.replace("else (1, 2, 3)", "else (1, 3)")
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        graph.build_from_ast(ast.parse(source))
        reduce_abstract_tensor_topology(graph)
        assert book.page(SOURCE_CALL_CLOSED_TUPLE_LENGTHS).rows() == ()
        assert book.page(SOURCE_FORMAL_CLOSED_TUPLE_LENGTHS).rows() == ()
    finally:
        end_identity_book(token)


def test_formal_length_domain_survives_reversed_definition_order():
    source = '''def root(flag: int) -> int:
    return use((1, 2), key_of(flag))
def use(first: tuple[int, ...], second: tuple[int, ...]) -> int:
    d = {}
    d[first] = 5
    d[second] = 7
    return d.get(first, 0) * 10 + d.get(second, 0) + len(d) * 100
def key_of(flag: int) -> tuple[int, ...]:
    return (1, 2) if flag else (1, 2, 3)
'''
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        graph.build_from_ast(ast.parse(source))
        reduce_abstract_tensor_topology(graph)
        rows = book.page(SOURCE_FORMAL_CLOSED_TUPLE_LENGTHS).rows()
        assert len(rows) == 1
        answer = require_length(rows[0][0], rows[0][1], book=book)
        assert isinstance(answer, UnresolvedSourceDemand)
        assert answer.alternatives == (2, 3)
        with pytest.raises(
            ValueError, match="native_tuple_key_arm_projection_missing",
        ):
            _field_slot_ops(graph.function_table.entry("use").graph.G)
    finally:
        end_identity_book(token)
