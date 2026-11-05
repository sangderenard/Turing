"""A runtime scalar tuple gives list(values) its complete booked row dtype."""

from dataclasses import replace

import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler import planner_concordance_posts as posts
from src.compiler.concordance_declarations import (
    CANONICAL_VALUE, PROCESS_GRAPH_OPERANDS, SEQUENCE_ROW_DTYPE, SOURCE_VALUE_CLASS,
    VALUE_TYPE_ABI,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import (
    CorrelationTable, EDGE_PAGE, current_identity_book,
)
from src.transmogrifier.ssa import IRModule


class _RowInspected(Exception):
    """Stop at the proven row before unrelated tensor lowering."""


@pytest.mark.parametrize("members", ("start", "start, start + 1", "start, start"))
def test_scalar_tuple_materializer_cites_every_runtime_member(
    monkeypatch, members,
):
    contract = ExtractionContract(
        "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [{
            "function": "root", "parameter": "start", "storage": "scalar",
            "dtype": "int64", "rank": 0, "python_type": "builtins.int",
        }], "records": {}, "bindings": [],
    })
    source = (
        "def make(values):\n"
        "    return AbstractTensor.tensor(list(values)).to_dtype('int64')\n"
        "def root(start: int):\n"
        f"    return make([{members}]).sum().item()\n"
    )
    captured = []
    original = posts.materialized_sequence_row_dtype

    def capture(graph, node_id):
        result = original(graph, node_id)
        if result[0] == "int64" and graph.G.graph.get("function_name") == "make":
            book = current_identity_book()
            scope = tuple(graph.G.graph["lexical_read_scope"])
            receipt = book.latest_ref(SEQUENCE_ROW_DTYPE, (scope, node_id))
            if receipt is not None and book.page(SEQUENCE_ROW_DTYPE).cell(
                receipt.row, receipt.column,
            ) == (("int64",), "specialized tuple scalar materialization"):
                captured.append((book, receipt))
                raise _RowInspected
        return result

    monkeypatch.setattr(posts, "materialized_sequence_row_dtype", capture)
    with pytest.raises(_RowInspected):
        lower_ast_source_to_ssa(
            source, "root", name="runtime_scalar_tuple_materialized_row",
            extraction_contract=contract,
            python_bindings={"AbstractTensor": AbstractTensor},
            tensor_ssa_reference=c_backend_repository_ssa_reference(),
            progress=lambda _: None, progress_bars=False,
        )
    assert captured
    book, receipt = captured[-1]
    assert receipt is not None
    assert book.page(SEQUENCE_ROW_DTYPE).cell(
        receipt.row, receipt.column,
    ) == (("int64",), "specialized tuple scalar materialization")
    module = IRModule({}, metadata={"identity_book": book})
    assert CorrelationTable._specialized_scalar_materialized_row_findings(
        module,
    ) == []

    # A two-member tuple cannot keep its row proof if one authored 'elts'
    # operand disappears from the source aggregate's exact operand page.
    if members != "start":
        tuple_ref, = (
            source for source, _ in book.edges_into(receipt)
            if source.page == PROCESS_GRAPH_OPERANDS and source.row[1] == 0
        )
        page = book.page(PROCESS_GRAPH_OPERANDS)
        original_fact = page.cells[(tuple_ref.row, tuple_ref.column)]
        assert len(original_fact) == 2
        page.cells[(tuple_ref.row, tuple_ref.column)] = original_fact[:1]
        try:
            assert CorrelationTable._specialized_scalar_materialized_row_findings(
                module,
            )
        finally:
            page.cells[(tuple_ref.row, tuple_ref.column)] = original_fact

    abi_ref = next(source for source, _ in book.edges_into(receipt)
                   if source.page == VALUE_TYPE_ABI)
    abi_page = book.page(VALUE_TYPE_ABI)
    abi_fact = abi_page.cell(abi_ref.row, abi_ref.column)
    abi_page.cells[(abi_ref.row, abi_ref.column)] = replace(
        abi_fact, physical_alternatives=(replace(
            abi_fact.physical_alternatives[0], dtype="float64",
        ),),
    )
    try:
        assert CorrelationTable._specialized_scalar_materialized_row_findings(
            module,
        )
    finally:
        abi_page.cells[(abi_ref.row, abi_ref.column)] = abi_fact

    edges = book.page(EDGE_PAGE)
    original_scope_rows = edges.scope_rows
    for missing_page in (SOURCE_VALUE_CLASS,):
        removed = {source.key for source, _ in book.edges_into(receipt)
                   if source.page == missing_page}
        assert removed
        monkeypatch.setattr(edges, "scope_rows", lambda scope: tuple(
            edge for edge in original_scope_rows(scope)
            if not (scope == receipt.key and edge[1] in removed)
        ))
        try:
            assert CorrelationTable._specialized_scalar_materialized_row_findings(
                module,
            )
        finally:
            monkeypatch.setattr(edges, "scope_rows", original_scope_rows)

    materializer = next(source for source, _ in book.edges_into(receipt)
                        if source.page == CANONICAL_VALUE
                        and source.row == receipt.row)
    monkeypatch.setattr(edges, "scope_rows", lambda scope: tuple(
        edge for edge in original_scope_rows(scope)
        if not (scope == receipt.key and edge[1] == materializer.key)
    ))
    try:
        assert CorrelationTable._specialized_scalar_materialized_row_findings(
            module,
        )
    finally:
        monkeypatch.setattr(edges, "scope_rows", original_scope_rows)
