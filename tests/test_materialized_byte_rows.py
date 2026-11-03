"""Exact builtin byte iteration refines a materialized sequence row."""

import inspect
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.common.tensors.compression import octet_huffman
from src.compiler import glsl_deployment_strategy as strategy
from src.compiler.concordance_declarations import (
    BUILTIN_ITERABLE_ELEMENT, SEQUENCE_ROW_DTYPE, SOURCE_PYTHON_IDENTITY,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable, current_identity_book
from src.compiler.planner_concordance_posts import materialized_sequence_row_dtype


def _policy(values):
    return ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({"records": {}, "bindings": [], "values": values})


def _materializer_id(graph, book):
    from src.common.tensors.topological_reducer import _source_numeric_scope

    scope = _source_numeric_scope(graph.G)
    matches = []
    for node_id, data in graph.G.nodes(data=True):
        row = (data.get("attributes") or {}).get("source_python_identity_row")
        if row is None:
            row = (scope, int(node_id))
        identity = book.latest_ref(SOURCE_PYTHON_IDENTITY, row)
        if identity is None:
            continue
        fact = book.page(SOURCE_PYTHON_IDENTITY).cell(identity.row, identity.column)
        if (fact[0] == "builtins.list"
                and (data.get("attributes") or {}).get("producer_kind")
                == "aggregate_materialization"):
            matches.append(int(node_id))
    assert len(matches) == 1
    return matches[0]


def test_octets_byte_materializer_posts_one_concorded_int64_row(monkeypatch):
    class Captured(Exception):
        pass

    observed = []
    original = strategy._build_shell_hierarchy_plan

    def capture(shell, *args, **kwargs):
        graph = getattr(shell, "process_graph", None)
        if graph is not None and graph.G.graph.get("function_name") == "_octets":
            book = current_identity_book()
            node_id = _materializer_id(graph, book)
            dtype, proof = materialized_sequence_row_dtype(graph, node_id)
            assert dtype == "int64" and proof
            receipt = proof[0]
            assert receipt.page == SEQUENCE_ROW_DTYPE
            assert book.page(SEQUENCE_ROW_DTYPE).cell(
                receipt.row, receipt.column,
            ) == (("int64",), "builtin iterable materialization")
            assert any(cell.page == BUILTIN_ITERABLE_ELEMENT for cell in proof)
            module = SimpleNamespace(metadata={"identity_book": book})
            assert CorrelationTable._materialized_byte_row_findings(module) == []
            page = book.page(SEQUENCE_ROW_DTYPE)
            original_fact = page.cells[(receipt.row, receipt.column)]
            try:
                page.cells[(receipt.row, receipt.column)] = (
                    ("float32",), original_fact[1],
                )
                assert CorrelationTable._materialized_byte_row_findings(module)
            finally:
                page.cells[(receipt.row, receipt.column)] = original_fact
            identity = next(cell for cell in proof
                            if cell.page == SOURCE_PYTHON_IDENTITY)
            identity_page = book.page(SOURCE_PYTHON_IDENTITY)
            identity_fact = identity_page.cells[(identity.row, identity.column)]
            try:
                identity_page.cells[(identity.row, identity.column)] = (
                    "builtins.set", *identity_fact[1:],
                )
                assert CorrelationTable._materialized_byte_row_findings(module)
            finally:
                identity_page.cells[(identity.row, identity.column)] = identity_fact
            observed.append(receipt)
            raise Captured
        return original(shell, *args, **kwargs)

    monkeypatch.setattr(strategy, "_build_shell_hierarchy_plan", capture)
    with pytest.raises(Captured):
        lower_ast_source_to_ssa(
            textwrap.dedent(inspect.getsource(octet_huffman._octets)),
            "_octets", python_bindings=vars(octet_huffman),
            source_module=octet_huffman.__name__,
            extraction_contract=_policy([{
                "function": "_octets", "parameter": "data", "storage": "span",
                "dtype": "int64", "rank": 1, "python_type": "builtins.bytes",
            }]),
            progress=lambda _: None, progress_bars=False,
        )
    assert len(observed) == 1


def test_generic_list_input_does_not_inherit_byte_row_contract(monkeypatch):
    class Captured(Exception):
        pass

    original = strategy._build_shell_hierarchy_plan
    observed = []

    def capture(shell, *args, **kwargs):
        graph = getattr(shell, "process_graph", None)
        if graph is not None and graph.G.graph.get("function_name") == "root":
            book = current_identity_book()
            node_id = _materializer_id(graph, book)
            assert materialized_sequence_row_dtype(graph, node_id) == (None, ())
            assert not book.page(BUILTIN_ITERABLE_ELEMENT).rows()
            observed.append(node_id)
            raise Captured
        return original(shell, *args, **kwargs)

    monkeypatch.setattr(strategy, "_build_shell_hierarchy_plan", capture)
    with pytest.raises(Captured):
        lower_ast_source_to_ssa(
            "def root(values):\n"
            "    return AbstractTensor.tensor(list(values))\n",
            "root", python_bindings={"AbstractTensor": octet_huffman.AbstractTensor},
            extraction_contract=_policy([{
                "function": "root", "parameter": "values", "storage": "span",
                "dtype": "int64", "rank": 1, "python_type": "builtins.list",
            }]),
            progress=lambda _: None, progress_bars=False,
        )
    assert len(observed) == 1
