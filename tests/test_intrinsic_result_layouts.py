"""Admitted intrinsic named results use the authored class field ledger."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.compiler.concordance_declarations import (
    CALL_RESULT_PROJECTION, SOURCE_NAMED_TUPLE_LAYOUT,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable, current_identity_book
from src.compiler.intrinsic_result_layouts import intrinsic_result_fields


def test_topk_named_result_uses_exact_authored_field_order(monkeypatch):
    import src.compiler.glsl_deployment_strategy as deployment

    class ReachedResultLayout(Exception):
        pass

    seen = []
    original = deployment._build_shell_hierarchy_plan

    def inspect(shell, *args, **kwargs):
        graph = shell.process_graph
        calls = tuple(
            int(node) for node, data in graph.G.nodes(data=True)
            if str(data.get("op") or "").casefold() == "topk"
        )
        if not calls:
            return original(shell, *args, **kwargs)
        assert len(calls) == 1
        call = calls[0]
        fields, proof = intrinsic_result_fields(graph, call)
        assert tuple(name for name, _ in fields) == ("values", "indices")
        assert all(cell is not None for cell in proof)
        assert fields[0][1] != fields[1][1]
        book = current_identity_book()
        module = SimpleNamespace(metadata={"identity_book": book})
        assert CorrelationTable._source_named_tuple_layout_findings(module) == []

        paths = deployment._post_call_result_projections(graph, call)
        selected = tuple(
            int(node) for node, data in graph.G.nodes(data=True)
            if str(data.get("op") or "").casefold() == "getattr"
            and (data.get("attributes") or {}).get("attribute") == "indices"
        )
        assert len(selected) == 1
        assert paths[(1,)] == selected[0]
        assert CorrelationTable._named_intrinsic_result_projection_findings(
            module,
        ) == []

        layout = proof[-1]
        layout_page = book.page(SOURCE_NAMED_TUPLE_LAYOUT)
        original_layout = layout_page.cells[(layout.row, layout.column)]
        try:
            base, members = original_layout
            layout_page.cells[(layout.row, layout.column)] = (
                base, tuple(reversed(members)),
            )
            assert CorrelationTable._source_named_tuple_layout_findings(module)
            assert CorrelationTable._named_intrinsic_result_projection_findings(
                module,
            )
        finally:
            layout_page.cells[(layout.row, layout.column)] = original_layout

        projection_page = book.page(CALL_RESULT_PROJECTION)
        projection_row = next(
            row for row in projection_page.rows()
            if row[1:] == (call, selected[0])
        )
        projection_cell = book.latest_ref(CALL_RESULT_PROJECTION, projection_row)
        original_path = projection_page.cells[
            (projection_row, projection_cell.column)
        ]
        try:
            projection_page.cells[(projection_row, projection_cell.column)] = (0,)
            assert CorrelationTable._named_intrinsic_result_projection_findings(
                module,
            )
        finally:
            projection_page.cells[
                (projection_row, projection_cell.column)
            ] = original_path
        seen.append(True)
        raise ReachedResultLayout

    monkeypatch.setattr(
        deployment, "_build_shell_hierarchy_plan", inspect,
    )
    policy = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "records": {}, "bindings": [],
        "values": [
            {"function": "root", "parameter": "values", "storage": "span",
             "dtype": "int64", "rank": 1,
             "python_type": "src.common.tensors.abstraction.AbstractTensor"},
            {"function": "root", "parameter": "k", "storage": "scalar",
             "dtype": "int64", "rank": 0, "python_type": "builtins.int"},
        ],
    })
    with pytest.raises(ReachedResultLayout):
        lower_ast_source_to_ssa(
            "def root(values,k):\n"
            "    return AbstractTensor.topk(values,k=k,dim=0).indices\n",
            "root", python_bindings={"AbstractTensor": AbstractTensor},
            name="topk_named_result_layout", extraction_contract=policy,
            progress=lambda _: None, progress_bars=False,
        )
    assert seen == [True]
