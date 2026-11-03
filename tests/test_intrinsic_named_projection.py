"""Named intrinsic result members retain their declared source slots."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.compiler import glsl_deployment_strategy as strategy
from src.compiler.concordance_declarations import (
    CALL_RESULT_PROJECTION, CLASS_FIELD_DECLARATION, PROCESS_GRAPH_OPERANDS,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable, current_identity_book


@pytest.mark.parametrize(("field", "position"), [("values", 0), ("indices", 1)])
def test_admitted_topk_named_projection_uses_authored_slot(monkeypatch, field, position):
    class Observed(Exception):
        pass

    original = strategy._build_shell_hierarchy_plan
    seen = []

    def capture(shell, *args, **kwargs):
        graph = getattr(shell, "process_graph", None)
        if graph is None or graph.G.graph.get("function_name") != "root":
            return original(shell, *args, **kwargs)
        calls = [int(node) for node, data in graph.G.nodes(data=True)
                 if str(data.get("op") or "").casefold() == "topk"]
        assert len(calls) == 1
        call, = calls
        projection, = [
            int(node) for node, data in graph.G.nodes(data=True)
            if str(data.get("op") or "").casefold() == "getattr"
            and (data.get("attributes") or {}).get("attribute") == field
            and (call, "value") in (data.get("parents") or ())
        ]
        assert strategy._tensor_descriptor_rule(graph, call) is None
        assert strategy._post_call_result_projections(graph, call) == {
            (position,): projection,
        }
        book = current_identity_book()
        receipt = book.latest_ref(
            CALL_RESULT_PROJECTION,
            (strategy._call_projection_scope(graph), call, projection),
        )
        assert receipt is not None
        causes = {cell for cell, _stage in book.edges_into(receipt)}
        assert any(cell.page == CLASS_FIELD_DECLARATION
                   and cell.row[-1] == field for cell in causes)
        assert any(cell.page == PROCESS_GRAPH_OPERANDS for cell in causes)
        module = SimpleNamespace(metadata={"identity_book": book})
        assert CorrelationTable._source_named_tuple_layout_findings(module) == []
        audit = CorrelationTable._named_intrinsic_result_projection_findings
        assert audit(module) == []
        page = book.page(CALL_RESULT_PROJECTION)
        key = (receipt.row, receipt.column)
        original_path = page.cells[key]
        try:
            page.cells[key] = (1 - position,)
            assert audit(module)
        finally:
            page.cells[key] = original_path
        seen.append((book, receipt))
        raise Observed

    monkeypatch.setattr(strategy, "_build_shell_hierarchy_plan", capture)
    policy = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "records": {}, "bindings": [], "values": [
            {"function": "root", "parameter": "values", "storage": "span",
             "dtype": "float64", "rank": 1,
             "python_type": "src.common.tensors.abstraction.AbstractTensor"},
            {"function": "root", "parameter": "k", "storage": "scalar",
             "dtype": "int64", "rank": 0, "python_type": "builtins.int"},
        ],
    })
    with pytest.raises(Observed):
        lower_ast_source_to_ssa(
            f"def root(values, k: int):\n"
            f"    return AbstractTensor.topk(values, k=k, dim=0).{field}\n",
            "root", python_bindings={"AbstractTensor": AbstractTensor},
            extraction_contract=policy, progress=lambda _: None,
            progress_bars=False,
        )
    assert len(seen) == 1
