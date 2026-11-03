"""A named intrinsic pair publishes ordered, concorded planner leaves."""

from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace

import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.compiler.concordance_declarations import (
    CALL_RESULT_PROJECTION, SOURCE_AGGREGATE_LAYOUT,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable, current_identity_book
from src.compiler.intrinsic_result_layouts import intrinsic_result_fields


@pytest.mark.parametrize("value_dtype", ["int64", "float64"])
def test_topk_result_pair_reuses_named_projection_and_keeps_pair_untyped(
    monkeypatch, value_dtype,
):
    import src.compiler.glsl_deployment_strategy as deployment

    class ReachedPublishedPair(Exception):
        pass

    observed = []
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
        authored = tuple(
            int(node) for node, data in graph.G.nodes(data=True)
            if str(data.get("op") or "").casefold() == "getattr"
            and (data.get("attributes") or {}).get("attribute") == "indices"
        )
        assert len(authored) == 1
        assert deployment._tensor_descriptor_rule(graph, call) is None
        published = deployment.publish_intrinsic_named_result_leaves(
            graph, call,
        )
        assert published is not None
        leaves, descriptors, proof = published
        assert len(leaves) == len(descriptors) == 2
        assert leaves[1] == authored[0]
        assert leaves[0] != leaves[1]
        assert tuple(item["dtype"] for item in descriptors) == (
            value_dtype, "int64",
        )
        assert deployment._tensor_descriptor_rule(graph, call) is None
        book = current_identity_book()
        layout = proof[-1]
        assert layout.page == SOURCE_AGGREGATE_LAYOUT
        fact = book.page(SOURCE_AGGREGATE_LAYOUT).cell(
            layout.row, layout.column,
        )
        assert tuple(member.value_id for member in fact.members) == leaves
        page = book.page(CALL_RESULT_PROJECTION)
        assert any(
            row[1:] == (call, authored[0]) and page.latest(row) == (1,)
            for row in page.rows()
        )
        module = SimpleNamespace(metadata={"identity_book": book})
        assert CorrelationTable._source_aggregate_layout_findings(module) == []
        assert CorrelationTable._named_intrinsic_result_projection_findings(
            module,
        ) == []
        declared, _named_proof = intrinsic_result_fields(graph, call)
        layout_page = book.page(SOURCE_AGGREGATE_LAYOUT)
        original_fact = layout_page.cells[(layout.row, layout.column)]
        try:
            first = original_fact.members[0]
            forged_first = replace(
                first,
                source_cells=tuple(
                    cell for cell in first.source_cells
                    if cell != declared[0][1]
                ),
            )
            layout_page.cells[(layout.row, layout.column)] = replace(
                original_fact,
                members=(forged_first, *original_fact.members[1:]),
            )
            assert CorrelationTable._source_aggregate_layout_findings(module)
        finally:
            layout_page.cells[(layout.row, layout.column)] = original_fact
        before = book.page(SOURCE_AGGREGATE_LAYOUT).history(layout.row)
        repeated = deployment.publish_intrinsic_named_result_leaves(
            graph, call,
        )
        assert repeated is not None and repeated[0] == leaves
        assert book.page(SOURCE_AGGREGATE_LAYOUT).history(layout.row) == before
        observed.append(True)
        raise ReachedPublishedPair

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
             "dtype": value_dtype, "rank": 1,
             "python_type": "src.common.tensors.abstraction.AbstractTensor"},
            {"function": "root", "parameter": "k", "storage": "scalar",
             "dtype": "int64", "rank": 0, "python_type": "builtins.int"},
        ],
    })
    with pytest.raises(ReachedPublishedPair):
        lower_ast_source_to_ssa(
            "def root(values,k):\n"
            "    return AbstractTensor.topk(values,k=k,dim=0).indices\n",
            "root", python_bindings={"AbstractTensor": AbstractTensor},
            name="topk_result_pair", extraction_contract=policy,
            progress=lambda _: None, progress_bars=False,
        )
    assert observed == [True]
