"""Named intrinsic layouts belong to each exact source read-scope fork."""

from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.common.tensors.topological_reducer import fork_read_scope
from src.compiler.concordance_declarations import (
    INGEST_SOURCE, PLANNER_TENSOR_SPECIALIZATION, READ_SCOPE_FORK,
    SOURCE_AGGREGATE_LAYOUT,
    SourceAggregateLayout,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import (
    CorrelationTable, Derived, Mode, Novel, ProjectedScopeRead,
    begin_identity_book, current_identity_book, end_identity_book,
)
from src.compiler.process_graph_fusion import extract_clean_process_subgraph


def test_two_specialized_topk_calls_publish_distinct_layout_rows(monkeypatch):
    import src.compiler.glsl_deployment_strategy as deployment

    source = (
        "def select(values: AbstractTensor, k: int):\n"
        "    return AbstractTensor.topk(values, k=k, dim=0).indices\n"
        "def root(a: AbstractTensor, b: AbstractTensor):\n"
        "    x = select(a, 1)\n"
        "    y = select(b, 2)\n"
        "    return x, y\n"
    )
    policy = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "records": {}, "bindings": [],
        "values": [
            {
                "function": "root", "parameter": name, "storage": "span",
                "dtype": "int64", "rank": 1, "shape": shape,
                "python_type": "src.common.tensors.abstraction.AbstractTensor",
            }
            for name, shape in (("a", [3]), ("b", [4]))
        ],
    })
    original = deployment.publish_intrinsic_named_result_leaves
    observed = []

    class BothLayoutsPublished(Exception):
        pass

    def capture(graph, node_id):
        result = original(graph, node_id)
        if result is None:
            return None
        scope = graph.G.graph["source_aggregate_layout_scope"]
        read_scope = graph.G.graph["lexical_read_scope"]
        book = current_identity_book()
        row = (scope, int(node_id))
        cell = book.latest_ref(SOURCE_AGGREGATE_LAYOUT, row)
        if cell is None:
            return result
        layout = book.page(SOURCE_AGGREGATE_LAYOUT).cell(
            cell.row, cell.column,
        )
        observed.append((scope, read_scope, row, layout, book))
        if len(observed) == 1:
            # An already-published parent is a stronger case than two
            # siblings whose authored parent has no layout yet.
            child = extract_clean_process_subgraph(
                graph, tuple(graph.G.nodes),
            )
            child_scope = child.G.graph["source_aggregate_layout_scope"]
            child_row = (child_scope, int(node_id))
            assert child.G.nodes[int(node_id)]["attributes"][
                "source_aggregate_layout_row"
            ] == child_row
            inherited = book.latest_ref(SOURCE_AGGREGATE_LAYOUT, child_row)
            assert book.page(SOURCE_AGGREGATE_LAYOUT).cell(
                inherited.row, inherited.column,
            ) == layout
            projected = original(child, node_id)
            assert projected is not None
            assert len(book.page(SOURCE_AGGREGATE_LAYOUT).history(
                child_row,
            )) == 2
            assert all(member.value_cell.row[0] == child_scope for member in
                       book.page(SOURCE_AGGREGATE_LAYOUT).latest(child_row).members)
            grandchild = extract_clean_process_subgraph(
                child, tuple(child.G.nodes),
            )
            grandchild_row = (
                grandchild.G.graph["source_aggregate_layout_scope"],
                int(node_id),
            )
            assert original(grandchild, node_id) is not None
            assert len(book.page(SOURCE_AGGREGATE_LAYOUT).history(
                grandchild_row,
            )) == 2
            assert CorrelationTable._source_aggregate_layout_findings(
                SimpleNamespace(metadata={"identity_book": book})
            ) == []

            bad_child = extract_clean_process_subgraph(
                graph, tuple(graph.G.nodes),
            )
            bad_row = (
                bad_child.G.graph["source_aggregate_layout_scope"],
                int(node_id),
            )
            bad_inherited = book.latest_ref(SOURCE_AGGREGATE_LAYOUT, bad_row)
            book.post(
                SOURCE_AGGREGATE_LAYOUT, bad_row,
                SourceAggregateLayout("list", layout.members),
                stage=PLANNER_TENSOR_SPECIALIZATION,
                provenance=Derived((bad_inherited,)), mode=Mode.REVISE,
            )
            with pytest.raises(ValueError, match="intrinsic named result disagrees"):
                original(bad_child, node_id)
        if len(observed) == 2:
            raise BothLayoutsPublished
        return result

    monkeypatch.setattr(
        deployment, "publish_intrinsic_named_result_leaves", capture,
    )
    with pytest.raises(BothLayoutsPublished):
        lower_ast_source_to_ssa(
            source, "root", name="two_topk_layout_forks",
            python_bindings={"AbstractTensor": AbstractTensor},
            extraction_contract=policy,
            tensor_ssa_reference=c_backend_repository_ssa_reference(),
            progress=lambda _message: None, progress_bars=False,
        )
    assert len(observed) == 2
    assert observed[0][2] != observed[1][2]
    for scope, read_scope, _row, layout, _book in observed:
        assert scope == read_scope
        assert tuple(member.value_id for member in layout.members) == (7, 4)
        assert all(member.value_cell.row[0] == scope
                   for member in layout.members)


def test_inherited_source_layout_reads_through_only_for_held_nodes():
    book, token = begin_identity_book()
    try:
        source_scope = book.mint_scope("layout-source", READ_SCOPE_FORK)
        fact = SourceAggregateLayout("tuple", ())
        source_cell = book.post(
            SOURCE_AGGREGATE_LAYOUT, (source_scope, 2), fact,
            stage=READ_SCOPE_FORK, provenance=Novel(INGEST_SOURCE, ()),
            mode=Mode.CONCORD,
        )
        book.post(
            SOURCE_AGGREGATE_LAYOUT, (source_scope, 3), fact,
            stage=READ_SCOPE_FORK, provenance=Novel(INGEST_SOURCE, ()),
            mode=Mode.CONCORD,
        )
        whole = SimpleNamespace(G=nx.DiGraph())
        whole.G.add_nodes_from((2, 3))
        copy = SimpleNamespace(G=nx.DiGraph())
        copy.G.add_node(2)
        copy.G.graph.update({
            "lexical_read_scope": source_scope,
            "source_aggregate_layout_scope": source_scope,
        })
        fork_read_scope(copy, "held-layout", source_graph=whole)
        forked = copy.G.graph["lexical_read_scope"]
        assert copy.G.graph["source_aggregate_layout_scope"] == forked
        page = book.page(SOURCE_AGGREGATE_LAYOUT)
        assert page.latest((forked, 2)) == fact
        inherited = book.latest_ref(SOURCE_AGGREGATE_LAYOUT, (forked, 2))
        assert (source_cell, READ_SCOPE_FORK) in book.edges_into(inherited)
        with pytest.raises(ProjectedScopeRead):
            page.latest((forked, 3))
    finally:
        end_identity_book(token)
