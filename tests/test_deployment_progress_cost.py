"""Diagnostic work stays bounded during a real deployment-selection pass."""
import ast

import pytest

from src.compiler import glsl_deployment_strategy as planner
from src.compiler.identity_concordance import begin_identity_book, end_identity_book
from src.common.tensors.topological_reducer import reduce_abstract_tensor_topology
from src.transmogrifier.graph.graph_express2 import ProcessGraph


def test_empty_control_partition_does_not_build_catalogue_membership(monkeypatch):
    from types import SimpleNamespace
    import networkx as nx
    graph = SimpleNamespace(G=nx.DiGraph())
    graph.G.add_node(1, type="Input", expr_obj=ast.Name(id="value", ctx=ast.Load()))
    scanned = []
    original = planner._branch_compartments

    def branches(graph):
        scanned.append(True)
        return original(graph)

    monkeypatch.setattr(planner, "_branch_compartments", branches)
    assert planner._control_partition_keys(graph, (), ()) == {}
    assert scanned == []


def test_callsite_progress_does_not_recount_catalogue_edges_per_event(monkeypatch):
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        graph.build_from_ast(ast.parse("def target(value):\n    return value + 1\n"))
        reduce_abstract_tensor_topology(graph)
        original_count = graph.G.number_of_edges
        counts = []

        def count(*args, **kwargs):
            counts.append(1)
            return original_count(*args, **kwargs)

        class Observed(Exception):
            pass

        def propagate(_graph, *, _progress, **_options):
            before = len(counts)
            for callsites in range(100):
                _progress({"specialization_state": "callsite-progress",
                           "callsites": callsites})
            assert len(counts) == before
            raise Observed

        monkeypatch.setattr(graph.G, "number_of_edges", count)
        monkeypatch.setattr(planner.time, "monotonic", lambda: 1.0)
        monkeypatch.setattr(planner, "_propagate_callsite_tensor_specializations", propagate)
        with pytest.raises(Observed):
            planner.strategize_shell_deployment(graph)
        events = graph.G.graph["deployment_selection_trace"]["events"]
        progress = [event for event in events if event["state"] == "progress"]
        assert [event["callsites"] for event in progress] == list(range(100))
        assert all(event["edges"] == original_count() for event in progress)
    finally:
        end_identity_book(token)
