"""Limits retain authored work; they only stop optional expansion."""
import ast
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from src.compiler.work_contract import (
    active_contract, expansion_policy, scoped_work_contract, with_expansion_policy,
)


def test_bootstrap_defaults_and_version_include_effective_depths(tmp_path):
    from src.compiler.whole_program_compiler_bootstrap import plan_whole_program_compiler
    first = plan_whole_program_compiler(tmp_path / "first")
    second = plan_whole_program_compiler(tmp_path / "second", expansion_policy={"unroll_depth": 1})
    assert first["expansion_policy"] == {"unroll_depth": 0, "inline_depth": 0, "custom_version_depth": 0}
    assert first["next_partition"]["version"] != second["next_partition"]["version"]
    assert json.loads((tmp_path / "first" / "plan.json").read_text()) == first
    for bad in (-1, True, 1.5):
        with pytest.raises(ValueError):
            with_expansion_policy(active_contract(), {"inline_depth": bad})


def test_environment_depths_are_independent(monkeypatch):
    monkeypatch.setenv("TURING_UNROLL_DEPTH", "0")
    monkeypatch.setenv("TURING_INLINE_DEPTH", "1")
    monkeypatch.setenv("TURING_CUSTOM_VERSION_DEPTH", "2")
    assert expansion_policy() == {"unroll_depth": 0, "inline_depth": 1, "custom_version_depth": 2}


def test_limit_returns_existing_callable_shell_without_losing_graph():
    import networkx as nx
    from src.compiler.glsl_deployment_strategy import _callsite_specialized_shell_type
    graph = SimpleNamespace(G=nx.DiGraph())
    graph.G.add_edge(7, 9, role="arg0")
    graph.G.graph["custom_version_generation"] = 2
    owner = SimpleNamespace(process_graph=graph)
    generic = type("Generic", (), {})
    for limit in (0, 1, 2):
        contract = with_expansion_policy(active_contract(), {"custom_version_depth": limit})
        with scoped_work_contract(contract):
            assert _callsite_specialized_shell_type(owner, 9, 42, generic, 256) is generic
        assert list(graph.G.edges(data=True)) == [(7, 9, {"role": "arg0"})]


def test_zero_inline_depth_retains_callee_and_call():
    from src.compiler.ir_identities import inline_host_linear_source_regions
    from src.transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue
    value = SSAValue(1, "float64")
    callee = Function("region", [value], {"entry": BasicBlock("entry", [
        Instr("Neg", [value], SSAValue(2, "float64"))])},
        metadata={"source_region_integral": {"owner": "root"}})
    call = Instr("Call", [value], None, attributes={"callee": "region"})
    caller = Function("root", [value], {"entry": BasicBlock("entry", [call])})
    functions = {"root": caller, "region": callee}
    with scoped_work_contract(with_expansion_policy(active_contract(), {"inline_depth": 0})):
        view, receipts = inline_host_linear_source_regions(functions)
    assert receipts == () and view == functions
    assert view["root"].blocks["entry"].instrs[0] is call
    assert view["region"] is callee


def test_zero_unroll_depth_retains_complete_loop_domain_and_body():
    from src.compiler.identity_concordance import begin_identity_book, end_identity_book
    from src.transmogrifier.graph.graph_express2 import ProcessGraph
    from src.common.tensors.topological_reducer import reduce_abstract_tensor_topology
    from src.compiler.loop_composer import LoopComposer, LoopBackendCapabilities, LoopStrategy
    _, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        graph.build_from_ast(ast.parse("def root(x):\n    for i in range(4):\n        x = x + i\n    return x\n"))
        reduce_abstract_tensor_topology(graph)
        graph = graph.function_table.entry("root").graph
        composer = LoopComposer(LoopBackendCapabilities("c", native_for=True, native_while=True, dynamic_bounds=True))
        original_nodes = tuple(graph.G.nodes)
        with scoped_work_contract(with_expansion_policy(active_contract(), {"unroll_depth": 0})):
            plan, = composer.discover(graph)
        assert plan.strategy is LoopStrategy.NATIVE_SOURCE
        assert plan.loop.trip_count == 4 and plan.loop.body_nodes
        assert tuple(graph.G.nodes) == original_nodes
        with scoped_work_contract(with_expansion_policy(active_contract(), {"unroll_depth": 1})):
            assert composer.plan(plan.loop, nesting_depth=0).strategy is LoopStrategy.UNROLL
            assert composer.plan(plan.loop, nesting_depth=1).strategy is LoopStrategy.NATIVE_SOURCE
    finally:
        end_identity_book(token)
