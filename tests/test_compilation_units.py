from __future__ import annotations

from types import SimpleNamespace

import networkx as nx

from src.compiler.compilation_units import (
    plan_compilation_units,
    record_compilation_unit_plan,
)
from src.compiler.shell_reference_tables import build_map_dependency_regions
from src.transmogrifier.function_table import FunctionTable


def _table(specification):
    table = FunctionTable()
    references = {
        name: table.declare(name, qualified_name=f"project.{name}")
        for name in specification
    }
    for name, callees in specification.items():
        graph = nx.DiGraph()
        graph.graph["function_name"] = name
        graph.add_node(0, type="Input", attributes={})
        for node_id, callee in enumerate(callees, 1):
            graph.add_node(
                node_id,
                type="Call",
                attributes={"callee_ref": references[callee].address},
            )
        table.resolve_graph(references[name], SimpleNamespace(G=graph))
    return table, references


def test_units_are_one_authored_call_each_and_dependencies_come_first():
    table, references = _table({"root": ("middle",), "middle": ("leaf",), "leaf": ()})

    plan = plan_compilation_units(table)

    assert [unit.qualified_names for unit in plan.units] == [
        ("project.leaf",),
        ("project.middle",),
        ("project.root",),
    ]
    assert plan.unit_for_reference(references["root"].address).dependency_units == (1,)
    assert plan.unit_for_reference(references["middle"].address).dependency_units == (0,)


def test_mutual_recursion_is_the_only_automatic_multi_call_unit():
    table, references = _table({"entry": ("left",), "left": ("right",), "right": ("left",)})

    plan = plan_compilation_units(table)

    assert [unit.qualified_names for unit in plan.units] == [
        ("project.left", "project.right"),
        ("project.entry",),
    ]
    recursive = plan.unit_for_reference(references["left"].address)
    assert recursive.recursive is True
    assert recursive.function_references == tuple(sorted((
        references["left"].address,
        references["right"].address,
    )))
    assert plan.unit_for_reference(references["entry"].address).dependency_units == (0,)


def test_plan_identity_uses_qualified_names_not_reference_order():
    first, _ = _table({"z": (), "a": ()})
    second, _ = _table({"a": (), "z": ()})

    first_names = [unit.qualified_names for unit in plan_compilation_units(first).units]
    second_names = [unit.qualified_names for unit in plan_compilation_units(second).units]

    assert first_names == second_names == [("project.a",), ("project.z",)]


def test_portable_plan_receipt_is_recorded_on_the_project_graph():
    table, references = _table({"root": ("leaf",), "leaf": ()})
    project = SimpleNamespace(function_table=table, G=nx.DiGraph())

    plan = record_compilation_unit_plan(project)

    receipt = project.G.graph["compilation_unit_plan"]
    assert receipt["schema"] == "turing.compilation-unit-plan.v1"
    assert receipt["order"] == "dependencies-first"
    assert receipt["units"][0]["qualified_names"] == ["project.leaf"]
    assert receipt["reference_to_unit"][str(references["root"].address)] == 1
    assert plan.to_mapping() == receipt


def test_runtime_closure_excludes_unrelated_catalogue_entries():
    table, references = _table({
        "root": ("middle",),
        "middle": ("leaf",),
        "leaf": (),
        "unrelated": (),
    })
    project = SimpleNamespace(function_table=table, G=nx.DiGraph())

    regions = build_map_dependency_regions(project, "root")

    assert regions.runtime == tuple(sorted((
        references["root"].address,
        references["middle"].address,
        references["leaf"].address,
    )))
    assert references["unrelated"].address not in regions.retained
