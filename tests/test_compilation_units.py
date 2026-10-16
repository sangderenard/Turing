from __future__ import annotations

import ast
from types import SimpleNamespace

import networkx as nx
import pytest

from src.compiler.identity_concordance import begin_identity_book, end_identity_book


@pytest.fixture(autouse=True)
def _isolated_project_book():
    # Each synthetic project owns its function-address declarations.
    _book, token = begin_identity_book()
    try:
        yield
    finally:
        end_identity_book(token)

from src.compiler.compilation_units import (
    compilation_growth_priorities,
    plan_compilation_units,
    record_compilation_unit_plan,
)
from src.compiler.shell_reference_tables import build_map_dependency_regions
from src.compiler.receiver_resolution_prepasses import (
    _resolve_grounded_method_references,
)
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


def test_growth_counts_shared_dependencies_once_and_retains_inward_chain():
    units = [
        {"source_nodes": 10, "dependency_units": []},
        {"source_nodes": 20, "dependency_units": [0]},
        {"source_nodes": 30, "dependency_units": [0]},
        {"source_nodes": 40, "dependency_units": [1, 2]},
        {"source_nodes": 95, "dependency_units": []},
    ]
    ranks = compilation_growth_priorities(units)
    assert ranks[3]["dependency_cone_source_nodes"] == 100
    assert ranks[0]["enclosing_unit_chain"] == [3, 2, 0]
    assert ranks[0]["enclosing_unit_count"] == 3
    assert ranks[0]["repeated_source_nodes_avoided"] == 30
    # Compiling the shared interior makes the unrelated cone the largest.
    remaining = compilation_growth_priorities(units, accepted_units=[0])
    assert remaining[0]["remaining_source_nodes"] == 0
    assert remaining[3]["dependency_cone_source_nodes"] == 90
    assert remaining[4]["enclosing_cone_source_nodes"] == 95
    # A partial/failed leaf is not passed as accepted and cannot shrink cost.
    assert compilation_growth_priorities(units)[3]["dependency_cone_source_nodes"] == 100


def test_growth_refuses_uncontracted_cycles_and_invalid_dependencies():
    with pytest.raises(ValueError, match="contracted"):
        compilation_growth_priorities([
            {"source_nodes": 1, "dependency_units": [1]},
            {"source_nodes": 1, "dependency_units": [0]},
        ])
    with pytest.raises(ValueError, match="outside"):
        compilation_growth_priorities([{"dependency_units": [2]}])


def test_authored_growth_triage_keeps_recursive_and_lexical_boundaries():
    from src.compiler.project_compilation_product import authored_compilation_growth_plan
    source = '''
def left(x):
    return right(x)
def right(x):
    return left(x)
def outer(x):
    def inner(y):
        return y + 1
    return inner(x)
'''
    plan = authored_compilation_growth_plan(source)
    assert plan["cut_authority"] == "resolved-function-table"
    assert plan["cost_basis"] == "authored-ast-nodes-estimate"
    recursive = next(unit for unit in plan["units"] if "left" in unit["qualified_names"])
    assert recursive["qualified_names"] == ["left", "right"]
    assert recursive["recursive"]
    owner = next(unit for unit in plan["units"] if unit["qualified_names"] == ["outer"])
    child = next(index for index, unit in enumerate(plan["units"])
                 if unit["qualified_names"] == ["outer.<locals>.inner"])
    assert child in owner["dependency_units"]


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


def test_source_class_constructor_is_resolved_before_unit_division():
    table = FunctionTable()
    caller_ref = table.declare("run", qualified_name="project.run")
    constructor_ref = table.declare(
        "__init__", qualified_name="project.IdentityBook.__init__",
    )
    constructor = nx.DiGraph()
    constructor.graph.update({
        "function_name": "__init__", "method_owner": "IdentityBook",
    })
    constructor.add_node(0, type="Input", attributes={})
    table.resolve_graph(constructor_ref, SimpleNamespace(G=constructor))

    caller = nx.DiGraph()
    caller.graph.update({
        "function_name": "run",
        "class_table": {
            "IdentityBook": {
                "methods": {"__init__": constructor_ref.address},
                "fields": ("detached",),
            },
        },
    })
    caller.add_node(
        9, type="Call", op="call", value_id=9,
        expr_obj=ast.parse("IdentityBook(detached=True)", mode="eval").body,
        attributes={"class_ref": "IdentityBook"}, parents=[],
    )
    table.resolve_graph(caller_ref, SimpleNamespace(G=caller))

    _resolve_grounded_method_references(table.entry(caller_ref).graph)
    attributes = caller.nodes[9]["attributes"]
    assert attributes["constructor_ref"] == constructor_ref.address
    assert attributes["constructor_resolution"] == "source-class-ref"

    plan = plan_compilation_units(table)
    constructor_unit = plan.reference_to_unit[constructor_ref.address]
    caller_unit = plan.unit_for_reference(caller_ref.address)
    assert caller_unit.dependency_units == (constructor_unit,)
    assert caller_unit.dynamic_call_nodes == ()


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
    first_names = [unit.qualified_names for unit in plan_compilation_units(first).units]
    # This is a second project, not conflicting declarations in the first.
    _book, token = begin_identity_book()
    try:
        second, _ = _table({"a": (), "z": ()})
        second_names = [unit.qualified_names for unit in plan_compilation_units(second).units]
    finally:
        end_identity_book(token)

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


def test_indivisible_edges_condense_the_convex_call_component_not_unrelated_work():
    table, references = _table({
        "island_root": ("island_left", "island_right", "island_free"),
        "island_left": ("island_tail",), "island_right": ("island_tail",),
        "island_tail": (), "island_free": (),
    })
    plan = plan_compilation_units(table, indivisible_edges=(
        (references["island_root"].address, references["island_left"].address),
        (references["island_left"].address, references["island_tail"].address),
    ))
    island = plan.unit_for_reference(references["island_root"].address)
    # The alternate root->right->tail path must join the contracted component;
    # otherwise its quotient has an island->right->island scheduling cycle.
    assert island.qualified_names == ("project.island_left", "project.island_right",
                                      "project.island_root", "project.island_tail")
    assert not island.recursive
    assert plan.unit_for_reference(references["island_free"].address).qualified_names == ("project.island_free",)
