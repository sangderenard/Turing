"""One value never stands for a tuple or an optional in an admitted unit.

Doc 34 section 1.4: unit 027 (``_fork_of_scope``, ``return row, column`` or
``return None``) was once ``compiled-unverified`` while exporting one member
of its optional pair.  Product accounting refuses that reading, and
``tools/compiler_bootstrap_status.py --check-product`` audits a finished run.
"""

from __future__ import annotations

import json
import pickle
from types import SimpleNamespace

import networkx as nx

from src.compiler.project_compilation_product import (
    _one_value_aggregate_return,
)
from tools.compiler_bootstrap_status import check_unit_result_layouts


def _source(kinds, sites, constants=()):
    graph = nx.DiGraph()
    for node_id, value in constants:
        graph.add_node(node_id, type="Constant", attributes={"value": value})
    graph.graph["return_container_kinds"] = dict(enumerate(kinds))
    graph.graph["return_slot_values"] = dict(enumerate(sites))
    return SimpleNamespace(G=graph)


def test_plain_value_sites_read_as_one_value():
    assert _one_value_aggregate_return(
        _source(["value", "value"], [(4,), (7,)], [(7, -1)])
    ) is None


def test_tuple_site_beside_a_value_site_is_refused():
    assert _one_value_aggregate_return(
        _source(["tuple", "value"], [(4, 5), (9,)], [(9, None)])
    ) is not None


def test_none_site_beside_a_value_site_is_refused():
    assert _one_value_aggregate_return(
        _source(["value", "value"], [(4,), (9,)], [(9, None)])
    ) == "a None return site beside a value site"


def _unit(tmp_path, name, published, returned, paths, kind):
    unit = tmp_path / "units" / name
    unit.mkdir(parents=True)
    value = lambda value_id: SimpleNamespace(id=value_id)
    function = SimpleNamespace(
        blocks={"entry": SimpleNamespace(instrs=[SimpleNamespace(
            op="Ret", args=[value(item) for item in returned],
        )])},
        metadata={
            "native_output_paths": tuple(
                ((index,), item) for index, item in enumerate(paths)
            ),
            "native_return_container_kind": kind,
        },
    )
    module = SimpleNamespace(functions={"f": function})
    with (unit / "repository-ssa.pkl").open("wb") as handle:
        pickle.dump((module, {"f": [value(item) for item in published]}, ["f"]), handle)
    (unit / "unit.json").write_text(json.dumps({
        "status": "compiled-unverified", "artifact": "repository-ssa.pkl",
        "qualified_names": [name],
    }))


def test_status_tool_reports_one_value_for_a_tuple(tmp_path):
    _unit(tmp_path, "unit_000", [33, 34], [33, 34], [33, 34], "tuple")
    _unit(tmp_path, "unit_001", [33], [33], [33], "tuple")
    report = check_unit_result_layouts(tmp_path)
    assert report["checked_exports"] == 2
    assert [row["unit"] for row in report["false_acceptance_findings"]] == [
        "unit_001"
    ]


def test_authored_effects_are_read_from_stores_and_discarded_calls():
    import ast

    from src.compiler.project_compilation_product import (
        _authored_effect_statements,
    )

    graph = nx.DiGraph()
    graph.add_node(0, type="Input", parents=[])
    graph.add_node(1, type="Call", parents=[(0, "operand")],
                   expr_obj=ast.parse("layer.count(key)").body[0].value)
    graph.add_node(2, type="Call", parents=[(0, "arg:0")],
                   expr_obj=ast.parse("len(xs)").body[0].value)
    graph.add_node(3, type="Add", parents=[(2, "lhs")])
    assert _authored_effect_statements(graph) == ("1: layer.count(key)",)
    graph.add_node(4, type="IndexedStore", parents=[(0, "base")])
    assert len(_authored_effect_statements(graph)) == 2
