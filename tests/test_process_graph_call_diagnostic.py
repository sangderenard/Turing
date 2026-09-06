import ast
import pickle
from types import SimpleNamespace

import networkx as nx

from tools.diagnose_process_graph_calls import inspect_saved_calls


def test_selected_loop_explains_effect_calls_and_missing_nodes(tmp_path):
    graph = nx.DiGraph()
    graph.graph["function_name"] = "draw"
    graph.add_node(10, attributes={"loop_state_effects": (
        {"effect_node_id": 20, "state_name": "display", "operator": "line",
         "effect_mode": "opaque"},
        {"effect_node_id": 21, "state_name": "display", "operator": "flip"},
    )})
    graph.add_node(20, expr_obj=ast.parse("display.line(a, b)").body[0].value,
                   attributes={"extraction_identity": "display.line",
                               "extraction_contract": {
                                   "rule_id": "display-native",
                                   "parameters": {"shell_abi": "display", "result_dtype": "opaque_ref"},
                               }})
    graph.add_node(30, expr_obj=ast.parse("unrelated()").body[0].value)
    saved = tmp_path / "graph.pkl"
    saved.write_bytes(pickle.dumps(SimpleNamespace(G=graph, levels=dict.fromkeys(graph))))

    report = inspect_saved_calls(saved, ids=(10,))

    assert [call["callsite_id"] for call in report["calls"]] == [20]
    assert report["calls"][0]["extraction_rule"] == "display-native"
    assert report["calls"][0]["extraction_parameters"] == {
        "shell_abi": "display", "result_dtype": "opaque_ref",
    }
    effects = report["loops"][0]["effects"]
    assert report["loops"][0]["loop_node_id"] == 10
    assert effects[0]["source"] == "display.line(a, b)"
    assert effects[0]["effect_mode"] == "opaque"
    assert effects[1]["effect_node_id"] == 21
    assert effects[1]["present"] is False
    assert effects[1]["source"] is None
