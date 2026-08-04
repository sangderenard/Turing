from src.compiler.card_graph import build_card_graph
from src.compiler.wasm_html_shell import emit_html_shell


def _map_ir():
    return {
        "callable_systems": {
            "classes": [{
                "identity": "Counter",
                "methods": [
                    {
                        "identity": "Counter.read",
                        "name": "read",
                        "function_reference": 3,
                        "parameters": [
                            {"name": "value", "role": "output", "dtype": "float"},
                        ],
                    },
                    {
                        "identity": "Counter.write",
                        "name": "write",
                        "function_reference": 4,
                        "parameters": [
                            {"name": "value", "role": "input", "dtype": "float"},
                        ],
                    },
                ],
            }],
        },
        "class_navigation": {
            "classes": [{
                "identity": "Counter",
                "members": [
                    {"identity": "Counter.read", "permissions": []},
                    {"identity": "Counter.write", "permissions": ["counter:write"]},
                ],
            }],
        },
    }


def _class_graph():
    return {
        "name": "counter-runtime",
        "modules": [
            {"name": "load", "entry": "run", "inputs": ["x"],
             "outputs": ["seam"], "value_type": "f64", "url": "load.wasm"},
            {"name": "store", "entry": "run", "inputs": ["seam"],
             "outputs": ["y"], "value_type": "f64", "url": "store.wasm"},
        ],
        "edges": [{
            "from": {"module": "load", "output": "seam"},
            "to": {"module": "store", "input": "seam"},
        }],
        "class_inventory": {"methods": [
            {"module": "load"}, {"module": "store"},
        ]},
    }


def test_map_ir_projects_all_cards_and_valid_potential_connections():
    graph = build_card_graph(_map_ir(), _class_graph())

    assert graph["abi"] == "turing.card-graph.v1"
    assert graph["paths"]["linear"] == ["load", "store"]
    assert graph["address_policy"] == {
        "arena": "outer-coordinator",
        "cache": "compiled-card",
        "execution": "read-head",
        "rebind": "every-traversal",
        "inputs": "alias",
        "outputs": "alias",
    }
    identities = {card["id"] for card in graph["cards"]}
    assert identities == {"Counter.read", "Counter.write", "load", "store"}
    assert any(
        edge["kind"] == "compatible-memory"
        and edge["from"] == "Counter.read"
        and edge["to"] == "Counter.write"
        for edge in graph["connections"]
    )
    resident = next(
        edge for edge in graph["connections"]
        if edge["kind"] == "resident-memory"
    )
    assert resident["bindings"][0]["rewrite"] == "alias"


def test_html_shell_embeds_card_graph_and_runtime_read_head():
    api = {
        "module": "cards", "language": "wasm", "entry": "run",
        "entry_points": [{"name": "run", "symbol": "run", "parameters": []}],
        "metadata": {},
    }
    html = emit_html_shell(api, map_ir=_map_ir(), class_graph=_class_graph()).html

    assert '"abi": "turing.card-graph.v1"' in html
    assert "class CardGraphReadHead" in html
    assert "window.TuringCardGraph" in html
    assert "PUNCH_CARD_MODULE_CACHE" in html
    assert "rebindCardAliases" in html
    assert "a previous traversal's address must never survive" in html
