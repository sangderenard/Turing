import ast
import json
import pickle
from pathlib import Path


root = Path(__file__).resolve().parents[2] / "build" / "full_formal_diagnostic"
missing = json.loads((root / "formals.json").read_text())
with (root / "resolved-process-graph.pkl").open("rb") as stream:
    saved = pickle.load(stream)

entries = list(saved.function_table._entries.values())


def entry_for(symbol):
    matches = [
        entry for entry in entries
        if f"__{entry.name}__" in symbol or symbol.endswith(f"__{entry.name}")
    ]
    return max(matches, key=lambda entry: len(entry.name))


for row in missing:
    entry = entry_for(row["function"])
    graph = entry.graph.G
    value_id = int(row["value_id"])
    if value_id not in graph:
        print(json.dumps({"function": entry.qualified_name, "value": value_id, "missing_from_graph": True}))
        continue
    data = graph.nodes[value_id]
    expression = data.get("expr_obj")
    controls = graph.graph.get("source_control_records") or {}
    mutations = graph.graph.get("source_sequence_mutation_records") or {}
    print(json.dumps({
        "function": entry.qualified_name,
        "value": value_id,
        "type": data.get("type"),
        "op": data.get("op"),
        "expr": type(expression).__name__ if expression is not None else None,
        "source": None if expression is None else ast.unparse(expression),
        "parents": data.get("parents"),
        "attributes": data.get("attributes"),
        "children": list(graph.successors(value_id)),
        "semantic_children": [
            int(child)
            for child, child_data in graph.nodes(data=True)
            if any(int(parent) == value_id for parent, _ in child_data.get("parents", ()))
        ],
        "child_details": [
            [
                int(child),
                graph.nodes[child].get("type"),
                type(graph.nodes[child].get("expr_obj")).__name__,
                None if graph.nodes[child].get("expr_obj") is None else ast.unparse(graph.nodes[child]["expr_obj"]),
            ]
            for child in sorted(set(graph.successors(value_id)) | {
                int(child)
                for child, child_data in graph.nodes(data=True)
                if any(int(parent) == value_id for parent, _ in child_data.get("parents", ()))
            })
        ],
        "predicate_for": [
            int(control_id) for control_id, record in controls.items()
            if record.get("predicate_id") == value_id
        ],
        "predicate_descendants": [
            [int(control_id), int(record["predicate_id"])]
            for control_id, record in controls.items()
            if record.get("predicate_id") in graph
            and __import__("networkx").has_path(graph, value_id, int(record["predicate_id"]))
        ],
        "mutation_argument_for": [
            int(effect_id) for effect_id, record in mutations.items()
            if value_id in record.get("argument_value_ids", ())
        ],
    }, default=str))
