"""Read exact call-boundary facts from trusted local ProcessGraph snapshots.

No source execution, specialization, shell construction, or graph repair is
performed. Pickle input must be a trusted compiler-generated artifact.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import pickle
from pathlib import Path


def inspect_saved_calls(path: Path, entry=None, ids=()):
    payload = Path(path).read_bytes()
    # Imports performed by the repository's pickle loader may print banners.
    with contextlib.redirect_stdout(io.StringIO()):
        graph = pickle.loads(payload)
    if entry and graph.G.graph.get("function_name") != entry:
        table = getattr(graph, "function_table", None)
        matches = [] if table is None else [
            item.graph for item in table
            if item.qualified_name == entry and item.graph is not None
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one exact authored entry {entry!r}, found {len(matches)}")
        graph = matches[0]
    selected = set(map(int, ids))
    loops = []
    effect_ids = set()
    for node, data in graph.G.nodes(data=True):
        attributes = data.get("attributes") or {}
        effects = attributes.get("loop_state_effects") or ()
        if not effects or (selected and int(node) not in selected):
            continue
        members = []
        for effect in effects:
            effect_id = int(effect["effect_node_id"])
            effect_ids.add(effect_id)
            effect_data = graph.G.nodes.get(effect_id, {})
            expression = effect_data.get("expr_obj")
            members.append({
                "effect_node_id": effect_id,
                "present": effect_id in graph.G,
                "state_name": effect.get("state_name"),
                "operator": effect.get("operator"),
                "effect_mode": effect.get("effect_mode", "opaque"),
                "source": ast.unparse(expression) if isinstance(expression, ast.AST) else None,
            })
        loops.append({"loop_node_id": int(node), "effects": members})
    # Selecting a rejected loop also selects the calls responsible for its
    # recorded effects. Missing effect nodes remain visible in the loop row.
    selected_calls = selected | effect_ids
    rows = []
    bindings = getattr(graph, "python_bindings", {}) or {}
    for node, data in graph.G.nodes(data=True):
        attributes = data.get("attributes") or {}
        expression = data.get("expr_obj")
        if selected and int(node) not in selected_calls:
            continue
        if not isinstance(expression, ast.Call) and not any(
            key in attributes for key in ("callee_ref", "external_callee_ref", "constructor_ref")
        ):
            continue
        name = attributes.get("static_python_reference")
        if name is None and isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name):
            name = expression.func.id
        bound = bindings.get(name)
        receipt = dict(attributes.get("extraction_contract") or {})
        arguments = []
        for parent, role in data.get("parents") or ():
            if str(role) in {"callee", "func", "definition"}:
                continue
            parent_data = graph.G.nodes.get(int(parent), {})
            parent_attributes = parent_data.get("attributes") or {}
            leaves = list(map(int, parent_attributes.get("aggregate_leaf_value_ids") or ()))
            arguments.append({
                "role": str(role), "value_id": int(parent),
                "present": int(parent) in graph.G,
                "type": parent_data.get("type"),
                "aggregate_kind": parent_attributes.get("aggregate_kind"),
                "leaf_ids": leaves,
                "missing_leaf_ids": [leaf for leaf in leaves if leaf not in graph.G],
                "structural_specialization": bool(parent_attributes.get("structural_specialization")),
            })
        rows.append({
            "callsite_id": int(node),
            "source": ast.unparse(expression) if isinstance(expression, ast.AST) else None,
            "source_span": data.get("source_span"),
            "references": {key: int(attributes[key]) for key in (
                "callee_ref", "external_callee_ref", "constructor_ref") if attributes.get(key) is not None},
            "extraction_action": attributes.get("extraction_action"),
            "extraction_identity": attributes.get("extraction_identity"),
            "extraction_rule": receipt.get("rule_id"),
            "extraction_parameters": dict(receipt.get("parameters") or {}),
            "binding_name": name,
            "python_binding_present": name in bindings,
            "python_binding_callable": callable(bound),
            "authored_source_exposed": callable(getattr(bound, "__turing_authored_source_callable__", None)),
            "arguments": arguments,
        })
    return {
        "schema": "turing.process-graph-call-diagnostic.v1",
        "artifact": str(Path(path).resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "function": graph.G.graph.get("function_name"),
        "levels_cover_nodes": set(graph.levels) == set(graph.G),
        "has_recursion_table": bool(graph.G.graph.get("recursion_table")),
        "calls": rows,
        "loops": loops,
        "limits": "Snapshot facts only: external references may resolve later; this does not prove native closure, binary emission, parity, or performance.",
    }
