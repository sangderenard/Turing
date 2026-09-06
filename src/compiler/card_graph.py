"""Project Map IR callables and deployed modules into a traversable card graph.

The graph is a loading/linking description, not an execution schedule.  Its
``connections`` enumerate legal ways a read head may move between cards.  A
separate ``paths`` member may name one preferred schedule (for example the
topological order of lowered kernel regions) without erasing the other routes.

Cards are immutable program descriptions.  Their input/output ports name
logical values; a runtime must resolve those ports to addresses in its own
arena whenever it crosses a connection.  Consequently this IR never embeds a
process-local pointer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _items(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _callables(map_ir: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    systems = map_ir.get("callable_systems") or {}
    if not isinstance(systems, Mapping):
        return [("file", item) for item in _items(systems)]
    found: list[tuple[str, Mapping[str, Any]]] = []
    for item in _items(systems.get("functions")):
        found.append(("file", item))
    file_scope = systems.get("file_scope") or {}
    if isinstance(file_scope, Mapping):
        for item in _items(file_scope.get("functions")):
            found.append(("file", item))
    for owner in _items(systems.get("classes")):
        owner_id = str(owner.get("identity") or owner.get("name") or "class")
        for item in _items(owner.get("methods")):
            found.append((owner_id, item))
    # Preserve first occurrence: site manifests often expose file functions in
    # both the compatibility list and the structured file-scope record.
    unique: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for owner, item in found:
        identity = str(item.get("identity") or item.get("name") or "")
        if identity:
            unique.setdefault(identity, (owner, item))
    return list(unique.values())


def _ports(parameters: Any) -> list[dict[str, Any]]:
    result = []
    for index, parameter in enumerate(_items(parameters)):
        role = str(parameter.get("role") or "input")
        result.append({
            "index": index,
            "name": str(parameter.get("name") or f"value_{index}"),
            "role": role,
            "dtype": str(parameter.get("dtype") or "Any"),
            "passing": str(parameter.get("passing") or "value"),
            "extent": parameter.get("extent"),
        })
    return result


def _compatible(source: Mapping[str, Any], target: Mapping[str, Any]) -> bool:
    source_type = str(source.get("dtype") or "Any")
    target_type = str(target.get("dtype") or "Any")
    return source_type == target_type or "Any" in {source_type, target_type}


def build_card_graph(
    map_ir: Mapping[str, Any] | None,
    class_graph: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return cards and every statically valid connection known to Graph IR.

    Map-level functions/methods become navigable cards.  Compatible typed
    output/input ports form *potential* links.  When lowered WebAssembly class
    modules are present, their compiler-authored edges form exact resident
    memory links and their inventory order becomes a preferred path.
    """

    mapping = dict(map_ir or {})
    deployed = dict(class_graph or {})
    cards: list[dict[str, Any]] = []
    connections: list[dict[str, Any]] = []

    for owner, item in _callables(mapping):
        identity = str(item.get("identity") or item.get("name"))
        cards.append({
            "id": identity,
            "kind": "method" if owner != "file" else "function",
            "owner": owner,
            "function_reference": item.get("function_reference"),
            "resource": item.get("page_url") or item.get("url"),
            "permissions": list(item.get("permissions") or ()),
            "ports": _ports(item.get("parameters")),
            "cache_key": identity,
        })

    navigation = mapping.get("class_navigation") or {}
    if isinstance(navigation, Mapping):
        for owner in _items(navigation.get("classes")):
            owner_id = str(owner.get("identity") or "")
            for member in _items(owner.get("members")):
                target = str(member.get("identity") or "")
                if target and any(card["id"] == target for card in cards):
                    connections.append({
                        "kind": "member",
                        "from": owner_id,
                        "to": target,
                        "permissions": list(member.get("permissions") or ()),
                    })

    # Type compatibility describes possibility, not order.  The runtime still
    # needs an explicit path/selector before it traverses one of these links.
    for source in cards:
        outputs = [p for p in source["ports"] if p["role"] in {"output", "result", "inout"}]
        for target in cards:
            if source["id"] == target["id"]:
                continue
            inputs = [p for p in target["ports"] if p["role"] in {"input", "inout"}]
            bindings = [
                {"from": output["index"], "to": input_["index"]}
                for output in outputs for input_ in inputs
                if _compatible(output, input_)
            ]
            if bindings:
                connections.append({
                    "kind": "compatible-memory",
                    "from": source["id"],
                    "to": target["id"],
                    "bindings": bindings,
                })

    modules = _items(deployed.get("modules"))
    for module in modules:
        module_id = str(module.get("name"))
        cards.append({
            "id": module_id,
            "kind": "wasm-card",
            "owner": str(deployed.get("name") or "compiled-program"),
            "entry": module.get("entry"),
            "resource": module.get("url"),
            "cache_key": module.get("cache_key") or module.get("url") or module_id,
            "ports": [
                {"index": index, "name": str(name), "role": role,
                 "dtype": str(module.get("value_type") or "f64"),
                 "passing": "arena-address", "extent": None}
                for role, names in (("input", module.get("inputs") or ()),
                                    ("output", module.get("outputs") or ()))
                for index, name in enumerate(names)
            ],
        })
    for edge in _items(deployed.get("edges")):
        source = edge.get("from") or {}
        target = edge.get("to") or {}
        connections.append({
            "kind": "resident-memory",
            "from": str(source.get("module")),
            "to": str(target.get("module")),
            "bindings": [{
                "from": source.get("output"),
                "to": target.get("input"),
                "rewrite": "alias",
            }],
        })

    inventory = deployed.get("class_inventory") or {}
    methods = _items(inventory.get("methods") if isinstance(inventory, Mapping) else ())
    linear_path = [str(method.get("module")) for method in methods]
    policy = {
        "arena": "outer-coordinator",
        "cache": "compiled-card",
        "execution": "read-head",
        "rebind": "every-traversal",
        "inputs": "alias",
        "outputs": "alias",
    }
    supplied_policy = deployed.get("external_link_policy") or {}
    if isinstance(supplied_policy, Mapping):
        policy.update({str(key): value for key, value in supplied_policy.items()})
    return {
        "abi": "turing.card-graph.v1",
        "address_policy": policy,
        "cards": cards,
        "connections": connections,
        "paths": {"linear": linear_path} if linear_path else {},
    }


__all__ = ["build_card_graph"]
