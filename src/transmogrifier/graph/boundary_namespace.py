"""Sparse filesystem boundaries for language/OOP-specific graph overrides.

This is an optional resolver beside the existing ProcessGraph ingestion seam,
not a second translator.  A namespace is searched first by language and then
by lexical source scope. Missing scope directories are skipped, permitting a
sparse hierarchy. Only declarative ``*.node.json`` records are read; nothing
from the namespace is imported or executed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping


_SEGMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_ACTIONS = frozenset({"schema", "spoof", "exclude"})
_COMMON_KEYS = frozenset({"version", "id", "action", "node_type"})
_ACTION_KEYS = {
    "schema": _COMMON_KEYS | {"role_schema"},
    "spoof": _COMMON_KEYS | {
        "match", "graph_match", "result",
    },
    "exclude": frozenset({"version", "id", "action", "target"}),
}
_RESULT_KEYS = frozenset({
    "type", "attributes", "attributes_from_node", "attributes_from_graph",
    "constant", "constant_from_node", "constant_from_graph",
})


class BoundaryNamespaceError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class BoundaryRule:
    identity: str
    action: str
    node_type: str | None
    payload: Mapping[str, Any]
    path: str
    layer: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BoundaryReceipt:
    rule_id: str
    action: str
    node_type: str
    path: str
    language: str
    scope: tuple[str, ...]

    def mapping(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "action": self.action,
            "node_type": self.node_type,
            "path": self.path,
            "language": self.language,
            "scope": self.scope,
        }


@dataclass(frozen=True, slots=True)
class BoundaryResolution:
    role_schema: Mapping[str, Any] | None = None
    special_case: Any = None
    receipt: BoundaryReceipt | None = None
    excluded_rule_ids: tuple[str, ...] = ()


def _frozen(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _frozen(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(item) for item in value)
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return f"<{type(value).__module__}.{type(value).__qualname__}>"


def _lookup(value: Any, path: str) -> Any:
    current = value
    if not path:
        return current
    for piece in path.split("."):
        if isinstance(current, Mapping):
            if piece not in current:
                raise KeyError(path)
            current = current[piece]
        elif isinstance(current, (tuple, list)) and piece.isdigit():
            current = current[int(piece)]
        else:
            current = getattr(current, piece)
    return current


def _matches(value: Any, predicates: Mapping[str, Any]) -> bool:
    for path, expected in predicates.items():
        try:
            actual = _lookup(value, str(path))
        except (AttributeError, IndexError, KeyError, TypeError):
            return False
        if actual != expected:
            return False
    return True


class BoundaryNamespace:
    """Validated, cached view of one language-first override tree."""

    def __init__(self, root: str | Path, language: str = "python") -> None:
        self.root = Path(root).resolve()
        self.language = self._segment(language, "language")
        self._layer_cache: dict[Path, tuple[BoundaryRule, ...]] = {}
        self._resolution_cache: dict[tuple[str, ...], tuple[BoundaryRule, ...]] = {}
        self._excluded_cache: dict[tuple[str, ...], tuple[str, ...]] = {}

    def __getstate__(self) -> dict[str, str]:
        return {"root": str(self.root), "language": self.language}

    def __setstate__(self, state: Mapping[str, str]) -> None:
        self.__init__(state["root"], state["language"])

    @staticmethod
    def _segment(value: Any, label: str) -> str:
        text = str(value)
        if not _SEGMENT.fullmatch(text):
            raise BoundaryNamespaceError(f"invalid {label} path segment: {text!r}")
        return text

    def _safe_child(self, parent: Path, segment: str) -> Path:
        child = (parent / self._segment(segment, "scope")).resolve()
        try:
            child.relative_to(self.root)
        except ValueError as error:
            raise BoundaryNamespaceError("boundary path escapes namespace root") from error
        return child

    def _read_layer(self, directory: Path, layer: tuple[str, ...]) -> tuple[BoundaryRule, ...]:
        cached = self._layer_cache.get(directory)
        if cached is not None:
            return cached
        if not directory.is_dir():
            self._layer_cache[directory] = ()
            return ()
        rules = []
        for source in sorted(directory.glob("*.node.json")):
            if source.is_symlink():
                raise BoundaryNamespaceError(f"boundary record may not be a symlink: {source}")
            try:
                payload = json.loads(source.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise BoundaryNamespaceError(f"cannot read boundary record {source}: {error}") from error
            if not isinstance(payload, dict):
                raise BoundaryNamespaceError(f"boundary record must be an object: {source}")
            action = str(payload.get("action", ""))
            if payload.get("version") != 1 or action not in _ACTIONS:
                raise BoundaryNamespaceError(f"invalid boundary version/action in {source}")
            unknown = set(payload) - _ACTION_KEYS[action]
            if unknown:
                raise BoundaryNamespaceError(f"unknown boundary keys in {source}: {sorted(unknown)}")
            identity = payload.get("id")
            if not isinstance(identity, str) or not identity:
                raise BoundaryNamespaceError(f"boundary record needs a nonempty id: {source}")
            node_type = payload.get("node_type")
            if action != "exclude" and (not isinstance(node_type, str) or not node_type):
                raise BoundaryNamespaceError(f"boundary record needs node_type: {source}")
            if action == "exclude" and not isinstance(payload.get("target"), str):
                raise BoundaryNamespaceError(f"exclude record needs exact target id: {source}")
            if action == "schema":
                schema = payload.get("role_schema")
                if not isinstance(schema, dict) or set(schema) != {"up", "down"}:
                    raise BoundaryNamespaceError(f"schema needs exact up/down maps: {source}")
                if not all(isinstance(schema[key], dict) for key in ("up", "down")):
                    raise BoundaryNamespaceError(f"schema up/down must be objects: {source}")
            if action == "spoof":
                result = payload.get("result")
                if not isinstance(result, dict) or not isinstance(result.get("type"), str):
                    raise BoundaryNamespaceError(f"spoof result needs a type: {source}")
                unknown_result = set(result) - _RESULT_KEYS
                if unknown_result:
                    raise BoundaryNamespaceError(
                        f"unknown spoof result keys in {source}: {sorted(unknown_result)}"
                    )
                if sum(
                    key in result for key in (
                        "constant", "constant_from_node", "constant_from_graph"
                    )
                ) > 1:
                    raise BoundaryNamespaceError(
                        f"spoof result has multiple constant sources: {source}"
                    )
                for key in ("attributes", "attributes_from_node", "attributes_from_graph"):
                    if key in result and not isinstance(result[key], dict):
                        raise BoundaryNamespaceError(f"spoof result {key} must be an object: {source}")
                for key in ("match", "graph_match"):
                    if key in payload and not isinstance(payload[key], dict):
                        raise BoundaryNamespaceError(f"spoof {key} must be an object: {source}")
            rules.append(BoundaryRule(
                identity,
                action,
                None if node_type is None else str(node_type),
                _frozen(payload),
                str(source),
                layer,
            ))
        result = tuple(rules)
        self._layer_cache[directory] = result
        return result

    def rules_for_scope(self, scope: tuple[str, ...]) -> tuple[BoundaryRule, ...]:
        scope = tuple(map(str, scope))
        cached = self._resolution_cache.get(scope)
        if cached is not None:
            return cached
        language_root = self._safe_child(self.root, self.language)
        active: dict[str, BoundaryRule] = {}
        excluded_ids: set[str] = set()
        directory = language_root
        layers = [(directory, ())]
        used_scope = []
        for raw_segment in scope:
            segment = self._segment(raw_segment, "scope")
            candidate = self._safe_child(directory, segment)
            if candidate.is_dir():
                directory = candidate
                used_scope.append(segment)
                layers.append((directory, tuple(used_scope)))
        for layer_directory, layer in layers:
            for rule in self._read_layer(layer_directory, layer):
                if rule.action == "exclude":
                    target = str(rule.payload["target"])
                    active.pop(target, None)
                    excluded_ids.add(target)
                else:
                    # Same exact id at a deeper level is an intentional override.
                    active[rule.identity] = rule
                    excluded_ids.discard(rule.identity)
        result = tuple(active.values())
        self._resolution_cache[scope] = result
        self._excluded_cache[scope] = tuple(sorted(excluded_ids))
        return result

    def graph_input(self, graph: Any) -> Mapping[str, Any]:
        metadata = getattr(getattr(graph, "G", None), "graph", {})
        return _frozen({
            "language": self.language,
            "class_definitions": tuple(sorted(metadata.get("class_definitions") or ())),
            "map_ir": dict(metadata.get("map_ir") or {}),
        })

    def resolve(self, node: Any, graph: Any) -> BoundaryResolution:
        scope = tuple(getattr(node, "_turing_source_scope", ()) or ())
        kind = type(node).__name__
        rules = self.rules_for_scope(scope)
        graph_input = self.graph_input(graph)
        excluded = self._excluded_cache.get(scope, ())
        for rule in reversed(rules):
            if rule.node_type != kind or rule.action != "spoof":
                continue
            if not _matches(node, rule.payload.get("match", {})):
                continue
            if not _matches(graph_input, rule.payload.get("graph_match", {})):
                continue
            result = rule.payload["result"]
            attributes = dict(result.get("attributes") or {})
            for name, path in dict(result.get("attributes_from_node") or {}).items():
                attributes[str(name)] = _lookup(node, str(path))
            for name, path in dict(result.get("attributes_from_graph") or {}).items():
                attributes[str(name)] = _lookup(graph_input, str(path))
            constant = result.get("constant")
            if "constant_from_node" in result:
                constant = _lookup(node, str(result["constant_from_node"]))
            if "constant_from_graph" in result:
                constant = _lookup(graph_input, str(result["constant_from_graph"]))
            from .node_special_cases import SpecialCase
            receipt = BoundaryReceipt(
                rule.identity, "spoof", kind, rule.path,
                self.language, scope,
            )
            return BoundaryResolution(
                special_case=SpecialCase(str(result["type"]), attributes, constant),
                receipt=receipt,
                excluded_rule_ids=excluded,
            )
        for rule in reversed(rules):
            if rule.node_type == kind and rule.action == "schema":
                receipt = BoundaryReceipt(
                    rule.identity, "schema", kind, rule.path,
                    self.language, scope,
                )
                return BoundaryResolution(
                    role_schema=rule.payload["role_schema"],
                    receipt=receipt,
                    excluded_rule_ids=excluded,
                )
        return BoundaryResolution(excluded_rule_ids=excluded)

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        language_root = self._safe_child(self.root, self.language)
        if not language_root.is_dir():
            return digest.hexdigest()
        for source in sorted(language_root.rglob("*.node.json")):
            if source.is_symlink():
                raise BoundaryNamespaceError(f"boundary record may not be a symlink: {source}")
            digest.update(str(source.relative_to(language_root)).replace("\\", "/").encode())
            digest.update(b"\0")
            digest.update(source.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()


__all__ = [
    "BoundaryNamespace",
    "BoundaryNamespaceError",
    "BoundaryReceipt",
    "BoundaryResolution",
    "BoundaryRule",
]
