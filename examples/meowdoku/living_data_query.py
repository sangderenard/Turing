"""Bounded, non-executing views of the existing Living Data identity graph.

This is transport/projection, not navigation physics. A dependency is not a
walkable passage. No supplied source, worker, WASM, or URL is executed or fetched.
"""
from collections import deque
import math

MAX_BODY = 4 * 1024 * 1024
MAX_NODES = 4096
MAX_EDGES = 16384


class QueryError(ValueError):
    pass


def _text(value, field):
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise QueryError(f"{field} must be a nonempty string of at most 1024 characters")
    return value


def _list(value, field):
    if not isinstance(value, list):
        raise QueryError(f"{field} must be an array")
    return value


def _integer(value, low, high, field):
    if type(value) is not int or not low <= value <= high:
        raise QueryError(f"{field} must be an integer from {low} to {high}")
    return value


class GraphView:
    """Index canonical regions/buildings/rooms/objects and declared edges only."""

    def __init__(self, model, layer="layout"):
        if layer not in ("layout", "world"):
            raise QueryError("layer must be layout or world")
        self.layer = layer
        if not isinstance(model, dict) or model.get("schema") != "abstract-ui-div-map-v0":
            raise QueryError("map must use abstract-ui-div-map-v0")
        self.nodes, self.edges, self.adj = {}, [], {}
        pending = deque([(model, "regions" if layer == "layout" else None)])
        # Explicit canonical containers: never discover nodes recursively inside
        # source, extensions, worker programs, or other untrusted payloads.
        while pending:
            node, children = pending.popleft()
            if not isinstance(node, dict):
                raise QueryError("node must be an object")
            identity = _text(node.get("identity"), "identity")
            if identity in self.nodes:
                raise QueryError("duplicate node identity")
            if len(self.nodes) >= MAX_NODES:
                raise QueryError("map exceeds node limit")
            self.nodes[identity] = node
            self.adj[identity] = []
            if children:
                next_children = {"regions": "buildings", "buildings": "rooms", "rooms": None}[children]
                for child in _list(node.get(children, []), children):
                    pending.append((child, next_children))
                if len(pending) + len(self.nodes) > MAX_NODES:
                    raise QueryError("map exceeds node limit")
        world = model.get("world", {})
        if not isinstance(world, dict):
            raise QueryError("world must be an object")
        for node in _list(world.get("objects", []), "objects") if layer == "world" else []:
            if not isinstance(node, dict):
                raise QueryError("object must be an object")
            identity = _text(node.get("identity"), "identity")
            if identity in self.nodes:
                raise QueryError("duplicate node identity")
            if len(self.nodes) >= MAX_NODES:
                raise QueryError("map exceeds node limit")
            self.nodes[identity] = node
            self.adj[identity] = []
        seen = set()
        def edge(source, target, relationship):
            source = _text(source, "edge source")
            target = _text(target, "edge target")
            relationship = _text(relationship, "relationship")
            key = (source, target, relationship)
            if key in seen:
                return
            if len(self.edges) >= MAX_EDGES:
                raise QueryError("map exceeds edge limit")
            seen.add(key)
            record = {"source": source, "target": target, "relationship": relationship}
            self.edges.append(record)
            for identity in {source, target}:
                self.adj.setdefault(identity, []).append(record)
        for identity, node in self.nodes.items():
            for dep in _list(node.get("dependencies", []), "dependencies"):
                if not isinstance(dep, dict):
                    raise QueryError("dependency must be an object")
                edge(identity, dep.get("target"), dep.get("relationship"))
        for track in _list(model.get("tracks", []), "tracks") if layer == "layout" else []:
            if not isinstance(track, dict):
                raise QueryError("track must be an object")
            edge(track.get("source"), track.get("target"), track.get("relationship"))

    def node(self, identity):
        node = self.nodes.get(identity)
        result = {"identity": identity, "known": node is not None}
        if node is not None:
            for key in ("kind", "name", "member_kind", "type_name"):
                if key in node and node[key] is not None:
                    if not isinstance(node[key], str) or len(node[key]) > 1024:
                        raise QueryError(f"{key} must be a string of at most 1024 characters")
                    result[key] = node[key]
            position = node.get("position")
            if position is not None:
                if not isinstance(position, dict) or any(
                    type(v) not in (int, float) or (type(v) is float and not math.isfinite(v))
                    for v in position.values()
                ) or set(position) - {"column", "row", "width", "height"}:
                    raise QueryError("position must contain finite document-layout coordinates")
                result["layout_position"] = dict(position)
        return result

    def query(self, identity, *, radius=1, limit=64, offset=0, edge_offset=0):
        _text(identity, "at")
        if identity not in self.nodes:
            raise QueryError("at must name a known node")
        _integer(radius, 0, 4, "radius")
        _integer(limit, 1, 256, "limit")
        _integer(offset, 0, MAX_NODES + MAX_EDGES * 2, "offset")
        _integer(edge_offset, 0, MAX_EDGES, "edge_offset")
        distances, queue, found = {identity: 0}, deque([identity]), []
        while queue:
            current = queue.popleft()
            distance = distances[current]
            if distance == radius:
                continue
            for edge in self.adj.get(current, []):
                other = edge["target"] if edge["source"] == current else edge["source"]
                if other in distances:
                    continue
                distances[other] = distance + 1
                found.append(other)
                # Unknown endpoints are references, not permission to infer a world.
                if other in self.nodes:
                    queue.append(other)
        page = found[offset:offset + limit]
        # Edge pages cover the whole horizon, independently of node pages.
        # Otherwise an edge joining nodes on different pages disappears.
        selected = {identity, *found}
        edges = [dict(e, traversability="unknown") for e in self.edges
                 if e["source"] in selected and e["target"] in selected]
        # Bound dense graph responses separately from the node page.
        return {"schema": "living-data-query.v1", "layer": self.layer, "at": self.node(identity),
                "nodes": [dict(self.node(n), distance=distances[n]) for n in page],
                "edges": edges[edge_offset:edge_offset + 256], "edge_total": len(edges),
                "next_edge_offset": edge_offset + 256 if edge_offset + 256 < len(edges) else None,
                "radius": radius, "total": len(found),
                "next_offset": offset + limit if offset + limit < len(found) else None,
                "knowledge": "caller-supplied graph; not actor perception or live state",
                "directions": {"basis": "declared directed edges", "vertical": "unknown",
                               "containment": "contained-by points outward; reverse points inward"},
                "capabilities": {"graph_query": True, "advance": False, "render": False}}


def query(payload):
    if not isinstance(payload, dict) or set(payload) - {"map", "at", "radius", "limit", "offset", "layer", "edge_offset"}:
        raise QueryError("expected map, at, and optional layer, radius, limit, offset, edge_offset")
    return GraphView(payload.get("map"), payload.get("layer", "layout")).query(
        payload.get("at"), radius=payload.get("radius", 1),
        limit=payload.get("limit", 64), offset=payload.get("offset", 0),
        edge_offset=payload.get("edge_offset", 0))
