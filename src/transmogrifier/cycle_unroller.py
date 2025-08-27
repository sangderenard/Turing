"""Utilities for removing trivial cycles from directed graphs.

This module introduces a small helper ``unroll_self_edges`` which replaces
self-loop edges ``(u, u)`` with a pair of versioned nodes.  Each occurrence of a
self-loop on node ``u`` generates two concrete nodes ``"{u}_v0"`` and
``"{u}_v1"``:

* ``{u}_v0`` receives all of the original incoming edges to ``u``.
* ``{u}_v1`` becomes the source for all outgoing edges from ``u``.
* The former self-loop edge is transformed into an edge from ``{u}_v0`` to
  ``{u}_v1``.

The suffix ``_vN`` denotes the *N*-th version of the node after unrolling.  All
attributes from the original node are copied to both versions, and each version
stores a ``"source"`` attribute pointing back to the originating node.  This
mapping allows downstream passes to relate versioned nodes back to their source
without maintaining a separate side table.
"""

from __future__ import annotations

from typing import Dict, Iterable, Set

import networkx as nx


def rebuild_parents_children(graph: nx.DiGraph) -> None:
    """Reconstruct ``parents`` and ``children`` node attributes from edges.

    Many passes expect each node to expose ``"parents"`` and ``"children"``
    lists of ``(node_id, edge_data)`` tuples.  After structural rewrites these
    attributes may become stale, so this helper wipes existing entries and
    rebuilds them directly from the graph's edge set.
    """

    for n in graph.nodes:
        graph.nodes[n]["parents"] = []
        graph.nodes[n]["children"] = []

    for u, v, data in graph.edges(data=True):
        payload = dict(data) if data else None
        graph.nodes[v]["parents"].append((u, payload))
        graph.nodes[u]["children"].append((v, payload))


def unroll_self_edges(graph: nx.DiGraph) -> nx.DiGraph:
    """Return a copy of ``graph`` with self-loops removed via node versioning.

    Any edge of the form ``(u, u)`` triggers the creation of two concrete nodes
    ``"{u}_v0"`` and ``"{u}_v1"``.  All incoming edges that originally targeted
    ``u`` are redirected to ``"{u}_v0"`` while all outgoing edges now originate
    from ``"{u}_v1"``.  The self-loop itself becomes an edge from the ``v0``
    version to the ``v1`` version, preserving any edge attributes.  Node
    attributes are copied verbatim to each version and a ``"source"`` attribute
    is added to link back to the original node identifier.

    Parameters
    ----------
    graph:
        The :class:`networkx.DiGraph` potentially containing self-loop edges.

    Returns
    -------
    nx.DiGraph
        A new directed graph with self-loop edges unrolled into versioned
        nodes.  Nodes that had no self-loop are copied unchanged (aside from the
        added ``"source"`` and ``"version"`` attributes).
    """

    unrolled = nx.DiGraph()
    self_loop_nodes = {n for n in graph.nodes if graph.has_edge(n, n)}

    # ------------------------------------------------------------------
    # Node creation with attribute preservation and source mapping
    # ------------------------------------------------------------------
    for node, attrs in graph.nodes(data=True):
        if node in self_loop_nodes:
            for version in (0, 1):
                new_name = f"{node}_v{version}"
                new_attrs = dict(attrs)
                new_attrs["source"] = node
                new_attrs["version"] = version
                unrolled.add_node(new_name, **new_attrs)
        else:
            new_attrs = dict(attrs)
            new_attrs["source"] = node
            new_attrs["version"] = 0
            unrolled.add_node(node, **new_attrs)

    # ------------------------------------------------------------------
    # Edge redirection
    # ------------------------------------------------------------------
    for u, v, attrs in graph.edges(data=True):
        if u == v:
            # Replace the self-loop with an edge from v0 -> v1
            unrolled.add_edge(f"{u}_v0", f"{u}_v1", **attrs)
            continue

        src = f"{u}_v1" if u in self_loop_nodes else u
        dst = f"{v}_v0" if v in self_loop_nodes else v
        unrolled.add_edge(src, dst, **attrs)

    # Store mapping back to original nodes for external use.
    source_map: Dict[str, str] = {}
    for n in unrolled.nodes:
        source_map[n] = unrolled.nodes[n]["source"]
    unrolled.graph["source_map"] = source_map

    rebuild_parents_children(unrolled)
    return unrolled


def sccs_with_cycles(graph: nx.DiGraph) -> Iterable[Set[str]]:
    """Yield strongly connected components that contain at least one cycle.

    Any SCC with more than one node necessarily contains a cycle.  A single
    node component only represents a cycle if the node has a self-loop.  The
    helper abstracts this logic and yields each qualifying component as a set
    of node identifiers.
    """

    for scc in nx.strongly_connected_components(graph):
        nodes = set(scc)
        if len(nodes) > 1:
            yield nodes
        else:
            n = next(iter(nodes))
            if graph.has_edge(n, n):
                yield nodes


def unroll_scc_once(graph: nx.DiGraph, scc: Set[str]) -> None:
    """Unroll a single strongly connected component in place.

    Each node ``u`` in ``scc`` is replaced by two versions ``u_v0`` and
    ``u_v1``.  Incoming edges from outside the component are redirected to
    ``u_v0`` while outgoing edges to the outside originate from ``u_v1``.
    Edges internal to the component now connect ``u_v0`` to ``v_v1`` which
    breaks the cycle by enforcing a hop between versions.  Original nodes are
    removed after rewiring.  Node attributes are copied verbatim and augmented
    with ``"source"`` and ``"version"`` metadata.
    """

    v0 = {u: f"{u}_v0" for u in scc}
    v1 = {u: f"{u}_v1" for u in scc}

    # Create versioned nodes with copied attributes
    for u in scc:
        attrs = dict(graph.nodes[u])
        for version, name in ((0, v0[u]), (1, v1[u])):
            new_attrs = dict(attrs)
            new_attrs["source"] = u
            new_attrs["version"] = version
            graph.add_node(name, **new_attrs)

    # Rewire edges
    for u in scc:
        for pred in list(graph.predecessors(u)):
            data = graph.get_edge_data(pred, u, default={})
            if pred not in scc:
                graph.add_edge(pred, v0[u], **data)
        for succ in list(graph.successors(u)):
            data = graph.get_edge_data(u, succ, default={})
            if succ not in scc:
                graph.add_edge(v1[u], succ, **data)
            else:
                graph.add_edge(v0[u], v1[succ], **data)

    # Stitch per-node same-iteration step
    for u in scc:
        graph.add_edge(v0[u], v1[u])

    # Remove originals
    graph.remove_nodes_from(scc)


def unroll_all_cycles_once(graph: nx.DiGraph) -> bool:
    """Unroll every cyclic strongly connected component exactly once.

    Returns ``True`` if any component was unrolled.
    """

    mutated = False
    for scc in list(nx.strongly_connected_components(graph)):
        nodes = set(scc)
        has_cycle = len(nodes) > 1 or any(graph.has_edge(n, n) for n in nodes)
        if not has_cycle:
            continue
        unroll_scc_once(graph, nodes)
        mutated = True
    if mutated:
        rebuild_parents_children(graph)
        graph.graph["source_map"] = {n: graph.nodes[n].get("source", n) for n in graph.nodes}
    return graph.graph["source_map"]
