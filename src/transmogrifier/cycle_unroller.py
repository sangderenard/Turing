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

from typing import Dict

import networkx as nx


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

    return unrolled
