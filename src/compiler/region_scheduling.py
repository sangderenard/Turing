"""Region scheduling: dependency order, topological region order/schedule and dependency levels over a ProcessGraph."""

from __future__ import annotations

import networkx as nx
from typing import Any

from .hierarchical_plan import PlanClosure, PlanLine


def _dependency_order(graph: Any) -> tuple[int, ...]:
    """Return DAG order, or stable condensation order for retained loops.

    The result is cached on the graph, keyed by a cheap (node count, edge
    count) fingerprint. During deployment planning the same shell graph is
    dependency-ordered repeatedly (once per ``_build_shell_hierarchy_plan``,
    which runs on shell construction and on every ``refresh_hierarchy_plan``),
    yet it is only read there -- so re-running the topological sort each time
    was pure repeated work. The fingerprint invalidates the cache the moment
    the graph gains or loses a node/edge, so a genuinely mutated graph is
    re-ordered.
    """

    G = graph.G
    semantic_parent_edges = tuple(
        (int(parent), int(node_id))
        for node_id, data in G.nodes(data=True)
        for parent, _role in (data.get("parents") or ())
        if int(parent) in G and int(parent) != int(node_id)
    )
    semantic_fingerprint = sum(
        ((left * 1_000_003) ^ right)
        for left, right in semantic_parent_edges
    )
    # Counts alone collide: removing two dead constants and adding two
    # edge-less member inputs leaves every count and the edge hash unchanged
    # while the node identity set differs, so a stale order would name nodes
    # that no longer exist.  The identity set is part of the fingerprint.
    node_fingerprint = sum(
        (int(node_id) * 1_000_003) ^ (int(node_id) >> 3) for node_id in G
    )
    fingerprint = (
        G.number_of_nodes(), G.number_of_edges(),
        len(semantic_parent_edges), semantic_fingerprint, node_fingerprint,
    )
    cached = G.graph.get("_dependency_order_cache")
    if cached is not None and cached[0] == fingerprint:
        return cached[1]
    dependency_graph = G.copy()
    dependency_graph.add_edges_from(semantic_parent_edges)
    try:
        order = tuple(nx.lexicographical_topological_sort(
            dependency_graph, key=lambda value_id: int(value_id)
        ))
    except nx.NetworkXUnfeasible:
        recursive = G.graph.get("recursion_table")
        if not recursive or set(graph.levels) != set(G):
            raise
        order = tuple(sorted(
            G,
            key=lambda node_id: (
                int(graph.levels[node_id]),
                int(node_id),
            ),
        ))
    G.graph["_dependency_order_cache"] = (fingerprint, order)
    return order


def _topological_region_order(shell: Any, candidate_regions: Any) -> tuple[int, ...]:
    return _topological_region_schedule(shell, candidate_regions)[0]


def _topological_region_schedule(shell: Any, candidate_regions: Any):
    """Order dispatch region indices to respect data dependencies.

    ``range(len(dispatch_subgraphs))`` -- discovery/creation order -- is
    NOT execution order: nothing about it guarantees a region discovered
    later doesn't produce a value a region discovered earlier reads (e.g.
    a loop that fills an array vs. a later whole-array copy of it). Two
    call sites here fed that naive numeric order straight into
    ``overlay_scheduled_control`` as the flat scheduled region sequence,
    which silently discarded any dependency edge between regions -- see
    tools/HANDOFF_fluid_c_shell.md and tools/DIFFERENTIAL_PHASES.md.

    A hierarchy item is discovered when the *first* member of its region is
    encountered.  That is a useful stable tie-break, but it is not a proof of
    atomic call order: another member can consume a value from a region whose
    first independent member appears later.  Build the region dependency DAG
    from canonical ProcessGraph ancestry and topologically order that instead.
    """

    candidates = tuple(int(index) for index in candidate_regions)
    candidate_set = set(candidates)
    plan = getattr(shell, "hierarchy_plan", None)
    plan_items = getattr(plan, "items", None) if plan is not None else None
    from_plan = tuple(
        int(item.name.split("_", 1)[1])
        for item in (plan_items or ())
        if isinstance(item, PlanClosure)
        and item.name.startswith("region_")
        and int(item.name.split("_", 1)[1]) in candidate_set
    )
    preference = tuple(dict.fromkeys((*from_plan, *candidates)))
    rank = {region: index for index, region in enumerate(preference)}
    graph = shell.process_graph
    nodes_by_region = {
        region: frozenset(map(
            int,
            shell.dispatch_subgraphs[region].G.graph.get(
                "deployment_nodes", (),
            ),
        ))
        for region in candidates
    }
    region_by_node = {
        node_id: region
        for region, nodes in nodes_by_region.items()
        for node_id in nodes
    }
    # A retained loop's own internal state-effect chain (for example two
    # mutually exclusive branches each doing ``ctrl.clamp_events += 1``)
    # legitimately closes a cycle in raw ProcessGraph ancestry: one
    # branch's result feeds the next iteration's read of the other. That
    # is exactly what ``recursion_table``/``control_members`` -- already
    # published for ``reduce_scheduled_shader_regions``, which drops edges
    # incident to those nodes before checking its own scheduling graph is
    # acyclic -- exists to name. This function walked raw, un-filtered
    # ancestry instead, so the very cycle the scheduler had already
    # discounted came back as an unresolvable region dependency cycle
    # (diagnosed via tools/repro_step_with_dt_control_used.py: regions
    # holding two branch-exclusive ``ctrl.clamp_events += 1`` computations
    # inside ``step_with_dt_control_used``'s retry loop). Apply the same
    # discount here instead of inventing a second rule for one graph.
    recursion_control_nodes = frozenset(
        int(node_id)
        for record in (graph.G.graph.get("recursion_table") or {}).values()
        if record.get("control_ir", True)
        for node_id in record.get("control_members", ())
        if int(node_id) in graph.G
    )
    ancestry_graph = (
        graph.G if not recursion_control_nodes
        else graph.G.copy()
    )
    if recursion_control_nodes:
        ancestry_graph.remove_edges_from(tuple(
            (left, right) for left, right in ancestry_graph.edges
            if left in recursion_control_nodes or right in recursion_control_nodes
        ))
    dependency_graph = nx.DiGraph()
    dependency_graph.add_nodes_from(candidates)
    for consumer, nodes in nodes_by_region.items():
        for node_id in nodes:
            if node_id not in ancestry_graph:
                continue
            for ancestor in nx.ancestors(ancestry_graph, node_id):
                producer = region_by_node.get(int(ancestor))
                if producer is not None and producer != consumer:
                    dependency_graph.add_edge(producer, consumer)
    # Closure captures are the compiler's explicit cross-unit ABI.  Some
    # structural operations preserve the canonical value correlation without
    # retaining a direct ProcessGraph edge after compartment extraction, so
    # graph ancestry alone can miss exactly the dependency that a native call
    # signature still exposes.  Order from the typed capture/result surface as
    # well; names and temporary discovery IDs are intentionally irrelevant.
    closures = {
        int(item.name.split("_", 1)[1]): item
        for item in (plan_items or ())
        if isinstance(item, PlanClosure)
        and item.name.startswith("region_")
        and int(item.name.split("_", 1)[1]) in candidate_set
    }
    producers_by_value: dict[int, set[int]] = {}
    for region, closure in closures.items():
        for line in closure.items:
            if not isinstance(line, PlanLine):
                continue
            for value_id in line.outputs:
                producers_by_value.setdefault(int(value_id), set()).add(region)
    for consumer, closure in closures.items():
        for value_id in closure.captures:
            for producer in producers_by_value.get(int(value_id), ()):
                if producer != consumer:
                    dependency_graph.add_edge(producer, consumer)
    try:
        result = tuple(nx.lexicographical_topological_sort(
            dependency_graph,
            key=lambda region: (rank.get(int(region), len(rank)), int(region)),
        ))
    except nx.NetworkXUnfeasible as error:
        cycles = tuple(nx.simple_cycles(dependency_graph))
        raise ValueError(
            "dispatch regions are not atomic compilation units: canonical "
            f"data dependencies cross region boundaries cyclically: {cycles!r}"
        ) from error
    import os as _os, sys as _sys
    if _os.environ.get("TURING_DEBUG_REGION_ORDER"):
        _fn = getattr(getattr(shell, "process_graph", None), "G", None)
        _fn = _fn.graph.get("function_name") if _fn is not None else None
        print(
            f"DEBUGTOPOORDER fn={_fn} plan_items_present={plan_items is not None} "
            f"candidates={candidates} result={result}",
            file=_sys.stderr,
        )
    return result, tuple(dependency_graph.edges)


def _control_dependency_value_ids(control: Any) -> frozenset[int]:
    from .control_source import control_dependency_value_ids

    return control_dependency_value_ids(control)


#: ``id(graph.G)`` -> (node count, levels, weak reference to that very
#: ``graph.G``).  The id alone is not an identity: callsite copies are made
#: and freed every specialization round, so a freed graph's id is recycled by
#: a new graph with (by chance) the same node count and the stale levels were
#: returned for it.  The weak reference is checked on every hit.
#: The entry is dropped when that graph dies (the weak reference's callback),
#: so the cache holds levels for LIVE graphs only: it used to keep one dict
#: per callsite copy ever made, for the life of the process.
_DEPENDENCY_LEVEL_CACHE: dict[int, tuple[int, dict[int, int], Any]] = {}


def release_dependency_levels() -> int:
    """Drop every cached level table; return how many there were.

    The levels are a pure function of the graph (``_dependency_levels``
    recomputes the same table), so this releases recomputable memory."""

    released = len(_DEPENDENCY_LEVEL_CACHE)
    _DEPENDENCY_LEVEL_CACHE.clear()
    return released


def _dependency_levels(graph: Any) -> dict[int, int]:
    """ASAP level per node, over the condensation so cycles stay defined.

    This is `ILPScheduler.compute_asap_levels` -- longest path from the roots
    -- with strongly connected components collapsed first, because a retained
    loop is irreducible recursion rather than an invalid schedule and every
    member of one is at the same causal position.
    """

    import weakref

    key = id(graph.G)
    cached = _DEPENDENCY_LEVEL_CACHE.get(key)
    if (
        cached is not None
        and cached[2]() is graph.G
        and cached[0] == graph.G.number_of_nodes()
    ):
        return cached[1]
    levels: dict[int, int] = {}
    try:
        import networkx as _nx

        condensed = _nx.condensation(graph.G)
        component_level: dict[int, int] = {}
        for component in _nx.topological_sort(condensed):
            parents = tuple(condensed.predecessors(component))
            component_level[component] = (
                0 if not parents
                else 1 + max(component_level[parent] for parent in parents)
            )
        for component, members in condensed.nodes(data="members"):
            depth = int(component_level.get(component, 0))
            for member in members or ():
                levels[int(member)] = depth
    except Exception:
        levels = {}
    def forget(reference: Any, key: int = key) -> None:
        held = _DEPENDENCY_LEVEL_CACHE.get(key)
        if held is not None and held[2] is reference:
            del _DEPENDENCY_LEVEL_CACHE[key]

    _DEPENDENCY_LEVEL_CACHE[key] = (
        graph.G.number_of_nodes(), levels, weakref.ref(graph.G, forget),
    )
    return levels
