"""Dispatch region partition: atomic region node order, shader-fusible node selection, precision-indivisible node groups and the closure routing dependencies."""

from __future__ import annotations

import networkx as nx
from typing import Any, Iterable, Mapping

from ..common.tensors.fused_ir import canonical_elementwise_op
from .region_scheduling import _dependency_order


def _atomic_region_node_order(
    graph: Any,
    topological_nodes: tuple[int, ...],
    region_by_node: Mapping[int, int],
) -> tuple[int, ...]:
    """Node walk order in which every dispatch region is one atomic unit.

    ``_dependency_order`` orders NODES.  The hierarchy plan emits a region
    closure when the walk meets the region's FIRST member, so a region whose
    members straddle another region's output was emitted before its
    producer: ``pi_update``'s region_1 holds the line-5 ``dt_prev``/``dt_pen``
    casts AND the line-13 ``self.acc`` update, which reads the field written
    from region_0's line-7 cast (``GetAttr`` ``after_write`` edge).  The
    plan listed region_1 first, the control lowering called it first, and
    the accumulator update ran on the stale field.  Contract each region to
    one unit, order the units topologically over the forward semantic
    parent edges between different units (with the same retained-loop
    control-member discount ``_topological_region_order`` applies), break
    ties by the first member's node position so every already-valid order
    is unchanged, and expand each unit back to its members in node order.
    """

    G = graph.G
    order_index = {
        int(node_id): index for index, node_id in enumerate(topological_nodes)
    }

    def unit_of(node_id: int) -> tuple[str, int]:
        region_index = region_by_node.get(int(node_id))
        if region_index is None:
            return ("node", int(node_id))
        return ("region", int(region_index))

    members: dict[tuple[str, int], list[int]] = {}
    for node_id in topological_nodes:
        members.setdefault(unit_of(int(node_id)), []).append(int(node_id))
    if not any(unit[0] == "region" for unit in members):
        return tuple(topological_nodes)
    recursion_control_nodes = frozenset(
        int(node_id)
        for record in (G.graph.get("recursion_table") or {}).values()
        if record.get("control_ir", True)
        for node_id in record.get("control_members", ())
        if int(node_id) in G
    )
    unit_graph = nx.DiGraph()
    unit_graph.add_nodes_from(members)
    for node_id in topological_nodes:
        node_id = int(node_id)
        if node_id in recursion_control_nodes:
            continue
        consumer = unit_of(node_id)
        for parent, _role in (G.nodes[node_id].get("parents") or ()):
            parent = int(parent)
            if (
                parent not in order_index
                or parent in recursion_control_nodes
                or order_index[parent] >= order_index[node_id]
            ):
                continue
            producer = unit_of(parent)
            if producer != consumer:
                unit_graph.add_edge(producer, consumer)
    first_position = {
        unit: order_index[unit_members[0]]
        for unit, unit_members in members.items()
    }
    try:
        ordered_units = tuple(nx.lexicographical_topological_sort(
            unit_graph, key=lambda unit: first_position[unit],
        ))
    except nx.NetworkXUnfeasible as error:
        cycles = tuple(
            tuple(unit for unit in cycle if unit[0] == "region")
            for cycle in nx.simple_cycles(unit_graph)
        )
        raise ValueError(
            "dispatch regions are not atomic compilation units: canonical "
            "data dependencies cross region boundaries cyclically: "
            f"{cycles!r}"
        ) from error
    return tuple(
        node_id
        for unit in ordered_units
        for node_id in members[unit]
    )


def _shader_fusible_node_ids(
    graph: Any, node_ids: Iterable[int],
) -> tuple[int, ...]:
    """Elementwise nodes that may share one generated shader body.

    Shape/layout changes, reductions, contractions, constructors and other
    native kernels remain executable, but each is a reserved dispatch region.
    The captured backend already has dedicated lowerings for those regions;
    admitting them to elementwise fixed-point fusion merely hides their
    ordering boundary and hands ``program_snippet`` an impossible mixed body.
    """

    fusible = []
    for node_id in node_ids:
        data = graph.G.nodes[int(node_id)]
        operation = str(
            (data.get("attributes") or {}).get("tensor")
            or (data.get("attributes") or {}).get("tensor_operation")
            or data.get("op") or data.get("type") or data.get("label")
        )
        try:
            canonical_elementwise_op(operation)
        except KeyError:
            continue
        fusible.append(int(node_id))
    return tuple(fusible)


def _precision_indivisible_node_groups(
    graph: Any,
    executable_node_ids: Iterable[int],
) -> tuple[tuple[int, ...], ...]:
    """Reserve each concorded Precision interval as one compiler region.

    Precision limbs are not independent region values. The source
    concordance is the authority for both boundaries and every
    precision-bearing interior value. Connected components of those facts
    are therefore the exact non-separable units supplied to the ordinary
    region reducer; source ranges and backend operator lists are irrelevant.
    """

    from .identity_concordance import current_identity_book

    executable = {
        int(node_id) for node_id in executable_node_ids
        if int(node_id) in graph.G
    }
    if not executable:
        return ()
    scope = str(
        graph.G.graph.get("source_numeric_scope")
        or graph.G.graph.get("function_name")
        or graph.G.graph.get("qualified_name")
        or "<module>"
    )
    book = current_identity_book()
    class_page = book.page("source_value_class_concordance")
    boundary_page = book.page("source_precision_boundary_concordance")
    precision_values = {
        int(row[1])
        for row in class_page.rows()
        for fact in (class_page.latest(row),)
        if (
            isinstance(row, tuple)
            and len(row) == 2
            and str(row[0]) == scope
            and int(row[1]) in executable
            and isinstance(fact, tuple)
            and len(fact) >= 2
            and str(fact[0]).rsplit(".", 1)[-1] == "Precision"
        )
    }
    boundaries = {
        int(row[1]): (str(fact[0]), int(fact[1]), int(fact[2]))
        for row in boundary_page.rows()
        for fact in (boundary_page.latest(row),)
        if (
            isinstance(row, tuple)
            and len(row) == 2
            and str(row[0]) == scope
            and int(row[1]) in graph.G
            and isinstance(fact, tuple)
            and len(fact) == 3
        )
    }
    members = precision_values | set(boundaries)
    if len(members) < 2:
        return ()

    semantic = nx.Graph()
    semantic.add_nodes_from(members)
    for node_id in members:
        for parent, _role in graph.G.nodes[node_id].get("parents", ()):
            parent = int(parent)
            if parent in members:
                semantic.add_edge(parent, node_id)
        for parent in graph.G.predecessors(node_id):
            if int(parent) in members:
                semantic.add_edge(int(parent), node_id)

    region_page = book.page("source_precision_region_concordance")
    groups = []
    order = {
        int(node_id): index
        for index, node_id in enumerate(_dependency_order(graph))
    }
    for component in nx.connected_components(semantic):
        component_boundaries = {
            node_id: boundaries[node_id]
            for node_id in component if node_id in boundaries
        }
        promotes = tuple(sorted(
            node_id for node_id, fact in component_boundaries.items()
            if fact[0] in {"promote", "pack"}
        ))
        collapses = tuple(sorted(
            node_id for node_id, fact in component_boundaries.items()
            if fact[0] == "collapse"
        ))
        # A helper may return Precision directly.  Its authored interval is
        # open at the function boundary rather than closed by ``collapse``;
        # the promotion and every concorded Precision result are still one
        # non-separable numerical region.  Requiring a collapse stranded the
        # promoted high limbs as anonymous frame formals in precisely these
        # helpers, so native execution read uninitialized storage.
        if not promotes:
            continue
        ordered = tuple(sorted(
            component.intersection(executable), key=order.__getitem__,
        ))
        if len(ordered) < 2:
            continue
        widths = tuple(sorted({
            int(fact[2]) for fact in component_boundaries.values()
        }))
        row = (scope, collapses[-1] if collapses else ordered[-1])
        proposed = (ordered, promotes, collapses, widths)
        incumbent = region_page.latest(row)
        if incumbent is not None and tuple(incumbent) != proposed:
            raise ValueError(
                "source precision region concordance disagreement for "
                f"{row!r}: recorded={incumbent!r}, proposed={proposed!r}"
            )
        if incumbent is None:
            region_page.set(row, 0, proposed)
        for node_id in ordered:
            graph.G.nodes[node_id].setdefault("attributes", {})[
                "source_precision_region"
            ] = row
        groups.append(ordered)
    return tuple(groups)


def _closure_routing_dependencies(
    graph: Any,
) -> tuple[frozenset[tuple[int, int]], frozenset[int]]:
    """Recover causal facts carried by closure routing rather than by edges.

    A call to a compiled function supplies that callee's free variables by
    name: the coordinator looks the name up in this graph's identity table and
    evaluates whichever definition precedes the call.  No parent reference or
    graph edge records that read, so region planning cannot see it.  Two
    consequences follow, and both are returned here.

    The routed definition is observed from outside the shader, so it is a
    compartment boundary value and must be published.  The call also depends
    on it, so a region may not internalize both the definition and any value
    produced by that call; doing so would make the region depend on its own
    unpublished result.  The returned edges are declared to the reducer as
    structural dependencies exactly like a path through a coordinator node.
    """

    identities = graph.G.graph.get("identity_table", {}) or {}
    if not identities:
        return frozenset(), frozenset()
    function_table = getattr(graph, "function_table", None)
    if function_table is None:
        return frozenset(), frozenset()

    def free_binding_names(reference: int) -> frozenset[str]:
        try:
            entry = function_table.entry(int(reference))
        except (KeyError, TypeError, ValueError):
            return frozenset()
        callee = getattr(entry, "graph", None)
        if callee is None:
            return frozenset()
        return frozenset(
            str(name)
            for _node_id, data in callee.G.nodes(data=True)
            if data.get("type") == "Input"
            for attributes in ((data.get("attributes") or {}),)
            if attributes.get("binding_kind") == "external"
            for name in (attributes.get("binding_name"),)
            if name is not None
        )

    order_index = {
        node_id: index
        for index, node_id in enumerate(_dependency_order(graph))
    }
    free_names: dict[int, frozenset[str]] = {}
    edges: set[tuple[int, int]] = set()
    routed: set[int] = set()
    for call_id, data in graph.G.nodes(data=True):
        attributes = data.get("attributes") or {}
        reference = attributes.get("callee_ref")
        if reference is None:
            reference = attributes.get("method_ref")
        if reference is None:
            continue
        names = free_names.get(int(reference))
        if names is None:
            names = free_binding_names(reference)
            free_names[int(reference)] = names
        call_position = order_index[call_id]
        for name in names:
            for definition in identities.get(name, ()):
                # Only a definition that can already have executed is a
                # candidate for this call site.  Declaring a later one would
                # invent a backward dependency the schedule cannot satisfy.
                if (
                    definition in order_index
                    and order_index[definition] < call_position
                ):
                    edges.add((definition, call_id))
                    routed.add(definition)
    return frozenset(edges), frozenset(routed)
