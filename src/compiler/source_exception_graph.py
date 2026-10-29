"""Exception-related edges projected from the completed source ProcessGraphs."""

from __future__ import annotations


def publish_source_exception_graph(source_graph):
    """Keep the existing node IDs and operand edges; mint no control nodes.

    A Try owns its body, handlers, else, and finalbody through the source
    graph's operand roles.  The path from each Raise back to those owners is
    retained so later control lowering can follow graph edges instead of
    recovering lexical nesting from an AST or a count of finalizers.
    """

    from .concordance_declarations import (
        REDUCTION, SOURCE_EXCEPTION_GRAPH, SourceExceptionGraph,
    )
    from .identity_concordance import Derived, Mode, current_identity_book
    from ..common.tensors.topological_reducer import node_identity_cell

    book = current_identity_book()
    wrappers = [source_graph]
    table = getattr(source_graph, "function_table", None)
    if table is not None:
        wrappers.extend(
            entry.graph for entry in table
            if getattr(entry, "graph", None) is not None
        )

    nodes = set()
    edges = set()
    for wrapper in wrappers:
        process = wrapper.G
        # ProcessGraph stores a child's authored operand in its parent's
        # ``parents`` list.  The directed exception view runs from owner to
        # child, preserving the same ID pair and operand role.
        children = {
            int(owner): tuple(
                (int(child), str(role))
                for child, role in data.get("parents") or ()
                if child in process
            )
            for owner, data in process.nodes(data=True)
        }
        kinds = {
            int(node): str(data.get("type") or "")
            for node, data in process.nodes(data=True)
        }
        try_ids = tuple(
            node for node, kind in kinds.items() if kind == "Try"
        )
        if not try_ids:
            continue

        def source_id(node):
            # A canonical integer is local to its read scope.  The Ref is
            # the exact identity already held by the concordance.
            return node_identity_cell(wrapper, node)

        def add_node(node):
            nodes.add(source_id(node))

        def add_edge(owner, child, role):
            add_node(owner)
            add_node(child)
            edges.add((source_id(owner), source_id(child), role))

        # The authored direct edges are the only ownership relationships.
        # Retain all Try arms and each handler's own body edges.
        for try_id in try_ids:
            add_node(try_id)
            for child, role in children[try_id]:
                if role in {"body", "handlers", "orelse", "finalbody"}:
                    add_edge(try_id, child, role)
                    if role == "handlers":
                        for member, member_role in children.get(child, ()):
                            if member_role in {"type", "name", "body"}:
                                add_edge(child, member, member_role)

        # Keep only paths already present in the ProcessGraph that connect
        # an exception arm to an explicit Raise.  No inferred transfer edge
        # or synthetic node is introduced here.
        visited = set()

        def raise_paths(node):
            if node in visited:
                return node in raise_reachable
            visited.add(node)
            reachable = kinds.get(node) == "Raise"
            for child, role in children.get(node, ()):
                if raise_paths(child):
                    add_edge(node, child, role)
                    reachable = True
            if reachable:
                raise_reachable.add(node)
            return reachable

        raise_reachable = set()
        for try_id in try_ids:
            raise_paths(try_id)

    ordered_edges = tuple(sorted(
        edges, key=lambda edge: (
            edge[0].page.name, repr(edge[0].row),
            edge[1].page.name, repr(edge[1].row), edge[2],
        ),
    ))
    fact = SourceExceptionGraph(
        nodes=tuple(sorted(
            nodes, key=lambda ref: (ref.page.name, repr(ref.row)),
        )),
        edges=tuple((owner, child) for owner, child, _ in ordered_edges),
        roles=tuple(role for _, _, role in ordered_edges),
    )
    scope = source_graph.G.graph.get("ingestion_value_scope")
    if scope is None:
        raise ValueError("source exception graph has no ingestion scope")
    if nodes:
        book.post(
            SOURCE_EXCEPTION_GRAPH, (scope,), fact, stage=REDUCTION,
            provenance=Derived(fact.nodes),
            mode=Mode.CONCORD,
        )
    source_graph.G.graph["source_exception_graph"] = fact
    return fact


def publish_ssa_exception_topology(module, source_graph):
    """Expose the existing exception graph as one SSA module object.

    The SSA row points to the source graph cell.  Neither the projection nor
    the module object creates a second copy of any node or edge identity.
    """

    from .concordance_declarations import (
        SOURCE_EXCEPTION_GRAPH, SSA_EXCEPTION_TOPOLOGY,
        SSA_EXCEPTION_TOPOLOGY_STAGE,
    )
    from .identity_concordance import Derived, Mode, current_identity_book
    from ..transmogrifier.ssa import SSAExceptionTopology

    scope = source_graph.G.graph.get("ingestion_value_scope")
    if scope is None:
        raise ValueError("SSA exception topology has no source scope")
    book = current_identity_book()
    source_cell = book.latest_ref(SOURCE_EXCEPTION_GRAPH, (scope,))
    if source_cell is None:
        return None
    graph = book.page(SOURCE_EXCEPTION_GRAPH).latest((scope,))
    ssa_cell = book.post(
        SSA_EXCEPTION_TOPOLOGY, (scope,), source_cell,
        stage=SSA_EXCEPTION_TOPOLOGY_STAGE,
        provenance=Derived((source_cell,)), mode=Mode.CONCORD,
    )
    topology = SSAExceptionTopology(
        ssa_cell, source_cell, graph.nodes, graph.edges, graph.roles,
    )
    module.exception_topology = topology
    return topology
