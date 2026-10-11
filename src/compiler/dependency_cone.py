"""Reachability over incumbent dependency relations, without another graph.

Callers supply exact identities and the relation's existing reader. The reader
may be NetworkX.successors/predecessors, a parent-record index, or book edges.
No names, inferred edges, identities, descriptor cache or book facts are added.
"""
from collections import deque
import heapq


def iter_dependency_cone(roots, dependents, *, include_roots=True, order=None):
    """Visit each reachable identity once, including cycles and shared tails.

    Default order is stable FIFO (root order, then relation-reader order), with
    O(reachable vertices + edges) work. ``order`` instead supplies a caller's
    incumbent sweep order: a newly reached later vertex runs in this sweep;
    an earlier vertex runs in the next. This reproduces scan-to-fixed-point
    visitation without scanning unrelated vertices or empty sweep prefixes.
    Ordered traversal costs O(order size + cone edges + cone vertices*log K).

    Roots are already changed/visited even when not emitted. They may be absent
    from ``order`` (a retired identity can still have resident consumers); every
    discovered dependent must occur in that order. The relation must stay
    stable during traversal. Ownership and invalidation remain with the caller.
    """
    roots = tuple(dict.fromkeys(roots))
    visited = set(roots)
    if include_roots:
        yield from roots
    if order is None:
        pending = deque(roots)
        while pending:
            source = pending.popleft()
            for target in dependents(source):
                if target not in visited:
                    visited.add(target)
                    pending.append(target)
                    yield target
        return

    vertices = tuple(order)
    rank = {node: index for index, node in enumerate(vertices)}
    if len(rank) != len(vertices):
        raise ValueError("dependency cone sweep order contains duplicate identities")
    queued = set()
    current, following = [], []

    def enqueue(source, position):
        for target in dependents(source):
            if target in visited or target in queued:
                continue
            ordinal = rank[target]
            queued.add(target)
            heapq.heappush(current if ordinal > position else following, ordinal)

    for source in roots:
        enqueue(source, -1)
    while current or following:
        while current:
            ordinal = heapq.heappop(current)
            node = vertices[ordinal]
            queued.remove(node)
            visited.add(node)
            yield node
            enqueue(node, ordinal)
        current, following = following, []


def parent_record_dependents(graph, *, excluded_roles=()):
    """Index this phase's exact ``parents`` records, in graph insertion order.

    This is an O(V+E) read index, not a persistent graph/cache. Rebuild after
    operand/topology changes. Parallel operand roles to one consumer do not
    duplicate that consumer; absent producer IDs are retained as keys.
    """
    excluded = frozenset(map(str, excluded_roles))
    consumers = {node: {} for node in graph}
    for node, data in graph.nodes(data=True):
        for parent, role in data.get("parents") or ():
            if str(role) not in excluded:
                consumers.setdefault(parent, {})[node] = None
    return {source: tuple(targets) for source, targets in consumers.items()}
