"""Frozen pre-cone descriptor withdrawal; test/probe oracle, not a backend."""


def original_invalidation(graph, source_ids, reason, *, _dependents=None):
    sources = {int(source_id) for source_id in source_ids}
    affected = set(sources)
    changed = True
    while changed:
        changed = False
        for node_id, data in graph.G.nodes(data=True):
            node_id = int(node_id)
            if node_id in affected:
                continue
            parents = tuple(
                int(parent) for parent, role in data.get("parents") or ()
                if str(role) not in {"callee", "func", "definition"}
            )
            causes = tuple(parent for parent in parents if parent in affected)
            if not causes:
                continue
            affected.add(node_id)
            data.pop("tensor", None)
            try:
                from src.compiler.identity_concordance import invalidate_proven_shape, shape_scope_of
                invalidate_proven_shape(
                    shape_scope_of(graph), int(data.get("value_id", node_id)),
                    int(causes[0]), str(reason),
                )
            except Exception:
                pass
            changed = True
