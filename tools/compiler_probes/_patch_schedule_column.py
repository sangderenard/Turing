"""Let the dependency schedule be the concordance's column.

The shape page was "first answer with extents wins, forever": I used
`len(history)` as the column, which is an arrival counter, so the page had no
notion of WHEN a fact was learned relative to anything else and could not
absorb a later, better-informed event.  A value annotated early kept that
annotation after a later stage proved better, which is the whole `matmul`
symptom and the reason patching individual snapshot sites only ever fixed
some of them.

`IdentityPage` is a row x COLUMN table and the column is meant to be a
position, so the position is now the node's level in the dependency graph --
the same ASAP level `ILPScheduler` computes, taken over the condensation so a
retained loop's feedback edges do not make it undefined.  A fact recorded at a
deeper level supersedes a shallower one because it was derived from strictly
more of the program; facts at the same level are concurrent, and two that
disagree there are a genuine conflict rather than a race.
"""

from pathlib import Path

ANCHOR = """def descriptor_states_a_shape(descriptor: Any) -> bool:"""

HELPER = '''_DEPENDENCY_LEVEL_CACHE: dict[int, tuple[int, dict[int, int]]] = {}


def _dependency_levels(graph: Any) -> dict[int, int]:
    """ASAP level per node, over the condensation so cycles stay defined.

    This is `ILPScheduler.compute_asap_levels` -- longest path from the roots
    -- with strongly connected components collapsed first, because a retained
    loop is irreducible recursion rather than an invalid schedule and every
    member of one is at the same causal position.
    """

    key = id(graph.G)
    cached = _DEPENDENCY_LEVEL_CACHE.get(key)
    if cached is not None and cached[0] == graph.G.number_of_nodes():
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
    _DEPENDENCY_LEVEL_CACHE[key] = (graph.G.number_of_nodes(), levels)
    return levels


def _deepest_fact(page: Any, row: Any) -> Any:
    """The fact recorded at the deepest causal position for this row."""

    recorded = page.history(row)
    if not recorded:
        return None
    return max(recorded, key=lambda entry: int(entry[0]))[1]


''' + ANCHOR

OLD_READ = """        proven = page.latest(row)
        if proven is not None and proven[0] != "conflicting":"""

NEW_READ = """        proven = _deepest_fact(page, row)
        if proven is not None and proven[0] != "conflicting":"""

OLD_WRITE = """            if extents and not dynamic:
                dtype = str((answer or {}).get("dtype") or "float64")
                previous = page.latest(row)
                if previous is None:
                    page.set(row, 0, ("proven", extents, dtype))
                elif previous[0] == "proven" and tuple(previous[1]) != extents:
                    # Two derivations prove two different shapes for one
                    # identity.  Record it and stop answering from the page
                    # rather than let whichever asked first speak for both.
                    page.set(
                        row, len(page.history(row)),
                        ("conflicting", extents, dtype),
                    )"""

NEW_WRITE = """            if extents and not dynamic:
                dtype = str((answer or {}).get("dtype") or "float64")
                level = int(_dependency_levels(graph).get(int(node_id), 0))
                previous = _deepest_fact(page, row)
                recorded = dict(page.history(row))
                if previous is None:
                    page.set(row, level, ("proven", extents, dtype))
                elif tuple(previous[1]) == extents:
                    # The same answer from deeper in the program is still the
                    # same answer; record it at its own position so the page
                    # shows how far it has been confirmed.
                    page.set(row, level, ("proven", extents, dtype))
                elif int(level) > max(
                    int(column) for column in recorded
                ):
                    # Derived from strictly more of the program than the
                    # incumbent, so it supersedes rather than conflicts.
                    page.set(row, level, ("proven", extents, dtype))
                else:
                    # Same position, different answer: concurrent and
                    # genuinely in conflict.
                    page.set(row, level, ("conflicting", extents, dtype))"""

path = Path(__file__).resolve().parents[2] / "src/compiler/glsl_deployment_strategy.py"
text = path.read_text(encoding="utf-8")
for anchor, replacement in (
    (ANCHOR, HELPER),
    (OLD_READ, NEW_READ),
    (OLD_WRITE, NEW_WRITE),
):
    assert text.count(anchor) == 1, (text.count(anchor), anchor[:50])
    text = text.replace(anchor, replacement)
path.write_text(text, encoding="utf-8")
print("the concordance column is the dependency level")
