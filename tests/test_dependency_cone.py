"""Exact cone order/cause and bounded edge-work proofs; no long compiler run."""
from types import SimpleNamespace
import copy
import random

import networkx as nx
import pytest

from src.compiler import glsl_deployment_strategy as strategy
from src.compiler import identity_concordance as concordance


@pytest.mark.parametrize("size", (32, 64, 128, 256))
def test_reverse_descriptor_chain_is_not_rescanned_per_changed_value(monkeypatch, size):
    edge_visits = [0]

    class CountedParents(list):
        def __iter__(self):
            for edge in super().__iter__():
                edge_visits[0] += 1
                yield edge

    graph = nx.DiGraph()
    for node in reversed(range(size)):
        graph.add_node(node, parents=CountedParents([] if node == 0 else [(node - 1, "value")]),
                       tensor={"shape": (4,), "dtype": "float64"})
    graph.graph["function_name"] = "reverse_descriptor_chain"
    events = []
    monkeypatch.setattr(concordance, "invalidate_proven_shape", lambda *args: events.append(args))
    monkeypatch.setattr(concordance, "shape_scope_of", lambda _graph: "reverse_descriptor_chain")
    strategy._invalidate_tensor_descriptor_dependents(SimpleNamespace(G=graph), (0,), "changed")
    assert len(events) == size - 1
    assert all("tensor" not in graph.nodes[node] for node in range(1, size))
    assert graph.nodes[0]["tensor"]["shape"] == (4,)
    assert edge_visits[0] == 2 * (size - 1)


def test_fifo_cone_visits_shared_tails_cycles_and_opaque_ids_once():
    from src.compiler.dependency_cone import iter_dependency_cone

    root, left, right, tail = ("scope", 1), "left", 2, ("scope", 3)
    relation = {root: (left, right, left), left: (tail,), right: (tail,), tail: (root,)}
    reads = []

    def dependents(node):
        reads.append(node)
        return relation[node]

    assert tuple(iter_dependency_cone((root, root), dependents)) == (root, left, right, tail)
    assert reads == [root, left, right, tail]
    assert tuple(iter_dependency_cone((root,), relation.__getitem__, include_roots=False)) == (left, right, tail)
    assert tuple(iter_dependency_cone((), relation.__getitem__)) == ()


def test_ordered_cone_replays_incumbent_sweeps_not_breadth_first():
    from src.compiler.dependency_cone import iter_dependency_cone

    # Root is retired, but its consumers remain. z causes an earlier b next
    # sweep, while x's later y runs this sweep. IDs themselves aren't sorted.
    relation = {"retired": ("z", "x"), "x": ("y",), "z": ("b", "y"),
                "y": (), "b": ("x",)}
    assert tuple(iter_dependency_cone(("retired",), relation.__getitem__,
                 order=("x", "b", "z", "y"), include_roots=False)) == ("x", "z", "y", "b")
    with pytest.raises(ValueError, match="duplicate"):
        tuple(iter_dependency_cone((), relation.__getitem__, order=("x", "x")))
    with pytest.raises(KeyError):
        tuple(iter_dependency_cone(("retired",), relation.__getitem__, order=("x",)))


def test_parent_record_index_respects_roles_not_weaker_networkx_edges():
    from src.compiler.dependency_cone import parent_record_dependents

    graph = nx.DiGraph()
    graph.add_node("consumer", parents=[("outside", "value"), ("outside", "receiver"),
                                         ("definition", "callee"), ("method", "function")])
    graph.add_edge("unrelated", "consumer")
    assert parent_record_dependents(graph, excluded_roles={"callee", "func", "definition"}) == {
        "consumer": (), "unrelated": (), "outside": ("consumer",), "method": ("consumer",),
    }


def _original_scan(graph, sources, reason):
    """Frozen pre-cone scan order and first-cause semantics."""
    affected = set(map(int, sources))
    events = []
    changed = True
    while changed:
        changed = False
        for node_id, data in graph.nodes(data=True):
            node_id = int(node_id)
            if node_id in affected:
                continue
            parents = tuple(int(parent) for parent, role in data.get("parents") or ()
                            if str(role) not in {"callee", "func", "definition"})
            causes = tuple(parent for parent in parents if parent in affected)
            if not causes:
                continue
            affected.add(node_id)
            data.pop("tensor", None)
            events.append(("oracle", int(data.get("value_id", node_id)), causes[0], reason))
            changed = True
    return events


@pytest.mark.parametrize("seed", range(40))
def test_invalidation_matches_original_event_order_cause_and_nodes(seed, monkeypatch):
    rng = random.Random(seed)
    graph = nx.DiGraph()
    nodes = list(range(12))
    rng.shuffle(nodes)
    for node in nodes:
        graph.add_node(node, value_id=node + 1000,
                       tensor={"shape": (node,), "dtype": "float64"},
                       parents=[(rng.choice(nodes + [-1]), rng.choice(("value", "receiver", "callee", "func", "definition")))
                                for _ in range(rng.randrange(5))])
    # These deliberately disagree with the operand records. Neither closure
    # nor cause may be reconstructed from this weaker/irrelevant adjacency.
    graph.add_edges_from((nodes[index], nodes[index + 1]) for index in range(11))
    sources = [rng.choice(nodes), -1]
    original = copy.deepcopy(graph)
    expected = _original_scan(original, sources, "changed")
    events = []
    monkeypatch.setattr(concordance, "invalidate_proven_shape", lambda *args: events.append(args))
    monkeypatch.setattr(concordance, "shape_scope_of", lambda _graph: "oracle")
    strategy._invalidate_tensor_descriptor_dependents(SimpleNamespace(G=graph), sources, "changed")
    assert events == expected
    assert dict(graph.nodes(data=True)) == dict(original.nodes(data=True))


def test_fifo_reevaluation_matches_original_scan_per_descendant():
    from src.compiler.dependency_cone import iter_dependency_cone, parent_record_dependents

    graph = nx.DiGraph()
    for node, parents in ((5, (3,)), (1, (0,)), (3, (1, 2)), (2, (0,)), (0, (5,))):
        graph.add_node(node, parents=[(parent, "value") for parent in parents])
    def original_dependents(source):
        return tuple(node for node, data in graph.nodes(data=True)
                     if any(parent == source for parent, _role in data["parents"]))
    pending = list(original_dependents(0))
    seen, expected = {0}, []
    while pending:
        node = pending.pop(0)
        if node in seen:
            continue
        seen.add(node)
        expected.append(node)
        pending.extend(original_dependents(node))
    index = parent_record_dependents(graph)
    assert tuple(iter_dependency_cone((0,), index.__getitem__, include_roots=False)) == tuple(expected)
