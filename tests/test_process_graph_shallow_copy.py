"""``copy.copy(ProcessGraph)`` is a shallow live copy: no binding round trip."""

import copy
import pickle
import time

import networkx as nx

from src.transmogrifier.graph.graph_express2 import ProcessGraph


class _Heavy:
    """A binding whose pickling is observable (a stand-in for a law piece)."""

    dumps = 0

    def __init__(self):
        self.payload = bytes(2_000_000)

    def __reduce__(self):
        type(self).dumps += 1
        return (_Heavy, ())


def _graph():
    graph = ProcessGraph(materialize_memory=False)
    graph.python_bindings = {"piece": _Heavy()}
    graph.G.add_edge(1, 2)
    return graph


def test_copy_shares_bindings_and_does_not_serialize_them():
    graph = _graph()
    before = _Heavy.dumps
    clone = copy.copy(graph)
    assert _Heavy.dumps == before
    assert clone.python_bindings is not graph.python_bindings
    assert clone.python_bindings["piece"] is graph.python_bindings["piece"]


def test_copy_is_independent_and_live():
    graph = _graph()
    clone = copy.copy(graph)
    clone.G = nx.DiGraph()
    clone.G.add_node(9)
    assert set(graph.G.nodes) == {1, 2}
    assert clone._graph_lock is not graph._graph_lock
    assert clone._graph_accessor is not graph._graph_accessor
    assert clone._graph_subscribers == [] and clone._graph_progress is None
    with clone._graph_lock:
        pass


def test_copy_is_not_quadratic_in_bindings():
    graph = _graph()
    start = time.perf_counter()
    for _ in range(50):
        copy.copy(graph)
    assert time.perf_counter() - start < 2.0


def test_checkpoint_pickle_path_is_unchanged():
    graph = _graph()
    restored = pickle.loads(pickle.dumps(graph))
    assert set(restored.G.edges) == {(1, 2)}
    assert isinstance(restored.python_bindings["piece"], _Heavy)
