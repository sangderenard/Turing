import networkx as nx

from src.common.tensors.graph_translator import GraphTranslator
from src.transmogrifier.ilpscheduler import ILPScheduler


class CountingScheduler(ILPScheduler):
    """ILPScheduler subclass that counts compute_levels invocations."""

    calls = 0

    def compute_levels(self, method, order):  # type: ignore[override]
        type(self).calls += 1
        return super().compute_levels(method, order)


def make_forward(trace):
    g = nx.DiGraph()
    g.add_node("a", op=lambda: trace.append("a"))
    g.add_node("b", op=lambda: trace.append("b"))
    g.add_edge("a", "b")
    return g


def make_backward(trace):
    g = nx.DiGraph()
    g.add_node("b", op=lambda: trace.append("b"))
    g.add_node("a", op=lambda: trace.append("a"))
    g.add_edge("b", "a")
    return g


def test_ilps_scheduler_interop():
    # Forward graph scheduling and execution
    f_trace: list[str] = []
    f_graph = make_forward(f_trace)
    f_trans = GraphTranslator(f_graph)

    CountingScheduler.calls = 0
    f_trans.schedule(CountingScheduler)
    f_trans.execute(CountingScheduler)
    f_trans.execute(CountingScheduler)
    assert f_trace == ["a", "b", "a", "b"]
    assert CountingScheduler.calls == 1
    assert {f_graph.nodes[n]["layer"] for n in f_graph} == {0, 1}

    # Backward graph scheduling and execution
    b_trace: list[str] = []
    b_graph = make_backward(b_trace)
    b_trans = GraphTranslator(b_graph)

    CountingScheduler.calls = 0
    b_trans.schedule(CountingScheduler)
    b_trans.execute(CountingScheduler)
    b_trans.execute(CountingScheduler)
    assert b_trace == ["b", "a", "b", "a"]
    assert CountingScheduler.calls == 1
    assert {b_graph.nodes[n]["layer"] for n in b_graph} == {0, 1}
