import networkx as nx
import pytest

from src.transmogrifier.cycle_unroller import (
    unroll_self_edges,
    unroll_all_cycles_once,
)
from src.transmogrifier.ilpscheduler import ILPScheduler


def test_unroll_self_edges_basic():
    g = nx.DiGraph()
    g.add_node("A", color="red")
    g.add_node("B")
    g.add_edge("A", "A", weight=1)
    g.add_edge("B", "A", weight=2)
    g.add_edge("A", "B", weight=3)

    unrolled = unroll_self_edges(g)

    assert set(unrolled.nodes) == {"A_v0", "A_v1", "B"}
    assert not unrolled.has_edge("A_v1", "A_v1")
    assert unrolled.has_edge("A_v0", "A_v1")
    assert unrolled.edges["A_v0", "A_v1"]["weight"] == 1
    assert unrolled.has_edge("B", "A_v0")
    assert unrolled.edges["B", "A_v0"]["weight"] == 2
    assert unrolled.has_edge("A_v1", "B")
    assert unrolled.edges["A_v1", "B"]["weight"] == 3
    assert unrolled.nodes["A_v0"]["color"] == "red"
    assert unrolled.nodes["A_v0"]["source"] == "A"
    assert unrolled.nodes["A_v1"]["version"] == 1


def test_unroll_all_cycles_once_multinode():
    g = nx.DiGraph()
    g.add_edge("X", "A")  # incoming from outside
    g.add_edge("A", "B")
    g.add_edge("B", "C")
    g.add_edge("C", "A")
    g.add_edge("C", "Y")  # outgoing to outside

    unroll_all_cycles_once(g)

    expected_nodes = {
        "A_v0",
        "A_v1",
        "B_v0",
        "B_v1",
        "C_v0",
        "C_v1",
        "X",
        "Y",
    }
    assert set(g.nodes) == expected_nodes
    assert g.has_edge("X", "A_v0")
    assert g.has_edge("A_v0", "B_v1")
    assert g.has_edge("B_v0", "C_v1")
    assert g.has_edge("C_v0", "A_v1")
    assert g.has_edge("C_v1", "Y")
    # Stitch edges
    for n in ["A", "B", "C"]:
        assert g.has_edge(f"{n}_v0", f"{n}_v1")
    assert nx.is_directed_acyclic_graph(g)


def test_compute_asap_levels_cycle_detection():
    g = nx.DiGraph()
    g.add_node("A", parents=[("B", None)])
    g.add_node("B", parents=[("A", None)])
    g.add_edge("A", "B")
    g.add_edge("B", "A")

    class Dummy:
        def __init__(self, G):
            self.G = G

    scheduler = ILPScheduler(Dummy(g))
    with pytest.raises(ValueError):
        scheduler.compute_asap_levels()
