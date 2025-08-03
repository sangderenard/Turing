from turing_provenance import ProvenanceGraph, ProvNode, ProvEdge
from ssa_builder import graph_to_ssa_with_loops


def _make_node(idx, op):
    return ProvNode(idx, op, (), {}, idx)


def test_simple_counter_phi():
    pg = ProvenanceGraph()
    pg._nodes = [
        ProvNode(0, 'init', (), {}, 0),
        ProvNode(1, 'cmp', (), {}, 1),
        ProvNode(2, 'inc', (), {}, 2),
    ]
    pg._edges = [
        ProvEdge(0,1,0),
        ProvEdge(1,2,0),
        ProvEdge(2,1,0),
    ]
    ssa = graph_to_ssa_with_loops(pg)
    assert '%n1.phi0' in ssa
    assert 'phi' in ssa


def test_rotate_until_pattern_phi():
    pg = ProvenanceGraph()
    pg._nodes = [
        ProvNode(0, 'start', (), {}, 0),
        ProvNode(1, 'check', (), {}, 1),
        ProvNode(2, 'rot', (), {}, 2),
    ]
    pg._edges = [
        ProvEdge(0,1,0),
        ProvEdge(1,2,0),
        ProvEdge(2,1,0),
    ]
    ssa = graph_to_ssa_with_loops(pg)
    assert '%n1.phi0' in ssa


def test_two_loops_phis():
    pg = ProvenanceGraph()
    pg._nodes = [
        ProvNode(0, 'i0', (), {}, 0),
        ProvNode(1, 'outer_chk', (), {}, 1),
        ProvNode(2, 'j0', (), {}, 2),
        ProvNode(3, 'inner_chk', (), {}, 3),
        ProvNode(4, 'inner_inc', (), {}, 4),
        ProvNode(5, 'outer_inc', (), {}, 5),
    ]
    pg._edges = [
        ProvEdge(0,1,0),
        ProvEdge(1,5,0),
        ProvEdge(5,1,0),
        ProvEdge(1,3,0),
        ProvEdge(2,3,0),
        ProvEdge(3,4,0),
        ProvEdge(4,3,0),
    ]
    ssa = graph_to_ssa_with_loops(pg)
    assert '%n1.phi0' in ssa
    assert '%n3.phi0' in ssa
