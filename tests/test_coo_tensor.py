
import pytest
import numpy as np
from src.common.tensors.abstract_graph_core import AbstractGraphCore
from src.common.tensors.abstraction import AbstractTensor


def test_coo_matrix_basic():
    # COO for a 3x3 matrix with 2 nonzero entries
    edge_index = AbstractTensor.get_tensor([[0, 1], [1, 2]])  # shape (2, 2)
    edge_weight = AbstractTensor.get_tensor([10, 20])         # shape (2,)
    shape = (3, 3)
    g = AbstractGraphCore(edge_index, edge_weight, shape)
    coo = g.coo
    assert coo.shape == (3, 3)
    assert np.all(coo.indices.data == np.array([[0, 1], [1, 2]]))
    assert np.all(coo.values.data == np.array([10, 20]))
    # Test to_dense
    dense = coo.to_dense()
    assert dense[0, 1] == 10
    assert dense[1, 2] == 20
    assert dense[0, 0] == 0
    # Test update and propagation
    coo.update(edge_weight=AbstractTensor.get_tensor([100, 200]))
    assert np.all(coo.values.data == np.array([100, 200]))
    # on_coo_update should update g.coo
    assert g.coo is coo
    # Test graph structure after COO update
    G = g.network
    # Should have index nodes (0,1), (1,2) and value nodes ("val", 10), ("val", 20)
    assert (0, 1) in G.nodes
    assert (1, 2) in G.nodes
    assert ("val", 10.0) in G.nodes
    assert ("val", 20.0) in G.nodes
    # Edges from index to value
    assert G.has_edge((0, 1), ("val", 10.0))
    assert G.has_edge((1, 2), ("val", 20.0))

def test_on_coo_update_syncs():
    edge_index = AbstractTensor.get_tensor([[0, 1], [1, 2]])
    edge_weight = AbstractTensor.get_tensor([1, 2])
    g = AbstractGraphCore(edge_index, edge_weight, (3, 3))
    coo = g.coo
    # Simulate external update
    new_idx = AbstractTensor.get_tensor([[0, 2], [2, 1]])
    new_val = AbstractTensor.get_tensor([5, 6])
    coo.update(edge_index=new_idx, edge_weight=new_val)
    # g.coo should still be the same object, but with updated data
    assert np.all(g.coo.indices.data == np.array([[0, 2], [2, 1]]))
    assert np.all(g.coo.values.data == np.array([5, 6]))
    # Graph should also be updated
    G = g.network
    assert (0, 2) in G.nodes
    assert (2, 1) in G.nodes
    assert ("val", 5.0) in G.nodes
    assert ("val", 6.0) in G.nodes
    assert G.has_edge((0, 2), ("val", 5.0))
    assert G.has_edge((2, 1), ("val", 6.0))

def test_on_graph_update_roundtrip():
    # Start with COO, update graph, sync back to COO
    edge_index = AbstractTensor.get_tensor([[0, 1], [1, 2]])
    edge_weight = AbstractTensor.get_tensor([10, 20])
    g = AbstractGraphCore(edge_index, edge_weight, (3, 3))
    G = g.network
    # Remove an edge and add a new one
    G.remove_edge((0, 1), ("val", 10.0))
    G.add_node((2, 0), type="index")
    G.add_node(("val", 99.0), type="value")
    G.add_edge((2, 0), ("val", 99.0))
    # Sync graph -> COO
    g.on_graph_update()
    # COO should now reflect the new edge set
    idx = g.coo.indices.data
    vals = g.coo.values.data
    assert (2, 0) in [tuple(idx[:, i]) for i in range(idx.shape[1])]
    assert 99.0 in vals
    assert not any((tuple(idx[:, i]) == (0, 1) and vals[i] == 10.0) for i in range(idx.shape[1]))

