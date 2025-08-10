import ctypes
from transmogrifier.graph.memory_graph import BitTensorMemoryGraph, NodeEntry
import pytest


@pytest.mark.xfail(reason="Memory graph allocation issue")
def test_add_and_retrieve_node():
    g = BitTensorMemoryGraph(1024)
    node_id = g.add_node(b'data', node_id=123)
    assert node_id == 123
    node = g.get_node(123)
    assert isinstance(node, NodeEntry)
    assert node.node_id == 123
    assert b'data' in bytes(node.node_data)
