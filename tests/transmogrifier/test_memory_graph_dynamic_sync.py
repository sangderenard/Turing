from transmogrifier.graph.memory_graph import BitTensorMemoryGraph


def test_dynamic_flag_syncs_with_memory():
    g = BitTensorMemoryGraph(1024, dynamic=False)
    assert g.dynamic is False
    assert g.hard_memory.dynamic is False
    g.dynamic = True
    assert g.dynamic is True
    assert g.hard_memory.dynamic is True
