from src.common.tensors.autoautograd.whiteboard_cache import WhiteboardCache
from src.common.tensors.autoautograd.whiteboard_runtime import run_op_and_grads_cached

class DummyNode:
    def __init__(self, sphere, version=0):
        self.sphere = sphere
        self.version = version

class DummySys:
    def __init__(self, nodes):
        self.nodes = nodes
    def impulse(self, *args, **kwargs):
        pass

def test_cache_hit_and_miss():
    nodes = {0: DummyNode(1.0), 1: DummyNode(2.0)}
    sys = DummySys(nodes)
    cache = WhiteboardCache()
    y1, g1, _ = run_op_and_grads_cached(sys, 'add', [0, 1], cache=cache)
    assert cache.misses == 1
    assert g1 == (1.0, 1.0)
    y2, g2, _ = run_op_and_grads_cached(sys, 'add', [0, 1], cache=cache)
    assert cache.hits == 1
    assert (y1, g1) == (y2, g2)
    nodes[0].version += 1
    y3, _, _ = run_op_and_grads_cached(sys, 'add', [0, 1], cache=cache)
    assert cache.misses == 2
    assert y3 == y1


def test_quantised_scale_and_residual():
    nodes = {0: DummyNode(1.0), 1: DummyNode(2.0)}
    sys = DummySys(nodes)
    cache = WhiteboardCache()
    run_op_and_grads_cached(sys, 'add', [0, 1], scale=1.0, residual=0.1, cache=cache)
    assert cache.misses == 1
    run_op_and_grads_cached(sys, 'add', [0, 1], scale=1.0, residual=0.1, cache=cache)
    assert cache.hits == 1
    run_op_and_grads_cached(sys, 'add', [0, 1], scale=1.0 + 2e-6, residual=0.1, cache=cache)
    assert cache.misses == 2
    run_op_and_grads_cached(sys, 'add', [0, 1], scale=1.0, residual=0.1 + 2e-6, cache=cache)
    assert cache.misses == 3


def test_gradient_alignment_mul():
    nodes = {0: DummyNode(3.0), 1: DummyNode(5.0)}
    sys = DummySys(nodes)
    cache = WhiteboardCache()
    y, grads, _ = run_op_and_grads_cached(sys, 'mul', [0, 1], cache=cache)
    assert y == 15.0
    assert grads == (5.0, 3.0)
