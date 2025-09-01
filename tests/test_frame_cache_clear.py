import numpy as np
from src.common.tensors.abstract_convolution.render_cache import FrameCache


def test_clear_preserves_composite_cache():
    cache = FrameCache()
    frame = np.zeros((2, 3), dtype=np.uint8)
    cache.enqueue("a", frame)
    cache.process_queue()
    grid = cache.compose_layout([["a"]])
    cache.clear()
    cached = cache.compose_layout([["a"]])
    assert np.array_equal(cached, grid)
