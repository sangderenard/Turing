import numpy as np
from src.common.tensors.abstract_convolution.render_cache import FrameCache


def test_compose_group_enforces_tile_size():
    cache = FrameCache()
    cache.enqueue("param0_grad", np.zeros((2, 3), dtype=np.uint8))
    cache.enqueue("param1_grad", np.zeros((3, 2), dtype=np.uint8))
    cache.process_queue()
    assert cache.tile_height == 16 and cache.tile_width == 24
    grid = cache.compose_group("grads")
    assert grid.shape == (16, 48)
