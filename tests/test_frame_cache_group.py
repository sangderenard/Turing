import numpy as np
from src.common.tensors.abstract_convolution.render_cache import FrameCache, add_vignette


def test_compose_group_pads_tiles_to_common_size():
    cache = FrameCache()
    cache.enqueue("param0_grad", np.zeros((2, 3), dtype=np.uint8))
    cache.enqueue("param1_grad", np.zeros((3, 2), dtype=np.uint8))
    cache.process_queue()
    grid = cache.compose_group("grads")
    # Stored grid remains at original resolution
    assert grid.shape == (3, 6)
    # Upscaling is deferred until rendering
    upscaled = add_vignette(grid)
    assert upscaled.shape == (24, 48)
