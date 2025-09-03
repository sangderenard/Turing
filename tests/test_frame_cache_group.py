import numpy as np
from src.common.tensors.abstract_convolution.render_cache import FrameCache, add_vignette


def test_compose_group_resizes_tiles_to_common_size():
    cache = FrameCache()
    cache.enqueue("param0_grad", np.zeros((2, 3), dtype=np.uint8))
    cache.enqueue("param1_grad", np.full((3, 2), 255, dtype=np.uint8))
    cache.process_queue()
    grid = cache.compose_group("grads")
    # Stored grid remains at original resolution
    assert grid.shape == (3, 6)
    # Second tile should be fully populated after resizing
    assert np.all(grid[:, 3:] == 255)
    # Upscaling is deferred until rendering
    upscaled = add_vignette(grid)
    assert upscaled.shape == (24, 48)
