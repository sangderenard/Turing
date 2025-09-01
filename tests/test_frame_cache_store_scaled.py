import numpy as np
from src.common.tensors.abstract_convolution.render_cache import FrameCache


def test_store_scaled_uses_preprocessed_frames():
    cache = FrameCache(store_scaled=True)
    frame = np.zeros((2, 3), dtype=np.uint8)
    cache.enqueue("a", frame)
    cache.process_queue()
    stored = cache.cache["a"][0]
    # add_vignette with default tile=8 expands dimensions by tile
    assert stored.shape == (16, 24)
    grid = cache.compose_layout([["a"]])
    # compose_layout should not apply vignette a second time
    assert grid.shape == stored.shape
