import numpy as np
from src.common.tensors.abstract_convolution.render_cache import add_vignette


def test_add_vignette_repeated_columns_match():
    frame = np.full((2, 3), 255, dtype=np.uint8)
    out = add_vignette(frame, tile=8)
    tile = 8
    for offset in range(tile):
        cols = out[:, offset::tile]
        assert np.all(cols == cols[:, [0]])

