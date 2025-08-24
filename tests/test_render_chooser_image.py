import numpy as np
import os
import pytest
from src.rendering.render_chooser import RenderChooser


@pytest.mark.skipif(os.name != "nt", reason="fast console requires Windows")
def test_render_chooser_accepts_image():
    chooser = RenderChooser(8, 4, mode="ascii")
    try:
        frame = np.zeros((4, 8), dtype=np.uint8)
        frame[1, 1] = 255
        chooser._render_ascii({"image": frame})
        q = chooser._ascii_queue
        assert q is not None
        out = q.get(timeout=1)
        assert out.strip() != ""
    finally:
        chooser.close()
