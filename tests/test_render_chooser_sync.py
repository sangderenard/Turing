import numpy as np
from src.rendering.render_chooser import RenderChooser


def test_render_chooser_sync_drains_buffers(capfd):
    rc = RenderChooser(2, 2, mode="ascii")
    try:
        frame = np.zeros((2, 2, 1), dtype=np.uint8)
        rc.render({"image": frame})
        rc.sync()
        print("PROMPT")
    finally:
        rc.close()
    out = capfd.readouterr().out
    assert out.endswith("PROMPT\n")
