import numpy as np
import os
import pytest
from src.rendering.render_chooser import RenderChooser


@pytest.mark.skipif(os.name != "nt", reason="fast console requires Windows")
def test_render_chooser_accepts_image(monkeypatch):
    chooser = RenderChooser(8, 4, mode="ascii")
    try:
        printer = chooser._ascii_printer
        assert printer is not None
        rendered = []
        monkeypatch.setattr(printer, "enqueue", rendered.append)
        monkeypatch.setattr(
            chooser.renderer,
            "to_ascii_diff",
            lambda **_kwargs: "rendered image",
        )
        frame = np.zeros((4, 8), dtype=np.uint8)
        frame[1, 1] = 255
        chooser._render_ascii({"image": frame})
        assert len(rendered) == 1
        assert rendered[0].strip() != ""
    finally:
        chooser.close()
