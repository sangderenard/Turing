from io import BytesIO
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from meowdoku.marble_calibration_qr import register_calibration_qr  # noqa: E402


def test_registers_known_module_square_in_png():
    rng = np.random.default_rng(72)
    grid = rng.integers(0, 2, (33, 33), dtype=np.uint8)
    grid[:4, :] = grid[-4:, :] = 0
    grid[:, :4] = grid[:, -4:] = 0
    rows = ["".join(map(str, row)) for row in grid]
    image = np.full((512, 512), 216, dtype=np.uint8)
    image[30:154, 310:434] = np.asarray(
        Image.fromarray((1 - grid) * 255).resize((124, 124)))
    out = BytesIO()
    Image.fromarray(image).save(out, format="PNG")

    result = register_calibration_qr(out.getvalue(), rows)
    assert result["corners"] == [
        {"x": 310, "y": 30}, {"x": 434, "y": 30},
        {"x": 434, "y": 154}, {"x": 310, "y": 154}]
    assert result["score"] > .9


def test_registers_full_card_nir_pattern():
    rng = np.random.default_rng(11)
    grid = rng.integers(0, 2, (33, 33), dtype=np.uint8)
    grid[:4, :] = grid[-4:, :] = 0
    grid[:, :4] = grid[:, -4:] = 0
    rows = ["".join(map(str, row)) for row in grid]
    image = np.zeros((512, 512), dtype=np.uint8)
    image[36:456, 43:463] = np.asarray(
        Image.fromarray(grid * 255).resize((420, 420), Image.Resampling.BILINEAR))
    out = BytesIO()
    Image.fromarray(image).save(out, format="PNG")

    result = register_calibration_qr(out.getvalue(), rows, full_card=True)
    assert result["method"] == "authored-qr-projective-v1"
    assert result["score"] > .8
    assert abs(result["corners"][0]["x"] - 43) < 3
    assert abs(result["corners"][0]["y"] - 36) < 3
