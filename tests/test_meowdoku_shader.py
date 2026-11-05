"""The Meowdoku board shader: compiled from the authored per-pixel law by the
existing targets, and compared against the Python version of that law.

* the capture lowers to a complete WebGL fragment module and a complete GLSL
  compute artifact from one program;
* the Python version (numpy backend) draws what the law says: an orange cat
  on a cat cell, the region colour on an empty cell, dark grid lines;
* headless Chrome draws the compiled fragment shader over the Python game's
  board state and matches the Python version within 8-bit quantization.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from meowdoku.page import emit_board_page  # noqa: E402
from meowdoku.session import MeowdokuSession  # noqa: E402
from meowdoku.shader import (  # noqa: E402
    FEED_NAMES, board_feeds, compile_shader, render_reference, session_feeds,
)

N, SIZE = 5, 100


def _chrome():
    for candidate in (
        shutil.which("chrome"), shutil.which("google-chrome"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.join(os.environ.get("LOCALAPPDATA", ""), r"Google\Chrome\Application\chrome.exe"),
    ):
        if candidate and Path(candidate).exists():
            return candidate
    return None


@pytest.fixture(scope="module")
def shader():
    compiled = compile_shader()
    assert compiled.webgl.complete, [item.format() for item in compiled.webgl.shortfalls]
    assert compiled.glsl.complete, compiled.glsl.shortfall_report()
    return compiled


def _state():
    columns = {
        "cat": np.zeros(N * N), "x_mark": np.zeros(N * N),
        "region": np.repeat(np.arange(N, dtype=np.float64), N),   # rows are regions
    }
    columns["cat"][2 * N + 2] = 1.0      # a cat in the middle cell
    columns["x_mark"][0] = 1.0           # an X in the corner
    return columns


def test_both_targets_emit_from_the_one_captured_program(shader):
    assert set(shader.feed_ids) == set(FEED_NAMES)
    assert shader.webgl.source.startswith("#version 300 es")
    assert shader.webgl.api.metadata["output_layout"] == "rgba"
    assert shader.webgl.api.metadata["input_sampling"] == "normalized"
    assert shader.webgl.source.count("uniform sampler2D") == len(FEED_NAMES)
    assert shader.glsl.target == "glsl" and "void main" in shader.glsl.source


def test_the_python_version_draws_the_board():
    feeds = board_feeds(_state(), N, 3.0, SIZE, SIZE)
    image = render_reference(feeds)
    assert image.shape == (SIZE, SIZE, 3)
    cell = SIZE // N
    middle = image[2 * cell + cell // 2, 2 * cell + cell // 2]
    np.testing.assert_allclose(middle, [0.96, 0.55, 0.16], atol=1e-9)   # the cat
    empty = image[cell // 2, 3 * cell + cell // 2]                        # row 0, col 3
    region_tone = 0.62 + 0.30 * np.sin(np.array([0.0, 2.0943951, 4.1887902]))
    np.testing.assert_allclose(empty, region_tone, atol=1e-9)
    line = image[cell, cell // 2]                                          # a horizontal grid line
    np.testing.assert_allclose(line, [0.12, 0.12, 0.14], atol=1e-9)
    corner = image[cell // 2, cell // 2]                                   # the X's centre
    np.testing.assert_allclose(corner, [0.30, 0.30, 0.32], atol=1e-9)
    dead = render_reference(board_feeds(_state(), N, 0.0, SIZE, SIZE))
    assert np.all(dead[cell // 2, 3 * cell + cell // 2] < empty)          # no hearts: dimmed


def test_session_feeds_come_from_the_python_game():
    session = MeowdokuSession(n=N, seed=0)
    feeds = session_feeds(session, SIZE, SIZE)
    assert feeds["cat"].shape == (N, N) and feeds["hearts"].item() == 3.0
    assert set(np.unique(feeds["region"])) == set(range(N))


@pytest.mark.skipif(_chrome() is None, reason="headless Chrome draws the shader")
def test_headless_chrome_matches_the_python_version(shader, tmp_path):
    session = MeowdokuSession(n=N, seed=0)
    feeds = session_feeds(session, SIZE, SIZE)
    page = tmp_path / "board.html"
    page.write_text(emit_board_page(shader, feeds, width=SIZE, height=SIZE), encoding="utf-8")
    completed = subprocess.run([
        _chrome(), "--headless=new", "--no-first-run", "--no-default-browser-check",
        "--disable-background-networking", "--enable-unsafe-swiftshader",
        "--use-angle=swiftshader", "--virtual-time-budget=10000",
        f"--user-data-dir={tmp_path / 'profile'}", "--dump-dom", page.resolve().as_uri(),
    ], capture_output=True, text=True, timeout=180)
    match = re.search(r'<pre id="readback">(.*?)</pre>', completed.stdout, re.S)
    assert match, completed.stdout[-800:] + completed.stderr[-800:]
    readback = json.loads(match.group(1))
    assert "error" not in readback, readback
    drawn = np.asarray(readback["rows"], dtype=np.float64).reshape(SIZE, SIZE, 3) / 255.0
    reference = render_reference(feeds)
    difference = np.abs(drawn - reference)
    # 8-bit quantization of the canvas is half a step; a pixel on a feature
    # edge may fall on either side of a comparison in float32, so the bound
    # is on the bulk, with a small allowance for edge pixels.
    assert np.mean(difference) < 0.01, float(np.mean(difference))
    assert np.mean(difference.max(axis=-1) > 0.02) < 0.02
