"""Meowdoku tiles through spectral-analyzer's glyph maker.

The scene-order package validates and compiles to geometry (fonts resolved,
outlines extruded, materials bound) without the renderer; the compositor
lays background-independent sprites over flat region tiles.  Rendering is
exercised only when the native kernels are built.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from meowdoku import tiles  # noqa: E402

pytestmark = pytest.mark.skipif(
    not tiles.SPECTRAL_ROOT.is_dir(), reason="spectral-analyzer checkout beside turing",
)


def test_the_tile_package_is_a_valid_scene_order_in_the_glyph_fonts():
    package = tiles.tile_order_package()
    jobs = {job["id"]: job for job in package["jobs"]}
    assert set(jobs) == {"meowdoku_cat", "meowdoku_x_mark", "meowdoku_heart"}
    assert jobs["meowdoku_cat"]["token"] == tiles.CAT
    assert jobs["meowdoku_cat"]["font"]["family"] == "Segoe UI Symbol"
    assert jobs["meowdoku_cat"]["geometry"]["material"] == "cat_orange"
    assert jobs["meowdoku_x_mark"]["geometry"]["material"] == "mark_slate"
    assert "cat_orange" in package["defaults"]["materials"]
    assert package["defaults"]["image"] == {"width": 128, "height": 128}


def test_the_tiles_compile_to_glyph_geometry_without_the_renderer(tmp_path):
    package_path = tiles.write_tile_package(tmp_path / "meowdoku_tiles.json")
    manifests = tiles.compile_tiles(package_path, tmp_path / "compiled")
    assert set(manifests) == {"meowdoku_cat", "meowdoku_x_mark", "meowdoku_heart"}
    for name, manifest in manifests.items():
        compiled = manifest["compile"]
        assert compiled["glyph_triangles"] > 0, name
    # The cat is a real outline (465 font vertices), not a missing-glyph box.
    assert manifests["meowdoku_cat"]["compile"]["glyph_triangles"] > 500
    assert manifests["meowdoku_cat"]["compile"]["glyph_triangles"] > manifests["meowdoku_x_mark"]["compile"]["glyph_triangles"]
    assert (tmp_path / "compiled" / "meowdoku_cat").is_dir()


def _synthetic_sprite(role, size=32, colour=(1.0, 0.5, 0.0)):
    yy, xx = np.mgrid[0:size, 0:size]
    alpha = (((yy - size / 2) ** 2 + (xx - size / 2) ** 2) < (size * 0.3) ** 2).astype(np.float64)
    premultiplied = alpha[..., None] * np.asarray(colour)
    additive = np.zeros((size, size, 3))
    return tiles.TileSprite(role, premultiplied, alpha, additive)


def test_the_compositor_lays_sprites_over_region_tiles_and_hearts():
    n = 5
    region = np.repeat(np.arange(n), n)
    cat = np.zeros(n * n); cat[2 * n + 2] = 1.0
    x_mark = np.zeros(n * n); x_mark[0] = 1.0
    sprites = {"cat": _synthetic_sprite("cat"), "x_mark": _synthetic_sprite("x_mark", colour=(0.3, 0.3, 0.3)),
               "heart": _synthetic_sprite("heart", colour=(0.8, 0.0, 0.0))}
    image = tiles.compose_board(cat, x_mark, region, hearts=2, sprites=sprites, cell_px=40)
    assert image.shape == (6 * 40, 5 * 40, 3)
    palette = tiles.region_palette(n)
    np.testing.assert_allclose(image[20, 3 * 40 + 20], palette[0], atol=1e-9)   # empty tile, region 0
    np.testing.assert_allclose(image[2 * 40 + 20, 2 * 40 + 20], [1.0, 0.5, 0.0])  # the cat sprite body
    np.testing.assert_allclose(image[20, 20], [0.3, 0.3, 0.3])                    # the X sprite body
    np.testing.assert_allclose(image[40, 100], 0.12)                              # a grid line
    np.testing.assert_allclose(image[5 * 40 + 20, 20], [0.8, 0.0, 0.0])           # heart 1 of 2
    assert image[5 * 40 + 20, 2 * 40 + 20].max() < 0.2                            # the lost third heart


@pytest.mark.skipif(not tiles.native_kernels_available(), reason="native kernels not built")
def test_rendering_produces_sprites(tmp_path):
    package_path = tiles.write_tile_package(tmp_path / "meowdoku_tiles.json", roles=("x_mark",))
    tiles.render_tiles(package_path, tmp_path / "rendered", jobs=("meowdoku_x_mark",))
    sprites = tiles.load_sprites(tmp_path / "rendered", tiles.tile_order_package(roles=("x_mark",)))
    assert "x_mark" in sprites and sprites["x_mark"].alpha.shape == (128, 128)
