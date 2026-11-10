"""The studio catalogue uses the spectral MaterialDatabase's actual records."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from meowdoku.material_catalog import MATERIAL_CATALOG  # noqa: E402
from meowdoku.tiles import SPECTRAL_ROOT  # noqa: E402

pytestmark = pytest.mark.skipif(not SPECTRAL_ROOT.is_dir(), reason="spectral-analyzer checkout beside turing")


def test_catalogue_builds_full_phong_and_pbr_records():
    sys.path.insert(0, str(SPECTRAL_ROOT))
    from material_db import MaterialDatabase

    db = MaterialDatabase()
    ids = {name: db.register(name, data) for name, data in MATERIAL_CATALOG.items()}
    tables = db.build_tensors()
    pbr, phong = tables["pbr"], tables["phong_compat"]

    assert len(ids) == pbr.shape[0] == phong.shape[0]
    assert len(set(ids.values())) == len(ids)
    assert pbr.shape[1] == 16 and phong.shape[1] == 8
    for name, data in MATERIAL_CATALOG.items():
        i = ids[name]
        assert set(("albedo", "roughness", "metallic", "transmission", "ior", "opacity",
                    "emission", "ambient", "spec_strength", "shininess", "grain", "inner_color")) <= data.keys()
        np.testing.assert_allclose(pbr[i, :3], data["albedo"])
        np.testing.assert_allclose(pbr[i, 3:8], [data[k] for k in
            ("roughness", "metallic", "transmission", "ior", "opacity")])
        np.testing.assert_allclose(phong[i, :4], [data[k] for k in
            ("ambient", "spec_strength", "shininess", "grain")])
        np.testing.assert_allclose(phong[i, 4:7], np.asarray(data["inner_color"]) * 0.55)

    for name in ("metal_brass", "metal_copper", "metal_steel", "marble_clear_glass",
                 "marble_milk_glass", "marble_cat_eye_blue", "marble_cat_eye_green",
                 "marble_cat_eye_amber"):
        assert name in ids
    assert len({tuple(pbr[ids[n], :4]) for n in ids if n.startswith("metal_")}) > 8
    assert pbr[ids["marble_clear_glass"], 5] > pbr[ids["marble_milk_glass"], 5]


def test_new_glass_defaults_use_distinct_dielectric_materials():
    shell = MATERIAL_CATALOG["marble_clear_glass"]
    assert shell["ior"] == 1.52
    for name in ("marble_clear_glass", "marble_cat_eye_blue",
                 "marble_cat_eye_green", "marble_cat_eye_amber"):
        material = MATERIAL_CATALOG[name]
        assert material["transmission"] == (1.0 if name == "marble_clear_glass" else 0.0)
        assert material["opacity"] == (0.0 if name == "marble_clear_glass" else 1.0)  # User-requested opaque Legacy inners.
        assert material["metallic"] == 0.0
        if name != "marble_clear_glass":
            assert material["ior"] == 1.54
            assert material["ior"] != shell["ior"]
    # Keep the optional artistic milky material and all its saved identities.
    assert MATERIAL_CATALOG["marble_milk_glass"]["ior"] == 1.48


def test_studio_finishes_are_distinct_opaque_surface_materials():
    sys.path.insert(0, str(SPECTRAL_ROOT))
    from material_db import MaterialDatabase

    names = ("white_plaster", "white_gloss_paint", "black_felt", "black_gloss_paint")
    db = MaterialDatabase()
    ids = {name: db.register(name, MATERIAL_CATALOG[name]) for name in names}
    tensors = db.build_tensors()
    for name in names:
        authored = MATERIAL_CATALOG[name]
        row = tensors["pbr"][ids[name]]
        assert authored["metallic"] == authored["transmission"] == 0
        assert authored["opacity"] == 1
        assert tuple(authored["emission"]) == (0, 0, 0)
        np.testing.assert_allclose(row[4:8], [0, 0, 1.5, 1])
        assert tensors["enamel"][ids[name], 0] == 0, "Opaque paint is not an interference-film substitute"
    for white in ("white_plaster", "white_gloss_paint"):
        assert MATERIAL_CATALOG[white]["albedo"] == (0.90, 0.90, 0.90)
    assert MATERIAL_CATALOG["white_plaster"]["albedo"] != MATERIAL_CATALOG["stucco_plaster"]["albedo"]
    for matte, gloss in (("white_plaster", "white_gloss_paint"), ("black_felt", "black_gloss_paint")):
        assert MATERIAL_CATALOG[matte]["roughness"] > 0.9
        assert MATERIAL_CATALOG[matte]["spec_strength"] < 0.05
        assert MATERIAL_CATALOG[gloss]["roughness"] < 0.1
        assert MATERIAL_CATALOG[gloss]["spec_strength"] > 0.8
        assert MATERIAL_CATALOG[gloss]["shininess"] > 200


def test_black_matte_paint_is_distinct_opaque_dielectric():
    from meowdoku.material_catalog import MATERIAL_CATALOG
    paint = MATERIAL_CATALOG["black_matte_paint"]
    assert paint["roughness"] == .8
    assert paint["transmission"] == 0 and paint["opacity"] == 1
    assert paint["metallic"] == 0 and paint["ior"] == 1.5
    assert paint != MATERIAL_CATALOG["black_felt"]
    assert paint != MATERIAL_CATALOG["black_gloss_paint"]
