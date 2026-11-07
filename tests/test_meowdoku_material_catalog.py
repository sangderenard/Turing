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
