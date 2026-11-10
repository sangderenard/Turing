"""Export registration preserves the IDs included in immutable mesh receipts."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from meowdoku.tiles import SPECTRAL_ROOT
from meowdoku.web_export import register_material_catalog

LEGACY_NAMES = (
    "cat x_mark heart metal_gold metal_silver metal_copper metal_brass metal_bronze "
    "metal_steel metal_chrome metal_aluminum metal_titanium metal_iron metal_oxidized_copper "
    "marble_clear_glass marble_milk_glass marble_cat_eye_blue marble_cat_eye_green "
    "marble_cat_eye_amber stone_onyx stone_jade ceramic_porcelain black_felt stucco_plaster "
    "calibration_white calibration_red calibration_green calibration_blue calibration_yellow "
    "calibration_magenta calibration_cyan calibration_gray20"
).split()

pytestmark = pytest.mark.skipif(not SPECTRAL_ROOT.is_dir(), reason="spectral-analyzer checkout beside turing")


def test_export_material_ids_and_private_slots_are_stable():
    sys.path.insert(0, str(SPECTRAL_ROOT))
    from material_db import MaterialDatabase

    db = MaterialDatabase()
    index, extra = register_material_catalog(db)
    assert {name: index[name] for name in LEGACY_NAMES} == dict(zip(LEGACY_NAMES, range(32)))
    assert extra == {"coated": 32, "paintA": 33, "paintB": 34}
    assert {name: index[name] for name in ("white_gloss_paint", "white_plaster", "black_gloss_paint")} == {
        "white_gloss_paint": 35, "white_plaster": 36, "black_gloss_paint": 37}
    assert index["black_matte_paint"] == 38
    assert index["polished_silver"] == 39
    assert index["polished_gold"] == 40
    assert len(index) == 45
    assert [index[f"ribbon_{i+1}"] for i in range(7)] == list(range(41,48))
    assert not set(index.values()) & set(extra.values())
    assert not any(name.startswith("meowdoku_studio_") for name in index)
    tables = db.build_tensors()
    for name in ("pbr", "phong_compat", "enamel", "texture_stack"):
        table = np.asarray(tables[name])
        assert table.shape[0] == 48
        assert np.isfinite(table).all()
        np.testing.assert_array_equal(table[32], table[index["marble_clear_glass"]])
        np.testing.assert_array_equal(table[33], table[index["ceramic_porcelain"]])
        np.testing.assert_array_equal(table[34], table[index["ceramic_porcelain"]])


def test_local_legacy_object_digest_is_unchanged_when_available():
    from meowdoku.marble_objects import Store

    directory = ROOT / "build" / "meowdoku" / "marble_objects"
    legacy = dict(zip(LEGACY_NAMES, range(32)))
    for path in directory.glob("*.json"):
        raw = path.read_bytes()
        value = json.loads(raw)
        if value["materials"]["index"] != legacy:
            continue
        restored = Store(directory).get(path.stem)
        assert restored["digest"] == path.stem
        assert restored["materials"]["extra"] == {"coated": 32, "paintA": 33, "paintB": 34}
        assert path.read_bytes() == raw
        return
    pytest.skip("no local 32-name saved object fixture")
