"""The lowering library registers a product from its own declared API + ABI."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))


def test_scan_registers_entity_product_idempotently(tmp_path):
    from meowdoku import entity_motion as M
    from meowdoku import lowering_library as L

    fn, names, outputs, stem = next(p for p in M.PREDICTORS if p[3] == "climb")
    products = tmp_path / "products"
    M._build_one(products, fn, names, outputs, stem)
    stray = products / "stray"
    stray.mkdir()
    (stray / "hand.api.yaml").write_text(
        "schema: turing-compiled-program-api-v1\nmodule: hand\nlanguage: c\nentry: f\nentry_points: []\nmetadata: {}\n",
        encoding="utf-8")
    db = tmp_path / "accounts.db"
    L.scan(products, db)
    abi = json.loads((products / f"{stem}.abi.json").read_text(encoding="utf-8"))
    entity = [c for c in L.cores(db) if c["family"] == "entity"]
    assert len(entity) == 1 and entity[0]["source_digest"] == abi["source_digest"]
    row = L.lowerings(db, core_id=entity[0]["core_id"])[0]
    assert row["backend"] == "wasm" and row["verdict"] == "built"
    assert row["product_sha256"] == hashlib.sha256((products / f"{stem}.wasm").read_bytes()).hexdigest()
    stray_rows = [r for r in L.lowerings(db) if r["api_path"] == "stray/hand.api.yaml"]
    assert len(stray_rows) == 1 and stray_rows[0]["verdict"] == "described"
    before = (len(L.cores(db)), len(L.lowerings(db)))
    L.scan(products, db)
    assert (len(L.cores(db)), len(L.lowerings(db))) == before == (2, 2)
