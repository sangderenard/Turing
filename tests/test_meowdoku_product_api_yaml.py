"""A Meowdoku wasm product carries its compiled-program API beside its JSON ABI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))


def test_entity_motion_product_api_matches_abi(tmp_path):
    from meowdoku import entity_motion as M

    fn, names, outputs, stem = next(p for p in M.PREDICTORS if p[3] == "climb")
    M._build_one(tmp_path, fn, names, outputs, stem)
    api = yaml.safe_load((tmp_path / f"{stem}.api.yaml").read_text(encoding="utf-8"))
    abi = json.loads((tmp_path / f"{stem}.abi.json").read_text(encoding="utf-8"))
    assert api["schema"] == "turing-compiled-program-api-v1"
    entry = api["entry_points"][0]
    parameters = entry["parameters"]
    inputs = [p["source_name"] for p in parameters if p["role"] in ("input", "inout")]
    results = [p["source_name"] for p in parameters if p["role"] in ("output", "inout")]
    assert inputs == abi["inputs"]
    # output names are not compared: the Function carries the lowering's names, the writer's authored ones are the .abi.json layer
    assert len(results) == len(abi["outputs"])
    assert [p["dtype"] for p in parameters if p["role"] in ("output", "inout")] == [
        p["dtype"] for p in abi["parameters"] if p["role"] == "output"
    ]


def test_fountain_wgsl_product_api_matches_wgsl_abi(tmp_path):
    from meowdoku import fountain_build as B

    smallest = min(B.SIZES, key=lambda tag: B.SIZES[tag][1])
    record = B.build_size(smallest, tmp_path, wasm=False)
    api = yaml.safe_load((tmp_path / f"fountain_{smallest}.api.yaml").read_text(encoding="utf-8"))
    abi = record["wgsl_abi"]
    meta = api["metadata"]
    assert api["schema"] == "turing-compiled-program-api-v1"
    assert [e["name"] for e in api["entry_points"]] == abi["entries"]
    # feeds and outputs are ids in the Function, names in the writer's layout: compared by slot, in order
    assert list(meta["feed_slots"].values()) == list(abi["feed_slots"].values())
    assert list(meta["output_slots"].values()) == list(abi["output_slots"].values())
    assert len(meta["output_span"]) == len(abi["outputs"])
    # the WGSL arena is one storage type (f32 under the "fast" contract): no per-value dtype is in the API to compare
