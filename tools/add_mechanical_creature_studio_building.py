"""Append a small Studio startup descriptor without reserializing the game model.

The caller supplies the definition produced from the captured Studio packet by
prepareMechanicalCreatureStudioBuilding. This does not publish, restart, compile,
or modify the original playable page. Existing generator remains untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "examples/meowdoku/web/mechanical_creature_studio_building.js"
SEAM = 'const model = JSON.parse(document.getElementById("abstract-ui-model").textContent);'


def patch_studio_building(source: bytes, definition: dict) -> tuple[bytes, dict]:
    if definition.get("schema") != "mechanical-creature-studio-building-v1":
        raise ValueError("expected captured Studio building definition")
    document = source.decode("utf-8")
    if document.count(SEAM) != 1:
        raise ValueError("original model initialization seam is not unique")
    helper = HELPER.read_bytes().decode("utf-8")
    payload = json.dumps(definition, ensure_ascii=True, separators=(",", ":")).replace("<", "\\u003c")
    insertion = ("\n" + helper + "\nconst mechanicalCreatureStudioBuildingDefinition = " + payload + ";\n"
                 "appendMechanicalCreatureStudioBuilding(model, mechanicalCreatureStudioBuildingDefinition);\n"
                 "globalThis.mechanicalCreatureStudioBuildingDefinition = mechanicalCreatureStudioBuildingDefinition;\n"
                 "globalThis.mechanicalCreatureStudioBuilding = Object.freeze({"
                 "assetIdentity: mechanicalCreatureStudioBuildingDefinition.assetIdentity,"
                 "placement: Object.freeze([...mechanicalCreatureStudioBuildingDefinition.placement]),"
                 "entrance: Object.freeze({...mechanicalCreatureStudioBuildingDefinition.renderEntrance}),"
                 "proxyIds: Object.freeze([...mechanicalCreatureStudioBuildingDefinition.proxyIdentities])});\n")
    if "</script" in insertion.lower():
        raise ValueError("unsafe script terminator in building helper")
    output = document.replace(SEAM, SEAM + insertion, 1)
    if output.replace(SEAM + insertion, SEAM, 1) != document:
        raise AssertionError("building append did not preserve the source exactly")
    result = output.encode("utf-8")
    return result, {"schema": "mechanical-creature-studio-building-patch-v1",
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "output_sha256": hashlib.sha256(result).hexdigest(),
        "original_page_recovered_exactly": True, "embedded_model_worker_unchanged": True,
        "definition_identity": definition["identity"], "proxy_count": len(definition["boxes"]),
        "compile_performed": False, "published": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("definition", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    protected = {args.source.resolve(), (ROOT / "build/meowdoku/web/mechanical_creature_play.html").resolve(),
                 (ROOT / "build/meowdoku/web/mechanical_creature_studio.html").resolve()}
    if args.output.resolve() in protected:
        parser.error("refusing to overwrite source or existing playable pages")
    output, receipt = patch_studio_building(args.source.read_bytes(), json.loads(args.definition.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(output)
    print(json.dumps({"output": str(args.output), **receipt}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
