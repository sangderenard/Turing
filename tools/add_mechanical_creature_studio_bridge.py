"""Add Studio frame exchange and receipt calls to either exact original version.

No worker, solver, camera or input implementation is replaced. For a previously
derived candidate, supply its explicitly verified expected_source_sha256. The
entire source is recovered exactly by reversing this insertion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "examples/meowdoku/web/mechanical_creature_studio_bridge.js"
BASE_HASHES = {
    "aa6cb6fb897d24e6bf769ca2dbfde3a0d3d0dbcdd64ddb87fd30173bc0f0d5e2",
    "2cd5c5e23f1491243346dd46586f5ebc10a4391187a4e991fa8f151f8dfd4ef4",
}
START = "  entityCycleTime = now;\n  updateViewportControls(dt);"
START_NEW = "  entityCycleTime = now;\n  applyMechanicalCreatureStudioEnvironment();\n  updateViewportControls(dt);"
END = "  updateShaderViewer();\n  updateViewportTelemetry(now);"
END_NEW = "  updateShaderViewer();\n  publishMechanicalCreatureStudioSnapshot(now,dt);\n  updateViewportTelemetry(now);"
FUNCTION = "function runEntityCycle(now) {"
READY = '          worker.postMessage({type: "colliders", colliders: shaderViewer.colliders});'
READY_NEW = READY + '\n          markMechanicalCreatureStudioColliderPublication();'


def patch_studio_bridge(source: bytes, *, expected_source_sha256: str | None = None) -> tuple[bytes, dict]:
    digest = hashlib.sha256(source).hexdigest()
    if (expected_source_sha256 is not None and digest != expected_source_sha256.lower()) or (
        expected_source_sha256 is None and digest not in BASE_HASHES
    ):
        raise ValueError(f"unverified original game or derived candidate: {digest}")
    document = source.decode("utf-8")
    if "const mechanicalCreatureStudioBridgeState" in document:
        raise ValueError("Studio bridge is already installed")
    separator = "\r\n" if "\r\n" in document else "\n"
    replacements = [(before.replace("\n", separator), after.replace("\n", separator))
                    for before, after in ((START, START_NEW), (END, END_NEW), (READY, READY_NEW))]
    helper = HELPER.read_bytes().decode("utf-8")
    if "</script" in helper.lower():
        raise ValueError("unsafe helper script terminator")
    for before, _ in replacements:
        if document.count(before) != 1:
            raise ValueError("original game frame seam is not unique")
    if document.count(FUNCTION) != 1:
        raise ValueError("original game cycle is not unique")
    output = document
    for before, after in replacements:
        output = output.replace(before, after, 1)
    output = output.replace(FUNCTION, helper + "\n" + FUNCTION, 1)
    restored = output.replace(helper + "\n" + FUNCTION, FUNCTION, 1)
    for before, after in reversed(replacements):
        restored = restored.replace(after, before, 1)
    if restored != document:
        raise AssertionError("runtime exchange changed unrelated original game bytes")
    encoded = output.encode("utf-8")
    return encoded, {"schema": "mechanical-creature-studio-bridge-patch-v1",
        "source_sha256": digest, "output_sha256": hashlib.sha256(encoded).hexdigest(),
        "source_recovered_exactly": True, "worker_model_controls_unchanged": True,
        "new_clock": False, "published": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-source-sha256")
    args = parser.parse_args()
    protected = {args.source.resolve(), (ROOT / "build/meowdoku/web/mechanical_creature_play.html").resolve(),
                 (ROOT / "build/dot-playable-tires-20261110/index.html").resolve(),
                 (ROOT / "build/dot-living-map-runtime-20261110/matched/index.html").resolve()}
    if args.output.resolve() in protected:
        parser.error("output must be a separate candidate")
    output, receipt = patch_studio_bridge(args.source.read_bytes(), expected_source_sha256=args.expected_source_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(output)
    print(json.dumps({"output": str(args.output), **receipt}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
