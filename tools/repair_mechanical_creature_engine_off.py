"""Guarded startup-only transfer of current source's engine-off initialization."""
import argparse
import hashlib
import json
from pathlib import Path

BASE = "abccd70968961b97737f390ec3673ac034d0ba11733ed8903128c55f0e18e29a"
REPLACEMENTS = (
    ("initializeVehicleFirstExperience();\ninitializeWorldPhysicsWasm();",
     "initializeVehicleFirstExperience();\nvehicleRuntime.electrical.ignitionOn=false;vehicleRuntime.electrical.starterEngaged=false;\ninitializeWorldPhysicsWasm();"),
    ("engine_angular_speed:Number(body.defaults?.engine_idle_angular_speed||0)};",
     "engine_angular_speed:body.energy?.ignitionOn===false?0:Number(body.defaults?.engine_idle_angular_speed||0)};"),
    ("body.wasmState={};body.effectiveThrottle=0;body.featheredThrottle=0;body.featheredThrottleVelocity=0;",
     "body.wasmState=body.energy?.ignitionOn===false?{engine_angular_speed:0}:{};body.effectiveThrottle=0;body.featheredThrottle=0;body.featheredThrottleVelocity=0;"),
)


def repair(data):
    if hashlib.sha256(data).hexdigest() != BASE:
        raise ValueError("Unknown HUD base")
    original = data.decode("utf-8")
    text = original
    for old, new in REPLACEMENTS:
        if text.count(old) != 1:
            raise ValueError("Nonunique engine-off seam")
        text = text.replace(old, new, 1)
    restored = text
    for old, new in reversed(REPLACEMENTS):
        if restored.count(new) != 1:
            raise ValueError("Nonunique reverse seam")
        restored = restored.replace(new, old, 1)
    if restored != original:
        raise ValueError("Reverse restoration failed")
    result = text.encode("utf-8")
    return result, {"base_sha256": BASE, "candidate_sha256": hashlib.sha256(result).hexdigest(),
                    "reverse_byte_restore": True, "replacements": len(REPLACEMENTS),
                    "scope": "Initial ignition/starter off and missing crank state seeded at zero; no kernel changes"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output must be a new candidate")
    result, report = repair(args.source.read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(result)
    args.output.with_suffix(".engine-off.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
