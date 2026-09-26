"""Compile the coupled validator simulation for the existing Python viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lanes", type=int, default=8)
    args = parser.parse_args()
    from src.compiler.vehicle_validator_simulation import build_simulation
    from src.compiler.work_contract import set_active_contract

    set_active_contract("develop")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        result = build_simulation(args.output, args.lanes,
                                  progress=lambda message: print(message, flush=True))
    except Exception as error:
        (args.output / "failure.json").write_text(json.dumps({
            "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.perf_counter() - started,
        }, indent=2), encoding="utf-8")
        raise
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
