"""Build the bespoke MechanicalCreature page through one checked output path."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.compiler.abstract_ui_div_map import project_class_to_div_map
from src.compiler.abstract_ui_vehicles import set_vehicle_build_progress_sink
from src.compiler.mechanical_creature import MechanicalCreatureWorld


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    destination = arguments.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")

    started = time.monotonic()
    def progress(stage: str, completed: int, total: int, detail: str) -> None:
        elapsed = time.monotonic() - started
        print(f"[vehicle {completed}/{total} · {elapsed:8.1f}s] {stage}: {detail}", flush=True)

    print("[projection · 0.0s] constructing introspective world", flush=True)
    set_vehicle_build_progress_sink(progress)
    try:
        page = project_class_to_div_map(MechanicalCreatureWorld)
    finally:
        set_vehicle_build_progress_sink(None)
    print(f"[projection · {time.monotonic() - started:.1f}s] rendering complete", flush=True)
    rendered = page.html
    if not isinstance(rendered, str) or "<html" not in rendered.lower():
        raise TypeError("MechanicalCreature projection did not return a complete HTML string")
    temporary.write_text(rendered, encoding="utf-8", newline="\n")
    os.replace(temporary, destination)
    print(len(rendered))
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
