"""Relink the behavior packet of a validated MechanicalCreature page cache."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.compiler.abstract_ui import javascript_with_system_root
from src.compiler.abstract_ui_div_map import DIV_MAP_JAVASCRIPT
from src.compiler.javascript_runtime_utilities import render_javascript_utilities
from src.compiler.state_loop_deployment import emit_javascript_physics_worker


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--legacy-fixed-kernels", action="store_true",
        help="keep one 1/120 step for a cached page whose specialized kernels baked that dt",
    )
    arguments = parser.parse_args()
    source = arguments.source.resolve()
    destination = arguments.output.resolve()
    document = source.read_text(encoding="utf-8")

    model_marker = "abstract-ui-model"
    marker_index = document.index(model_marker)
    model_start = document.index(">", marker_index) + 1
    model_end = document.index("</script>", model_start)
    model = json.loads(document[model_start:model_end])
    identity = str(model["identity"])
    expected = "introspection-world:python:src.compiler.mechanical_creature.MechanicalCreature"
    if identity != expected:
        raise ValueError(f"refusing to relink unexpected world {identity!r}")

    if arguments.legacy_fixed_kernels:
        worker_source = emit_javascript_physics_worker()
        marker = "PHYSICS_SUBSTEPS=3,SUBSTEP_DT=FIXED_DT/PHYSICS_SUBSTEPS"
        if marker not in worker_source:
            raise ValueError("could not locate the three-substep worker marker")
        worker_source = worker_source.replace(
            marker, "PHYSICS_SUBSTEPS=1,SUBSTEP_DT=FIXED_DT", 1,
        )
        model["loop_deployment"]["workers"][0]["source"] = worker_source
        encoded_model = json.dumps(model, ensure_ascii=False, separators=(",", ":"))
        if "</script>" in encoded_model.lower():
            raise ValueError("updated model contains an unsafe script terminator")
        document = document[:model_start] + encoded_model + document[model_end:]
        model_end = model_start + len(encoded_model)

    script_open = document.index("<script>", model_end)
    script_start = script_open + len("<script>")
    script_end = document.index("</script>", script_start)
    behavior = javascript_with_system_root(
        render_javascript_utilities((
            "turing.wasm.registry",
            "turing.world.registry",
            "turing.revision.channel",
        )) + "\n\n" + DIV_MAP_JAVASCRIPT,
        timer_identity=f"{identity}/timer",
    )
    if "</script>" in behavior.lower():
        raise ValueError("behavior packet contains an unsafe script terminator")
    relinked = document[:script_start] + behavior + document[script_end:]
    if relinked.count(model_marker) != document.count(model_marker):
        raise ValueError("relink changed the embedded model boundary")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(relinked, encoding="utf-8", newline="\n")
    os.replace(temporary, destination)
    print(len(relinked))
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
