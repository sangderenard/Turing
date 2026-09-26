"""Compile the vectorized Python/AbstractTensor vehicle graph for native use.

No vehicle-specific C is accepted as input.  The sole C translation unit in
the output directory is emitted from the Python -> ProcessGraph -> SSA path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compiler.vehicle_balloon_tire_program import balloon_tire_python_program
from src.compiler.vehicle_native_graph_program import vehicle_native_graph_python_program
from src.compiler.vehicle_python_compilation import (
    emit_vehicle_python_graph_c,
    vehicle_python_compilation_inputs,
)


def _sha256(value: str | bytes) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "build" / "vehicle_native_parity",
    )
    parser.add_argument("--no-launch", action="store_true")
    parser.add_argument("--double-double", action="store_true")
    parser.add_argument("--resume-canonical", action="store_true")
    parser.add_argument("--game-validator", action="store_true")
    args = parser.parse_args()
    if args.double_double:
        raise SystemExit(
            "two-limb promotion must be applied to the combined Python graph; "
            "the former per-C-kernel widening path is intentionally disabled"
        )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs = vehicle_python_compilation_inputs()
    tire = balloon_tire_python_program()
    graph = vehicle_native_graph_python_program()
    source_path = output / "vehicle_graph.abstract_tensor.py"
    source_path.write_text(inputs.source, encoding="utf-8")

    def progress(message: str) -> None:
        print(message, flush=True)

    print("[1/4] Python source fixed; capturing vectorized vehicle graph", flush=True)
    artifact = emit_vehicle_python_graph_c(progress=progress)
    c_path = output / "vehicle_native_graph_tick.c"
    c_path.write_text(artifact.source, encoding="utf-8")
    print("[2/4] repository SSA emitted one compiler-owned C call closure", flush=True)
    artifact.compile(output)
    library = artifact.library_path
    if library is None or not library.is_file():
        raise RuntimeError("compiler-owned vehicle library was not produced")
    print(f"[3/4] native vehicle graph compiled: {library}", flush=True)
    manifest = {
        "schema": "turing.vehicle-python-graph-deployment.v2",
        "authority": {
            "source_language": "python",
            "tensor_api": "AbstractTensor",
            "pipeline": ["python-ast", "ProcessGraph", "repository-SSA", "C"],
            "vehicle_specific_c_input": False,
            "legacy_c_linked": False,
            "python_sha256": _sha256(inputs.source),
            "emitted_c_sha256": _sha256(artifact.source),
        },
        "entrypoint": artifact.name,
        "library": library.name,
        "batch_capacity": 8,
        "vector_axes": list(graph.vector_axes),
        "tire": {
            "input_names": list(tire.input_names),
            "output_names": list(tire.output_names),
            "state_scalar_count_per_lane": tire.state_scalar_count,
            "vertex_count": tire.vertex_count,
            "face_count": tire.face_count,
            "contact": "deformed-balloon-skin hard-surface CCD",
        },
        "time_integration": {
            "outer_rate_hz": 120 if args.game_validator else None,
            "regular_substeps": 3 if args.game_validator else 48,
            "wall_time_independent": True,
        },
        "native_presentation": {
            "status": "separate generic ABI consumer",
            "vehicle_math_in_presentation_shell": False,
            "auto_launch": False,
        },
    }
    manifest_path = output / "vehicle_native.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[4/4] wrote authority manifest: {manifest_path}", flush=True)
    if not args.no_launch:
        print(
            "The old viewer is not launched because it calls the retired C ABI; "
            "attach the generic compiled-buffer viewer to this manifest.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
