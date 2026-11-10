"""Link an emitted game transition into the existing native diagnostic shell.

This consumes the exact named input capture from the actual game worker. It
does not supply default physics, a replacement game loop, or a new host.
The executable replays one captured transition, not an interactive game.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.compiler.compiled_program_api import CompiledProgramAPI, EntryPoint, Parameter
from src.compiler.fortran_c_shell import NativeObjectSection, compile_native_program_c_shell
from src.compiler.ssa_fortran_backend import fortran_compiler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--frame", required=True, type=Path)
    parser.add_argument("--emit", action="store_true", help="emit current game SSA through its existing C backend")
    args = parser.parse_args()
    directory = args.build.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if args.emit:
        from src.compiler.abstract_ui_vehicles import compile_symbolic_vehicle_physics_c
        print("Emitting current game transition with existing C backend", flush=True)
        artifact = compile_symbolic_vehicle_physics_c()
        (directory / (artifact.name + ".c")).write_text(artifact.source)
        (directory / "vehicle-abi.json").write_text(json.dumps({
            "name": artifact.name, "input_names": artifact.input_names,
            "output_names": artifact.output_names}, indent=2))
    abi = json.loads((directory / "vehicle-abi.json").read_text())
    frame = json.loads(args.frame.read_text())
    inputs = abi["input_names"]
    outputs = abi["output_names"]
    # Refuse mismatched snapshots. No positional guesses or missing defaults.
    if set(inputs) != set(frame["effectiveInput"]):
        raise ValueError("captured input identities differ from emitted native ABI")
    if set(outputs) != set(frame["output"]):
        raise ValueError("captured output identities differ from emitted native ABI")
    source = directory / (abi["name"] + ".c")
    obj = directory / (abi["name"] + ".o")
    fc = fortran_compiler()
    if fc is None:
        raise RuntimeError("existing native C shell's adjacent GCC was not found")
    cc = Path(fc).with_name("gcc.exe" if sys.platform == "win32" else "gcc")
    print("Compiling emitted C with", cc, flush=True)
    subprocess.run([str(cc), "-O0", "-fno-sanitize=all", "-c",
                    str(source), "-o", str(obj)], check=True)
    parameters = tuple(Parameter(name, role, "float64", "double", "c_double",
                                 "reference", (count,), source_name=name)
                       for name, role, count in (("input", "input", len(inputs)),
                                                  ("output", "output", len(outputs))))
    api = CompiledProgramAPI(abi["name"], "c", abi["name"],
                            (EntryPoint(abi["name"], abi["name"], "numerical", parameters),),
                            metadata={"named_input_slots": inputs, "named_output_slots": outputs,
                                      "scope": "captured-game-transition-not-playable-world"})
    product = compile_native_program_c_shell(
        api, {"input": np.array([frame["effectiveInput"][name] for name in inputs])},
        directory / "shell", object_sections=(NativeObjectSection(obj, "c"),),
        name="mechanical_creature_frame", entrypoint=abi["name"])
    result = product.run(frames=1)
    print(result.stdout, end="", flush=True)
    observed = np.fromfile(product.final_outputs_path, dtype=np.float64)
    if len(observed) != len(outputs) or not np.isfinite(observed).all():
        raise RuntimeError("native shell returned invalid output count or nonfinite state")
    reference = np.array([frame["output"][name] for name in outputs])
    differences = {name: {"native": float(actual), "wasm": float(expected),
                          "absolute_difference": abs(float(actual - expected))}
                   for name, actual, expected in zip(outputs, observed, reference)
                   if actual != expected}
    receipt = {"executable": str(product.executable_path), "frames": 1,
               "finite_outputs": len(outputs), "exact_matches": len(outputs) - len(differences),
               "differences": differences, "playable_world": False}
    (directory / "native-shell-receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
