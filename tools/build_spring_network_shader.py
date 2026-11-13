"""Publish the spring row's authored law as SPIR-V and direct Vulkan GLSL.

The SPIR-V product owns the binding/launch contract. GLSL lowers independently
from the same scalar repository SSA function and is compiled by
glslangValidator. Neither path implements the physics a second time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.compiler.shader_deployment import publish_spirv_product
from src.compiler.spring_network_law import (
    compile_spring_network_compute_shader, compile_spring_network_step,
    emit_spring_network_resident_glsl,
)
from src.compiler.ssa_glsl_compute_backend import emit_ssa_scalar_to_glsl_compute


def _run(args: list[str]) -> None:
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{' '.join(args)}: {result.stderr or result.stdout}")


def build(directory: Path, *, local_size: int = 64,
          steps_per_loadout: int = 4,
          target_completion_ns: int = 20_000_000) -> dict:
    if local_size not in (64, 128, 256, 512, 1024):
        raise ValueError("spring shader local size must be 64, 128, 256, 512, or 1024")
    if not 1 <= steps_per_loadout <= 4096 or target_completion_ns < 0:
        raise ValueError("invalid spring resident work contract")
    shader = compile_spring_network_compute_shader(local_size=local_size)
    if not shader.complete:
        raise RuntimeError(f"spring law shader refused: {shader.shortfalls}")
    product = publish_spirv_product(directory, shader)
    manifest = json.loads(product.manifest_path.read_text(encoding="utf-8"))
    if not manifest["complete"] or product.module_path is None:
        raise RuntimeError(f"spring SPIR-V product incomplete: {manifest['shortfalls']}")
    validator = shutil.which("glslangValidator")
    if validator is None:
        raise RuntimeError("glslangValidator is required")
    glsl = directory / "spring_network_step.comp"
    glsl.write_text(emit_ssa_scalar_to_glsl_compute(
        compile_spring_network_step().function, name="spring_network_step",
        local_size=local_size,
    ), encoding="utf-8", newline="\n")
    roundtrip = directory / "spring_network_step.direct-glsl.spv"
    _run([validator, "-V", "--target-env", "vulkan1.1", "-S", "comp",
          str(glsl), "-o", str(roundtrip)])
    resident_glsl = directory / "spring_network_step.resident.comp"
    resident_glsl.write_text(emit_spring_network_resident_glsl(local_size=local_size),
                             encoding="utf-8", newline="\n")
    resident_spv = directory / "spring_network_step.resident.spv"
    _run([validator, "-V", "--target-env", "vulkan1.1", "-S", "comp",
          str(resident_glsl), "-o", str(resident_spv)])
    result = {
        "schema": "turing.spring-network-shader.v1",
        "law": "spring_network_step",
        "numeric_dtype": "float64",
        "local_size_x": local_size,
        "source_of_truth": "src.compiler.spring_network_law.spring_network_equations",
        "spirv_product": product.manifest_path.name,
        "deployment": product.deployment,
        "glsl": glsl.name,
        "glsl_compiled_spirv": roundtrip.name,
        "resident_glsl": resident_glsl.name,
        "resident_spirv": resident_spv.name,
        "resident_control": "u32 element_count, u32 row_stride_scalar_lanes",
        "resident_neighbor_source": "feed17 current position with feed5..8 boundary masks",
        "resident_work_contract": {
            "schema": "nodus.shader-work.v1",
            "steps_per_loadout": steps_per_loadout,
            "target_completion_ns": target_completion_ns,
            "resident_state_feedback": True,
            "position_current_binding": 17,
            "velocity_current_binding": 23,
            "position_next_binding": 24,
            "velocity_next_binding": 25,
        },
        "glsl_emitter": "ssa_glsl_compute_backend.emit_ssa_scalar_to_glsl_compute",
        "sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (product.module_path, glsl, roundtrip,
                         resident_glsl, resident_spv)
        },
    }
    (directory / "spring_network_step.glsl.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--local-size", type=int, default=64)
    parser.add_argument("--steps-per-loadout", type=int, default=4)
    parser.add_argument("--target-completion-ns", type=int, default=20_000_000)
    args = parser.parse_args()
    print(json.dumps(build(args.directory, local_size=args.local_size,
                           steps_per_loadout=args.steps_per_loadout,
                           target_completion_ns=args.target_completion_ns), indent=2))
