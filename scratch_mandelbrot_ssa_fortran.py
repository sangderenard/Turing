"""Scratch: compile the real (non-stub) Mandelbrot recording program -- the
same ProcessGraph/entrypoint demo_mandelbrot_fusion.py's real AVI-encoding
animate_glsl() uses -- to SSA, then to Fortran, without ever touching a GL
context.  Reports per-frame Fortran execution time, or the honest shortfall
report if the recording program (which includes MJPEG/AVI packet encoding,
not just the escape-time solve) doesn't fully lower.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from src.common.tensors.accelerator_backends.demo_mandelbrot_ssa import (
    build_parametric_mandelbrot_glsl_deployment,
    normalized_plane,
    dream_parameters,
    _open_control_stream,
)
from src.compiler.precompile_to_ssa import (
    lower_precompile_and_control_to_ssa,
)

WIDTH = HEIGHT = 48
ITERATIONS = 24
FRAME_COUNT = 6
TIMELINE_FPS = 60.0

unit_x, unit_y = normalized_plane(WIDTH, HEIGHT)
audio = _open_control_stream(None, gain=1.0, duration=FRAME_COUNT / TIMELINE_FPS)

controls_by_name = {
    name: []
    for name in (
        "center_x", "center_y", "span", "family_mix",
        "julia_x", "julia_y", "palette_phase", "color_drive",
    )
}
center = complex(-0.743643887, 0.131825904)
span = 0.004
travel = 0.0
for frame_index in range(FRAME_COUNT):
    logical_time = frame_index / TIMELINE_FPS
    controls = audio.sample(logical_time)
    travel += (1.0 / TIMELINE_FPS) * 1.0 * (0.28 + 0.20 * 1.35 * controls.loudness)
    animated_center, animated_span, family_mix, julia_constant = dream_parameters(
        center, span, travel,
        bass=controls.bass, low_mid=controls.low_mid, high_mid=controls.high_mid,
        reaction=0.20, zoom_rate=0.0,
    )
    controls_by_name["center_x"].append(animated_center.real)
    controls_by_name["center_y"].append(animated_center.imag)
    controls_by_name["span"].append(animated_span)
    controls_by_name["family_mix"].append(family_mix)
    controls_by_name["julia_x"].append(julia_constant.real)
    controls_by_name["julia_y"].append(julia_constant.imag)
    controls_by_name["palette_phase"].append(0.028 * travel)
    controls_by_name["color_drive"].append(0.52)

tensor_controls = {
    name: np.asarray(values, dtype=np.float32)
    for name, values in controls_by_name.items()
}
static_feeds = {
    "unit_x": unit_x, "unit_y": unit_y,
    "width": WIDTH, "height": HEIGHT, "iterations": ITERATIONS,
}

def batch_feeds(start, stop):
    return {
        **static_feeds,
        **{name: values[start:stop] for name, values in tensor_controls.items()},
    }

print("building deployment (real mandelbrot_recording_program entrypoint,"
      " same graph demo_mandelbrot_fusion.py's AVI root uses)...", flush=True)
deployment, _ = build_parametric_mandelbrot_glsl_deployment(
    ITERATIONS,
    entrypoint="mandelbrot_recording_program",
)
print(
    f"program : {deployment.source_node_count} ProcessGraph nodes -> "
    f"{deployment.primitive_count} scheduled nodes in "
    f"{deployment.dispatch_count} deployment regions",
    flush=True,
)

import pickle

CACHE_PATH = Path(__file__).resolve().parent / ".turing-cache" / "mandelbrot_precompile_cache.pkl"

if CACHE_PATH.exists():
    print(f"loading cached precompile/control/region_programs from {CACHE_PATH}", flush=True)
    precompile, control, region_programs = pickle.loads(CACHE_PATH.read_bytes())
else:
    t0 = time.perf_counter()
    deployment.compile_process_graph()
    print(f"compile_process_graph: {time.perf_counter()-t0:.3f}s", flush=True)

    t0 = time.perf_counter()
    deployment.capture_fused_programs(batch_feeds(0, 1), precompile_only=True)
    print(f"capture_fused_programs(precompile_only=True): {time.perf_counter()-t0:.3f}s"
          " -- no GLSL context touched", flush=True)

    precompile = deployment.compiled_shell_program
    control = deployment.shell_control_program
    if precompile is None or control is None:
        raise RuntimeError("no numerical/control precompile boundary produced")

    region_programs = {}
    for region_key, captured in deployment.captured_region_programs.items():
        region_index = (
            region_key[-2] if isinstance(region_key, tuple) and len(region_key) >= 2
            else region_key
        )
        if isinstance(region_index, bool) or not isinstance(region_index, int):
            raise RuntimeError(f"invalid precompile region key {region_key!r}")
        region_programs[int(region_index)] = captured

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_bytes(pickle.dumps((precompile, control, region_programs)))
    print(f"cached precompile/control/region_programs to {CACHE_PATH}", flush=True)

lowering = lower_precompile_and_control_to_ssa(
    precompile, control,
    numerical_name="mandelbrot_recording_numerical",
    control_name="mandelbrot_recording_control",
    region_programs=region_programs,
)

print("---- SSA lowering report ----")
print(f"precompile steps: {len(precompile.program.steps)}")
print(f"valid_precompile: {lowering.validation.valid_precompile}")
print(f"ssa_compatible: {lowering.validation.ssa_compatible}")
print(f"complete: {lowering.complete}")
print(lowering.shortfall_report())
print("functions in module:", sorted(lowering.module.functions.keys()) if hasattr(lowering.module, "functions") else lowering.module)

if not lowering.complete:
    print("\nSTOPPING: SSA lowering is not complete; will not attempt Fortran emission.")
    deployment.release()
    audio.close()
    sys.exit(1)

from src.compiler.ssa_fortran_backend import emit_module, compile_module

# The lowered module carries the C/LLVM backend's own kernel bodies
# (sum_double, matmul_double, ... -- they hold llvm_opcode/GetElementPtr/Load
# pointer instructions) alongside the Mandelbrot program itself. Those are an
# existing kernel library, not part of this program; Fortran has its own
# intrinsics for the same work and must not be asked to re-implement LLVM's.
# Only the program's own functions are emitted.
PROGRAM_FUNCTIONS = (
    "mandelbrot_recording_numerical",
    "mandelbrot_recording_control",
    "numerical_region_1",
)
from src.transmogrifier.ssa import IRModule

program_module = IRModule(
    {name: lowering.module.functions[name] for name in PROGRAM_FUNCTIONS}
)
module = emit_module(program_module, name="mandelbrot_recording_fortran", outputs={})
print("\n---- Fortran emission ----")
print("complete:", module.complete)
if not module.complete:
    for s in module.shortfalls:
        print("-", s.format())
else:
    out = Path(__file__).resolve().parent / ".turing-cache" / "mandelbrot_recording.f90"
    out.write_text(module.source, encoding="utf-8")
    lines = module.source.splitlines()
    print(f"emitted {len(lines):,} lines / {len(module.source):,} chars -> {out}")
    t0 = time.perf_counter()
    try:
        library = compile_module(module, directory=out.parent)
        print(f"gfortran compile: {time.perf_counter()-t0:.2f}s -> {library}")
    except Exception as exc:
        print(f"gfortran compile FAILED after {time.perf_counter()-t0:.2f}s:")
        print(str(exc)[:4000])

deployment.release()
audio.close()
