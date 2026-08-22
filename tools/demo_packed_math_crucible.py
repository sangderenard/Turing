"""Crush one field of metric tensors through every packed realization.

The demo exists to be *checked*, not admired.  A single W x H field of 3x3
symmetric metric tensors is built once, then eigendecomposed by every route
the newly packed mathematical library offers -- the interpreted AbstractTensor
sweep, the same sweep with every rotation routed to a compiled BLAS ``rot``,
and (optionally) one whole-graph native artifact that answers an entire
eigendecomposition in a single launch.  Every lane is scored against NumPy on
the identical inputs, so a lane that is fast and wrong reports as wrong.

The BLAS stage is deliberately run on more than one surface.  The kernel bank
serves the NATIVE surface -- one authored scalar loop on one core -- and
``glsl_blas_deployment`` lowers the SAME authored source into standalone
compute-shader products that own their own GL context.  Reporting only the
native number understates the packs by roughly three orders of magnitude.

It also prints where the packs do NOT reach.  The trigonometry object is
captured at one literal shape and its artifact carries no runtime extents, so
it answers correctly at that shape and silently returns that shape for any
other; several linalg methods refuse to cook at all, with a stated compiler
shortfall.  Those refusals are part of the demo's output because a coverage
claim that only reports its successes is the one thing this tree's admission
machinery exists to prevent.

Run::

    python -m tools.demo_packed_math_crucible --width 72 --height 72

Cook the packs it reads first (a few minutes, and linalg is expected to fail
partway -- the demo reads whatever completed)::

    python -m tools.rebuild_all_standard_objects --output build/demo-packs --sizes 3
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors import linalg


RULE = "=" * 78


def _banner(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}", flush=True)


def _seconds(value: float) -> str:
    if value < 1.0e-3:
        return f"{value * 1.0e6:8.1f} us"
    if value < 1.0:
        return f"{value * 1.0e3:8.2f} ms"
    if value < 120.0:
        return f"{value:8.2f} s "
    return f"{value / 60.0:8.2f} m "


# --------------------------------------------------------------------------
# 1. What is actually packed
# --------------------------------------------------------------------------


def report_inventory(packs: Path) -> dict[str, dict]:
    """Read every cooked standard object and state its real coverage."""

    _banner("1. PACK INVENTORY -- what cooked, at what shape, and what refused")
    records: dict[str, dict] = {}
    if not packs.is_dir():
        print(f"  no cooked packs at {packs}")
        print("  build them: python -m tools.rebuild_all_standard_objects "
              f"--output {packs} --sizes 3")
        return records

    for child in sorted(packs.iterdir()):
        manifest = child / "manifest.json"
        if not manifest.is_file():
            if child.is_dir():
                print(f"\n  {child.name:14s} INCOMPLETE -- cook did not publish "
                      "a manifest (see the refusal it raised)")
            continue
        record = json.loads(manifest.read_text(encoding="utf-8"))
        records[child.name] = record
        methods = record["methods"]
        print(f"\n  {child.name}  ({len(methods)} methods, "
              f"product {record['product_id'][:12]})")
        print(f"    {'method':10s} {'forward':14s} {'shape coverage':26s} rows")
        for method in methods:
            name = method["name"]
            forward = record["artifacts"][name]["parametric_forward"]
            rows = sum(
                1 for row in record["deployment_matrix"] if row["method"] == name
            )
            if forward["kind"] == "kernel_bank":
                error = forward["verification"]["worst_abs_error"]
                coverage = f"parametric, admitted {error:.1e}"
            else:
                artifact = forward["artifact"]
                extents = artifact.get("extent_order") or ()
                shapes = artifact.get("buffer_shapes") or ()
                coverage = (
                    f"runtime extents ({len(extents)})" if extents
                    else f"FROZEN at {shapes[0] if shapes else '?'}"
                )
            print(f"    {name:10s} {forward['kind']:14s} {coverage:26s} {rows}")
    return records


def report_trig_shape_coverage(packs: Path) -> None:
    """Install the trigonometry pack and measure the shapes it truly serves."""

    _banner("2. TRIGONOMETRY PACK -- installed onto AbstractTensor, then probed")
    path = packs / "trigonometry"
    if not (path / "manifest.json").is_file():
        print(f"  no cooked trigonometry object at {path}")
        return

    from src.compiler.mathematical_library_product import _python_loader

    loader = types.ModuleType("packed_standard_object_loader")
    exec(compile(_python_loader(), "<generated loader>", "exec"), loader.__dict__)
    pack = loader.CompiledStandardObject(
        path, json.loads((path / "manifest.json").read_text(encoding="utf-8")),
    )

    authored = AbstractTensor.sin
    pack.install(AbstractTensor)
    print(f"  installed: AbstractTensor.sin rebound = "
          f"{AbstractTensor.sin is not authored}")
    print(f"  packs held by the class: "
          f"{len(AbstractTensor._installed_operator_packs)}")
    print(f"\n    {'length':>8s}  {'result':>10s}  verdict")
    try:
        for length in (2, 4, 5, 16, 4096):
            values = np.linspace(0.1, 1.2, length)
            got = np.asarray(
                AbstractTensor.get_tensor(values).sin().tolist(), dtype=float,
            ).ravel()
            want = np.sin(values)
            if got.shape != want.shape:
                verdict = (f"SILENTLY WRONG SHAPE -- asked {want.shape[0]}, "
                           f"got {got.shape[0]}")
            else:
                verdict = f"max|err| = {np.max(np.abs(got - want)):.3e}"
            print(f"    {length:8d}  {str(got.shape):>10s}  {verdict}")
    finally:
        pack.uninstall(AbstractTensor)
    print(f"  uninstalled: authored sin restored = "
          f"{AbstractTensor.sin is authored}")
    print("\n  The artifact's extent_order is empty: its trip count is the "
          "literal\n  capture shape, but install() rebinds the operator for "
          "every shape.\n  The field below is therefore built on the authored "
          "path, not this pack.")


# --------------------------------------------------------------------------
# 2. The field
# --------------------------------------------------------------------------


def build_field(width: int, height: int) -> tuple[np.ndarray, float]:
    """One 3x3 symmetric metric tensor per cell, from AbstractTensor trig.

    Every trigonometric value below is produced by the authored AbstractTensor
    operator surface -- the same source the trigonometry pack was cooked from.
    """

    _banner(f"3. FIELD -- {width} x {height} = {width * height} metric tensors")
    grid_u, grid_v = np.meshgrid(
        np.linspace(-np.pi, np.pi, width, dtype=np.float64),
        np.linspace(-np.pi, np.pi, height, dtype=np.float64),
        indexing="ij",
    )
    u = AbstractTensor.get_tensor(grid_u.ravel())
    v = AbstractTensor.get_tensor(grid_v.ravel())

    started = time.perf_counter()
    fibres = (
        (u.cos(), u.sin(), (v * 0.5).sin().tanh()),
        ((u + v).sin(), (u - v).cos(), (v * 0.75).tanh()),
        ((u * 0.5).sin() * v.cos(), (v * 0.5).cos(), (u * 0.25).sin()),
    )
    weights = (
        (u.sin() * v.sin()) * 0.5 + 1.25,
        (u * 0.5).cos() * 0.4 + 0.9,
        (v * 0.5).sin() * 0.3 + 0.6,
    )
    elapsed = time.perf_counter() - started

    cells = width * height
    metric = np.zeros((cells, 3, 3), dtype=np.float64)
    for fibre, weight in zip(fibres, weights):
        direction = np.stack(
            [np.asarray(component.tolist(), dtype=np.float64).ravel()
             for component in fibre],
            axis=1,
        )
        scale = np.asarray(weight.tolist(), dtype=np.float64).ravel()
        metric += scale[:, None, None] * (
            direction[:, :, None] * direction[:, None, :]
        )
    metric += np.eye(3)[None, :, :] * 0.05

    print(f"  built with the authored trig surface in {_seconds(elapsed)}")
    print(f"  symmetry residual  max|M - M^T| = "
          f"{np.max(np.abs(metric - np.transpose(metric, (0, 2, 1)))):.3e}")
    smallest = np.min(np.linalg.eigvalsh(metric))
    print(f"  smallest eigenvalue anywhere      = {smallest:.6f} "
          f"({'positive definite' if smallest > 0 else 'INDEFINITE'})")
    return metric, elapsed


# --------------------------------------------------------------------------
# 3. The BLAS pack, on real sizes
# --------------------------------------------------------------------------


def report_blas(metric: np.ndarray, gemm_size: int) -> dict[str, float]:
    """Route genuine reductions through the admission-verified BLAS pack."""

    _banner("4. BLAS PACK -- native launches, scored against NumPy")
    from src.compiler.kernel_bank import LaunchCoordinator, open_blas_bank

    started = time.perf_counter()
    coordinator = LaunchCoordinator(open_blas_bank(), specialize_missing=False)
    print(f"  bank opened in {_seconds(time.perf_counter() - started)}")

    def timed(call, repetitions=3):
        """Route once to pay compile/selection, then time the steady state."""

        call()
        samples = []
        for _repetition in range(repetitions):
            started = time.perf_counter()
            result = call()
            samples.append(time.perf_counter() - started)
        return result, float(np.median(samples))

    scores: dict[str, float] = {}
    field = np.ascontiguousarray(metric.reshape(metric.shape[0], 9))
    flat = np.ascontiguousarray(field.ravel())

    energy, dot_seconds = timed(
        lambda: coordinator.launch("dot", x=flat, y=flat, n=flat.size)
    )
    reference = float(flat @ flat)
    scores["dot"] = abs(energy - reference) / abs(reference)
    print(f"    dot   n={flat.size:<9d} {_seconds(dot_seconds)}  "
          f"relative error {scores['dot']:.3e}")

    probe = np.ascontiguousarray(np.linspace(-1.0, 1.0, 9))
    projection, gemv_seconds = timed(
        lambda: coordinator.launch(
            "gemv", A=field, x=probe, y=np.zeros(field.shape[0]),
            alpha=1.0, beta=0.0, m=field.shape[0], n=9,
        )
    )
    scores["gemv"] = float(np.max(np.abs(projection - field @ probe)))
    print(f"    gemv  {field.shape[0]}x9{'':6s} {_seconds(gemv_seconds)}  "
          f"max|err| {scores['gemv']:.3e}")

    rng = np.random.default_rng(20260821)
    left = np.ascontiguousarray(rng.standard_normal((gemm_size, gemm_size)))
    right = np.ascontiguousarray(rng.standard_normal((gemm_size, gemm_size)))
    product, gemm_seconds = timed(
        lambda: coordinator.launch(
            "gemm", A=left, B=right, C=np.zeros((gemm_size, gemm_size)),
            alpha=1.0, beta=0.0, m=gemm_size, n=gemm_size, k=gemm_size,
        )
    )
    oracle, oracle_seconds = timed(lambda: left @ right)
    scores["gemm"] = float(np.max(np.abs(product - oracle)))
    flops = 2.0 * gemm_size ** 3
    print(f"    gemm  {gemm_size}^3{'':7s} {_seconds(gemm_seconds)}  "
          f"max|err| {scores['gemm']:.3e}")
    print(f"          packed {flops / gemm_seconds / 1e9:6.2f} GF/s   "
          f"numpy {flops / oracle_seconds / 1e9:6.2f} GF/s   "
          f"(numpy is {gemm_seconds / oracle_seconds:.1f}x faster here -- this "
          f"is the\n          NATIVE surface, one scalar loop on one core; "
          f"see 4b for the shader surface)")
    print("\n  These kernels are authored with `n` as a real loop bound, so the\n"
          "  artifact carries runtime extents and any size is served.")
    return scores, gemm_seconds, oracle_seconds


def report_shader_gemm(gemm_size: int, output: Path, iterations: int,
                       native_seconds: float, numpy_seconds: float) -> None:
    """The same authored GEMM lowered to a compute shader and run on the GPU.

    The native lane above is one surface of four.  ``glsl_blas_deployment``
    emits a standalone OpenGL product from the SAME authored source -- Python
    builds and measures it, then the executable owns its own hidden context and
    links neither Python nor Turing.  Two variants are built: the source-order
    algorithm exactly as authored, and the compiler-selected tiled identity.
    """

    _banner("4b. SHADER SURFACE -- the same GEMM as a compute-shader product")
    from tools.demo_glsl_blas_pair import run_comparison

    try:
        report = run_comparison(
            gemm_size, gemm_size, gemm_size,
            output=output, iterations=iterations,
        )
    except Exception as error:
        text = str(error).replace("\n", " ")
        print(f"  no GPU lane -- {type(error).__name__}: {text[:220]}")
        return

    flops = 2.0 * gemm_size ** 3
    labels = {
        "source_algorithm": "GPU, source-order (as authored)",
        "glslblas_gemm": "GPU, compiler-tiled identity",
    }
    print(f"    {'lane':34s} {'time':>12s} {'GF/s':>10s}")
    print(f"    {'CPU, native scalar loop':34s} "
          f"{_seconds(native_seconds):>12s} "
          f"{flops / native_seconds / 1e9:>10.2f}")
    print(f"    {'CPU, numpy (tuned BLAS)':34s} "
          f"{_seconds(numpy_seconds):>12s} "
          f"{flops / numpy_seconds / 1e9:>10.2f}")
    fastest = native_seconds
    device = ""
    for record in report["deployments"]:
        measurement = record.get("measurement")
        if measurement is None:
            continue
        seconds = float(measurement["elapsed_ms"]) / 1000.0
        fastest = min(fastest, seconds)
        device = measurement.get("opengl", device)
        print(f"    {labels.get(record['variant'], record['variant']):34s} "
              f"{_seconds(seconds):>12s} "
              f"{float(measurement['gflops']):>10.2f}")
    equivalence = report.get("equivalence", {})
    print(f"\n    device {device}")
    print(f"    the two shader variants agree: max_abs "
          f"{equivalence.get('max_abs')}, allclose {equivalence.get('allclose')}")
    print(f"    fastest shader lane is {native_seconds / fastest:.0f}x the "
          f"native scalar lane and {numpy_seconds / fastest:.0f}x numpy")
    print("    NOTE dtype: the shader arena is float32, both CPU lanes are "
          "float64.\n    The shader-vs-shader ratio is the like-for-like "
          "number; the CPU\n    comparisons carry that caveat.")


# --------------------------------------------------------------------------
# 4. The crunch
# --------------------------------------------------------------------------


def _oracle(metric: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    started = time.perf_counter()
    values, vectors = np.linalg.eigh(metric)
    return values, vectors, time.perf_counter() - started


def _lane_jacobi(metric: np.ndarray, sweeps: int, cells: int):
    """The interpreted AbstractTensor sweep -- sampled, then extrapolated."""

    sample = metric[:cells]
    started = time.perf_counter()
    values = np.stack([
        np.asarray(
            linalg.eigh(
                AbstractTensor.get_tensor(cell), sweeps=sweeps, method="jacobi",
            )[0].tolist(),
            dtype=np.float64,
        ).ravel()
        for cell in sample
    ])
    return values, time.perf_counter() - started


def _lane_blas(metric: np.ndarray, sweeps: int):
    """Every Jacobi rotation issued as one compiled ``rot`` launch."""

    started = time.perf_counter()
    values = np.empty((metric.shape[0], 3), dtype=np.float64)
    vectors = np.empty_like(metric)
    for index, cell in enumerate(metric):
        w, V = linalg.eigh(
            AbstractTensor.get_tensor(cell), sweeps=sweeps, method="blas",
        )
        values[index] = np.asarray(w.tolist(), dtype=np.float64).ravel()
        vectors[index] = np.asarray(V.tolist(), dtype=np.float64)
    return values, vectors, time.perf_counter() - started


def _compile_whole_graph_eigh(directory: Path, sweeps: int):
    """One native artifact for an entire eigendecomposition."""

    from src.common.tensors.mathematical_library import LINALG_LIBRARY
    from src.common.tensors.source_realization import authored_source_realization
    from src.compiler.llvm_training_runtime import compile_native_graph_forward
    from src.compiler.standard_object_linalg import LINALG_CAPTURE_CONTRACTS
    from src.compiler.standard_object_tensor_capture import capture_tensor_method

    with authored_source_realization():
        capture = capture_tensor_method(
            linalg.eigh,
            LINALG_CAPTURE_CONTRACTS["eigh"],
            {"n": 3, "sweeps": sweeps, "method": "jacobi"},
            name="crucible_eigh",
        )
    return compile_native_graph_forward(
        capture.output,
        bindings=capture.bindings,
        source=LINALG_LIBRARY.method("eigh").source,
        name="crucible_eigh",
        directory=directory,
    )


def _lane_whole_graph(metric: np.ndarray, forward):
    """Prepare the ABI once, then refill the input buffer per cell."""

    from src.compiler.ssa_llvm_backend import prepare_artifact_execution

    input_id = int(next(iter(forward.input_value_ids.values())))
    # prepare_artifact_execution binds a feed BY REFERENCE, so the buffer we
    # refill below must be our own copy -- feeding a view of `metric` would
    # overwrite the caller's field one cell at a time.  This is the same
    # aliasing hazard docs/KERNEL_BANK_DESIGN.md section 4.5 records against
    # CompiledVariant.run.
    execution = prepare_artifact_execution(
        forward.artifact,
        {input_id: np.array(metric[0], dtype=np.float64, copy=True)},
    )
    feed = execution.buffers[input_id]
    output_id = int(forward.output_value_ids[0])

    started = time.perf_counter()
    values = np.empty((metric.shape[0], 3), dtype=np.float64)
    for index, cell in enumerate(metric):
        feed[...] = cell
        execution.run()
        values[index] = np.asarray(
            execution.buffers[output_id], dtype=np.float64,
        ).ravel()[:3]
    return values, time.perf_counter() - started


def report_eigh(metric: np.ndarray, sweeps: int, jacobi_cells: int,
                whole_graph_directory: Path | None):
    _banner(f"5. SPECTRAL CRUNCH -- {metric.shape[0]} eigendecompositions, "
            f"{sweeps} sweeps each")
    cells = metric.shape[0]
    reference, _vectors, oracle_seconds = _oracle(metric)
    print(f"  numpy oracle (vectorized over all cells): "
          f"{_seconds(oracle_seconds)}")

    lanes: list[dict] = []

    sampled = max(1, min(jacobi_cells, cells))
    print(f"\n  lane: jacobi (interpreted AbstractTensor) -- "
          f"sampling {sampled} of {cells} cells")
    values, seconds = _lane_jacobi(metric, sweeps, sampled)
    error = float(np.max(np.abs(values - reference[:sampled])))
    per_cell = seconds / sampled
    lanes.append({
        "lane": "jacobi (interpreted)", "cells": sampled, "seconds": seconds,
        "per_cell": per_cell, "error": error, "projected": per_cell * cells,
    })
    print(f"    {_seconds(seconds)} for {sampled} cells  -> "
          f"{_seconds(per_cell)}/cell   max|err| {error:.3e}")
    print(f"    the whole field on this lane would take "
          f"{_seconds(per_cell * cells)}")

    print(f"\n  lane: blas-routed (one compiled `rot` launch per rotation) -- "
          f"all {cells} cells")
    blas_values, blas_vectors, seconds = _lane_blas(metric, sweeps)
    error = float(np.max(np.abs(blas_values - reference)))
    lanes.append({
        "lane": "blas-routed pack", "cells": cells, "seconds": seconds,
        "per_cell": seconds / cells, "error": error, "projected": seconds,
    })
    print(f"    {_seconds(seconds)} for {cells} cells  -> "
          f"{_seconds(seconds / cells)}/cell   max|err| {error:.3e}")

    if whole_graph_directory is not None:
        print(f"\n  lane: whole-graph pack (one launch per cell)")
        started = time.perf_counter()
        try:
            forward = _compile_whole_graph_eigh(whole_graph_directory, sweeps)
        except Exception as error_value:
            text = str(error_value).replace("\n", " ")
            print(f"    REFUSED after {_seconds(time.perf_counter() - started)}"
                  f" -- {type(error_value).__name__}: {text[:220]}")
        else:
            print(f"    compiled in {_seconds(time.perf_counter() - started)} "
                  f"-> {forward.artifact.library_path.name}")
            values, seconds = _lane_whole_graph(metric, forward)
            error = float(np.max(np.abs(np.sort(values, axis=1) - reference)))
            lanes.append({
                "lane": "whole-graph pack", "cells": cells,
                "seconds": seconds, "per_cell": seconds / cells,
                "error": error, "projected": seconds,
            })
            print(f"    {_seconds(seconds)} for {cells} cells  -> "
                  f"{_seconds(seconds / cells)}/cell   max|err| {error:.3e}")

    print("\n  identity checks against the linalg surface (packed BLAS lane):")
    sample = min(64, cells)
    trace_error = 0.0
    det_error = 0.0
    for index in range(sample):
        tensor = AbstractTensor.get_tensor(metric[index])
        trace_error = max(trace_error, abs(
            linalg.trace(tensor).item() - float(np.sum(blas_values[index]))
        ))
        det_error = max(det_error, abs(
            linalg.det(tensor).item() - float(np.prod(blas_values[index]))
        ))
    print(f"    max |linalg.trace(M) - sum(lambda)|     = {trace_error:.3e}")
    print(f"    max |linalg.det(M)   - prod(lambda)|    = {det_error:.3e}")
    print("    (read through .item(); float(tensor) truncates -- "
          "AbstractTensor\n     defines __index__ and no __float__)")

    return lanes, blas_values, blas_vectors


# --------------------------------------------------------------------------
# 5. Make it visible
# --------------------------------------------------------------------------


def render(values: np.ndarray, vectors: np.ndarray, width: int, height: int,
           destination: Path, scale: int) -> Path:
    """Colour every cell by principal orientation and anisotropy."""

    _banner("6. RENDER")
    principal = vectors[:, :, 2]
    orientation = AbstractTensor.get_tensor(
        principal[:, 1] / (principal[:, 0] + 1.0e-12)
    ).atan()
    angle = np.asarray(orientation.tolist(), dtype=np.float64).ravel()
    hue = (angle / np.pi + 0.5) % 1.0

    largest = values[:, 2]
    smallest = values[:, 0]
    anisotropy = (largest - smallest) / (largest + smallest + 1.0e-12)
    magnitude = largest / max(float(np.max(largest)), 1.0e-12)

    from matplotlib.colors import hsv_to_rgb

    hsv = np.stack([
        hue.reshape(width, height),
        np.clip(anisotropy, 0.0, 1.0).reshape(width, height),
        np.clip(0.15 + 0.85 * magnitude, 0.0, 1.0).reshape(width, height),
    ], axis=-1)
    rgb = (hsv_to_rgb(hsv) * 255.0).astype(np.uint8)
    rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)

    from PIL import Image

    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.transpose(rgb, (1, 0, 2))).save(destination)
    print(f"  hue = principal-eigenvector orientation (linalg + trig atan)")
    print(f"  saturation = anisotropy, value = largest eigenvalue")
    print(f"  wrote {destination}  ({rgb.shape[1]}x{rgb.shape[0]})")
    return destination


def report_scoreboard(lanes: list[dict]) -> None:
    _banner("7. SCOREBOARD")
    print(f"  {'lane':24s} {'cells':>7s} {'per cell':>12s} "
          f"{'whole field':>13s} {'max|err| vs numpy':>19s}")
    for lane in lanes:
        print(f"  {lane['lane']:24s} {lane['cells']:7d} "
              f"{_seconds(lane['per_cell']):>12s} "
              f"{_seconds(lane['projected']):>13s} "
              f"{lane['error']:>19.3e}")
    if len(lanes) > 1:
        base = lanes[0]["per_cell"]
        print()
        for lane in lanes[1:]:
            print(f"  {lane['lane']:24s} is {base / lane['per_cell']:8.1f}x "
                  f"the interpreted sweep")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=72)
    parser.add_argument("--height", type=int, default=72)
    parser.add_argument("--sweeps", type=int, default=24)
    parser.add_argument(
        "--jacobi-cells", type=int, default=8,
        help="cells to run on the interpreted lane before extrapolating",
    )
    parser.add_argument("--gemm", type=int, default=256)
    parser.add_argument(
        "--no-gpu", action="store_true",
        help="skip the standalone compute-shader GEMM products",
    )
    parser.add_argument("--gpu-iterations", type=int, default=20)
    parser.add_argument("--packs", type=Path, default=ROOT / "build" / "demo-packs")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "build" / "packed-math-crucible",
    )
    parser.add_argument("--scale", type=int, default=6)
    parser.add_argument(
        "--whole-graph", action="store_true",
        help="also compile linalg.eigh whole-graph (minutes) and launch it per cell",
    )
    arguments = parser.parse_args(argv)

    print(RULE)
    print("PACKED MATH CRUCIBLE -- one field, every realization, all scored")
    print(RULE)

    report_inventory(arguments.packs)
    report_trig_shape_coverage(arguments.packs)
    metric, _field_seconds = build_field(arguments.width, arguments.height)
    _scores, native_seconds, numpy_seconds = report_blas(metric, arguments.gemm)
    if not arguments.no_gpu:
        report_shader_gemm(
            arguments.gemm, arguments.output / "glsl", arguments.gpu_iterations,
            native_seconds, numpy_seconds,
        )
    lanes, values, vectors = report_eigh(
        metric, arguments.sweeps, arguments.jacobi_cells,
        (arguments.output / "whole-graph") if arguments.whole_graph else None,
    )
    render(
        values, vectors, arguments.width, arguments.height,
        arguments.output / "crucible.png", arguments.scale,
    )
    report_scoreboard(lanes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
