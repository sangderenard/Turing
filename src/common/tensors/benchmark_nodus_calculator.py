"""Compare one tensor chain across eager, fused-GPU, and prepared execution.

The workload is deliberately restricted to the shared primitive vocabulary:

    sigmoid(x) = 1 / (1 + exp(-x))

One forward trace is captured as a ``FusedProgram``. Setup/allocation is
reported separately. NumPy, Torch, and C replay its steps through
``ProgramRunner``; Nodus receives its canonical textual transport and lowers
the value ids to prepared calculator tensors; GLSL lowers the same object to
one shader. Process startup is not charged to Nodus compute samples.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .abstraction import AbstractTensor as AT
from .fused_ir import (
    ordered_feed_ids,
    serialize_elementwise_fused_program,
)

ProgressCallback = Callable[[str, str, dict | None], None]


def _sigmoid(value):
    """The one ordinary AbstractTensor workload captured and run everywhere."""

    return 1.0 / (1.0 + (-value).exp())


def _capture_sigmoid_program():
    from .accelerator_backends.c_primitive_program import compile_elementwise_tape
    from .autograd import autograd
    from .numpy_backend import NumPyTensorOperations

    with autograd.forward_capture() as tape:
        source = NumPyTensorOperations.tensor(np.asarray([-1.0, 1.0]))
        output = _sigmoid(source)
    return compile_elementwise_tape(tape, output)


def _input_values(elements: int) -> np.ndarray:
    indices = np.arange(elements, dtype=np.int64)
    return ((indices % 1024) - 512).astype(np.float64) / 128.0


def _synchronize_torch(tensor) -> None:
    try:
        import torch
    except ImportError:
        return
    data = getattr(tensor, "data", None)
    device = getattr(data, "device", None)
    device_type = getattr(device, "type", str(device))
    if device is not None and device_type == "cuda":
        torch.cuda.synchronize(device)


def _find_nodus_executable(explicit: Path | None = None) -> Path:
    if explicit is not None:
        candidates = [explicit]
    elif os.environ.get("NODUS_CALCULATOR_BENCHMARK"):
        candidates = [Path(os.environ["NODUS_CALCULATOR_BENCHMARK"])]
    else:
        workspace = Path(__file__).resolve().parents[4]
        name = (
            "nodus_precomposed_calculator_benchmark.exe"
            if os.name == "nt"
            else "nodus_precomposed_calculator_benchmark"
        )
        candidates = [
            workspace / "nodus" / "build" / "Release" / name,
            workspace / "nodus" / "build" / name,
        ]
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file():
            return resolved
    rendered = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Nodus prepared-calculator benchmark executable was not found. "
        "Build target nodus_precomposed_calculator_benchmark or pass "
        f"--nodus-executable. Checked:\n  {rendered}"
    )


def _benchmark_abstracttensor_backend(
    backend: str,
    *,
    captured,
    elements: int,
    warmup: int,
    repeats: int,
    torch_device: str,
) -> dict:
    from .abstract_nn import ProgramRunner
    from .autograd import autograd

    host = _input_values(elements)
    expected = 1.0 / (1.0 + np.exp(-host))
    device = torch_device if backend == "torch" else None
    feed_ids = ordered_feed_ids(captured.program)
    output_name = next(iter(captured.program.outputs))
    runner = ProgramRunner(captured.program)

    started = perf_counter()
    with AT.use_backend(backend, device):
        value = AT.tensor(host, dtype="float64", device=device)
    setup_sec = perf_counter() - started

    def execute():
        feeds = {feed_id: value for feed_id in feed_ids}
        with autograd.no_grad():
            return runner(feeds, training=False)[output_name]

    result = None
    with AT.use_backend(backend, device):
        for _ in range(warmup):
            result = execute()
        if result is not None:
            _synchronize_torch(result)

        timings = []
        for _ in range(repeats):
            _synchronize_torch(value)
            started = perf_counter()
            result = execute()
            _synchronize_torch(result)
            timings.append(perf_counter() - started)

        values = np.asarray(result.tolist(), dtype=np.float64).reshape(-1)

    difference = np.abs(values - expected)
    return {
        "backend": backend,
        "device": device or "cpu",
        "execution": "captured_fused_program_eager",
        "dtype": "float64",
        "program_steps": len(captured.program.steps),
        "elements": elements,
        "warmup": warmup,
        "repeats": repeats,
        "setup_sec": setup_sec,
        "median_sec": float(np.median(timings)),
        "min_sec": float(np.min(timings)),
        "max_sec": float(np.max(timings)),
        "checksum": float(values.sum()),
        "first": float(values[0]),
        "middle": float(values[elements // 2]),
        "last": float(values[-1]),
        "max_abs_error": float(difference.max(initial=0.0)),
    }


def _benchmark_nodus(
    executable: Path,
    *,
    captured,
    elements: int,
    warmup: int,
    repeats: int,
) -> dict:
    expected = 1.0 / (1.0 + np.exp(-_input_values(elements)))
    with tempfile.TemporaryDirectory(prefix="nodus-fused-program-") as temp:
        temp_path = Path(temp)
        program_path = temp_path / "program.fused"
        output_path = temp_path / "output.f64"
        program_path.write_text(
            serialize_elementwise_fused_program(captured.program),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [
                str(executable),
                "--program",
                str(program_path),
                "--output-bin",
                str(output_path),
                "--elements",
                str(elements),
                "--warmup",
                str(warmup),
                "--repeats",
                str(repeats),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        values = np.fromfile(output_path, dtype=np.float64)
    if values.shape != (elements,):
        raise RuntimeError(
            "Nodus benchmark returned "
            f"{values.size} values for {elements} elements"
        )
    lines = [line.strip() for line in completed.stdout.splitlines()]
    payloads = [line for line in lines if line.startswith("{")]
    if not payloads:
        raise RuntimeError(
            "Nodus benchmark produced no JSON result:\n" + completed.stdout
        )
    row = json.loads(payloads[-1])
    difference = np.abs(values - expected)
    row.update(
        checksum=float(values.sum(dtype=np.float64)),
        first=float(values[0]),
        middle=float(values[elements // 2]),
        last=float(values[-1]),
        max_abs_error=float(difference.max(initial=0.0)),
    )
    return row


def _benchmark_glsl(
    *,
    captured,
    elements: int,
    warmup: int,
    repeats: int,
) -> dict:
    """Benchmark the developing GLSL fused-program path without modifying it.

    The input and output remain GPU-resident across samples. We synchronize
    explicitly so wall-clock samples cannot merely measure command submission.
    Buffer setup, compilation/first-dispatch, and final readback are reported
    separately.
    """
    from .accelerator_backends import glsl_backend as glsl

    overall_started = perf_counter()
    context_started = perf_counter()
    context = glsl.require_gl_context()
    context_setup_sec = perf_counter() - context_started

    from OpenGL import GL
    from OpenGL.raw.GL.VERSION.GL_3_3 import glGetQueryObjectui64v

    host = _input_values(elements).astype(np.float32)
    expected = 1.0 / (1.0 + np.exp(-host.astype(np.float64)))
    program = captured.program
    feed_id = next(iter(program.feeds))

    feed = glsl.GLChunk.from_numpy(host)
    upload_started = perf_counter()
    feed.to_gpu()
    GL.glFinish()
    upload_sec = perf_counter() - upload_started

    output = glsl.GLChunk(host.shape)
    output_allocation_started = perf_counter()
    output.to_gpu()
    GL.glFinish()
    output_allocation_sec = perf_counter() - output_allocation_started

    cache_before = glsl.shader_cache_stats()
    compile_started = perf_counter()
    glsl.execute_program(program, {feed_id: feed}, out=output)
    GL.glFinish()
    compile_first_dispatch_sec = perf_counter() - compile_started
    cache_after_compile = glsl.shader_cache_stats()
    setup_sec = perf_counter() - overall_started

    for _ in range(warmup):
        glsl.execute_program(program, {feed_id: feed}, out=output)
        GL.glFinish()

    generated_queries = np.asarray(GL.glGenQueries(1)).reshape(-1)
    query = int(generated_queries[0])
    timings = []
    gpu_timings = []
    try:
        for _ in range(repeats):
            GL.glBeginQuery(GL.GL_TIME_ELAPSED, query)
            started = perf_counter()
            glsl.execute_program(program, {feed_id: feed}, out=output)
            GL.glEndQuery(GL.GL_TIME_ELAPSED)
            GL.glFinish()
            timings.append(perf_counter() - started)
            gpu_nanoseconds = ctypes.c_uint64()
            glGetQueryObjectui64v(
                query, GL.GL_QUERY_RESULT, ctypes.byref(gpu_nanoseconds)
            )
            gpu_timings.append(gpu_nanoseconds.value / 1_000_000_000.0)

        readback_started = perf_counter()
        values = np.asarray(output.numpy(), dtype=np.float32).reshape(-1)
        readback_sec = perf_counter() - readback_started
    finally:
        output.release()
        feed.release()
        GL.glDeleteQueries(1, [query])

    difference = np.abs(values.astype(np.float64) - expected)
    cache_after = glsl.shader_cache_stats()
    renderer = str(context.get("renderer", "OpenGL GPU"))
    return {
        "backend": "glsl",
        "device": renderer,
        "execution": "fused_shader_resident_io",
        "dtype": "float32",
        "program_steps": len(captured.program.steps),
        "elements": elements,
        "warmup": warmup,
        "repeats": repeats,
        "setup_sec": setup_sec,
        "context_setup_sec": context_setup_sec,
        "upload_sec": upload_sec,
        "output_allocation_sec": output_allocation_sec,
        "compile_first_dispatch_sec": compile_first_dispatch_sec,
        "readback_sec": readback_sec,
        "median_sec": float(np.median(timings)),
        "min_sec": float(np.min(timings)),
        "max_sec": float(np.max(timings)),
        "gpu_median_sec": float(np.median(gpu_timings)),
        "gpu_min_sec": float(np.min(gpu_timings)),
        "gpu_max_sec": float(np.max(gpu_timings)),
        "checksum": float(values.sum(dtype=np.float64)),
        "first": float(values[0]),
        "middle": float(values[elements // 2]),
        "last": float(values[-1]),
        "max_abs_error": float(difference.max(initial=0.0)),
        "context_source": context.get("source", "unknown"),
        "gpu_vendor": context.get("vendor", "unknown"),
        "gpu_renderer": renderer,
        "gl_version": context.get("version", "unknown"),
        "shader_cache_misses": (
            cache_after_compile["misses"] - cache_before["misses"]
        ),
        "shader_cache_hits": cache_after["hits"] - cache_before["hits"],
    }


def _annotate_result(
    row: dict,
    *,
    elements: int,
    reference_checksum: float,
) -> None:
    row["checksum_delta"] = abs(row["checksum"] - reference_checksum)
    if row["backend"] == "glsl":
        parity_tolerance = 2e-6
        checksum_tolerance = max(1e-5, elements * 2e-7)
    else:
        parity_tolerance = 2e-12
        checksum_tolerance = max(2e-10, elements * 2e-13)
    row["parity_tolerance"] = parity_tolerance
    row["checksum_tolerance"] = checksum_tolerance
    row["parity_ok"] = (
        row["max_abs_error"] <= parity_tolerance
        and row["checksum_delta"] <= checksum_tolerance
    )
    row["million_elements_per_sec"] = (
        elements / row["median_sec"] / 1_000_000.0
    )
    row["million_primitive_ops_per_sec"] = (
        row["program_steps"] * elements / row["median_sec"] / 1_000_000.0
    )
    gpu_median_sec = row.get("gpu_median_sec")
    if gpu_median_sec is not None and gpu_median_sec > 0.0:
        row["gpu_million_primitive_ops_per_sec"] = (
            row["program_steps"] * elements / gpu_median_sec / 1_000_000.0
        )
        row["host_overhead_sec"] = row["median_sec"] - gpu_median_sec
    else:
        row["gpu_million_primitive_ops_per_sec"] = np.nan
        row["host_overhead_sec"] = np.nan


def run_comparison(
    *,
    backends=("numpy", "torch", "c"),
    elements=262144,
    warmup=10,
    repeats=50,
    torch_device="cpu",
    include_glsl=False,
    include_nodus=True,
    nodus_executable: Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Run each target serially and return parity/timing rows."""
    if elements <= 0 or warmup < 0 or repeats <= 0:
        raise ValueError("elements/repeats must be positive and warmup nonnegative")

    expected = 1.0 / (1.0 + np.exp(-_input_values(elements)))
    reference_checksum = float(expected.sum(dtype=np.float64))
    captured = _capture_sigmoid_program()
    rows: list[dict] = []

    def run_one(label: str, benchmark: Callable[[], dict]) -> None:
        if progress_callback is not None:
            progress_callback("start", label, None)
        row = benchmark()
        _annotate_result(
            row,
            elements=elements,
            reference_checksum=reference_checksum,
        )
        rows.append(row)
        if progress_callback is not None:
            progress_callback("complete", label, row)

    for backend in backends:
        run_one(
            backend,
            lambda backend=backend: _benchmark_abstracttensor_backend(
                backend,
                captured=captured,
                elements=elements,
                warmup=warmup,
                repeats=repeats,
                torch_device=torch_device,
            ),
        )
    if include_glsl:
        run_one(
            "glsl",
            lambda: _benchmark_glsl(
                captured=captured,
                elements=elements,
                warmup=warmup,
                repeats=repeats,
            ),
        )
    if include_nodus:
        executable = _find_nodus_executable(nodus_executable)
        run_one(
            "nodus_calculator",
            lambda: _benchmark_nodus(
                executable,
                captured=captured,
                elements=elements,
                warmup=warmup,
                repeats=repeats,
            ),
        )

    return pd.DataFrame(rows)


def _row_label(row: pd.Series) -> str:
    backend = str(row["backend"])
    device = str(row.get("device", "cpu"))
    if backend == "torch":
        return f"Torch ({device})"
    if backend == "glsl":
        renderer = device.split("/")[0]
        return f"GLSL ({renderer})"
    if backend == "nodus_calculator":
        return "Nodus prepared"
    return backend.capitalize()


def plot_comparison(
    table: pd.DataFrame,
    output: Path,
    *,
    show: bool = False,
) -> Path:
    """Render steady-state, GPU, setup, transfer, and throughput timings."""
    try:
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for benchmark graphs; install it or pass "
            "--no-plot"
        ) from exc

    labels = [_row_label(row) for _, row in table.iterrows()]
    x = np.arange(len(table), dtype=np.float64)
    palette = {
        "numpy": "#4C78A8",
        "torch": "#EE854A",
        "c": "#55A868",
        "glsl": "#8172B3",
        "nodus_calculator": "#C44E52",
    }
    colors = [palette.get(str(name), "#777777") for name in table["backend"]]

    figure, axes = plt.subplots(2, 2, figsize=(16, 10))
    steady, setup, throughput, exact = axes.flat

    medians = table["median_sec"].to_numpy(dtype=float)
    minimums = table["min_sec"].to_numpy(dtype=float)
    maximums = table["max_sec"].to_numpy(dtype=float)
    steady.bar(x, medians, color=colors, alpha=0.82, label="host median")
    steady.errorbar(
        x,
        medians,
        yerr=np.vstack((medians - minimums, maximums - medians)),
        fmt="none",
        ecolor="#222222",
        capsize=4,
        linewidth=1.2,
        label="host min–max",
    )
    gpu_median_column = table.get(
        "gpu_median_sec",
        pd.Series(np.nan, index=table.index, dtype=float),
    )
    gpu_mask = gpu_median_column.notna().to_numpy()
    if gpu_mask.any():
        gpu_rows = table.loc[gpu_mask]
        gpu_x = x[gpu_mask]
        gpu_medians = gpu_rows["gpu_median_sec"].to_numpy(dtype=float)
        gpu_mins = gpu_rows["gpu_min_sec"].to_numpy(dtype=float)
        gpu_maxes = gpu_rows["gpu_max_sec"].to_numpy(dtype=float)
        steady.errorbar(
            gpu_x,
            gpu_medians,
            yerr=np.vstack((gpu_medians - gpu_mins, gpu_maxes - gpu_medians)),
            fmt="D",
            color="#111111",
            markerfacecolor="#FFD166",
            capsize=4,
            label="GLSL device min/median/max",
        )
    steady.set_yscale("log")
    steady.set_xticks(x, labels, rotation=18, ha="right")
    steady.set_ylabel("seconds per completed run (log scale)")
    steady.set_title("Steady-state execution")
    steady.grid(axis="y", which="both", alpha=0.22)
    steady.legend(fontsize=8)

    phase_specs = (
        ("setup_sec", "setup total"),
        ("context_setup_sec", "GL context"),
        ("compile_first_dispatch_sec", "compile + first"),
        ("upload_sec", "input upload"),
        ("output_allocation_sec", "output allocation"),
        ("readback_sec", "readback"),
    )
    width = 0.12
    offsets = (np.arange(len(phase_specs)) - (len(phase_specs) - 1) / 2) * width
    for offset, (column, label) in zip(offsets, phase_specs):
        if column not in table:
            continue
        values = table[column].to_numpy(dtype=float)
        mask = np.isfinite(values) & (values > 0.0)
        if mask.any():
            setup.bar(x[mask] + offset, values[mask], width=width, label=label)
    setup.set_yscale("log")
    setup.set_xticks(x, labels, rotation=18, ha="right")
    setup.set_ylabel("seconds (log scale)")
    setup.set_title("One-time and transfer costs")
    setup.grid(axis="y", which="both", alpha=0.22)
    setup.legend(fontsize=8, ncols=2)

    host_throughput = table["million_primitive_ops_per_sec"].to_numpy(dtype=float)
    throughput.bar(
        x - 0.18,
        host_throughput,
        width=0.36,
        color=colors,
        alpha=0.82,
        label="completed host-visible",
    )
    gpu_throughput = table[
        "gpu_million_primitive_ops_per_sec"
    ].to_numpy(dtype=float)
    gpu_throughput_mask = np.isfinite(gpu_throughput)
    if gpu_throughput_mask.any():
        throughput.bar(
            x[gpu_throughput_mask] + 0.18,
            gpu_throughput[gpu_throughput_mask],
            width=0.36,
            color="#FFD166",
            edgecolor="#222222",
            label="GLSL device timer",
        )
    throughput.set_yscale("log")
    throughput.set_xticks(x, labels, rotation=18, ha="right")
    throughput.set_ylabel("million primitive operations / second")
    throughput.set_title("Throughput from the same captured program")
    throughput.grid(axis="y", which="both", alpha=0.22)
    throughput.legend(fontsize=8)

    timing_columns = (
        ("setup_sec", "setup"),
        ("compile_first_dispatch_sec", "compile"),
        ("upload_sec", "upload"),
        ("output_allocation_sec", "out alloc"),
        ("readback_sec", "readback"),
        ("min_sec", "host min"),
        ("median_sec", "host median"),
        ("max_sec", "host max"),
        ("gpu_median_sec", "GPU median"),
    )

    def milliseconds(value: object) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "—"
        return "—" if not np.isfinite(numeric) else f"{numeric * 1_000:.3f}"

    cell_text = [
        [milliseconds(row.get(column)) for column, _ in timing_columns]
        for _, row in table.iterrows()
    ]
    exact.axis("off")
    timing_table = exact.table(
        cellText=cell_text,
        rowLabels=labels,
        colLabels=[label for _, label in timing_columns],
        loc="center",
        cellLoc="right",
    )
    timing_table.auto_set_font_size(False)
    timing_table.set_fontsize(7.5)
    timing_table.scale(1.0, 1.45)
    exact.set_title("Exact timing summary (milliseconds)", pad=18)

    elements = int(table["elements"].iloc[0])
    figure.suptitle(
        f"AbstractTensor / Nodus execution benchmark — {elements:,} values",
        fontsize=16,
    )
    figure.text(
        0.5,
        0.012,
        "Backends run serially. Host timings are synchronized. GLSL is float32; "
        "the other displayed paths are float64. GPU timer excludes Python/driver "
        "submission overhead.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.96))
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
    return output


def _print_progress(event: str, label: str, row: dict | None) -> None:
    if event == "start":
        print(f"\n[{label}] running alone...", flush=True)
        return
    if row is None:
        return
    gpu = row.get("gpu_median_sec")
    gpu_text = f", GPU={gpu * 1_000:.3f} ms" if gpu is not None else ""
    print(
        f"[{label}] median={row['median_sec'] * 1_000:.3f} ms"
        f"{gpu_text}, parity={'ok' if row['parity_ok'] else 'FAILED'}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", default="numpy,torch,c")
    parser.add_argument("--elements", type=int, default=262144)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--torch-device", default="cpu")
    parser.add_argument("--nodus-executable", type=Path)
    parser.add_argument("--without-glsl", action="store_true")
    parser.add_argument("--without-nodus", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("benchmark_timing.png"),
        help="PNG timing graph destination (default: benchmark_timing.png)",
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="open the graph after saving it",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    table = run_comparison(
        backends=tuple(filter(None, args.backends.split(","))),
        elements=args.elements,
        warmup=args.warmup,
        repeats=args.repeats,
        torch_device=args.torch_device,
        include_glsl=not args.without_glsl,
        include_nodus=not args.without_nodus,
        nodus_executable=args.nodus_executable,
        progress_callback=None if args.quiet else _print_progress,
    )
    columns = [
        "backend",
        "device",
        "execution",
        "elements",
        "setup_sec",
        "context_setup_sec",
        "compile_first_dispatch_sec",
        "median_sec",
        "min_sec",
        "max_sec",
        "million_elements_per_sec",
        "million_primitive_ops_per_sec",
        "gpu_median_sec",
        "gpu_million_primitive_ops_per_sec",
        "host_overhead_sec",
        "upload_sec",
        "output_allocation_sec",
        "readback_sec",
        "max_abs_error",
        "checksum_delta",
        "parity_ok",
    ]
    print(table.reindex(columns=columns).to_string(index=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
        print(args.output.resolve())
    if not args.no_plot:
        graph = plot_comparison(table, args.plot, show=args.show_plot)
        print(graph)
    if not table["parity_ok"].all():
        raise RuntimeError("one or more backend parity checks failed")


if __name__ == "__main__":
    main()
