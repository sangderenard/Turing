"""Bounded self-compiled descriptor unit: parity, overhead and real serial use.

No allocation tracing, alternate compiler, package installation or N4 run.
The parent terminates this entire probe after 55 seconds. Native tests use
their existing ten-second child boundary. All receipts print inline.
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main(directory):
    import json
    from statistics import median
    from time import perf_counter
    from src.compiler import fortran_c_shell
    from src.compiler.native_compiler_accelerators import (
        BROADCAST_SHAPES, CompilerAcceleratorRegistry, _python_broadcast_shapes,
        compile_and_register_broadcast_shapes_accelerator,
        compiler_accelerators,
    )
    from tools.compiler_probes.probe_dt_unit_interfaces import main as dt_probe

    registry = CompilerAcceleratorRegistry()
    registry.register_fallback(BROADCAST_SHAPES, _python_broadcast_shapes, name="probe:eager-shapes")
    started = perf_counter()
    provider = compile_and_register_broadcast_shapes_accelerator(
        Path(directory) / "shape-unit", registry=registry, progress=lambda _message: None,
    )
    print("SHAPE_BUILD_SECONDS=" + str(perf_counter() - started), flush=True)

    def measure(run, pairs, repetitions):
        expected = _python_broadcast_shapes(pairs)
        assert run(pairs) == expected
        samples = []
        for _ in range(3):
            started = perf_counter()
            for _ in range(repetitions):
                run(pairs)
            samples.append((perf_counter() - started) / repetitions)
        return median(samples)

    for count, rank, repetitions in ((1, 2, 100), (256, 2, 20), (4096, 8, 4), (4096, 32, 2)):
        pairs = [((2,) * rank, (1,) * rank)] * count
        eager = measure(_python_broadcast_shapes, pairs, repetitions)
        native = measure(provider.run, pairs, repetitions)
        print("SHAPE_ABI_BENCHMARK=" + json.dumps({
            "rows": count, "rank": rank, "eager_seconds": eager,
            "native_seconds": native, "speedup": eager / native,
            "includes": "alignment, packing, ABI allocation, execution, result unpacking",
        }), flush=True)

    if "--benchmark-only" in sys.argv:
        return

    # Count the exact admitted operator calls in the ordinary serial compiler,
    # not all descriptor queries. The original owner still publishes facts.
    canonical = fortran_c_shell.lower_ast_source_to_ssa
    active = {"tier": None, "calls": 0, "shape_seconds": 0.0, "lower_seconds": 0.0}

    def observed_lower(*args, **kwargs):
        kwargs["progress"] = lambda _message: None
        started = perf_counter()
        result = canonical(*args, **kwargs)
        active["lower_seconds"] += perf_counter() - started
        return result

    def observed_shapes(pairs):
        pairs = tuple(pairs)
        active["calls"] += len(pairs)
        started = perf_counter()
        try:
            return (provider.run(pairs) if active["tier"] == "native"
                    else _python_broadcast_shapes(pairs))
        finally:
            active["shape_seconds"] += perf_counter() - started

    fortran_c_shell.lower_ast_source_to_ssa = observed_lower
    original_native = compiler_accelerators._native.get(BROADCAST_SHAPES)
    try:
        for tier in ("python", "native", "python"):
            active.update(tier=tier, calls=0, shape_seconds=0.0, lower_seconds=0.0)
            compiler_accelerators.register_native(
                BROADCAST_SHAPES, observed_shapes, name="probe:observed-shapes",
                library_path=provider.library_path,
            )
            started = perf_counter()
            dt_probe(Path(directory) / f"serial-{tier}-{int(started * 1000)}", proposal=True, serial=True)
            print("SERIAL_DT_SHAPE_RECEIPT=" + json.dumps({
                **active, "total_seconds": perf_counter() - started,
                "native_outputs": [0.9, 0.25], "partitioned": False,
            }), flush=True)
    finally:
        fortran_c_shell.lower_ast_source_to_ssa = canonical
        compiler_accelerators.unregister_native(BROADCAST_SHAPES)
        if original_native is not None:
            compiler_accelerators._native[BROADCAST_SHAPES] = original_native


if __name__ == "__main__":
    if "--child" in sys.argv:
        main(sys.argv[1])
    else:
        raise SystemExit(subprocess.run(
            [sys.executable, "-u", __file__, sys.argv[1], "--child",
             *(["--benchmark-only"] if "--benchmark-only" in sys.argv else [])], timeout=55,
        ).returncode)
