"""Correctness-gated BLAS rows from one repository-SSA authority."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _median(operation, repetitions: int) -> float:
    samples = []
    for _ in range(max(1, repetitions)):
        started = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - started)
    return float(statistics.median(samples))


def _identity(module) -> str:
    payload = {
        name: [
            [instruction.op, [int(value.id) for value in instruction.args],
             None if instruction.res is None else int(instruction.res.id)]
            for block in function.blocks.values()
            for instruction in block.instrs
        ]
        for name, function in sorted(module.functions.items())
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--contract", default="fast")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--ssa-step-limit", type=int, default=0,
        help="SSA diagnostic guard (0 derives a finite GEMM-sized limit)",
    )
    parser.add_argument("--root", type=Path, default=ROOT / "build" / "blas-bakeoff")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.kernel_bank import blas_kernel_specs, _specialize_source
    from src.compiler.work_contract import set_active_contract

    spec = blas_kernel_specs()["gemm"]
    sizes = {"m": args.size, "n": args.size, "k": args.size}
    source = _specialize_source(spec, sizes)
    source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    lower_started = time.perf_counter()
    set_active_contract(args.contract)
    dll_directory = None
    try:
        module, outputs, _exports = lower_ast_source_to_ssa(
            source, "gemm", name="backend_bakeoff",
        )
        lower_seconds = time.perf_counter() - lower_started
        entry = "backend_bakeoff__gemm"
        function = module.functions[entry]
        identifiers = {
            str(name): int(identifier)
            for name, identifier in function.metadata["parameter_names"]
        }
        ssa_sha = _identity(module)

        rng = np.random.default_rng(23)
        a = rng.standard_normal(args.size * args.size)
        b = rng.standard_normal(args.size * args.size)
        c = rng.standard_normal(args.size * args.size)
        alpha, beta = 1.7, 0.3
        expected = (
            alpha * (a.reshape(args.size, args.size)
                     @ b.reshape(args.size, args.size)).reshape(-1)
            + beta * c
        )
        common = {
            "source_sha256": source_sha,
            "ssa_sha256": ssa_sha,
            "contract": args.contract,
            "sizes": sizes,
            "lower_seconds": lower_seconds,
            "machine": platform.platform(),
            "python": sys.version.split()[0],
        }
        rows = []

        def record(backend, status, **fields):
            rows.append({"backend": backend, "status": status, **common, **fields})

        # Repository SSA interpreter: executable authority, no compilation.
        try:
            from src.compiler.ssa_reference_evaluator import SSAReferenceEvaluator

            def ssa_call():
                feeds = {
                    identifiers["A"]: a.copy(), identifiers["B"]: b.copy(),
                    identifiers["C"]: c.copy(), identifiers["alpha"]: alpha,
                    identifiers["beta"]: beta,
                }
                # This is a known finite m*n*k loop nest. Raise the
                # diagnostic interpreter's runaway guard in proportion to
                # that work; compiled backends never use this limit.
                step_limit = args.ssa_step_limit or max(
                    5_000_000, 64 * args.size ** 3,
                )
                result = SSAReferenceEvaluator(
                    module, step_limit=step_limit,
                ).run(entry, feeds)
                return np.asarray(result.values[identifiers["C"]])

            started = time.perf_counter(); produced = ssa_call()
            first = time.perf_counter() - started
            error = float(np.max(np.abs(produced - expected)))
            if error > 1.0e-9:
                record("ssa-reference", "wrong", worst_abs_error=error)
            else:
                record(
                    "ssa-reference", "timed", compile_seconds=0.0,
                    first_launch_seconds=first,
                    relaunch_median_seconds=_median(ssa_call, 1),
                    compute_median_seconds=None, repetitions=1,
                    worst_abs_error=error,
                )
        except Exception as error:
            record("ssa-reference", "error", reason=f"{type(error).__name__}: {error}")

        # Optimizing LLVM: same module and entry, reusable execution buffers.
        try:
            from src.compiler.ssa_llvm_backend import (
                compile_artifact, emit_ssa_function_to_llvm,
                prepare_artifact_execution,
            )
            started = time.perf_counter()
            artifact = emit_ssa_function_to_llvm(module, entry)
            if not artifact.complete:
                raise RuntimeError("; ".join(item.reason for item in artifact.shortfalls[:3]))
            native = compile_artifact(artifact, directory=args.root / "llvm")
            compile_seconds = time.perf_counter() - started
            base_feeds = {
                identifiers["A"]: a.copy(), identifiers["B"]: b.copy(),
                identifiers["C"]: c.copy(), identifiers["alpha"]: alpha,
                identifiers["beta"]: beta,
            }
            started = time.perf_counter()
            execution = prepare_artifact_execution(native, base_feeds)
            execution.run()
            first = time.perf_counter() - started
            produced = np.asarray(execution.buffers[identifiers["C"]]).copy()
            error = float(np.max(np.abs(produced - expected)))

            def llvm_relaunch():
                fresh = prepare_artifact_execution(native, {
                    key: (value.copy() if isinstance(value, np.ndarray) else value)
                    for key, value in base_feeds.items()
                })
                fresh.run()

            c_buffer = np.asarray(execution.buffers[identifiers["C"]])
            def llvm_compute():
                c_buffer[...] = c
                execution.run()

            record(
                "llvm", "timed" if error <= 1.0e-9 else "wrong",
                compile_seconds=compile_seconds,
                first_launch_seconds=first,
                relaunch_median_seconds=_median(llvm_relaunch, args.repetitions),
                compute_median_seconds=_median(llvm_compute, args.repetitions),
                repetitions=args.repetitions, worst_abs_error=error,
                toolchain="ziglang clang",
            )
        except Exception as error:
            record("llvm", "error", reason=f"{type(error).__name__}: {error}")

        # Fortran: emit and compile the SAME IRModule, bind from its API record.
        try:
            from src.compiler.ssa_fortran_backend import (
                compile_module, emit_module, fortran_compiler,
            )
            started = time.perf_counter()
            emitted = emit_module(
                module, name="backend_bakeoff_fortran",
                outputs={entry: outputs[entry]}, extra_roots=(entry,),
                progress=lambda _message: None,
            )
            if not emitted.complete:
                raise RuntimeError(emitted.shortfalls[0].format())
            library_path = compile_module(
                emitted, directory=args.root / "fortran", standalone=False,
            )
            compile_seconds = time.perf_counter() - started
            compiler = fortran_compiler()
            if os.name == "nt":
                dll_directory = os.add_dll_directory(str(Path(compiler).parent))
            dll = ctypes.CDLL(str(library_path))
            endpoint = emitted.api.entry_point(entry)
            native_entry = getattr(dll, endpoint.symbol)
            pointer = ctypes.POINTER(ctypes.c_double)
            values = {"A": a, "B": b, "C": c, "alpha": alpha, "beta": beta}
            native_entry.argtypes = [
                ctypes.c_int32 if parameter.role == "extent"
                else pointer if parameter.passing == "reference"
                else ctypes.c_double
                for parameter in endpoint.parameters
            ]

            def fortran_args(c_value):
                bound = {**values, "C": c_value}
                return [
                    args.size * args.size if parameter.role == "extent"
                    else bound[parameter.source_name].ctypes.data_as(pointer)
                    if parameter.passing == "reference"
                    else bound[parameter.source_name]
                    for parameter in endpoint.parameters
                ]

            first_c = c.copy()
            started = time.perf_counter(); native_entry(*fortran_args(first_c))
            first = time.perf_counter() - started
            error = float(np.max(np.abs(first_c - expected)))
            def fortran_relaunch():
                fresh = c.copy(); native_entry(*fortran_args(fresh))
            steady_c = c.copy(); steady_args = fortran_args(steady_c)
            def fortran_compute():
                steady_c[...] = c; native_entry(*steady_args)
            record(
                "fortran", "timed" if error <= 1.0e-9 else "wrong",
                compile_seconds=compile_seconds,
                first_launch_seconds=first,
                relaunch_median_seconds=_median(fortran_relaunch, args.repetitions),
                compute_median_seconds=_median(fortran_compute, args.repetitions),
                repetitions=args.repetitions, worst_abs_error=error,
                toolchain=str(compiler),
            )
        except Exception as error:
            record("fortran", "unsupported", reason=f"{type(error).__name__}: {error}")

        # The direct C emitter is callable but intentionally scalar/one-block.
        try:
            from src.compiler.ssa_c_backend import emit_ssa_function_to_c
            started = time.perf_counter()
            artifact = emit_ssa_function_to_c(module, entry, entry_name="bakeoff_c")
            elapsed = time.perf_counter() - started
            if artifact.complete:
                record("direct-c", "unsupported", emission_seconds=elapsed,
                       reason="array/control ABI has no comparable runner")
            else:
                record("direct-c", "unsupported", emission_seconds=elapsed,
                       reason="; ".join(x.reason for x in artifact.shortfalls[:3]))
        except Exception as error:
            record("direct-c", "error", reason=f"{type(error).__name__}: {error}")

        # WebGPU can emit this SSA vocabulary, but this CLI has no synchronized
        # device runner for the repository array ABI; never report emit time as compute.
        try:
            from src.compiler.ssa_webgpu_backend import emit_module as emit_webgpu
            started = time.perf_counter()
            emitted = emit_webgpu(module, name=entry, outputs={entry: outputs[entry]},
                                  count=args.size * args.size)
            elapsed = time.perf_counter() - started
            record(
                "webgpu", "unsupported", emission_seconds=elapsed,
                reason=("no synchronized executable runner for this SSA ABI"
                        if emitted.complete else emitted.shortfalls[0].format()),
            )
        except Exception as error:
            record("webgpu", "unsupported", reason=f"{type(error).__name__}: {error}")

        report = {
            "schema": "turing.blas-backend-bakeoff.v1",
            "oracle": "numpy", "rows": rows,
        }
        output = args.output or args.root / "bakeoff.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        for row in rows:
            compute = row.get("compute_median_seconds")
            rate = (
                2 * args.size ** 3 / compute / 1.0e9
                if compute else None
            )
            print(f"{row['backend']:<14} {row['status']:<11} "
                  f"err={row.get('worst_abs_error', '-')} "
                  f"compute={compute if compute is not None else '-'} "
                  f"GF/s={rate if rate is not None else '-'}")
            if row.get("reason"): print(" " * 16 + row["reason"])
        print(f"wrote {output}")
        return 0 if all(row["status"] != "wrong" for row in rows) else 1
    finally:
        if dll_directory is not None:
            dll_directory.close()
        set_active_contract(None)


if __name__ == "__main__":
    raise SystemExit(main())
