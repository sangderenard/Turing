"""Profile the CI-pinned compiled Jacobi EIGH and installed ROT route."""
from __future__ import annotations

import argparse
import ast
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _test_kernel() -> str:
    tree = ast.parse(
        (ROOT / "tests" / "test_compiled_linalg.py").read_text(encoding="utf-8")
    )
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "JACOBI_EIGH"
            for target in node.targets
        ):
            return str(ast.literal_eval(node.value))
    raise RuntimeError("tests/test_compiled_linalg.py no longer defines JACOBI_EIGH")


def _median(operation, repetitions):
    samples = []
    result = None
    for _ in range(max(1, repetitions)):
        started = time.perf_counter(); result = operation()
        samples.append(time.perf_counter() - started)
    return float(statistics.median(samples)), result


def _matrix(size: int):
    rng = np.random.default_rng(100 + size)
    q, _r = np.linalg.qr(rng.standard_normal((size, size)))
    spectrum = np.linspace(1.0, 2.0, size)
    return q @ np.diag(spectrum) @ q.T, spectrum


def _errors(matrix, spectrum, values, vectors):
    values = np.asarray(values, dtype=float).reshape(-1)
    vectors = np.asarray(vectors, dtype=float).reshape(matrix.shape)
    return {
        "eigenvalue_max_abs": float(np.max(np.abs(np.sort(values) - spectrum))),
        "orthogonality_max_abs": float(np.max(np.abs(
            vectors.T @ vectors - np.eye(len(values))
        ))),
        "residual_max_abs": float(np.max(np.abs(
            matrix @ vectors - vectors @ np.diag(values)
        ))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=(3, 4, 6, 8))
    parser.add_argument("--sweeps", type=int, default=12)
    parser.add_argument("--tol", type=float, default=1.0e-15)
    parser.add_argument("--contract", default="fast")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--root", type=Path, default=ROOT / "build" / "eigh-profile")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    from src.compiler.ssa_llvm_backend import prepare_artifact_execution
    from tools.benchmark_blas_vs_numpy import compile_kernel

    source = _test_kernel()
    started = time.perf_counter()
    native, identifiers, _outputs, _returns = compile_kernel(
        "jacobi_eigh", source, args.contract, args.root / "compiled_whole",
    )
    compile_seconds = time.perf_counter() - started
    rows = []

    for size in args.sizes:
        matrix, spectrum = _matrix(size)
        base = {
            identifiers["a"]: matrix.reshape(-1),
            identifiers["v"]: np.zeros(size * size),
            identifiers["w"]: np.zeros(size),
            identifiers["n"]: size,
            identifiers["sweeps"]: args.sweeps,
            identifiers["eps"]: args.tol,
        }
        started = time.perf_counter()
        execution = prepare_artifact_execution(native, {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in base.items()
        })
        execution.run()
        first_seconds = time.perf_counter() - started
        errors = _errors(
            matrix, spectrum,
            execution.buffers[identifiers["w"]],
            execution.buffers[identifiers["v"]],
        )

        def relaunch():
            fresh = prepare_artifact_execution(native, {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in base.items()
            })
            fresh.run()

        a_buffer = np.asarray(execution.buffers[identifiers["a"]])
        v_buffer = np.asarray(execution.buffers[identifiers["v"]])
        w_buffer = np.asarray(execution.buffers[identifiers["w"]])
        def compute():
            a_buffer[...] = matrix.reshape(-1)
            v_buffer[...] = 0.0
            w_buffer[...] = 0.0
            execution.run()

        relaunch_seconds, _ = _median(relaunch, args.repetitions)
        compute_seconds, _ = _median(compute, args.repetitions)
        rows.append({
            "algorithm": "compiled-whole-jacobi-test-kernel",
            "contract": args.contract,
            "size": size, "compile_seconds": compile_seconds,
            "first_call_seconds": first_seconds,
            "relaunch_median_seconds": relaunch_seconds,
            "warm_median_seconds": compute_seconds,
            "errors": errors,
        })

        from src.common.tensors.abstraction import AbstractTensor
        from src.common.tensors.abstraction_methods import eigen as eigen_module
        eigh = eigen_module.eigh
        tensor = AbstractTensor.get_tensor(matrix.tolist())
        for method in ("jacobi", "blas"):
            def operation(method=method):
                return eigh(
                    tensor, sweeps=args.sweeps, tol=args.tol,
                    sort=True, method=method,
                )
            started = time.perf_counter(); first_result = operation()
            method_first = time.perf_counter() - started
            warm_seconds, result = _median(operation, args.repetitions)
            errors = _errors(
                matrix, spectrum,
                np.asarray(result[0], dtype=float),
                np.asarray(result[1], dtype=float),
            )
            row = {
                "algorithm": f"abstract-tensor-{method}",
                "size": size, "compile_seconds": None,
                "first_call_seconds": method_first,
                "relaunch_median_seconds": None,
                "warm_median_seconds": warm_seconds,
                "errors": errors,
                "contract": "abstract-tensor-default",
            }
            if method == "blas":
                core_n = max(9, size)
                variant, route = eigen_module._rot_launcher().bank.select(
                    "rot", sizes={"n": core_n}, contract=None,
                    compile_missing=False,
                )
                row["contract"] = variant.contract or "develop"
                row["native_rotation"] = {
                    "route": route,
                    "module_key": variant.key,
                    "problem_n": size,
                    "prebaked_n": core_n,
                    "parameter_ids_by_name": variant.id_by_name,
                }
            rows.append(row)

    for row in rows:
        row.update({
            "status": (
                "timed" if max(row["errors"].values()) <= 1.0e-9 else "wrong"
            ),
            "sweeps": args.sweeps, "tolerance": args.tol,
            "spectrum": "orthogonal Q @ diag(linspace(1,2,n)) @ Q.T",
        })
    report = {
        "schema": "turing.eigh-profile.v1",
        "source_authority": "tests/test_compiled_linalg.py::JACOBI_EIGH",
        "rows": rows,
        "backward_policy": (
            "no opaque EIGH backward is installed; the BLAS route refuses "
            "autograd semantics and the Jacobi route retains primitive tape ops"
        ),
    }
    output = args.output or args.root / "profile.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for row in rows:
        print(f"n={row['size']:<3} {row['algorithm']:<36} "
              f"{row['warm_median_seconds']*1e3:9.3f} ms "
              f"err={max(row['errors'].values()):.3e} {row['status']}")
    print(f"wrote {output}")
    return 0 if all(row["status"] == "timed" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
