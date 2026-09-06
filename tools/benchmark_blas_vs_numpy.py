"""Profile compiled AbstractTensor.blas kernels against NumPy, two ways:

  DIRECT   -- the compiled kernel's own steady-state ``execution.run()``
              cost vs. the equivalent raw NumPy call, at matching sizes.
  OPERATOR -- the AbstractTensor-level operators that are BLAS-shaped and
              are candidate future call-sites for these kernels (``AT.dot``,
              ``AT @ AT``, elementwise ``alpha * x + y``), run eagerly on
              the NumPy backend, vs. the same raw NumPy call. This is the
              same measurement the eigh compilation effort made
              (``docs/FUNCTION_TO_DEPLOYMENT_HANDOFF.md`` section 1: "~89%
              of the time is AbstractTensor dispatch, not arithmetic") --
              it quantifies the dispatch tax these operators currently pay,
              which is the argument for eventually routing them through a
              compiled kernel.

Timing methodology matches ``tools/bench_native_step.py``: compile/prepare
once, warm up once, then time many repeated calls and divide by the
repetition count -- steady state, not first-call (which includes lazy
imports, cache population, etc).

Run:  python tools/benchmark_blas_vs_numpy.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.common.tensors.blas import KERNELS


def _time(fn, reps: int, *, warmup: int = 1) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def compile_kernel(name: str, source: str, contract: str | None, build_dir: Path,
                    *, tag_suffix: str = ""):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_llvm_backend import compile_artifact, emit_ssa_function_to_llvm
    from src.compiler.work_contract import set_active_contract

    # Windows keeps a compiled DLL locked while it is loaded (execution.run()
    # dlopen's it); reusing one output path across sizes makes the SECOND
    # compile fail with an lld-link write error against a file the process
    # still has open (the exact trap tools/bench_native_step.py's own
    # comments warn about). Every distinct (kernel, contract, size) gets
    # its own build subdirectory.
    tag = f"blas_bench_{name}_{contract or 'develop'}{tag_suffix}"
    set_active_contract(contract)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            module, outputs, _exports = lower_ast_source_to_ssa(source, name, name=tag)
        entrypoint = f"{tag}__{name}"
        fn = module.functions[entrypoint]
        artifact = emit_ssa_function_to_llvm(module, entrypoint)
        if not artifact.complete:
            raise RuntimeError(
                f"{name}: {len(artifact.shortfalls)} shortfall(s): "
                + "; ".join(s.reason[:100] for s in artifact.shortfalls[:3])
            )
        native = compile_artifact(artifact, directory=build_dir / tag)
    finally:
        set_active_contract(None)
    table = dict(fn.metadata.get("parameter_names") or fn.metadata.get("value_names") or ())
    output_names = {p for p, _ in (fn.metadata.get("named_outputs") or ())}
    ret_ids = tuple(int(v.id) for v in (outputs.get(entrypoint) or ()))
    return native, {str(k): int(v) for k, v in table.items()}, output_names, ret_ids


def flops_for(name: str, sizes: dict) -> int:
    if name in ("scal",):
        return sizes["n"]
    if name in ("axpy",):
        return 2 * sizes["n"]
    if name in ("dot",):
        return 2 * sizes["n"]
    if name == "gemv":
        return 2 * sizes["m"] * sizes["n"]
    if name == "gemm":
        return 2 * sizes["m"] * sizes["n"] * sizes["k"]
    raise ValueError(name)


def run_direct(name: str, source: str, arity, sizes: dict, contract, build_dir, reps: int,
                *, tag_suffix: str = ""):
    from src.compiler.ssa_llvm_backend import prepare_artifact_execution

    rng = np.random.default_rng(7)
    m, n, k = sizes.get("m", sizes.get("n", 4)), sizes["n"], sizes.get("k", 4)
    alpha, beta = 1.7, 0.6

    sample = {}
    for p in arity:
        if p == "m":
            sample[p] = m
        elif p == "n":
            sample[p] = n
        elif p == "k":
            sample[p] = k
        elif p == "alpha":
            sample[p] = alpha
        elif p == "beta":
            sample[p] = beta
        elif p == "A":
            rows = m if "m" in arity else n
            cols = k if "k" in arity else n
            sample[p] = rng.uniform(-1, 1, size=rows * cols)
        elif p == "B":
            sample[p] = rng.uniform(-1, 1, size=k * n)
        elif p == "C":
            sample[p] = rng.uniform(-1, 1, size=m * n)
        elif p == "y":
            sample[p] = rng.uniform(-1, 1, size=(m if "m" in arity else n))
        elif p == "x":
            sample[p] = rng.uniform(-1, 1, size=n)

    native, id_by_name, output_names, ret_ids = compile_kernel(
        name, source, contract, build_dir, tag_suffix=tag_suffix,
    )
    feeds = {id_by_name[p]: sample[p] for p in arity}
    execution = prepare_artifact_execution(native, feeds)
    compiled_time = _time(execution.run, reps)

    # NumPy equivalent, same sizes, same data.
    npx = {p: (np.array(v, dtype=np.float64) if isinstance(v, np.ndarray) else v)
           for p, v in sample.items()}
    if name == "scal":
        fn = lambda: np.multiply(npx["alpha"], npx["x"], out=npx["y"])
    elif name == "axpy":
        fn = lambda: np.add(npx["alpha"] * npx["x"], npx["y"], out=npx["y"])
    elif name == "dot":
        fn = lambda: np.dot(npx["x"], npx["y"])
    elif name == "gemv":
        A2 = npx["A"].reshape(m, n)
        fn = lambda: np.add(
            alpha * (A2 @ npx["x"]), beta * npx["y"], out=npx["y"],
        )
    elif name == "gemm":
        A2 = npx["A"].reshape(m, k)
        B2 = npx["B"].reshape(k, n)
        C2 = npx["C"].reshape(m, n)
        fn = lambda: np.add(alpha * (A2 @ B2), beta * C2, out=C2)
    numpy_time = _time(fn, reps)

    flops = flops_for(name, {"m": m, "n": n, "k": k})
    return compiled_time, numpy_time, flops


def run_operator(name: str, sizes: dict, reps: int):
    """AbstractTensor-level operator dispatch on the NumPy backend, vs raw
    NumPy at the same sizes -- the "operators that should use these
    kernels" comparison."""

    from src.common.tensors.numpy_backend import NumPyTensorOperations
    from src.common.tensors import linalg as AT_linalg

    rng = np.random.default_rng(7)
    m, n, k = sizes.get("m", sizes.get("n", 4)), sizes["n"], sizes.get("k", 4)

    def AT(data):
        return NumPyTensorOperations.get_tensor(data)

    if name in ("scal", "axpy"):
        x = rng.uniform(-1, 1, size=n)
        y = rng.uniform(-1, 1, size=n)
        atx, aty = AT(x.copy()), AT(y.copy())
        alpha = 1.7
        if name == "scal":
            op = lambda: alpha * atx
        else:
            op = lambda: alpha * atx + aty
        npx, npy = x.copy(), y.copy()
        npfn = (lambda: alpha * npx) if name == "scal" else (lambda: alpha * npx + npy)
    elif name == "dot":
        x = rng.uniform(-1, 1, size=n)
        y = rng.uniform(-1, 1, size=n)
        atx, aty = AT(x.copy()), AT(y.copy())
        op = lambda: AT_linalg.dot(atx, aty)
        npfn = lambda: np.dot(x, y)
    elif name in ("gemv", "gemm"):
        A = rng.uniform(-1, 1, size=(m, k if name == "gemm" else n))
        B = (rng.uniform(-1, 1, size=(k, n)) if name == "gemm"
             else rng.uniform(-1, 1, size=n))
        atA, atB = AT(A.copy()), AT(B.copy())
        op = lambda: atA @ atB
        npfn = lambda: A @ B
    else:
        raise ValueError(name)

    operator_time = _time(op, reps)
    numpy_time = _time(npfn, reps)
    return operator_time, numpy_time


LADDER = {
    "scal": [{"n": 256}, {"n": 4096}, {"n": 65536}],
    "axpy": [{"n": 256}, {"n": 4096}, {"n": 65536}],
    "dot": [{"n": 256}, {"n": 4096}, {"n": 65536}],
    "gemv": [{"m": 64, "n": 64}, {"m": 256, "n": 256}, {"m": 1024, "n": 1024}],
    "gemm": [{"m": 64, "n": 64, "k": 64}, {"m": 128, "n": 128, "k": 128},
             {"m": 256, "n": 256, "k": 256}],
}
CONTRACTS_BY_KERNEL = {
    "scal": (None,), "axpy": (None,), "dot": (None,),
    "gemv": (None, "fast"), "gemm": (None, "fast"),
}
REPS_BY_KERNEL = {
    "scal": 200, "axpy": 200, "dot": 200, "gemv": 50, "gemm": 10,
}


def main() -> int:
    build_dir = ROOT / "build" / "blas-bench"
    build_dir.mkdir(parents=True, exist_ok=True)
    by_name = {entry[1]: entry for entry in KERNELS}

    print("=" * 100)
    print("DIRECT: compiled kernel vs. raw NumPy (steady-state execution.run(), same sizes)")
    print("=" * 100)
    header = (f"{'kernel':<6} {'contract':<8} {'size':<22} {'compiled ms':>12} "
              f"{'numpy ms':>10} {'compiled GF/s':>14} {'numpy GF/s':>11} {'ratio':>8}")
    print(header)
    for _level, name, source, _reference, arity in KERNELS:
        reps = REPS_BY_KERNEL[name]
        for sizes in LADDER[name]:
            for contract in CONTRACTS_BY_KERNEL[name]:
                size_suffix = "_" + "_".join(f"{k}{v}" for k, v in sorted(sizes.items()))
                try:
                    compiled_t, numpy_t, flops = run_direct(
                        name, source, arity, sizes, contract, build_dir, reps,
                        tag_suffix=size_suffix,
                    )
                except Exception as error:
                    print(f"{name:<6} {contract or 'develop':<8} {str(sizes):<22} "
                          f"FAILED: {type(error).__name__}: {str(error)[:80]}")
                    continue
                size_str = ",".join(f"{k}={v}" for k, v in sizes.items())
                compiled_gflops = flops / compiled_t / 1e9
                numpy_gflops = flops / numpy_t / 1e9
                ratio = numpy_t / compiled_t
                print(f"{name:<6} {contract or 'develop':<8} {size_str:<22} "
                      f"{compiled_t*1e3:12.4f} {numpy_t*1e3:10.4f} "
                      f"{compiled_gflops:14.3f} {numpy_gflops:11.3f} "
                      f"{ratio:7.3f}x")

    print()
    print("=" * 100)
    print("OPERATOR: AbstractTensor op (NumPy backend, eager) vs. raw NumPy, same sizes")
    print("=" * 100)
    print(f"{'kernel':<6} {'size':<22} {'AT op ms':>10} {'numpy ms':>10} "
          f"{'AT/numpy overhead':>18}")
    for name in ("scal", "axpy", "dot", "gemv", "gemm"):
        reps = REPS_BY_KERNEL[name]
        for sizes in LADDER[name]:
            try:
                operator_t, numpy_t = run_operator(name, sizes, reps)
            except Exception as error:
                print(f"{name:<6} {str(sizes):<22} FAILED: "
                      f"{type(error).__name__}: {str(error)[:80]}")
                continue
            size_str = ",".join(f"{k}={v}" for k, v in sizes.items())
            print(f"{name:<6} {size_str:<22} {operator_t*1e3:10.4f} "
                  f"{numpy_t*1e3:10.4f} {operator_t/numpy_t:17.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
