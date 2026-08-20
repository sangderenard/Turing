"""NumPy versus the FASTEST VARIANT WE CAN BAKE, for every BLAS kernel.

One command, one question: at this size, what is the best thing this
compiler can produce, and how does it compare to NumPy calling its own
vendor BLAS?

    python tools/benchmark_bakeoff.py                  # every kernel, full ladder
    python tools/benchmark_bakeoff.py gemm gemv        # named kernels only
    python tools/benchmark_bakeoff.py --quick          # smallest rung of each
    python tools/benchmark_bakeoff.py --candidates     # every variant, not just the winner

WHAT "BAKE" MEANS HERE. Each kernel is compiled once per point in a
variant matrix and the winner is whichever is fastest AND correct:

  contract   ``develop`` (no contract) vs ``fast`` (FMA contraction and
             the rest of the work contract's fast preset).
  bake       ``parametric`` -- one artifact serving every size, loop bounds
             passed in -- vs ``sized`` -- the size literals baked into the
             source before lowering, which is the tree's first
             argument-baking specializer (``kernel_bank._specialize_source``)
             and what makes a monomorphic kernel possible at all.

NO UNVERIFIED CANDIDATE CAN WIN. Every variant's output is compared
against the NumPy result at the BENCHMARK size before its time is allowed
to count, and a disagreeing variant is reported as REFUSED with its error
rather than ranked. This is not ceremony: baking a size literal is the
exact condition under which this compiler is known to silently dead-store
a write-only array parameter (``blas.py`` authoring rule 2), and gemm's
sized bake is a pinned open defect. A bakeoff that timed a kernel
computing the wrong thing would report its best number for its worst
artifact.

METHODOLOGY matches ``tools/benchmark_blas_vs_numpy.py`` and
``tools/bench_native_step.py``: compile and prepare once, warm up, then
time many repeated ``execution.run()`` calls and divide. Steady state --
first-call cost (lazy imports, cache population, dlopen) is excluded from
both sides. The compiled column is therefore the KERNEL's cost, not a call
site's; the dispatch tax an AbstractTensor operator pays on top is what
``benchmark_blas_vs_numpy.py``'s OPERATOR section measures, and it is the
larger number at small sizes.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.common.tensors.blas import KERNELS


# The size ladder. Kept identical to benchmark_blas_vs_numpy.py so the two
# tools' numbers are comparable rather than nearly comparable.
LADDER = {
    "scal": [{"n": 256}, {"n": 4096}, {"n": 65536}],
    "axpy": [{"n": 256}, {"n": 4096}, {"n": 65536}],
    "dot": [{"n": 256}, {"n": 4096}, {"n": 65536}],
    "rot": [{"n": 256}, {"n": 4096}, {"n": 65536}],
    "gemv": [{"m": 64, "n": 64}, {"m": 256, "n": 256}, {"m": 1024, "n": 1024}],
    "gemm": [{"m": 64, "n": 64, "k": 64}, {"m": 128, "n": 128, "k": 128},
             {"m": 256, "n": 256, "k": 256}],
}
REPS = {"scal": 200, "axpy": 200, "dot": 200, "rot": 200,
        "gemv": 50, "gemm": 10}
CONTRACTS = (None, "fast")
BAKES = ("parametric", "sized")

#: Useful arithmetic per call, for GF/s. rot is 6n: a multiply-add per
#: element per output, twice, plus the sign folded into them.
FLOPS = {
    "scal": lambda s: s["n"],
    "axpy": lambda s: 2 * s["n"],
    "dot": lambda s: 2 * s["n"],
    "rot": lambda s: 6 * s["n"],
    "gemv": lambda s: 2 * s["m"] * s["n"],
    "gemm": lambda s: 2 * s["m"] * s["n"] * s["k"],
}


def _time(fn, reps: int, *, warmup: int = 1) -> float:
    for _ in range(warmup):
        fn()
    started = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - started) / reps


def _sample(arity, sizes, rng):
    m = sizes.get("m", sizes.get("n"))
    n = sizes["n"]
    k = sizes.get("k", n)
    values = {}
    for parameter in arity:
        if parameter in ("m", "n", "k"):
            values[parameter] = int(sizes[parameter])
        elif parameter == "alpha":
            values[parameter] = 1.7
        elif parameter == "beta":
            values[parameter] = 0.6
        elif parameter == "c":
            values[parameter] = 0.8
        elif parameter == "s":
            values[parameter] = 0.6
        elif parameter == "A":
            rows = m if "m" in arity else n
            cols = k if "k" in arity else n
            values[parameter] = rng.uniform(-1, 1, size=rows * cols)
        elif parameter == "B":
            values[parameter] = rng.uniform(-1, 1, size=k * n)
        elif parameter == "C":
            values[parameter] = rng.uniform(-1, 1, size=m * n)
        elif parameter == "y":
            values[parameter] = rng.uniform(
                -1, 1, size=(m if "m" in arity else n))
        elif parameter == "x":
            values[parameter] = rng.uniform(-1, 1, size=n)
        else:
            raise ValueError(f"no sample rule for {parameter!r}")
    return values


def _numpy_oracle(name, sample, sizes):
    """The NumPy equivalent: the timed baseline AND the correctness oracle.

    Returns ``(call, expected)`` where ``call`` is a zero-argument closure
    over private copies (so timing it thousands of times cannot corrupt the
    verification data) and ``expected`` maps each output parameter to the
    value the compiled kernel must produce.
    """

    m = sizes.get("m", sizes.get("n"))
    n = sizes["n"]
    k = sizes.get("k", n)
    v = {p: (np.array(value, dtype=np.float64)
             if isinstance(value, np.ndarray) else value)
         for p, value in sample.items()}

    if name == "scal":
        expected = {"y": v["alpha"] * v["x"]}
        call = lambda: np.multiply(v["alpha"], v["x"], out=v["y"])
    elif name == "axpy":
        expected = {"y": v["alpha"] * v["x"] + v["y"]}
        call = lambda: np.add(v["alpha"] * v["x"], v["y"], out=v["y"])
    elif name == "dot":
        expected = {"return": float(np.dot(v["x"], v["y"]))}
        call = lambda: np.dot(v["x"], v["y"])
    elif name == "rot":
        expected = {
            "x": v["c"] * v["x"] + v["s"] * v["y"],
            "y": v["c"] * v["y"] - v["s"] * v["x"],
        }
        call = lambda: (v["c"] * v["x"] + v["s"] * v["y"],
                        v["c"] * v["y"] - v["s"] * v["x"])
    elif name == "gemv":
        A = v["A"].reshape(m, n)
        expected = {"y": v["alpha"] * (A @ v["x"]) + v["beta"] * v["y"]}
        call = lambda: np.add(
            v["alpha"] * (A @ v["x"]), v["beta"] * v["y"], out=v["y"])
    elif name == "gemm":
        A = v["A"].reshape(m, k)
        B = v["B"].reshape(k, n)
        C = v["C"].reshape(m, n)
        expected = {"C": (v["alpha"] * (A @ B) + v["beta"] * C).reshape(-1)}
        call = lambda: np.add(v["alpha"] * (A @ B), v["beta"] * C, out=C)
    else:
        raise ValueError(name)
    return call, expected


def _build(name, source, contract, bake, sizes, build_dir):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm,
    )
    from src.compiler.work_contract import set_active_contract

    if bake == "sized":
        # The tree's own argument-baking specializer, not a second one
        # written here: whatever it does for the kernel bank is exactly
        # what gets measured.
        from src.compiler.kernel_bank import KernelSpec, _specialize_source

        spec = KernelSpec(
            name=name, source=source, function_name=name, reference=None,
            parameter_order=(), size_parameters=tuple(sizes),
            example_inputs=None,
        )
        source = _specialize_source(spec, sizes)

    # Windows holds a loaded DLL open, so every distinct variant needs its
    # own output directory or the second compile fails with an lld-link
    # write error against a file this process still has mapped.
    suffix = "_".join(f"{key}{value}" for key, value in sorted(sizes.items()))
    tag = f"bakeoff_{name}_{contract or 'develop'}_{bake}_{suffix}"
    set_active_contract(contract)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            module, outputs, _exports = lower_ast_source_to_ssa(
                source, name, name=tag,
            )
        entrypoint = f"{tag}__{name}"
        function = module.functions[entrypoint]
        artifact = emit_ssa_function_to_llvm(module, entrypoint)
        if not artifact.complete:
            raise RuntimeError(
                f"{len(artifact.shortfalls)} emission shortfall(s): "
                + "; ".join(s.reason[:90] for s in artifact.shortfalls[:2])
            )
        native = compile_artifact(artifact, directory=build_dir / tag)
    finally:
        set_active_contract(None)

    table = dict(
        function.metadata.get("parameter_names")
        or function.metadata.get("value_names") or ()
    )
    return (
        native,
        {str(key): int(value) for key, value in table.items()},
        {str(p) for p, _v in (function.metadata.get("named_outputs") or ())},
        tuple(int(value.id) for value in (outputs.get(entrypoint) or ())),
    )


def _measure(name, source, arity, sizes, contract, bake, reps, build_dir,
             expected, tolerance):
    """One candidate: build it, VERIFY it, then time it. In that order."""

    from src.compiler.ssa_llvm_backend import prepare_artifact_execution

    native, id_by_name, _output_names, ret_ids = _build(
        name, source, contract, bake, sizes, build_dir,
    )
    rng = np.random.default_rng(7)
    sample = _sample(arity, sizes, rng)
    # A sized bake drops its size parameters from the signature entirely --
    # they are literals now -- so feed by what the artifact actually binds.
    feeds = {
        id_by_name[p]: (np.array(sample[p], dtype=np.float64, copy=True)
                        if isinstance(sample[p], np.ndarray) else sample[p])
        for p in arity if p in id_by_name
    }
    missing = [p for p in arity if p not in id_by_name
               and isinstance(sample[p], np.ndarray)]
    if missing:
        raise RuntimeError(
            f"array parameter(s) {missing!r} vanished from the signature; "
            "a dead-stored output is the known cause"
        )
    execution = prepare_artifact_execution(native, feeds)
    execution.run()

    worst = 0.0
    culprit = None
    for parameter, oracle in expected.items():
        if parameter == "return":
            if not ret_ids:
                raise RuntimeError("no Ret record for a returning kernel")
            produced = float(
                np.asarray(execution.buffers[ret_ids[0]]).reshape(-1)[0])
        elif parameter in id_by_name:
            produced = np.asarray(execution.buffers[id_by_name[parameter]])
        else:
            raise RuntimeError(f"output {parameter!r} is not a bound buffer")
        error = float(np.max(np.abs(
            np.asarray(produced, dtype=np.float64)
            - np.asarray(oracle, dtype=np.float64)
        )))
        if error > worst:
            worst, culprit = error, parameter
    if worst > tolerance:
        raise RuntimeError(
            f"disagrees with NumPy by {worst:.3e} on output {culprit!r}"
        )

    # Re-feed before timing: the verification run mutated the in-place
    # buffers, and an axpy timed on its own accumulating output measures a
    # different (growing) problem.
    for parameter, value in sample.items():
        if parameter in id_by_name and isinstance(value, np.ndarray):
            np.asarray(execution.buffers[id_by_name[parameter]])[...] = value
    return _time(execution.run, reps), worst


def main() -> int:
    parser = argparse.ArgumentParser(
        description="NumPy vs the fastest bakeable variant, per kernel/size",
    )
    parser.add_argument("kernels", nargs="*", help="default: all of them")
    parser.add_argument("--quick", action="store_true",
                        help="smallest ladder rung of each kernel only")
    parser.add_argument("--candidates", action="store_true",
                        help="print every variant, not just the winner")
    parser.add_argument("--tolerance", type=float, default=1e-9,
                        help="max |compiled - numpy| a candidate may show")
    parser.add_argument("--build-dir", type=Path,
                        default=ROOT / "build" / "bakeoff")
    arguments = parser.parse_args()

    by_name = {entry[1]: entry for entry in KERNELS}
    order = ("scal", "axpy", "dot", "rot", "gemv", "gemm")
    selected = arguments.kernels or [n for n in order if n in by_name]
    unknown = [name for name in selected if name not in by_name]
    if unknown:
        parser.error(f"unknown kernel(s) {unknown!r}; have {sorted(by_name)}")
    arguments.build_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 108)
    print("BAKEOFF: raw NumPy vs the fastest CORRECT variant this compiler "
          "can bake (steady state, same sizes/data)")
    print("=" * 108)
    print(f"{'kernel':<6} {'size':<20} {'numpy ms':>10} {'numpy GF/s':>11} "
          f"{'best ms':>10} {'best GF/s':>10} {'speedup':>9}  best variant")
    print("-" * 108)

    refusals = []
    for name in selected:
        _level, _name, source, _reference, arity = by_name[name]
        reps = REPS[name]
        ladder = LADDER[name][:1] if arguments.quick else LADDER[name]
        for sizes in ladder:
            rng = np.random.default_rng(7)
            sample = _sample(arity, sizes, rng)
            numpy_call, expected = _numpy_oracle(name, sample, sizes)
            numpy_time = _time(numpy_call, reps)
            flops = FLOPS[name](sizes)
            size_text = ",".join(f"{k}={v}" for k, v in sizes.items())

            results = []
            for contract in CONTRACTS:
                for bake in BAKES:
                    label = f"{contract or 'develop'}/{bake}"
                    outcome = None
                    # Back-to-back compiles race zig's compiler_rt /
                    # mingw-w64 cache ("failed to check cache"), a known
                    # transient this suite trips precisely because it
                    # builds many variants in a row. Retried ONCE and only
                    # for that signature -- a wrong-answer refusal is
                    # never retried, since repeating it would only hide it.
                    for attempt in (1, 2):
                        try:
                            outcome = _measure(
                                name, source, arity, sizes, contract, bake,
                                reps, arguments.build_dir, expected,
                                arguments.tolerance,
                            )
                            break
                        except Exception as failure:
                            detail = f"{type(failure).__name__}: {failure}"
                            transient = "failed to check cache" in str(failure)
                            if transient and attempt == 1:
                                continue
                            refusals.append((
                                name, size_text, label,
                                ("(retried once) " if transient else "")
                                + detail[:110],
                            ))
                            break
                    if outcome is None:
                        continue
                    elapsed, error = outcome
                    results.append((elapsed, label, error))

            if not results:
                print(f"{name:<6} {size_text:<20} {numpy_time * 1e3:10.4f} "
                      f"{flops / numpy_time / 1e9:11.3f} "
                      f"{'--':>10} {'--':>10} {'--':>9}  "
                      f"NO CORRECT VARIANT (see refusals)")
                continue

            results.sort()
            best_time, best_label, best_error = results[0]
            print(f"{name:<6} {size_text:<20} {numpy_time * 1e3:10.4f} "
                  f"{flops / numpy_time / 1e9:11.3f} "
                  f"{best_time * 1e3:10.4f} {flops / best_time / 1e9:10.3f} "
                  f"{numpy_time / best_time:8.3f}x  {best_label} "
                  f"(|err| {best_error:.1e})")
            if arguments.candidates:
                for elapsed, label, error in results[1:]:
                    print(f"{'':<6} {'':<20} {'':>10} {'':>11} "
                          f"{elapsed * 1e3:10.4f} "
                          f"{flops / elapsed / 1e9:10.3f} "
                          f"{numpy_time / elapsed:8.3f}x  {label} "
                          f"(|err| {error:.1e})")

    if refusals:
        print()
        print("=" * 108)
        print("REFUSED -- built but not ranked. A variant that does not "
              "reproduce NumPy's answer never wins on speed.")
        print("=" * 108)
        for name, size_text, label, reason in refusals:
            print(f"{name:<6} {size_text:<20} {label:<18} {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
