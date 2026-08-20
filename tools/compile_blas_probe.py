"""Compile one AbstractTensor BLAS kernel and check it against its oracle.

Same calling convention as ``docs/FUNCTION_TO_DEPLOYMENT_HANDOFF.md``
section 2 (proven on the eigh compilation effort, one day before this
script): ``lower_ast_source_to_ssa`` -> ``emit_ssa_function_to_llvm`` ->
``compile_artifact`` -> ``prepare_artifact_execution``. Bind by
``artifact.buffer_order``, never by a remembered id or by position alone
(section 2.1) -- this script prints the binding it derived so a disagreement
is visible rather than silently trusted.

No pytest, no test harness -- run it directly and read the report:

    python tools/compile_blas_probe.py            # smallest kernel (scal)
    python tools/compile_blas_probe.py dot
    python tools/compile_blas_probe.py gemv --n 6 --m 5
    python tools/compile_blas_probe.py --all       # walk the whole ladder

This is a diagnostic probe, not a test suite: it reports LOWER / EMIT / RUN /
EQUIVALENT the way ``tools/compile_re_probe.py`` and
``tools/translation_scorecard.py`` do, and it does not fix anything it finds.
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

STAGES = ("LOWER", "EMIT", "RUN", "EQUIVALENT")


def _sample_args(names: tuple[str, ...], *, m: int, n: int, k: int, seed: int):
    """Concrete, small, non-trivial inputs for one kernel's authored order."""

    rng = np.random.default_rng(seed)
    values: dict[str, object] = {}
    for name in names:
        if name == "m":
            values[name] = m
        elif name == "n":
            values[name] = n
        elif name == "k":
            values[name] = k
        elif name == "alpha":
            values[name] = 1.5
        elif name == "beta":
            values[name] = 0.5
        elif name == "A":
            rows = m if "m" in names else n
            cols = k if "k" in names else n
            values[name] = rng.uniform(-2.0, 2.0, size=rows * cols)
        elif name == "B":
            values[name] = rng.uniform(-2.0, 2.0, size=k * n)
        elif name == "C":
            values[name] = rng.uniform(-2.0, 2.0, size=m * n)
        elif name == "y":
            # gemv's y has length m (one per output row); scal/axpy have no
            # 'm' parameter at all and their y has length n.
            length = m if "m" in names else n
            values[name] = rng.uniform(-2.0, 2.0, size=length)
        elif name == "x":
            length = n
            values[name] = rng.uniform(-2.0, 2.0, size=length)
        else:
            raise ValueError(f"no sample rule for parameter {name!r}")
    return values


def probe(level: int, name: str, source: str, reference, arity: tuple[str, ...],
          *, m: int, n: int, k: int, build_dir: Path) -> tuple[str, str]:
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )

    tag = f"blas_probe_{name}"
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            module, outputs, _exports = lower_ast_source_to_ssa(
                source, name, name=tag,
            )
        except Exception as error:
            return "LOWER", f"{type(error).__name__}: {str(error)[:200]}"
    print(f"  [{name}] lowered in {time.time() - t0:.2f}s")

    entrypoint = f"{tag}__{name}"
    if entrypoint not in module.functions:
        return "LOWER", (
            f"{entrypoint!r} missing; compiled names: "
            f"{sorted(module.functions)[:10]}"
        )
    fn = module.functions[entrypoint]
    # ``parameter_names`` (equivalently ``value_names``) is the authoritative
    # authored-name -> SSA-id table this compiler actually populates for a
    # plain source function -- NOT a positional zip against ``fn.args``,
    # whose order need not match authored order (measured: it doesn't, for
    # every kernel here). Section 2.1's warning generalizes: derive the
    # binding from a real table, never from an assumed order.
    table = fn.metadata.get("parameter_names") or fn.metadata.get(
        "value_names"
    ) or ()
    id_by_name = {str(param): int(value_id) for param, value_id in table}
    missing = [param for param in arity if param not in id_by_name]
    if missing:
        return "EMIT", (
            f"no value id in parameter_names/value_names for {missing!r}; "
            f"table={table!r}"
        )
    print(f"  [{name}] name -> value id: {id_by_name}")

    try:
        artifact = emit_ssa_function_to_llvm(module, entrypoint)
    except Exception as error:
        return "EMIT", f"{type(error).__name__}: {str(error)[:200]}"
    if not artifact.complete:
        detail = "; ".join(s.reason[:120] for s in artifact.shortfalls[:5])
        return "EMIT", f"{len(artifact.shortfalls)} shortfall(s): {detail}"
    print(f"  [{name}] emit complete: buffer_order={artifact.buffer_order}")

    try:
        native = compile_artifact(artifact, directory=build_dir / name)
    except Exception as error:
        return "RUN", f"compile: {type(error).__name__}: {str(error)[:200]}"

    sample = _sample_args(arity, m=m, n=n, k=k, seed=1234)
    reference_args = {
        param: (np.array(sample[param], copy=True)
                if isinstance(sample[param], np.ndarray) else sample[param])
        for param in arity
    }
    expected = reference(*(reference_args[param] for param in arity))

    feeds = {}
    for param in arity:
        value_id = id_by_name.get(param)
        if value_id is None:
            return "RUN", f"no value id resolved for parameter {param!r}"
        feeds[value_id] = sample[param]
    try:
        execution = prepare_artifact_execution(native, feeds)
        execution.run()
    except Exception as error:
        return "RUN", f"{type(error).__name__}: {str(error)[:200]}"

    # In-place kernels (scal/axpy/gemv/gemm) publish their result by
    # mutating a fed-in array; dot returns a fresh scalar via Ret. Decide
    # which by ``named_outputs`` -- the function's own record of which
    # authored name the return value is -- not by guessing a parameter
    # name: ``dot``'s own ``y`` is a read-only INPUT vector, and a
    # name-based guess collides with it (measured: this cost a false
    # 9.8e-01 "disagreement" that was actually a probe bug, not the
    # compiler's).
    output_names = {
        str(param) for param, _value_id in (fn.metadata.get("named_outputs") or ())
    }
    inout_param = next(
        (p for p in ("y", "C") if p in id_by_name and p in output_names),
        None,
    )
    if inout_param is not None:
        produced = np.asarray(execution.buffers[id_by_name[inout_param]])
        expected_array = np.asarray(expected)
        worst = float(np.max(np.abs(produced - expected_array)))
    else:
        ret_values = outputs.get(entrypoint) or ()
        if not ret_values:
            return "RUN", "no Ret values recorded for a non-inout kernel"
        result_id = int(ret_values[0].id)
        if result_id not in execution.buffers:
            return "RUN", (
                f"result id {result_id} not in buffer_order "
                f"{artifact.buffer_order}"
            )
        produced = float(np.asarray(execution.buffers[result_id]).reshape(-1)[0])
        worst = abs(produced - float(expected))

    print(f"  [{name}] worst |produced - expected| = {worst:.3e}")
    if worst > 1e-9:
        return "EQUIVALENT", f"worst disagreement {worst:.3e}"
    return "PASSED", f"max disagreement {worst:.1e}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel", nargs="?", default="scal")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument(
        "--build", type=Path, default=ROOT / "build" / "blas-probe",
    )
    args = parser.parse_args()

    by_name = {entry[1]: entry for entry in KERNELS}
    targets = (
        KERNELS if args.all
        else (by_name.get(args.kernel),)
    )
    if targets[0] is None:
        print(f"unknown kernel {args.kernel!r}; known: {sorted(by_name)}")
        return 2

    args.build.mkdir(parents=True, exist_ok=True)
    overall_ok = True
    for level, name, source, reference, arity in targets:
        print(f"--- level {level}: {name} {arity} ---")
        stage, detail = probe(
            level, name, source, reference, arity,
            m=args.m, n=args.n, k=args.k, build_dir=args.build,
        )
        marker = "ok  " if stage == "PASSED" else "STOP"
        print(f"{marker} {stage:<11} {detail}")
        overall_ok = overall_ok and stage == "PASSED"
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
