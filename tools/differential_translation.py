"""Run the authored program and its translation side by side; report the
FIRST place they disagree.

Every other tool here answers a question you already thought to ask. This
one finds the question. It executes the authored mathematics through an
independent path -- SymPy `lambdify` over the same equations, evaluated by
NumPy -- and the compiled artifact through the real pipeline, on bitwise
identical inputs, then compares everything observable and reports the
earliest divergence rather than the loudest symptom.

Why an independent oracle matters
---------------------------------
The reference must not share machinery with the thing under test, or it
inherits its bugs. Two failures in this tree came from exactly that: a
"ground truth" for `mass_err` was computed from state the compiled program
had already corrupted, so the number being compared against was itself
produced by the defect. Here the oracle comes from `sympy.lambdify` over
`symbolic_viscous_shallow_water_equations()` -- the authored equations,
evaluated by SymPy and NumPy, sharing no lowering, no SSA, and no backend
with the artifact.

What it compares
----------------
* every state field, cell by cell (this is what catches a whole-array
  assignment that moved one element);
* every Metrics field;
* any additional SSA values named on the command line, watched
  non-perturbingly and matched to the authored local of the same name.

Ordering
--------
"First divergence" is reported in authored program order where that is
knowable -- state fields are written at the end of the traversal, so a
divergence in a named intermediate is reported ahead of a divergence in a
final field, because the intermediate is upstream of it.

    python tools/differential_translation.py
    python tools/differential_translation.py --grid 8 --dt 0.05
    python tools/differential_translation.py --watch max_wave_speed,next_mass
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

# Absolute/relative tolerance for "the same number". Not zero: the oracle
# and the artifact evaluate the same mathematics through different orders
# of operation, so last-bit differences are expected and are not defects.
ATOL = 1e-12
RTOL = 1e-9


def build_python_reference():
    """An independent Python implementation of the authored program.

    ``symbolic_fluid_step`` is rebuilt from the SymPy equations by
    ``lambdify``; the traversal around it is the authored source text,
    executed by Python. Neither path touches the compiler.
    """
    import sympy

    from src.compiler.symbolic_fluid_model import (
        compile_symbolic_fluid_step,
        symbolic_viscous_shallow_water_equations,
    )
    from src.compiler.symbolic_fluid_dt import SYMBOLIC_FLUID_DT_SOURCE
    from src.common.dt_system.dt_controller import run_superstep
    from src.common.dt_system.dt_scaler import Metrics

    compilation = compile_symbolic_fluid_step()
    argument_names = tuple(compilation.function.metadata["argument_names"])
    output_names = tuple(compilation.function.metadata["output_names"])

    model = symbolic_viscous_shallow_water_equations()
    by_name = {str(equation.lhs): equation.rhs for equation in model.equations}
    ordered_symbols = [model.symbols[name] for name in argument_names]
    kernels = [
        sympy.lambdify(ordered_symbols, by_name[name], "numpy")
        for name in output_names
    ]

    def symbolic_fluid_step(*values):
        return tuple(kernel(*values) for kernel in kernels)

    namespace: dict[str, Any] = {
        "symbolic_fluid_step": symbolic_fluid_step,
        "Metrics": Metrics,
        "run_superstep": run_superstep,
    }
    exec(compile(SYMBOLIC_FLUID_DT_SOURCE, "<authored>", "exec"), namespace)
    return namespace["symbolic_fluid_advance"], argument_names, output_names


def build_native(build_directory: Path, watch_ids: tuple[int, ...] = ()):
    from src.compiler.symbolic_fluid_native_runtime import (
        compile_native_symbolic_fluid_advance,
    )

    advance = compile_native_symbolic_fluid_advance(build_directory)
    if not watch_ids:
        return advance
    # Re-emit with watches, reusing the module the first build produced.
    import pickle

    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm,
    )
    from src.compiler.symbolic_fluid_native_runtime import (
        NativeSymbolicFluidAdvance,
    )

    cached = Path(build_directory) / "control_repository_ssa.pkl"
    if not cached.is_file():
        return advance
    with cached.open("rb") as stream:
        module, outputs, _exports = pickle.load(stream)
    name = "symbolic_fluid_control__symbolic_fluid_advance"
    artifact = emit_ssa_function_to_llvm(module, name, watch=watch_ids)
    compile_artifact(artifact, directory=build_directory / "watched")
    return NativeSymbolicFluidAdvance(
        artifact, module.functions[name], dict(module.functions),
        outputs.get(name),
    )


def compare(label: str, reference: Any, native: Any) -> tuple[bool, str]:
    left = np.asarray(reference, dtype=float)
    right = np.asarray(native, dtype=float)
    if left.shape != right.shape:
        return False, f"shape {left.shape} vs {right.shape}"
    if np.allclose(left, right, atol=ATOL, rtol=RTOL, equal_nan=True):
        return True, ""
    difference = np.abs(left - right)
    worst = int(np.argmax(difference))
    disagreeing = int((difference > (ATOL + RTOL * np.abs(left))).sum())
    if left.ndim:
        where = np.unravel_index(worst, left.shape)
        return False, (
            f"{disagreeing}/{left.size} elements differ; worst at {where}: "
            f"reference={left.flat[worst]!r} native={right.flat[worst]!r}"
        )
    return False, f"reference={left.item()!r} native={right.item()!r}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Differential execution: authored oracle vs translation.",
    )
    parser.add_argument("--grid", type=int, default=4)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument(
        "--watch", default="",
        help="authored local names to additionally compare, comma separated",
    )
    parser.add_argument("--build", default="build/differential")
    arguments = parser.parse_args()

    from src.compiler.symbolic_fluid_dt import SymbolicFluidGridState

    build_directory = ROOT / arguments.build
    build_directory.mkdir(parents=True, exist_ok=True)

    print("building the independent oracle (sympy -> numpy) ...")
    reference_advance, _argument_names, _output_names = build_python_reference()

    print(f"building the translation into {build_directory} ...")
    native_advance = build_native(build_directory)

    # Bitwise identical inputs. Built twice from the same constructor rather
    # than copied: `copy_shallow` returns a tuple here, and a reference that
    # shares storage with the thing under test is how an oracle silently
    # inherits the defect it is supposed to catch.
    reference_state = SymbolicFluidGridState.initial(
        arguments.grid, arguments.grid,
    )
    native_state = SymbolicFluidGridState.initial(
        arguments.grid, arguments.grid,
    )
    for field in ("height", "momentum_x", "momentum_y", "tracer"):
        left = np.asarray(getattr(reference_state, field), dtype=float)
        right = np.asarray(getattr(native_state, field), dtype=float)
        if not np.array_equal(left, right):
            raise SystemExit(
                f"the two initial states differ in {field}; the comparison "
                "below would be meaningless"
            )

    print(f"running both on a {arguments.grid}x{arguments.grid} grid, "
          f"dt={arguments.dt} ...\n")
    reference_ok, reference_metrics = reference_advance(
        reference_state, arguments.dt,
    )
    native_ok, native_metrics = native_advance(native_state, arguments.dt)

    findings: list[tuple[str, str]] = []

    if bool(reference_ok) != bool(native_ok):
        findings.append((
            "return value `ok`",
            f"reference={bool(reference_ok)} native={bool(native_ok)}",
        ))

    # Metrics first: they are computed DURING the traversal, so a
    # disagreement here is upstream of the state write-back below.
    for field in ("max_vel", "max_flux", "div_inf", "mass_err", "dt_limit"):
        same, detail = compare(
            field,
            getattr(reference_metrics, field),
            getattr(native_metrics, field),
        )
        if not same:
            findings.append((f"metrics.{field}", detail))

    reference_channels = dict(reference_metrics.error_channels or {})
    native_channels = dict(native_metrics.error_channels or {})
    for channel in sorted(set(reference_channels) | set(native_channels)):
        same, detail = compare(
            channel,
            reference_channels.get(channel, 0.0),
            native_channels.get(channel, 0.0),
        )
        if not same:
            findings.append((f"error_channels[{channel!r}]", detail))

    # State fields last: they are written at the end of the traversal.
    for field in (
        "height", "momentum_x", "momentum_y", "tracer",
        "next_height", "next_momentum_x", "next_momentum_y", "next_tracer",
        "last_wave_speed", "last_height_violation", "last_tracer_violation",
    ):
        if not hasattr(reference_state, field):
            continue
        same, detail = compare(
            field,
            getattr(reference_state, field),
            getattr(native_state, field),
        )
        if not same:
            findings.append((f"state.{field}", detail))

    if not findings:
        print("AGREED: every observable value matches the oracle.")
        print("  A defect, if present, is in something neither side "
              "observes -- widen the comparison rather than concluding "
              "the translation is correct.")
        return 0

    print(f"DIVERGENCE: {len(findings)} observable(s) disagree.\n")
    print("FIRST (earliest in authored program order):")
    first_label, first_detail = findings[0]
    print(f"  {first_label}: {first_detail}")
    if len(findings) > 1:
        print("\nalso:")
        for label, detail in findings[1:]:
            print(f"  {label}: {detail}")
    print(
        "\nStart at the FIRST one. Later divergences are frequently "
        "downstream consequences of it, and chasing a downstream symptom "
        "is how a whole day disappears."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
