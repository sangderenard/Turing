"""One table: every representation against every other, on identical inputs.

`differential_translation.py` answers "is the translation faithful end to
end". This answers the question that actually routes a defect: **which
pair of representations first disagrees**, across all of them at once.

    representation   what it is
    --------------   ----------------------------------------------------
    oracle           sympy.lambdify over the authored equations (NumPy)
    ssa              repository SSA run by the reference evaluator
    llvm             the compiled LLVM artifact
    fortran          the Fortran / C-shell executable

Read a row as a claim about a LAYER, not about a number:

    oracle vs ssa      disagree -> lowering changed the meaning
    ssa vs llvm        disagree -> LLVM emission changed the meaning
    ssa vs fortran     disagree -> Fortran emission changed the meaning
    llvm vs fortran    disagree -> at least one backend is wrong, and the
                                   one matching `ssa` is the right one

That last row is the cheapest real signal in the table: two backends from
one SSA disagreeing is unambiguous, needs no oracle, and immediately names
which side to read. It is how the one-element array bug was finally
attributed -- Fortran emitted a whole-array assignment from the same SSA
that LLVM rendered as a single scalar store.

A cell is not a pass/fail verdict. It is `max |a-b|` over the compared
observables plus the name of the worst one, because "differs" without
"where" starts another blind hunt.

    python tools/differential_matrix.py
    python tools/differential_matrix.py --grid 8 --dt 0.05
    python tools/differential_matrix.py --skip fortran      # slow one

Availability is reported, never faked: a representation that cannot be
built for this program prints why and its row stays empty. An empty cell
means "not measured"; it never means "agrees".
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

REPRESENTATIONS = ("oracle", "ssa", "llvm", "fortran")

# Observables compared across every representation. Metrics first because
# they are computed DURING the traversal, so a disagreement there is
# upstream of the state written at the end.
METRIC_FIELDS = ("max_vel", "max_flux", "div_inf", "mass_err", "dt_limit")
STATE_FIELDS = (
    "height", "momentum_x", "momentum_y", "tracer",
    "next_height", "next_momentum_x", "next_momentum_y", "next_tracer",
    "last_wave_speed", "last_height_violation", "last_tracer_violation",
)


def _observables(state: Any, metrics: Any) -> dict[str, np.ndarray]:
    found: dict[str, np.ndarray] = {}
    for field in METRIC_FIELDS:
        value = getattr(metrics, field, None)
        if value is not None:
            found[f"metrics.{field}"] = np.asarray(value, dtype=float)
    for name, value in dict(getattr(metrics, "error_channels", {}) or {}).items():
        found[f"channel.{name}"] = np.asarray(value, dtype=float)
    for field in STATE_FIELDS:
        if hasattr(state, field):
            found[f"state.{field}"] = np.asarray(
                getattr(state, field), dtype=float,
            )
    return found


def _fresh_state(grid: int):
    from src.compiler.symbolic_fluid_dt import SymbolicFluidGridState

    return SymbolicFluidGridState.initial(grid, grid)


# -- representations ------------------------------------------------------


def run_oracle(grid: int, dt: float) -> dict[str, np.ndarray]:
    from differential_translation import build_python_reference

    advance, _arguments, _outputs = build_python_reference()
    state = _fresh_state(grid)
    _ok, metrics = advance(state, dt)
    return _observables(state, metrics)


def run_llvm(grid: int, dt: float, build: Path) -> dict[str, np.ndarray]:
    from src.compiler.symbolic_fluid_native_runtime import (
        compile_native_symbolic_fluid_advance,
    )

    advance = compile_native_symbolic_fluid_advance(build)
    state = _fresh_state(grid)
    _ok, metrics = advance(state, dt)
    return _observables(state, metrics)


def run_ssa(grid: int, dt: float, build: Path) -> dict[str, np.ndarray]:
    """Execute the repository SSA itself.

    Deliberately raises unless the evaluator can bind the whole frame: a
    partially-bound run produces numbers that look like findings and are
    not. See ssa_reference_evaluator's own status note.
    """
    import pickle

    from src.compiler.ssa_reference_evaluator import (
        SSAReferenceEvaluator, bind_program_abi_arguments,
    )

    cached = build / "control_repository_ssa.pkl"
    if not cached.is_file():
        # Any lowering of this program will do for a differential; prefer
        # the newest so a stale one cannot quietly stand in for the
        # compiler as it exists now.
        candidates = sorted(
            (ROOT / "build").glob("*/control_repository_ssa.pkl"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise RuntimeError(
                "no lowered SSA found under build/; run "
                "`python -m src.compiler.symbolic_fluid_direct_control "
                "--output build/<name>` first"
            )
        cached = candidates[0]
    with cached.open("rb") as stream:
        module, _outputs, _exports = pickle.load(stream)
    name = "symbolic_fluid_control__symbolic_fluid_advance"
    function = module.functions[name]
    state = _fresh_state(grid)
    # Use the canonical binder, not a private one. This column previously
    # rebuilt binding by hand from `program_abi_field`, which bound the
    # state fields and the two extents and nothing else -- so it never
    # bound `dt` and reported "13 formals unbound", refusing to produce the
    # one column that answers the routing question. Binding by declared
    # identity, and finding an unaccounted parameter through the callee
    # formal it feeds, is what the compiled runtime already does.
    arguments, unbound = bind_program_abi_arguments(
        function,
        record=state,
        named={"dt": float(dt), "height_count": grid, "width_count": grid},
        functions=module.functions,
    )
    # An unbound formal is not automatically a problem, and not automatically
    # fine. A formal that some call declares as an OUTPUT is an in-place
    # result cell: the program writes it before anything reads it, so the
    # scratch zero is never observed. Anything else unbound is a real input
    # the binder could not identify, and a zero standing in for it would be
    # manufactured evidence -- so that still refuses.
    written = {
        int(output)
        for block in function.blocks.values()
        for instruction in block.instrs
        for output in (instruction.attributes.get("output_ids") or ())
    }
    # A formal nothing reads cannot carry a wrong value into a result, so a
    # scratch zero there is unobservable rather than manufactured. This
    # program declares five such formals, all anonymous and unaccounted.
    read = {
        int(argument.id)
        for block in function.blocks.values()
        for instruction in block.instrs
        for argument in instruction.args
    }
    genuinely_unbound = sorted((set(unbound) - written) & read)
    if genuinely_unbound:
        raise RuntimeError(
            f"{len(genuinely_unbound)} INPUT formals unbound "
            f"(e.g. {genuinely_unbound[:6]}); the SSA column refuses to "
            "report partially-bound numbers"
        )
    SSAReferenceEvaluator(module).run(name, arguments)
    # The evaluator mutates caller-owned storage exactly as the compiled ABI
    # does, so the state fields it wrote are readable here. Metrics are NOT
    # reconstructed, so this column reports the state observables only --
    # `compare` intersects keys, and a metric absent from one side is left
    # out of the comparison rather than counted as agreement.
    return {
        key: value for key, value in _observables(state, _NoMetrics()).items()
        if key.startswith("state.")
    }


class _NoMetrics:
    """Stands in where this column has no Metrics to report.

    Deliberately empty rather than zero-filled: a zero would be compared
    and would read as a disagreement of exactly the true value, which is
    the kind of manufactured evidence this tree has been bitten by before.
    """

    error_channels: dict = {}


def run_fortran(grid: int, dt: float, build: Path) -> dict[str, np.ndarray]:
    raise RuntimeError(
        "the Fortran column runs the C-shell executable, which reports a "
        "summary rather than the full state; not yet wired up"
    )


# -- comparison -----------------------------------------------------------


def compare(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray],
) -> tuple[float, str, int]:
    """(worst absolute difference, which observable, how many differ)."""
    worst_value = 0.0
    worst_name = "-"
    differing = 0
    for key in sorted(set(left) & set(right)):
        a, b = left[key], right[key]
        if a.shape != b.shape:
            return float("inf"), f"{key} (shape {a.shape} vs {b.shape})", 1
        gap = float(np.max(np.abs(a - b))) if a.size else 0.0
        if gap > 1e-12:
            differing += 1
        if gap > worst_value:
            worst_value, worst_name = gap, key
    return worst_value, worst_name, differing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=4)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--build", default="build/matrix")
    parser.add_argument(
        "--skip", default="",
        help="representations to skip, comma separated",
    )
    arguments = parser.parse_args()
    skipped = {
        item.strip() for item in arguments.skip.split(",") if item.strip()
    }
    build = ROOT / arguments.build
    build.mkdir(parents=True, exist_ok=True)

    builders: dict[str, Callable[[], dict[str, np.ndarray]]] = {
        "oracle": lambda: run_oracle(arguments.grid, arguments.dt),
        "llvm": lambda: run_llvm(arguments.grid, arguments.dt, build),
        "ssa": lambda: run_ssa(arguments.grid, arguments.dt, build),
        "fortran": lambda: run_fortran(arguments.grid, arguments.dt, build),
    }

    results: dict[str, dict[str, np.ndarray]] = {}
    unavailable: dict[str, str] = {}
    # LLVM first: it produces the lowered pickle the SSA column needs.
    for label in ("oracle", "llvm", "ssa", "fortran"):
        if label in skipped:
            unavailable[label] = "skipped on request"
            continue
        print(f"building {label} ...")
        try:
            results[label] = builders[label]()
        except Exception as error:  # noqa: BLE001 - reported, never hidden
            unavailable[label] = f"{type(error).__name__}: {error}"

    print()
    print(f"grid {arguments.grid}x{arguments.grid}, dt={arguments.dt}")
    print("cells are max |a-b| over shared observables; blank = not measured")
    print()

    available = [name for name in REPRESENTATIONS if name in results]
    width = 12
    header = "".ljust(width) + "".join(name.ljust(width) for name in available)
    print(header)
    print("-" * len(header))
    for row in available:
        line = row.ljust(width)
        for column in available:
            if row == column:
                line += "-".ljust(width)
                continue
            worst, _name, _count = compare(results[row], results[column])
            line += (f"{worst:.3e}" if worst else "0").ljust(width)
        print(line)

    print()
    findings = []
    for index, row in enumerate(available):
        for column in available[index + 1:]:
            worst, name, count = compare(results[row], results[column])
            if worst > 1e-12:
                findings.append((worst, row, column, name, count))
    if findings:
        print("WORST OBSERVABLE PER DISAGREEING PAIR")
        for worst, row, column, name, count in sorted(findings, reverse=True):
            print(
                f"  {row:8} vs {column:8}  {worst:.6e}  in {name} "
                f"({count} observable(s) differ)"
            )
        print()
        print("ROUTING: oracle-vs-ssa blames lowering, ssa-vs-backend blames "
              "that backend's emission, and backend-vs-backend means the one "
              "matching ssa is right.")
    elif len(available) > 1:
        print("every available pair agrees to 1e-12.")

    if unavailable:
        print()
        print("NOT MEASURED (a blank cell never means 'agrees'):")
        for label, reason in unavailable.items():
            print(f"  {label:8} {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
