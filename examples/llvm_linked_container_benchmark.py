"""A small state container whose update law is an LLVM link.

The container is deliberately ordinary authored Python: four evolving columns
(``amount``, ``temperature``, ``foam``, and ``spin``) plus three controls.  Its
update law is compiled once as an :class:`LLVMPiece`; the containing
``advance_container`` program calls that piece by name and is compiled with
``compose_native_package(..., link="static")``.  The resulting DLL therefore
contains the C shell and the separately emitted LLVM law linked into it.

Run from the repository root::

    py -3.11 examples/llvm_linked_container_benchmark.py

The timing is an end-to-end repeated-step benchmark.  ``standalone link`` is
the important lane: Turing's native host owns the hot loop and calls the LLVM
symbol without returning to Python between steps.  ``linked LLVM (ctypes)``
reuses one prepared execution but crosses Python on every step, while
``eager LLVM piece`` additionally exercises the standalone piece's normal
Python-call boundary.  NumPy is the readable reference implementation.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.compiler.native_package import (  # noqa: E402
    compose_native_package,
    piece_from_law,
)
from src.compiler.symbolic_equation_compiler import (  # noqa: E402
    compile_sympy_equations,
)


STATE_NAMES = ("amount", "temperature", "foam", "spin")


@dataclass
class LittleContainer:
    """The host-side view of the caller-owned native buffers."""

    amount: np.ndarray
    temperature: np.ndarray
    foam: np.ndarray
    spin: np.ndarray
    feed: np.ndarray
    ambient: np.ndarray
    dt: np.ndarray

    @classmethod
    def seeded(cls, batch: int, seed: int = 7) -> "LittleContainer":
        rng = np.random.default_rng(seed)
        return cls(
            amount=rng.uniform(0.6, 1.4, batch),
            temperature=rng.uniform(18.0, 26.0, batch),
            foam=rng.uniform(0.0, 0.3, batch),
            spin=rng.uniform(0.2, 1.2, batch),
            feed=rng.uniform(0.04, 0.12, batch),
            ambient=rng.uniform(19.0, 23.0, batch),
            dt=np.full(batch, 0.002, dtype=np.float64),
        )

    def clone(self) -> "LittleContainer":
        return LittleContainer(**{
            name: value.copy() for name, value in vars(self).items()
        })

    def columns(self) -> dict[str, np.ndarray]:
        return dict(vars(self))


def numpy_step(container: LittleContainer) -> None:
    """Readable truth for the law's simultaneous update."""

    amount = container.amount
    temperature = container.temperature
    foam = container.foam
    spin = container.spin
    dt = container.dt
    amount_next = amount + dt * (container.feed - 0.08 * amount)
    temperature_next = temperature + dt * (
        0.18 * (container.ambient - temperature)
        + 0.035 * spin * spin
        + 0.12 * amount
    )
    foam_next = foam + dt * (0.22 * spin - 0.31 * foam)
    spin_next = spin + dt * (
        0.40 * container.feed - 0.17 * spin - 0.025 * foam * spin
    )
    container.amount[...] = amount_next
    container.temperature[...] = temperature_next
    container.foam[...] = foam_next
    container.spin[...] = spin_next


def authored_law():
    """The same four equations, authored once as the compiler's symbolic truth."""

    amount, temperature, foam, spin = sp.symbols(
        "amount temperature foam spin"
    )
    feed, ambient, dt = sp.symbols("feed ambient dt")
    equations = [
        sp.Eq(
            sp.Symbol("amount_next"),
            amount + dt * (feed - sp.Float("0.08") * amount),
            evaluate=False,
        ),
        sp.Eq(
            sp.Symbol("temperature_next"),
            temperature
            + dt
            * (
                sp.Float("0.18") * (ambient - temperature)
                + sp.Float("0.035") * spin * spin
                + sp.Float("0.12") * amount
            ),
            evaluate=False,
        ),
        sp.Eq(
            sp.Symbol("foam_next"),
            foam + dt * (sp.Float("0.22") * spin - sp.Float("0.31") * foam),
            evaluate=False,
        ),
        sp.Eq(
            sp.Symbol("spin_next"),
            spin
            + dt
            * (
                sp.Float("0.40") * feed
                - sp.Float("0.17") * spin
                - sp.Float("0.025") * foam * spin
            ),
            evaluate=False,
        ),
    ]
    return compile_sympy_equations(equations, name="little_container_law")


def container_source(argument_names: tuple[str, ...]) -> str:
    """Handwritten container body, with the piece's declared argument order."""

    arguments = ", ".join(argument_names)
    return f'''\
def advance_container({arguments}):
    amount_next, temperature_next, foam_next, spin_next = little_container_law({arguments})
    amount[...] = amount_next
    temperature[...] = temperature_next
    foam[...] = foam_next
    spin[...] = spin_next
    return amount, temperature, foam, spin
'''


def time_steps(label: str, steps: int, operation: Callable[[], None]) -> tuple[str, float]:
    gc.collect()
    started = time.perf_counter()
    for _ in range(steps):
        operation()
    return label, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=ROOT / "build" / "llvm_linked_container_benchmark",
    )
    args = parser.parse_args()
    if args.batch < 1 or args.steps < 1 or args.warmup < 0:
        parser.error("batch and steps must be positive; warmup must be non-negative")

    args.build_dir.mkdir(parents=True, exist_ok=True)
    piece_dir = args.build_dir / "piece"
    package_dir = args.build_dir / "linked"
    piece = piece_from_law(
        authored_law(),
        "little_container_law",
        args.batch,
        directory=piece_dir,
        optimization="O2",
    )
    package = compose_native_package(
        container_source(piece.argument_names),
        "advance_container",
        {"little_container_law": piece},
        piece.argument_names,
        args.batch,
        directory=package_dir,
        name="linked_little_container",
        optimization="O2",
        link="static",
    )

    initial = LittleContainer.seeded(args.batch)
    standalone = package.compile_standalone(
        args.build_dir / "standalone",
        package.feeds(initial.columns()),
        optimization="O2",
        link="static",
    )

    # Parity is checked over many evolving steps, not only the first call.
    reference = initial.clone()
    linked_columns = initial.clone().columns()
    linked = package.prepare_execution(package.feeds(linked_columns))
    parity_steps = 100
    for _ in range(parity_steps):
        numpy_step(reference)
        linked.run()
    errors = {
        name: float(np.max(np.abs(
            linked.buffers[package.parameter_ids[name]] - getattr(reference, name)
        )))
        for name in STATE_NAMES
    }
    for name in STATE_NAMES:
        np.testing.assert_allclose(
            linked.buffers[package.parameter_ids[name]],
            getattr(reference, name),
            rtol=2e-13,
            atol=2e-13,
        )

    # Independent states make the three timing lanes comparable.
    numpy_state = initial.clone()
    eager_state = initial.clone()
    timed_linked = package.prepare_execution(package.feeds(initial.clone().columns()))

    def eager_step() -> None:
        columns = eager_state.columns()
        outputs = piece(*(columns[name] for name in piece.argument_names))
        for name, output in zip(piece.output_names, outputs):
            if name.endswith("_next") and name[:-5] in STATE_NAMES:
                getattr(eager_state, name[:-5])[...] = output

    for _ in range(args.warmup):
        numpy_step(numpy_state)
        eager_step()
        timed_linked.run()
    if args.warmup:
        standalone.run(frames=args.warmup)

    # This call crosses Python only twice: process launch and completion.  The
    # generated C host owns every timed frame in between and calls the linked
    # LLVM-containing entry directly.
    def time_standalone() -> tuple[str, float]:
        gc.collect()
        started = time.perf_counter()
        standalone.run(frames=args.steps)
        return "standalone link", time.perf_counter() - started

    timings = [
        time_steps("NumPy", args.steps, lambda: numpy_step(numpy_state)),
        time_steps("eager LLVM piece", args.steps, eager_step),
        time_steps("linked LLVM (ctypes)", args.steps, timed_linked.run),
        time_standalone(),
    ]
    standalone_seconds = dict(timings)["standalone link"]

    print("little container: amount + temperature + foam + spin")
    print(f"batch={args.batch:,} steps={args.steps:,} warmup={args.warmup:,}")
    print(f"piece library:   {piece.artifact.library_path}")
    print(f"container DLL:   {package.library_path}")
    print(f"standalone EXE:  {standalone.executable_path}")
    print("linked LLVM:     " + ", ".join(
        symbol for symbol, _llvm_ir in package.linked_llvm
    ))
    print("max abs error:   " + ", ".join(
        f"{name}={error:.3e}" for name, error in errors.items()
    ))
    print()
    for label, seconds in timings:
        ns_per_cell_step = seconds * 1e9 / (args.steps * args.batch)
        relative = seconds / standalone_seconds
        print(
            f"{label:18s} {seconds:9.4f} s  "
            f"{ns_per_cell_step:9.2f} ns/cell-step  {relative:7.2f}x standalone"
        )


if __name__ == "__main__":
    main()
