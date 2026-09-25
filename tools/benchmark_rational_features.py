"""Cost table for repeated tensor division across numeric feature sets.

The workload is the motivating rational use case: apply a sequence of
nonzero tensor divisors, then observe the value once at the boundary.  Build,
division-chain, and collapse time are measured separately so a rational's
deferred division is not hidden inside a single aggregate number.

All eight subsets of {complex, rational, precision} are represented.  Real
and complex rows use their own ordinary-tensor baseline because complex
division is a different amount of arithmetic even without a wrapper.

Usage::

    py -3.11 tools/benchmark_rational_features.py
    py -3.11 tools/benchmark_rational_features.py --size 4096 --steps 8
    py -3.11 tools/benchmark_rational_features.py --limbs 3 --repeats 7
"""
from __future__ import annotations

import argparse
import gc
import math
import pathlib
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.common.tensors.extended_precision import (  # noqa: E402
    ComplexPrecision,
    ComplexRational,
    ComplexRationalPrecision,
    Precision,
    Rational,
    RationalPrecision,
)
from src.common.tensors.abstraction import AbstractTensor  # noqa: E402


def _tensor(values: list[float] | list[complex]) -> AbstractTensor:
    return AbstractTensor.get_tensor(values)


def _observed(value: Any) -> Any:
    return value.collapse() if hasattr(value, "collapse") else value


def _values(value: Any) -> list[float] | list[complex]:
    observed = value.tolist()
    return observed if isinstance(observed, list) else [observed]


def _chain(value: Any, divisors: list[Any]) -> Any:
    for divisor in divisors:
        value = value / divisor
    return value


def _median_ms(function: Callable[[], Any], warmups: int, repeats: int) -> float:
    for _ in range(warmups):
        function()
    samples = []
    for _ in range(repeats):
        gc.collect()
        started = time.perf_counter()
        function()
        samples.append((time.perf_counter() - started) * 1_000.0)
    return statistics.median(samples)


@dataclass(frozen=True)
class Case:
    name: str
    features: str
    domain: str
    coefficient_tensors: int
    build: Callable[[], tuple[Any, list[Any]]]
    reference: list[float] | list[complex]


@dataclass(frozen=True)
class Result:
    case: Case
    build_ms: float
    chain_ms: float
    collapse_ms: float
    maximum_error: float


def _inputs(size: int, steps: int) -> tuple[
    list[float], list[list[float]], list[complex], list[list[complex]]
]:
    position = (
        [0.0] if size == 1
        else [-1.0 + 2.0 * index / (size - 1) for index in range(size)]
    )
    real_start = [1.0 + 0.125 * point for point in position]
    real_divisors = [
        [
            1.0 + math.ldexp(1.0, -9) * math.sin(
                (index + 1.0) * point + 0.25 * index
            )
            for point in position
        ]
        for index in range(steps)
    ]
    complex_start = [
        real + 1j * (0.0625 * point)
        for real, point in zip(real_start, position)
    ]
    complex_divisors = [
        [
            real + 1j * math.ldexp(1.0, -10) * math.cos(
                (index + 1.0) * point - 0.125 * index
            )
            for real, point in zip(divisor, position)
        ]
        for index, divisor in enumerate(real_divisors)
    ]
    return real_start, real_divisors, complex_start, complex_divisors


def _reference(start: list[Any], divisors: list[list[Any]]) -> list[Any]:
    return _values(_chain(_tensor(start), [_tensor(value) for value in divisors]))


def _cases(size: int, steps: int, limbs: int) -> list[Case]:
    real_start, real_divisors, complex_start, complex_divisors = _inputs(
        size, steps
    )
    real_reference = _reference(real_start, real_divisors)
    complex_reference = _reference(complex_start, complex_divisors)

    def ordinary(values, divisors):
        return _tensor(values), [_tensor(value) for value in divisors]

    def precision(values, divisors):
        return (
            Precision.of(_tensor(values), limbs),
            [Precision.of(_tensor(value), limbs) for value in divisors],
        )

    def rational(values, divisors):
        return (
            Rational.of(_tensor(values)),
            [Rational.of(_tensor(value)) for value in divisors],
        )

    def rational_precision(values, divisors):
        return (
            RationalPrecision.of(_tensor(values), limbs),
            [RationalPrecision.of(_tensor(value), limbs) for value in divisors],
        )

    def complex_precision(values, divisors):
        return (
            ComplexPrecision.of(_tensor(values), limbs),
            [ComplexPrecision.of(_tensor(value), limbs) for value in divisors],
        )

    def complex_rational(values, divisors):
        return (
            ComplexRational.of(_tensor(values)),
            [ComplexRational.of(_tensor(value)) for value in divisors],
        )

    def complex_rational_precision(values, divisors):
        return (
            ComplexRationalPrecision.of(_tensor(values), limbs),
            [
                ComplexRationalPrecision.of(_tensor(value), limbs)
                for value in divisors
            ],
        )

    return [
        Case("ordinary real", "-", "real", 1,
             lambda: ordinary(real_start, real_divisors), real_reference),
        Case(f"Precision[{limbs}]", "P", "real", limbs,
             lambda: precision(real_start, real_divisors), real_reference),
        Case("Rational", "R", "real", 2,
             lambda: rational(real_start, real_divisors), real_reference),
        Case(f"RationalPrecision[{limbs}]", "R+P", "real", 2 * limbs,
             lambda: rational_precision(real_start, real_divisors),
             real_reference),
        Case("ordinary complex", "C", "complex", 1,
             lambda: ordinary(complex_start, complex_divisors),
             complex_reference),
        Case(f"ComplexPrecision[{limbs}]", "C+P", "complex", 2 * limbs,
             lambda: complex_precision(complex_start, complex_divisors),
             complex_reference),
        Case("ComplexRational", "C+R", "complex", 4,
             lambda: complex_rational(complex_start, complex_divisors),
             complex_reference),
        Case(f"ComplexRationalPrecision[{limbs}]", "C+R+P", "complex",
             4 * limbs,
             lambda: complex_rational_precision(
                 complex_start, complex_divisors
             ), complex_reference),
    ]


def _measure(case: Case, warmups: int, repeats: int) -> Result:
    build_ms = _median_ms(case.build, warmups, repeats)
    start, divisors = case.build()
    chain_call = lambda: _chain(start, divisors)
    chain_ms = _median_ms(chain_call, warmups, repeats)
    result = chain_call()
    collapse_call = lambda: _observed(result)
    collapse_ms = _median_ms(collapse_call, warmups, repeats)
    produced = _values(collapse_call())
    error = max(
        (abs(actual - expected)
         for actual, expected in zip(produced, case.reference)),
        default=0.0,
    )
    scale = max((abs(value) for value in case.reference), default=0.0)
    if len(produced) != len(case.reference) or error > 5e-15 + 5e-13 * scale:
        raise AssertionError(
            f"{case.name} failed the reference check: max error {error:.3e}"
        )
    return Result(case, build_ms, chain_ms, collapse_ms, error)


def _print_table(results: list[Result]) -> None:
    baselines = {
        result.case.domain: result.chain_ms + result.collapse_ms
        for result in results
        if result.case.name.startswith("ordinary ")
    }
    print("| Variant | Set | Stored tensors | Build ms | Chain ms | Collapse ms | Observe x | Max abs error |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for result in results:
        observed_ms = result.chain_ms + result.collapse_ms
        slowdown = observed_ms / baselines[result.case.domain]
        print(
            f"| {result.case.name} | {result.case.features} | "
            f"{result.case.coefficient_tensors} | {result.build_ms:.3f} | "
            f"{result.chain_ms:.3f} | {result.collapse_ms:.3f} | "
            f"{slowdown:.2f}x | {result.maximum_error:.3e} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--limbs", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--backend", default="numpy",
        help="backend selected through AbstractTensor.use_backend (default: numpy)",
    )
    arguments = parser.parse_args()
    if min(arguments.size, arguments.steps, arguments.limbs,
           arguments.repeats) < 1 or arguments.warmups < 0:
        parser.error("size, steps, limbs, and repeats must be positive")

    with AbstractTensor.use_backend(arguments.backend):
        resolved = type(AbstractTensor.get_tensor([0.0])).__name__
        print(
            f"Python {platform.python_version()} | {platform.system()} "
            f"{platform.machine()} | AbstractTensor backend {resolved}"
        )
        print(
            f"AbstractTensor eager surface | {arguments.size} elements | "
            f"{arguments.steps} divisions | {arguments.limbs} limbs | "
            f"median of {arguments.repeats} runs after "
            f"{arguments.warmups} warmup(s)"
        )
        results = [
            _measure(case, arguments.warmups, arguments.repeats)
            for case in _cases(arguments.size, arguments.steps, arguments.limbs)
        ]
    _print_table(results)


if __name__ == "__main__":
    main()
