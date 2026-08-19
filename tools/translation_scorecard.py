"""How far does a program of given complexity get, and where does it stop?

A translation is a journey with stages, and "it works" is not a useful answer
for a pipeline whose failures are silent. This scores journeys instead: for
each authored program, how far along did it get, and at which stage did it
stop?

    LOWER        the canonical source compiler produced an IRModule
    MATERIALIZE  every function came back as Python
    EXECUTE      the emitted Python ran
    EQUIVALENT   it computed what the authored Python computes

The last stage is the one that matters and the one no compiler self-check
provides. Every defect this scorecard currently records reached EXECUTE --
they lower, they materialize, they run, and they are wrong. A journey that
stops at EQUIVALENT is exactly the failure class that costs days here, so the
scorecard reports the stage rather than a pass/fail.

Levels are ordered by what they demand of the compiler, not by line count, so
the first failing level names the frontier.

Run it::

    python tools/translation_scorecard.py
"""

from __future__ import annotations

import inspect
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STAGES = ("LOWER", "MATERIALIZE", "EXECUTE", "EQUIVALENT")


@dataclass(frozen=True)
class Journey:
    """One authored program and how to check what it should compute."""

    level: int
    name: str
    source: str
    probes: tuple[tuple[float, ...], ...]
    authored: Callable[..., float]
    note: str = ""


def _helper_preamble(body: str) -> str:
    """Every case keeps a second function.

    ``aot_compile``'s docstring records that the entrypoint must call at least
    one other function defined in the same source; a bare top-level function
    hits an unrelated failure in the graph coordinator. That is a property of
    the harness, not of the level being measured, so it is held constant.
    """

    return "def helper(a):\n    return a * 1.0\n\n" + body


def _loop(update: str, extra: str = "") -> str:
    return _helper_preamble(
        f"def train(w, n{extra}):\n"
        "    total = helper(w)\n"
        "    for _ in range(n):\n"
        f"{update}"
        "        total = w\n"
        "    return total\n"
    )


JOURNEYS: tuple[Journey, ...] = (
    Journey(
        0, "straight-line arithmetic",
        _helper_preamble("def train(x):\n    return helper(x) * 3.0 - 2.0\n"),
        ((1.0,), (-2.5,), (0.0,)),
        lambda x: x * 3.0 - 2.0,
    ),
    Journey(
        1, "two parameters",
        _helper_preamble("def train(x, y):\n    return helper(x) * y + x - y\n"),
        ((2.0, 3.0), (-1.0, 4.0)),
        lambda x, y: x * y + x - y,
    ),
    Journey(
        2, "nested calls",
        "def inner(a):\n    return a + 1.0\n\n"
        "def middle(a):\n    return inner(a) * 2.0\n\n"
        "def train(x):\n    return middle(x) - inner(x)\n",
        ((1.0,), (5.0,)),
        lambda x: (x + 1.0) * 2.0 - (x + 1.0),
        note="dead frame-storage formal; linking never completed",
    ),
    Journey(
        3, "division and power",
        _helper_preamble("def train(x):\n    return helper(x) / 4.0 + x ** 2.0\n"),
        ((2.0,), (3.0,)),
        lambda x: x / 4.0 + x ** 2.0,
        note="fixed: dependency-ordered region scheduling",
    ),
    Journey(
        4, "loop, one carried value",
        _loop("        next_w = w * 0.9\n        w = next_w\n"),
        ((2.0, 3), (1.0, 0), (5.0, 7)),
        lambda w, n: _repeat(w, n, lambda v: v * 0.9),
    ),
    Journey(
        5, "loop, compound body",
        _loop("        next_w = (w * 2.0 - 1.0) * 0.5\n        w = next_w\n"),
        ((3.0, 2), (1.0, 5)),
        lambda w, n: _repeat(w, n, lambda v: (v * 2.0 - 1.0) * 0.5),
        note="",
    ),
    Journey(
        6, "loop, two carried values",
        _loop(
            "        next_m = m * 0.5 + w\n"
            "        next_w = w - 0.1 * next_m\n"
            "        m = next_m\n"
            "        w = next_w\n",
            extra=", m",
        ),
        ((1.0, 4, 0.0), (3.0, 6, -1.0)),
        lambda w, n, m: _two_carried(w, m, n),
        note="fixed: parameters a loop reads materialize pre-snapshot",
    ),
    Journey(
        7, "adam-shaped loop, three carried values",
        _helper_preamble(
            "def train(w, m, v, g, n):\n"
            "    total = helper(w)\n"
            "    for _ in range(n):\n"
            "        next_m = 0.9 * m + 0.1 * g\n"
            "        next_v = 0.999 * v + 0.001 * (g * g)\n"
            "        denominator = (next_v ** 0.5) + 1e-8\n"
            "        next_w = w - 0.05 * next_m / denominator\n"
            "        m = next_m\n"
            "        v = next_v\n"
            "        w = next_w\n"
            "        total = w\n"
            "    return total\n"
        ),
        ((1.0, 0.0, 0.0, 0.3, 5), (2.0, 0.1, 0.01, -0.2, 10), (0.5, 0.0, 0.0, 1.0, 0)),
        lambda w, m, v, g, n: _adam(w, m, v, g, n),
        note="the optimizer update itself, compiled; the goal shape",
    ),
    Journey(
        8, "loop, carried value through a call",
        "def update(a):\n    return a - 0.05 * a\n\n"
        "def train(w, n):\n"
        "    total = update(w)\n"
        "    for _ in range(n):\n"
        "        next_w = update(w)\n"
        "        w = next_w\n"
        "        total = w\n"
        "    return total\n",
        ((2.0, 3),),
        lambda w, n: _repeat(w, n, lambda v: v - 0.05 * v),
        note="round trip through a call: the body region is elided",
    ),
)


def _repeat(value: float, count: int, step: Callable[[float], float]) -> float:
    for _ in range(int(count)):
        value = step(value)
    return value


def _adam(w: float, m: float, v: float, g: float, n: int) -> float:
    for _ in range(int(n)):
        next_m = 0.9 * m + 0.1 * g
        next_v = 0.999 * v + 0.001 * (g * g)
        next_w = w - 0.05 * next_m / ((next_v ** 0.5) + 1e-8)
        m, v, w = next_m, next_v, next_w
    return w


def _two_carried(w: float, m: float, n: int) -> float:
    for _ in range(int(n)):
        next_m = m * 0.5 + w
        next_w = w - 0.1 * next_m
        m, w = next_m, next_w
    return w


def score(journey: Journey) -> tuple[str, str]:
    """Return ``(stage_reached, detail)`` for one journey."""

    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_python_materializer import materialize_ir_module

    prefix = f"sc{journey.level}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            module, _outputs, _exports = lower_ast_source_to_ssa(
                journey.source, "train", name=prefix
            )
        except Exception as error:
            return "LOWER", f"{type(error).__name__}: {str(error)[:70]}"

    try:
        emitted, skipped = materialize_ir_module(module)
    except Exception as error:
        return "MATERIALIZE", f"{type(error).__name__}: {str(error)[:70]}"
    if skipped:
        return "MATERIALIZE", f"not materialized: {sorted(skipped)}"

    namespace: dict[str, Any] = {}
    try:
        exec(compile(emitted, "<scorecard>", "exec"), namespace)
        compiled = namespace[f"{prefix}__train"]
        formals = list(inspect.signature(compiled).parameters)
        authored_names = list(inspect.signature(journey.authored).parameters)
        results = []
        for probe in journey.probes:
            bound = dict(zip(authored_names, probe))
            if set(formals) == set(authored_names):
                results.append(compiled(**bound))
            elif len(formals) == len(probe):
                results.append(compiled(*probe))
            else:
                return "EXECUTE", (
                    f"{len(probe)} authored parameters became {len(formals)}: "
                    f"{formals}"
                )
    except Exception as error:
        return "EXECUTE", f"{type(error).__name__}: {str(error)[:70]}"

    worst = 0.0
    for probe, produced in zip(journey.probes, results):
        expected = journey.authored(*probe)
        worst = max(worst, abs(float(produced) - float(expected)))
    if worst > 1e-9:
        return "EQUIVALENT", f"worst disagreement {worst:.3e}"
    return "PASSED", f"max disagreement {worst:.1e}"


def report() -> list[tuple[Journey, str, str]]:
    return [(journey, *score(journey)) for journey in JOURNEYS]


def main() -> int:
    rows = report()
    width = max(len(journey.name) for journey, _stage, _detail in rows)
    print(f"{'lvl':>3}  {'journey':<{width}}  {'stopped at':<12}  detail")
    print("-" * (3 + 2 + width + 2 + 12 + 2 + 44))
    for journey, stage, detail in rows:
        marker = "ok  " if stage == "PASSED" else "STOP"
        print(f"{journey.level:>3}  {journey.name:<{width}}  "
              f"{marker} {stage:<7}  {detail}")
        if journey.note and stage != "PASSED":
            print(f"{'':>3}  {'':<{width}}  {'':<12}  -> {journey.note}")
    passed = sum(1 for _journey, stage, _detail in rows if stage == "PASSED")
    print()
    print(f"{passed}/{len(rows)} journeys equivalent end to end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
