"""Timing and sub-ulp accuracy for the compiled signal cores.

Sweeps the permutation: core x limb width x load size, measuring wall time
per element and the FULL ulp distribution rather than a worst case. A worst
case over a few points says almost nothing -- a core can be correctly
rounded on 999 arguments out of 1000 and still be wrong, and "wrong" is what
matters. So the accuracy column reports the fraction of arguments that come
back correctly rounded, alongside the tail.

Accuracy is measured against exact rational evaluation of the SAME
polynomial (``fractions.Fraction``, no floating point anywhere in the
reference), so what is measured is the arithmetic's error and not the
series' truncation. Truncation is ``signal_symbolic.order_for``'s job and is
already bounded there; conflating the two would credit or blame this
pipeline for a different decision.

Timing and accuracy use different sample counts on purpose: timing wants
large arrays to amortise call overhead, exact rational comparison costs
milliseconds per point, and running both at the same size would mean either
a useless timing or an hour of Fractions.

Usage::

    python tools/benchmark_precision_cores.py
    python tools/benchmark_precision_cores.py --cores sin cos --sizes 1000 100000
    python tools/benchmark_precision_cores.py --widths 1 2 4 --accuracy-samples 400
"""
from __future__ import annotations

import argparse
import pathlib
import random
import re
import statistics
import sys
import time
from fractions import Fraction

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.common.tensors import signal_symbolic as _proof  # noqa: E402
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402
from src.compiler.ir_identities import (  # noqa: E402
    carry_precision_through_ssa,
    lower_precision_operations,
    mark_precision_sections,
    reduce_precision_operations,
)
from src.compiler.ssa_llvm_backend import (  # noqa: E402
    compile_artifact,
    emit_ssa_function_to_llvm,
    prepare_artifact_execution,
)
from src.transmogrifier.ssa import IRModule  # noqa: E402

#: One ulp of a binary64 significand, as a RELATIVE spacing. Half of this is
#: the correctly-rounded bound: an error below it means no other double is
#: closer to the true value, which is the only accuracy claim worth making.
ULP = 2.0 ** -52


def core_names() -> tuple[str, ...]:
    return tuple(sorted(_proof.CORE_RADII))


def _pack_source(name: str, width: int) -> tuple[str, str, tuple[str, ...], int]:
    """The core plus a loop shell, annotated at ``width`` limbs.

    The shell is what makes timing meaningful: one call covers the whole
    array, so the measurement is the arithmetic rather than the cost of
    reaching it.
    """

    degree = _proof.order_to_degree(
        name, _proof.order_for(name, _proof.CORE_RADII[name], digits=17)
    )
    body = _proof.materialised_source(name, degree)
    header = body.splitlines()[0]
    core = header.split("(")[0][4:].strip()
    parameters = tuple(re.findall(r"\w+", header.split("(")[1].split(")")[0]))
    coefficients = tuple(p for p in parameters if p[0] == "c" and p[1:].isdigit())

    if width > 1:
        body = (
            "def " + core + "("
            + ", ".join(p + ": Precision[%d]" % width for p in parameters)
            + "):" + body[len(header):]
        )

    structure = _proof.TRANSCENDENTALS[name]["structure"]
    structural = "z * z" if structure in ("odd", "even") else "z"
    annotate = ": Precision[%d]" % width if width > 1 else ""

    # The core is INLINED into the loop rather than called.
    #
    # Not a convenience: precision does not cross a call boundary yet. A
    # call's actual arguments are bound while parameters are being widened,
    # which is before any local's limbs have been computed, so a locally
    # derived precision value (here `s` and `z`) arrives at the callee with
    # its high limb duplicated into the low one. The standalone core measures
    # 0.47 ulp; the same core reached through a call returns denormal noise.
    #
    # Inlining is also what a deployed kernel looks like, so this measures
    # the intended shape rather than working around it -- but the call path
    # is a real gap and is recorded as one.
    statements = [
        line for line in body.splitlines()[1:] if line.strip()
    ]
    returned = statements[-1].split("return")[-1].strip()
    inlined = [("    " + line) for line in statements[:-1]]

    shell = "\n".join((
        "",
        "def %s_pack(x%s, y%s, n, %s):" % (
            core, annotate, annotate,
            ", ".join(c + annotate for c in coefficients),
        ),
        "    for i in range(n):",
        "        z = x[i]",
        "        s = %s" % structural,
        *inlined,
        "        y[i] = %s" % returned,
        "    return y",
    ))
    header_line = body.splitlines()[0]
    return shell, core + "_pack", parameters, degree



def build(name: str, width: int, root: pathlib.Path):
    """Compile one core at one width and return what is needed to run it."""

    text, entry, parameters, degree = _pack_source(name, width)
    directory = root / ("%s_w%d" % (name, width))
    directory.mkdir(parents=True, exist_ok=True)

    module = lower_ast_source_to_ssa(text, entry)
    functions = getattr(module, "functions", None) or module[0].functions
    identities = {}
    if width > 1:
        carry_precision_through_ssa(functions)
        identities = reduce_precision_operations(functions)
        mark_precision_sections(functions)
        lower_precision_operations(functions)
    if not isinstance(module, IRModule):
        module = IRModule(functions)

    wrapper = [k for k in functions if k.endswith("__" + entry)][0]
    artifact = emit_ssa_function_to_llvm(module, wrapper)
    if artifact.shortfalls:
        raise RuntimeError(
            "%s w%d: %s" % (name, width, artifact.shortfalls[0].reason[:80])
        )
    return (
        compile_artifact(artifact, directory=directory),
        artifact,
        _roles(functions, wrapper),
        parameters,
        degree,
        identities,
    )


def _roles(functions, wrapper) -> dict:
    """Which formal is the input array, which the output, which the count.

    DERIVED, not assumed. The emitted formal order is not the authored one
    -- a pack written ``(x, y, n, c0...)`` arrives as ``(n, x, y, c0...)`` --
    and nothing in a formal's type distinguishes the input array from the
    output, since both are float64 of shape (). Guessing that layout is what
    made every earlier reading of these kernels wrong.

    An array is told apart by what is done THROUGH it: the base of a
    GetElementPtr feeding a Load is read, the base of one feeding a Store is
    written. The count is the only integer. Whatever is left is a
    coefficient, and those keep their relative order.
    """

    read: set[int] = set()
    written: set[int] = set()
    for function in functions.values():
        for block in function.blocks.values():
            pointers = {
                int(i.res.id): int(i.args[0].id)
                for i in block.instrs
                if str(i.op) == "GetElementPtr" and i.args and i.res is not None
            }
            for instruction in block.instrs:
                if str(instruction.op) == "Load" and instruction.args:
                    base = pointers.get(int(instruction.args[0].id))
                    if base is not None:
                        read.add(base)
                elif (
                    str(instruction.op) == "Store"
                    and len(instruction.args) >= 2
                ):
                    base = pointers.get(int(instruction.args[1].id))
                    if base is not None:
                        written.add(base)

    formals = list(functions[wrapper].args)
    roles = {"coefficients": [], "extras": []}
    for value in formals:
        identity = int(value.id)
        if str(value.dtype or "").startswith("int"):
            roles["count"] = value
        elif identity in written:
            roles["output"] = value
        elif identity in read:
            roles["input"] = value
        else:
            roles["coefficients"].append(value)
    return roles


def _feeds(roles, width, coefficients, x, y, count):
    """Bind the formals by ROLE rather than by position."""

    feeds = {
        int(roles["count"].id): np.int32(count),
        int(roles["input"].id): x,
        int(roles["output"].id): y,
    }
    for value, coefficient in zip(roles["coefficients"], coefficients):
        feeds[int(value.id)] = np.float64(coefficient)
    # A coefficient formal beyond the authored count is an appended limb.
    # Zero: an exactly-representable input has nothing below its leading
    # double, and a coefficient captured at one limb is measured AS one limb.
    for value in roles["coefficients"][len(coefficients):]:
        feeds[int(value.id)] = np.float64(0.0)
    return feeds


def measure(name, width, root, sizes, repeats, accuracy_samples):
    built, artifact, roles, parameters, degree, identities = build(
        name, width, root
    )
    exact = _proof.structured_coefficients(name, degree)
    coefficients = [float(c) for c in exact]
    structure = _proof.TRANSCENDENTALS[name]["structure"]
    radius = _proof.CORE_RADII[name]
    output_id = int(roles["output"].id)

    def call(x, count):
        y = np.zeros(count * width, dtype=np.float64)
        feeds = _feeds(roles, width, coefficients, x, y, count)
        execution = prepare_artifact_execution(built, feeds)
        execution.run()
        return np.asarray(execution.buffers[output_id]).ravel()

    timings = {}
    for count in sizes:
        x = np.random.default_rng(11).uniform(
            -radius * 0.98, radius * 0.98, count * width
        )
        y = np.zeros(count * width, dtype=np.float64)
        # Prepared ONCE and outside the timed region. Allocating the public
        # ABI and marshalling buffers costs hundreds of microseconds; leaving
        # it inside made a compiled Horner look like 248 ns per element, some
        # forty times its actual cost, because at small counts the setup IS
        # the measurement.
        execution = prepare_artifact_execution(
            built, _feeds(roles, width, coefficients, x, y, count)
        )
        execution.run()  # warm pages and any lazy binding
        best = float("inf")
        for _ in range(repeats):
            start = time.perf_counter()
            execution.run()
            best = min(best, time.perf_counter() - start)
        timings[count] = best / count

    # ONE call for every accuracy point. Calling per point limited this to a
    # couple of dozen samples, which cannot distinguish a core that is
    # correctly rounded from one that is correctly rounded on the twenty
    # arguments that were looked at. The kernel is an array kernel; using it
    # as one costs nothing and buys four thousand points.
    rng = random.Random(7)
    points = [rng.uniform(-radius * 0.98, radius * 0.98)
              for _ in range(accuracy_samples)]
    packed = np.zeros(accuracy_samples * width, dtype=np.float64)
    packed[::width] = points
    produced = call(packed, accuracy_samples)

    errors: list[float] = []
    exact_hits = 0
    counted = 0
    for index, z in enumerate(points):
        structural = z * z if structure in ("odd", "even") else z
        accumulated = Fraction(0)
        for coefficient in reversed(exact):
            accumulated = accumulated * Fraction(structural) + coefficient
        truth = accumulated * Fraction(z) if structure == "odd" else accumulated
        if not truth:
            continue
        # A limbed result is the SUM of its limbs; comparing only the high
        # limb would score the representation on half of itself.
        got = Fraction(0)
        for limb in range(width):
            value = float(produced[index * width + limb])
            if value == value:  # NaN poisons the Fraction, and says enough
                got += Fraction(value)
            else:
                got = None
                break
        if got is None:
            errors.append(float("inf"))
            counted += 1
            continue
        counted += 1
        # Bit-exact means: no other double is closer to the true value. That
        # is the only accuracy claim worth making, and it is not the same as
        # a small average.
        if float(truth) == float(got):
            exact_hits += 1
        errors.append(float(abs(got - truth) / abs(truth)) / ULP)
    return timings, errors, (exact_hits, counted)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", nargs="*", default=[
        "sin", "cos", "exp", "atan", "tanh", "sinh", "asin",
    ])
    parser.add_argument("--widths", nargs="*", type=int, default=[1, 2])
    parser.add_argument("--sizes", nargs="*", type=int,
                        default=[1_000, 10_000, 100_000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--accuracy-samples", type=int, default=120)
    parser.add_argument("--scratch", default=None)
    options = parser.parse_args()

    root = pathlib.Path(
        options.scratch or (pathlib.Path(__file__).resolve().parent
                            / "_precision_benchmark")
    )
    sizes = list(options.sizes)

    print("ns per element" + " " * 8 + "".join(
        "%12s" % ("n=%d" % n) for n in sizes
    ) + "   corr.rounded   median   p99      max")
    print("-" * (24 + 12 * len(sizes) + 44))

    for name in options.cores:
        for width in options.widths:
            label = "%s w%d" % (name, width)
            try:
                timings, errors, _identities = measure(
                    name, width, root, sizes, options.repeats,
                    options.accuracy_samples,
                )
            except Exception as error:  # noqa: BLE001 -- report, never hide
                print("%-22s %s" % (label, type(error).__name__ + ": "
                                    + str(error)[:60]))
                continue
            correct = sum(1 for e in errors if e <= 0.5) / max(len(errors), 1)
            ordered = sorted(errors)
            median = statistics.median(ordered) if ordered else float("nan")
            p99 = ordered[int(len(ordered) * 0.99) - 1] if ordered else float("nan")
            print("%-22s%s   %10.1f%%%9.3f%9.3f%9.3f" % (
                label,
                "".join("%12.1f" % (timings[n] * 1e9) for n in sizes),
                correct * 100.0, median, p99, max(ordered or [float("nan")]),
            ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
