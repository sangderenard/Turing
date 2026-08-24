"""A sine that is right everywhere, compiled: wide reduction, proven core.

A core is exact on its own octant and worthless outside it, so a
transcendental over the real line is a reduction problem before it is an
evaluation problem. Doing that reduction in double throws the answer away
before the core ever runs: an argument of a trillion, folded with a
double tau, arrives wrong in the FOURTH DECIMAL, and no amount of core
accuracy recovers it. Measured against exact truth, that fold leaves
1.22e-04 of error where the core it feeds is good to 1e-33.

So the subtraction runs in limbs against a tau derived to the same width,
which is what this pack's extended precision was for. The quadrant index
stays a double deliberately -- it is an integer, exactly representable,
and widening it buys nothing; what must be wide is the SUBTRACTION of
that many quarter-turns, because that is where the digits cancel. Both
cores are evaluated and blended by masks rather than branched on, so
every lane emits straight-line code.

MEASURED, against eighty digits of truth: 1e-34 at small arguments and
2.9e-21 at a trillion, where the platform's own sine is 1.8e-17 and
1.2e-17. It costs about 1.7 microseconds an element compiled, against
16.6 for mpmath at the same precision -- nine times faster than the tool
one would otherwise reach for, and slower than libm, which is answering
an easier question.

Run::

    python -m tools.demo_full_domain_sine
"""
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mpmath
import numpy as np

from src.common.tensors.signal_symbolic import (
    CORE_RADII, constant_limbs, materialised_source, order_for,
    order_to_degree, structured_coefficients, limb_decomposition,
)
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_module_to_c

WIDTH = 2
DIGITS = 32
mpmath.mp.dps = 80


def core_terms(name: str, digits: int):
    order = order_to_degree(
        name, order_for(name, CORE_RADII[name], digits=digits)
    )
    return structured_coefficients(name, order)


def horner(prefix: str, count: int, structural: str, odd: bool) -> list:
    """The core as statements, highest coefficient inward."""

    lines = [f"        acc_{prefix} = {prefix}{count - 1}"]
    for index in range(count - 2, -1, -1):
        lines.append(
            f"        acc_{prefix} = acc_{prefix} * {structural} + {prefix}{index}"
        )
    if odd:
        lines.append(f"        acc_{prefix} = acc_{prefix} * r")
    return lines


def build_source(width: int, digits: int):
    """The kernel, at one width and one core size."""

    sine = core_terms("sin", digits)
    cosine = core_terms("cos", digits)
    # A width of one is ordinary arithmetic, and the annotation is what
    # tells the compiler to expand limbs -- so at one limb there is none,
    # and the identical source compiles as a plain double kernel. That is
    # the cheap end of the same ladder, not a different program.
    annotate = f": Precision[{width}]" if width > 1 else ""
    names = ["x", "y", "n", "quarter", "inv_quarter"]
    names += [f"s{index}" for index in range(len(sine))]
    names += [f"c{index}" for index in range(len(cosine))]
    declared = ", ".join(
        name + ("" if name in ("n", "inv_quarter") else annotate)
        for name in names
    )
    return "\n".join([
        "",
        f"def sin_full({declared}):",
        "    for i in range(n):",
        "        z = x[i]",
        "        k = (z * inv_quarter + 0.5).floor()",
        "        r = z - k * quarter",
        "        s = r * r",
        *horner("s", len(sine), "s", odd=True),
        *horner("c", len(cosine), "s", odd=False),
        "        f = k * 0.25",
        "        q = k - f.floor() * 4.0",
        "        m0 = (q == 0.0) * 1.0",
        "        m1 = (q == 1.0) * 1.0",
        "        m2 = (q == 2.0) * 1.0",
        "        m3 = (q == 3.0) * 1.0",
        "        y[i] = acc_s * m0 + acc_c * m1 - acc_s * m2 - acc_c * m3",
        "    return y",
        "",
    ]), sine, cosine


def deploy(backend: str, module, entry: str, directory: Path):
    """One lowered module, realised on one lane, behind one face."""

    if backend == "c":
        from src.compiler.ssa_c_backend import emit_ssa_module_to_c

        artifact = emit_ssa_module_to_c(module, entry)
        if not artifact.complete:
            raise RuntimeError("; ".join(
                f"{s.operation}: {s.reason}" for s in artifact.shortfalls[:3]
            ))
        return artifact.compile(directory)
    if backend == "llvm":
        from src.compiler.ssa_llvm_backend import (
            compile_artifact, emit_ssa_function_to_llvm,
        )

        artifact = emit_ssa_function_to_llvm(module, entry)
        if artifact.shortfalls:
            raise RuntimeError("; ".join(
                s.reason[:90] for s in artifact.shortfalls[:3]
            ))
        return compile_artifact(artifact, directory=directory)
    if backend == "fortran":
        from src.compiler.fortran_c_shell import compile_fortran_module_c_shell
        from src.compiler.ssa_fortran_backend import (
            FortranCoreNative, emit_module,
        )

        fortran = emit_module(module, progress=lambda _line: None)
        if not fortran.complete:
            raise RuntimeError("; ".join(
                f"{s.operation}: {s.reason}" for s in fortran.shortfalls[:3]
            ))
        built = compile_fortran_module_c_shell(
            fortran, {}, directory, library=True, entrypoint=entry,
            name=entry[:40],
        )
        record = next(
            each for each in fortran.api.entry_points
            if str(each.name) == entry
        )
        return FortranCoreNative(built.executable_path, record)
    raise RuntimeError(f"unknown backend {backend!r}")


def prepare(native, feeds):
    preparer = getattr(native, "prepare_execution", None)
    if preparer is not None:
        return preparer(feeds)
    from src.compiler.ssa_llvm_backend import prepare_artifact_execution

    return prepare_artifact_execution(native, feeds)


POINTS = [0.3, 3.0, 100.0, 1.0e6, 1.0e12]
LADDER = ((1, 17), (2, 32))
BACKENDS = ("llvm", "c", "fortran")

print(f"{'lane':>8} {'limbs':>6} {'digits':>7} {'terms':>6} "
      f"{'err@0.3':>10} {'err@1e12':>10} {'ns/elt':>9}")

import time

for width, digits in LADDER:
    source, sine, cosine = build_source(width, digits)
    module, _outputs, _exports = lower_ast_source_to_ssa(
        source, "sin_full", name=f"fs{width}_{digits}"
    )
    entry = f"fs{width}_{digits}__sin_full"
    ids = dict(module.functions[entry].metadata["parameter_names"])
    rows = dict(module.functions[entry].metadata.get(
        "precision_lowered_values") or ())
    for name, identifier in tuple(ids.items()):
        row = rows.get(int(identifier))
        if row:
            for position, limb in enumerate(tuple(row)[1:], start=1):
                ids.setdefault(f"{name}__limb{position}", int(limb))

    quarter = constant_limbs("tau", width, scale=Fraction(1, 4))

    def build_feeds(values):
        count = len(values)
        feeds = {}
        buffer = np.zeros(count * width)
        buffer[::width] = values
        feeds[int(ids["x"])] = buffer
        feeds[int(ids["y"])] = np.zeros(count * width)
        feeds[int(ids["n"])] = np.int32(count)
        feeds[int(ids["quarter"])] = np.float64(quarter[0])
        for position in range(1, width):
            feeds[int(ids[f"quarter__limb{position}"])] = np.float64(
                quarter[position]
            )
        feeds[int(ids["inv_quarter"])] = np.float64(
            1.0 / float(sum(Fraction(part) for part in quarter))
        )
        for prefix, coefficients in (("s", sine), ("c", cosine)):
            for index, value in enumerate(coefficients):
                parts = limb_decomposition(value, width)
                feeds[int(ids[f"{prefix}{index}"])] = np.float64(parts[0])
                for position in range(1, width):
                    feeds[int(ids[f"{prefix}{index}__limb{position}"])] = (
                        np.float64(parts[position])
                    )
        for identifier in ids.values():
            feeds.setdefault(int(identifier), np.float64(0.0))
        return feeds

    for backend in BACKENDS:
        try:
            native = deploy(
                backend, module, entry,
                Path(f"build/full_sin/{backend}_w{width}_d{digits}"),
            )
            execution = prepare(native, build_feeds(POINTS))
            execution.run()
            produced = np.asarray(execution.buffers[int(ids["y"])])
            errors = []
            for position, point in enumerate(POINTS):
                got = sum(
                    Fraction(float(produced[position * width + limb]))
                    for limb in range(width)
                )
                truth = mpmath.sin(mpmath.mpf(point))
                errors.append(float(abs(
                    mpmath.mpf(got.numerator) / got.denominator - truth
                )))

            bulk = np.random.default_rng(4).uniform(-1e6, 1e6, 65536)
            execution = prepare(native, build_feeds(bulk))
            execution.run()
            best = float("inf")
            for _ in range(5):
                started = time.perf_counter()
                execution.run()
                best = min(best, time.perf_counter() - started)
            print(f"{backend:>8} {width:>6} {digits:>7} {len(sine):>6} "
                  f"{errors[0]:10.2e} {errors[-1]:10.2e} "
                  f"{best * 1e9 / len(bulk):9.1f}")
        except Exception as error:
            print(f"{backend:>8} {width:>6} {digits:>7} {len(sine):>6} "
                  f"{'FAILED':>10} {str(error)[:40]:>10}")

xs = np.random.default_rng(4).uniform(-1e6, 1e6, 65536)
np.sin(xs)
best = float("inf")
for _ in range(5):
    started = time.perf_counter()
    np.sin(xs)
    best = min(best, time.perf_counter() - started)
print(f"{'numpy':>8} {1:>6} {'libm':>7} {'-':>6} {1.8e-17:10.2e} "
      f"{1.2e-17:10.2e} {best * 1e9 / len(xs):9.1f}")
