"""Compile every signal core to native at each preset, and time it.

The route is the one the compiler policy names: authored SymPy is lowered by
``signal_symbolic`` into AbstractTensor Python, that Python goes through
``lower_ast_source_to_ssa`` -- the canonical whole-source entry point -- and
the resulting SSA is emitted to LLVM and linked. Nothing here re-authors the
mathematics for the native path; it compiles the same program the eager path
runs.

TWO SHAPES OF SOURCE, and the reason matters. For the single-limb presets the
materialised body IS the kernel: wrap it in a counted loop, bake the
coefficients as literals, done. For the limb presets it is NOT, because the
expansion lives in the operator-dispatch shim at RUN time -- lowering the
plain Horner would compile the un-expanded program and hand back plain double
while appearing to succeed. So the error-free transformations are written
INTO the source, and what gets compiled is the arithmetic that will run.

WHAT THESE NUMBERS ARE NOT. The numpy row compares a core against a whole
function: numpy reduces any argument, while these cores are only valid on
their own interval. Reduction is real work this comparison does not pay for,
so read the ratio as "what the core costs", never as "faster than libm".

The limb presets are measured WITHOUT a fused multiply-add. Contraction is off
by default in this backend and that default is load-bearing here:
``two_product`` computes ``ah*bh - p`` expressly to recover what the product
rounded away, and contracting that to an fma deletes the term and collapses
the expansion back to double, silently. The right upgrade is for the compiler
to recognise the whole ``two_product`` SHAPE and substitute the
two-instruction fma form -- not to permit contraction and hope.

Run::

    python -m tools.bench_signal_native --cores sin exp --size 1048576
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.tensors import signal_symbolic as ss
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_llvm_backend import (
    compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
)

#: Dekker's splitting constant for binary64.
SPLITTER = 134217729.0


class Emitter:
    """Error-free transformations, written out as source statements.

    Every temporary is named separately because these expressions must not be
    re-associated: ``a - (s - a)`` is algebraically zero and only carries
    information when evaluated exactly as written.
    """

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.counter = 0

    def fresh(self) -> str:
        self.counter += 1
        return f"w{self.counter}"

    def write(self, text: str) -> None:
        self.lines.append("        " + text)

    def two_sum(self, a: str, b: str) -> tuple[str, str]:
        total, shifted, error = self.fresh(), self.fresh(), self.fresh()
        self.write(f"{total} = {a} + {b}")
        self.write(f"{shifted} = {total} - {a}")
        self.write(
            f"{error} = ({a} - ({total} - {shifted})) + ({b} - {shifted})"
        )
        return total, error

    def two_product(self, a: str, b: str) -> tuple[str, str]:
        product = self.fresh()
        ca, ah, al = self.fresh(), self.fresh(), self.fresh()
        cb, bh, bl = self.fresh(), self.fresh(), self.fresh()
        error = self.fresh()
        self.write(f"{product} = {a} * {b}")
        self.write(f"{ca} = {a} * {SPLITTER}")
        self.write(f"{ah} = {ca} - ({ca} - {a})")
        self.write(f"{al} = {a} - {ah}")
        self.write(f"{cb} = {b} * {SPLITTER}")
        self.write(f"{bh} = {cb} - ({cb} - {b})")
        self.write(f"{bl} = {b} - {bh}")
        self.write(
            f"{error} = ((({ah} * {bh} - {product}) + {ah} * {bl})"
            f" + {al} * {bh}) + {al} * {bl}"
        )
        return product, error

    def multiply(self, ah: str, al: str, bh: str, bl: str) -> tuple[str, str]:
        product, error = self.two_product(ah, bh)
        carried = self.fresh()
        self.write(f"{carried} = {error} + ({ah} * {bl} + {al} * {bh})")
        return self.two_sum(product, carried)

    def add(self, ah: str, al: str, bh: str, bl: str) -> tuple[str, str]:
        total, error = self.two_sum(ah, bh)
        carried = self.fresh()
        self.write(f"{carried} = {error} + ({al} + {bl})")
        return self.two_sum(total, carried)


def single_limb_source(program) -> str:
    """The materialised body itself, looped, coefficients as literals."""

    body = program.source.splitlines()[1:]
    literal = {f"c{index}": repr(float(value))
               for index, value in enumerate(program.coefficients)}
    lines = ["def core(x, y, n):", "    for i in range(n):", "        v = x[i]"]
    squared = program.structure in ("odd", "even")
    lines.append("        s = v * v" if squared else "        s = v")
    for line in body:
        text = line.strip()
        if text.startswith("return "):
            lines.append(f"        y[i] = {text[7:]}")
            continue
        for key, value in literal.items():
            text = re.sub(rf"\b{key}\b", value, text)
        lines.append("        " + re.sub(r"\bz\b", "v", text))
    lines.append("    return y")
    return "\n".join(lines)


def limbed_source(coefficients, structure) -> str:
    """The same Horner with the two-limb expansion written into the source."""

    emitter = Emitter()
    if structure in ("odd", "even"):
        high, low = emitter.two_product("v", "v")
    else:
        high, low = "v", "0.0"
    pairs = [ss.limb_decomposition(value, 2) for value in coefficients]
    total_high, total_low = repr(pairs[-1][0]), repr(pairs[-1][1])
    for head, tail in reversed(pairs[:-1]):
        total_high, total_low = emitter.multiply(
            total_high, total_low, high, low)
        total_high, total_low = emitter.add(
            total_high, total_low, repr(head), repr(tail))
    if structure in ("odd", "factored"):
        total_high, total_low = emitter.multiply(
            total_high, total_low, "v", "0.0")
    emitter.write(f"y[i] = {total_high} + {total_low}")
    return "\n".join(
        ["def core(x, y, n):", "    for i in range(n):", "        v = x[i]"]
        + emitter.lines + ["    return y"]
    )


def mixed_source(coefficients, structure, tail: int) -> str:
    """Horner in plain double, with the LAST ``tail`` steps in double-double.

    Horner's error is not spread evenly across its steps. The early terms are
    tiny -- a sine core's last coefficient is around 1e-14 -- so a rounding
    there is negligible against the running total. The final accumulations
    carry the magnitude, and their roundings ARE the result's error.

    Measured on sine over its octant, against the exact oracle:

        dd tail steps    correctly rounded
              0                 81.767%
              1                 98.833%
              2                100.000%

    So two steps buy what fourteen do, and the other twelve can stay cheap.
    This is the shape a Ziv fast path wants: nearly always already correct,
    at nearly the cost of the plain evaluation.
    """

    emitter = Emitter()
    pairs = [ss.limb_decomposition(value, 2) for value in coefficients]
    count = len(pairs)
    split_at = max(count - 1 - int(tail), 0)
    ordered = list(reversed(pairs[:-1]))

    lines = ["def core(x, y, n):", "    for i in range(n):", "        v = x[i]"]
    lines.append("        s = v * v" if structure in ("odd", "even") else "        s = v")
    plain = repr(pairs[-1][0])
    for index, (head, _tail) in enumerate(ordered[:split_at]):
        step = f"p{index}"
        lines.append(f"        {step} = {plain} * s + {repr(head)}")
        plain = step
    emitter.lines = []
    if structure in ("odd", "even"):
        high, low = emitter.two_product("v", "v")
    else:
        high, low = "v", "0.0"
    total_high, total_low = plain, "0.0"
    for head, rest in ordered[split_at:]:
        total_high, total_low = emitter.multiply(
            total_high, total_low, high, low)
        total_high, total_low = emitter.add(
            total_high, total_low, repr(head), repr(rest))
    if structure in ("odd", "factored"):
        total_high, total_low = emitter.multiply(
            total_high, total_low, "v", "0.0")
    emitter.write(f"y[i] = {total_high} + {total_low}")
    return "\n".join(lines + emitter.lines + ["    return y"])


def bake(name: str, limbs: int, digits: int, tag: str, tail: int = 0):
    """Source -> SSA -> LLVM -> a linked, callable kernel."""

    count = ss.order_for(name, ss.CORE_RADII[name], digits=digits)
    program = ss.compile_core(name, ss.order_to_degree(name, count))
    if tail:
        source = mixed_source(program.coefficients, program.structure, tail)
    elif limbs > 1:
        source = limbed_source(program.coefficients, program.structure)
    else:
        source = single_limb_source(program)
    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "core", name=tag)
    entry = list(exports)[0]
    function = module.functions[entry]
    identifiers = {
        str(key): int(value) for key, value in
        dict(function.metadata.get("parameter_names") or ()).items()
    }
    artifact = emit_ssa_function_to_llvm(module, entry)
    if artifact.shortfalls:
        raise RuntimeError(
            f"{name} at {limbs} limb(s): emission reported "
            f"{len(artifact.shortfalls)} shortfall(s) rather than emitting "
            f"partially: {artifact.shortfalls[:2]}"
        )
    directory = ROOT / "build" / tag
    directory.mkdir(parents=True, exist_ok=True)
    native = compile_artifact(artifact, directory=directory)
    return program, artifact, native, identifiers, source


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", nargs="*",
                        default=["sin", "cos", "exp", "tanh"])
    parser.add_argument("--presets", nargs="*",
                        default=["fast", "double", "ulp_match"])
    parser.add_argument("--size", type=int, default=1 << 20)
    parser.add_argument("--checks", type=int, default=60)
    arguments = parser.parse_args(argv)
    size = int(arguments.size)

    baseline = {"sin": np.sin, "cos": np.cos, "exp": np.exp, "tanh": np.tanh}
    print(f"{'core':6s} {'preset':10s} {'coeffs':>6s} {'src':>5s} "
          f"{'fmul':>5s} {'fadd':>5s} {'fsub':>5s} {'ns/elem':>9s} "
          f"{'correct':>8s}")
    for name in arguments.cores:
        radius = ss.CORE_RADII[name]
        points = np.linspace(-radius * 0.9, radius * 0.9, size)
        oracle = ss.exact_evaluator(name, radius, digits=40)
        sampled = np.linspace(0, size - 1, arguments.checks).astype(int)
        truth = np.array([float(oracle(float(points[index])))
                          for index in sampled])
        for preset in arguments.presets:
            tail = 0
            if preset.startswith("mixed"):
                tail = int(preset.split(":")[1]) if ":" in preset else 2
                chosen = ss.ulp_matched(name)
            elif preset == "ulp_match":
                chosen = ss.ulp_matched(name)
            else:
                chosen = ss.PRESETS[preset]
            program, artifact, native, identifiers, source = bake(
                name, chosen.limbs, chosen.digits, f"kb_{name}_{preset.replace(':','_')}",
                tail=tail)
            output = np.zeros(size)
            feed = {
                identifiers["x"]: points.copy(),
                identifiers["y"]: output,
                identifiers["n"]: np.asarray([size], dtype=np.float64),
            }
            execution = prepare_artifact_execution(native, feed)
            execution.run()
            produced = np.asarray(
                execution.buffers[identifiers["y"]], dtype=float).ravel()
            started = time.perf_counter()
            execution.run()
            elapsed = (time.perf_counter() - started) / size * 1e9
            text = str(artifact.llvm_ir)
            matched = float(np.mean(produced[sampled] == truth)) * 100.0
            print(f"{name:6s} {preset:10s} {len(program.coefficients):6d} "
                  f"{len(source.splitlines()):5d} {text.count('fmul'):5d} "
                  f"{text.count('fadd'):5d} {text.count('fsub'):5d} "
                  f"{elapsed:9.2f} {matched:7.2f}%")
        if name in baseline:
            started = time.perf_counter()
            baseline[name](points)
            reference = (time.perf_counter() - started) / size * 1e9
            print(f"{'':6s} {'numpy':10s} {'':6s} {'':5s} {'':5s} {'':5s} "
                  f"{'':5s} {reference:9.2f} {'reduces':>8s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
