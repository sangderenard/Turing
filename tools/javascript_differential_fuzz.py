"""Differential fuzzer: generated JavaScript functions, Turing native vs node.

The JavaScript counterpart of ``differential_fuzz.py``.  Each program is a
random function over two Number parameters (``a``, ``b``) and two Boolean
parameters (``p``, ``q``) built from ``const`` bindings, an optional
early-return ``if`` and typed expression snippets.  It is lowered with
``lower_javascript_source_to_ssa``, compiled through the C backend and run on
rows of ordinary and special values (``NaN``, ``-0``, ``Infinity``), and
compared bit for bit with node (``javascript_corpus_typed_check``).

Statuses: PASS, WRONG, TYPE, LOUD (a named refusal or emission shortfall),
FINDING (anything else: lowering/emission crash, run failure, SEGV).
WRONG and TYPE are emergencies: stop and minimize.

Programs using an implementation-approximated operation (``Math.sin``,
``**``, ...) are judged within the oracle's ulp tier.  The grammar also
includes spellings the frontend refuses today (``%``, ``Math.sign``,
``Math.max``, ``&&`` on Numbers) so that each
refusal stays visible as a LOUD cluster rather than silently vanishing.

Usage: python tools/javascript_differential_fuzz.py OUT SEED COUNT
"""

from __future__ import annotations

import collections
import multiprocessing
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from javascript_corpus_typed_check import check_program  # noqa: E402


INPUT_ROWS = (
    "[[1, 2, true, false], [-0, 0, false, true], [NaN, 3, true, true], "
    "[Infinity, -Infinity, false, false], [0.1, 0.2, true, false], "
    "[-7, 2, false, true], [1e308, 1e-308, true, true], [0, NaN, false, false], "
    "[-1, Infinity, true, false], [1, NaN, false, true]]"
)

NUMBER_LITERALS = ("0", "1", "2", "0.5", "3", "1e308", "1e-308", "10")
# Implementation-approximated: the oracle judges them by an ulp tier.
APPROXIMATED_CALLS = ("Math.sin", "Math.cos", "Math.exp", "Math.log", "Math.atan")
# Refused today; kept so their refusals stay counted.
LOUD_NUMBER_CALLS = ("Math.sign",)
LOUD_NUMBER_BINARY = ("%",)


class Generator:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.numbers: list[str] = ["a", "b"]
        self.booleans: list[str] = ["p", "q"]

    def number(self, depth: int) -> str:
        rng = self.rng
        if depth <= 0 or rng.random() < 0.25:
            return rng.choice(self.numbers + list(NUMBER_LITERALS[:4]))
        choice = rng.random()
        if choice < 0.40:
            op = rng.choice(("+", "-", "*", "/"))
            return f"({self.number(depth - 1)} {op} {self.number(depth - 1)})"
        if choice < 0.50:
            return f"(-{self.number(depth - 1)})"
        if choice < 0.62:
            call = rng.choice(
                ("Math.floor", "Math.ceil", "Math.trunc", "Math.abs", "Math.sqrt")
            )
            return f"{call}({self.number(depth - 1)})"
        if choice < 0.80:
            return (
                f"({self.condition(depth - 1)} ? {self.number(depth - 1)} : "
                f"{self.number(depth - 1)})"
            )
        if choice < 0.88:
            return rng.choice(NUMBER_LITERALS)
        if choice < 0.94:
            call = rng.choice(APPROXIMATED_CALLS)
            return f"{call}({self.number(depth - 1)})"
        if choice < 0.97:
            return f"({self.number(depth - 1)} ** {self.number(depth - 1)})"
        # Rare on purpose: one refused spelling sinks the whole program.
        loud = rng.randrange(4)
        if loud == 0:
            call = rng.choice(LOUD_NUMBER_CALLS)
            return f"{call}({self.number(depth - 1)})"
        if loud == 1:
            op = rng.choice(LOUD_NUMBER_BINARY)
            return f"({self.number(depth - 1)} {op} {self.number(depth - 1)})"
        if loud == 2:
            # ``&&`` on Numbers returns an operand, not a boolean.
            return f"({self.number(depth - 1)} && {self.number(depth - 1)})"
        return f"Math.max({self.number(depth - 1)}, {self.number(depth - 1)})"

    def boolean(self, depth: int) -> str:
        rng = self.rng
        if depth <= 0 or rng.random() < 0.2:
            return rng.choice(self.booleans + ["true", "false"])
        choice = rng.random()
        if choice < 0.35:
            op = rng.choice(("<", "<=", ">", ">=", "===", "!=="))
            return f"({self.number(depth - 1)} {op} {self.number(depth - 1)})"
        if choice < 0.50:
            op = rng.choice(("&&", "||"))
            return f"({self.boolean(depth - 1)} {op} {self.boolean(depth - 1)})"
        if choice < 0.62:
            return f"(!{self.boolean(depth - 1)})"
        if choice < 0.74:
            return f"(!{self.number(depth - 1)})"
        if choice < 0.84:
            return f"Number.isNaN({self.number(depth - 1)})"
        if choice < 0.92:
            return f"({self.boolean(depth - 1)} === {self.boolean(depth - 1)})"
        return (
            f"({self.condition(depth - 1)} ? {self.boolean(depth - 1)} : "
            f"{self.boolean(depth - 1)})"
        )

    def condition(self, depth: int) -> str:
        # Conditions may be Numbers (JavaScript truthiness) or Booleans.
        if self.rng.random() < 0.35:
            return self.number(depth)
        return self.boolean(depth)

    def program(self) -> str:
        rng = self.rng
        lines = []
        for index in range(rng.randint(0, 3)):
            if rng.random() < 0.7:
                name = f"n{index}"
                lines.append(f"  const {name} = {self.number(3)};")
                self.numbers.append(name)
            else:
                name = f"t{index}"
                lines.append(f"  const {name} = {self.boolean(3)};")
                self.booleans.append(name)
        returns_number = rng.random() < 0.7
        value = (lambda: self.number(3)) if returns_number else (
            lambda: self.boolean(3)
        )
        if rng.random() < 0.4:
            lines.append(f"  if ({self.condition(2)}) {{ return {value()}; }}")
        lines.append(f"  return {value()};")
        body = "\n".join(lines)
        return (
            "/** @param {number} a @param {number} b "
            "@param {boolean} p @param {boolean} q */\n"
            f"function root(a, b, p, q) {{\n{body}\n}}\n"
        )


def _isolated(source: str) -> str:
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    child = context.Process(
        target=lambda: queue.put(check_program(source, INPUT_ROWS))
    )
    child.start()
    try:
        result = queue.get(timeout=600)
    except Exception:
        result = None
    child.join(10)
    if result is None:
        return f"SEGV exit={child.exitcode}"
    return result


def classify(status: str) -> str:
    head = status.split(" ", 1)[0]
    if head == "PASS":
        return "PASS"
    if head == "FAIL":
        if "WRONG" in status:
            return "WRONG"
        if "TYPE" in status:
            return "TYPE"
        return "FINDING"
    if head in {"REFUSED", "EMIT-INCOMPLETE"}:
        return "LOUD"
    return "FINDING"


def main(argv: list[str]) -> int:
    out = pathlib.Path(argv[1])
    seed = int(argv[2])
    count = int(argv[3])
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    totals: collections.Counter[str] = collections.Counter()
    for index in range(count):
        source = Generator(rng).program()
        path = out / f"js_{seed}_{index:03d}.js"
        path.write_text(f"{source}// INPUTS {INPUT_ROWS}\n", encoding="utf-8")
        status = _isolated(source)
        verdict = classify(status)
        totals[verdict] += 1
        print(f"{path.name}: {verdict} {status[:400]}", flush=True)
    print("summary:", " ".join(f"{k}={v}" for k, v in sorted(totals.items())))
    return 1 if totals["WRONG"] or totals["TYPE"] else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
