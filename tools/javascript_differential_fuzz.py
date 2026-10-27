"""Differential fuzzer: generated JavaScript functions, Turing native vs node.

The JavaScript counterpart of ``differential_fuzz.py``.  Each program is a
random function over two Number parameters (``a``, ``b``) and two Boolean
parameters (``p``, ``q``) built from ``const``/``let`` bindings,
reassignments (``=``, ``op=``, ``++``/``--``) under nested ``if``/``else``
and bounded ``for``/``while`` loops (with guarded ``break``/``continue``,
labeled ones to enclosing loops, and early ``return``),
an optional early-return ``if`` and typed expression snippets.  It is lowered with
``lower_javascript_source_to_ssa``, compiled through the C backend and run on
rows of ordinary and special values (``NaN``, ``-0``, ``Infinity``), and
compared bit for bit with node (``javascript_corpus_typed_check``).

Statuses: PASS, WRONG, TYPE, LOUD (a named refusal or emission shortfall),
FINDING (anything else: lowering/emission crash, run failure, SEGV).
WRONG and TYPE are emergencies: stop and minimize.

Programs using an implementation-approximated operation (``Math.sin``,
``**``, ...) are judged within the oracle's ulp tier.  ``%``,
``Math.sign``, ``Math.max``/``Math.min`` and ``&&``/``||`` on Numbers
are exact programs, judged bit for bit.  The grammar also includes a
spelling the frontend refuses today (``&&`` on a Number and a Boolean,
whose result has no single kind) so that its refusal stays visible as a
LOUD cluster rather than silently vanishing.

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
            op = rng.choice(("+", "-", "*", "/", "%"))
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
        if choice < 0.96:
            return f"({self.number(depth - 1)} ** {self.number(depth - 1)})"
        if choice < 0.975:
            return f"Math.sign({self.number(depth - 1)})"
        if choice < 0.985:
            call = rng.choice(("Math.max", "Math.min"))
            return f"{call}({self.number(depth - 1)}, {self.number(depth - 1)})"
        if choice < 0.997:
            # ``&&``/``||`` on Numbers return an operand, not a boolean.
            op = rng.choice(("&&", "||"))
            return f"({self.number(depth - 1)} {op} {self.number(depth - 1)})"
        # Rare on purpose: one refused spelling sinks the whole program.
        return f"({self.number(depth - 1)} && {self.boolean(depth - 1)})"

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

    def result(self, depth: int) -> str:
        """A returned value of the function's one result kind."""

        return self.number(depth) if self.returns_number else self.boolean(depth)

    def condition(self, depth: int) -> str:
        # Conditions may be Numbers (JavaScript truthiness) or Booleans.
        if self.rng.random() < 0.35:
            return self.number(depth)
        return self.boolean(depth)

    def assignment(self, depth: int) -> str:
        """One statement rebinding a ``let`` (or a parameter) of its own kind."""

        rng = self.rng
        if self.mutable_numbers and (not self.mutable_booleans or rng.random() < 0.75):
            name = rng.choice(self.mutable_numbers)
            form = rng.random()
            if form < 0.15:
                return f"{name}{rng.choice(('++', '--'))};"
            if form < 0.55:
                op = rng.choice(("+=", "-=", "*=", "/=", "%="))
                return f"{name} {op} {self.number(depth)};"
            return f"{name} = {self.number(depth)};"
        name = rng.choice(self.mutable_booleans)
        return f"{name} = {self.boolean(depth)};"

    def block(self, depth: int, indent: str) -> list[str]:
        """Statements that rebind outer names, possibly with a nested if."""

        rng = self.rng
        lines = []
        # Names declared in this block (a while counter) end with it.
        visible = list(self.numbers)
        for _ in range(rng.randint(1, 2)):
            choice = rng.random()
            if self.loops and choice < 0.12:
                # ``continue`` would skip a while loop's trailing counter
                # increment, so only a ``for`` loop gets one.
                # The innermost loop, or (named) an enclosing labeled one.
                targets = [self.loops[-1]] + [
                    loop for loop in self.loops if loop[1] is not None
                ]
                kind, label = rng.choice(targets)
                if (kind, label) == self.loops[-1] and rng.random() < 0.6:
                    label = None
                exit_ = "break" if kind == "while" or (
                    rng.random() < 0.5
                ) else "continue"
                if label is not None:
                    exit_ += f" {label}"
                if rng.random() < 0.3:
                    exit_ = f"return {self.result(1)}"
                lines.append(f"{indent}if ({self.condition(1)}) {{ {exit_}; }}")
            elif not self.loops and choice < 0.1:
                # An early return in an arm; the other paths fall through
                # into the rest of the function.
                lines.append(
                    f"{indent}if ({self.condition(1)}) {{ return {self.result(1)}; }}"
                )
            elif depth > 0 and choice < 0.18:
                lines.extend(self.switch_statement(depth - 1, indent))
            elif depth > 0 and choice < 0.25:
                lines.extend(self.loop_statement(depth - 1, indent))
            elif depth > 0 and choice < 0.5:
                lines.extend(self.if_statement(depth - 1, indent))
            else:
                lines.append(f"{indent}{self.assignment(2)}")
        self.numbers[:] = visible
        return lines

    def loop_statement(self, depth: int, indent: str) -> list[str]:
        """A bounded for/while loop rebinding outer names in its body.

        The trip count is at most a small literal, optionally cut short by
        a data comparison, so every program terminates in node.
        """

        rng = self.rng
        self.loop_index += 1
        counter = f"i{self.loop_index}"
        limit = rng.choice(("0", "1", "3", "4"))
        guard = f"{counter} < {limit}"
        if rng.random() < 0.4:
            guard += f" && {counter} < {rng.choice(('a', 'b'))}"
        self.numbers.append(counter)
        label = f"L{self.loop_index}" if rng.random() < 0.35 else None
        prefix = f"{label}: " if label else ""
        form = rng.random()
        if form < 0.15:
            # A do loop tests after its body; it gets no continue, which
            # would skip the trailing counter increment.
            lines = [f"{indent}let {counter} = 0;", f"{indent}{prefix}do {{"]
            self.loops.append(("while", label))
            lines.extend(self.block(depth, indent + "  "))
            lines.append(f"{indent}  {counter}++;")
            self.loops.pop()
            lines.append(f"{indent}}} while ({guard});")
            return lines
        if form < 0.7:
            lines = [
                f"{indent}{prefix}for ({rng.choice(('let', 'let', 'var'))} "
                f"{counter} = 0; {guard}; {counter}++) {{"
            ]
            self.loops.append(("for", label))
            lines.extend(self.block(depth, indent + "  "))
        else:
            lines = [
                f"{indent}let {counter} = 0;", f"{indent}{prefix}while ({guard}) {{",
            ]
            self.loops.append(("while", label))
            lines.extend(self.block(depth, indent + "  "))
            lines.append(f"{indent}  {counter}++;")
        self.loops.pop()
        lines.append(f"{indent}}}")
        if "for (" in lines[0]:
            # A for-head let is scoped to the loop (a var is not read after).
            self.numbers.remove(counter)
        return lines

    def switch_statement(self, depth: int, indent: str) -> list[str]:
        """A switch over a small number, with fallthrough and a default."""

        rng = self.rng
        lines = [f"{indent}switch ({self.number(1)}) {{"]
        labels = rng.sample(("0", "1", "2", "-0", "0.5"), rng.randint(1, 3))
        if rng.random() < 0.6:
            labels.insert(rng.randint(0, len(labels)), None)
        for value in labels:
            lines.append(f"{indent}  {'default' if value is None else 'case ' + value}:")
            lines.append(f"{indent}    {self.assignment(2)}")
            if rng.random() < 0.6:
                lines.append(f"{indent}    break;")
        lines.append(f"{indent}}}")
        return lines

    def if_statement(self, depth: int, indent: str) -> list[str]:
        lines = [f"{indent}if ({self.condition(2)}) {{"]
        lines.extend(self.block(depth, indent + "  "))
        if self.rng.random() < 0.6:
            lines.append(f"{indent}}} else {{")
            lines.extend(self.block(depth, indent + "  "))
        lines.append(f"{indent}}}")
        return lines

    def program(self) -> str:
        rng = self.rng
        lines = []
        # Parameters are reassignable too.
        self.mutable_numbers = ["a", "b"] if rng.random() < 0.3 else []
        self.mutable_booleans = []
        self.loop_index = 0
        self.loops: list[tuple[str, str | None]] = []
        self.returns_number = rng.random() < 0.7
        for index in range(rng.randint(0, 4)):
            # A var declared up front with an initializer reads like a let.
            binding = rng.choice(("let", "let", "const", "const", "var"))
            if rng.random() < 0.7:
                name = f"n{index}"
                lines.append(f"  {binding} {name} = {self.number(3)};")
                self.numbers.append(name)
                if binding != "const":
                    self.mutable_numbers.append(name)
            else:
                name = f"t{index}"
                lines.append(f"  {binding} {name} = {self.boolean(3)};")
                self.booleans.append(name)
                if binding != "const":
                    self.mutable_booleans.append(name)
            if (self.mutable_numbers or self.mutable_booleans) and rng.random() < 0.5:
                form = rng.random()
                if form < 0.35:
                    lines.extend(self.loop_statement(1, "  "))
                elif form < 0.65:
                    lines.extend(self.if_statement(1, "  "))
                else:
                    lines.append(f"  {self.assignment(3)}")
        value = lambda: self.result(3)  # noqa: E731
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
