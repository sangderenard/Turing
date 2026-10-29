"""Value-and-type check of JavaScript programs against real node.

The JavaScript counterpart of ``native_corpus_typed_check.py``.  Every
``c*.js`` in DIR declares a function ``root`` (parameter kinds in its JSDoc)
and ends with one comment line of input rows, written as JavaScript so that
``NaN``, ``Infinity`` and ``-0`` are spellable::

    // INPUTS [[1, 2], [NaN, -0], [Infinity, true]]

node evaluates the rows and ``root`` on each.  The program is lowered with
``lower_javascript_source_to_ssa``, emitted through ``emit_ssa_to_c``,
compiled, and called on the same rows.  Values travel between node and
Python as IEEE-754 bit patterns, so nothing is rounded on the way: a number
matches only when its bits match (any NaN matches any NaN; ``-0`` does not
match ``0``).  The returned kind must match too: a JavaScript number is a
float buffer, a boolean a bool buffer.

Programs whose implementation-approximated operations are all unary Math
calls are judged bit for bit against node replaying with those calls
answered by the C library (``pinned_node_rows``); programs using ``**`` or
``Math.pow`` are judged within a ulp tier (``oracle_ulps``).

Statuses: PASS, FAIL (WRONG or TYPE rows), REFUSED (named lowering
shortfalls -- loud, never a wrong answer), LOWER-FAIL, EMIT-INCOMPLETE,
EMIT-FAIL, NODE-FAIL.

Usage: python tools/javascript_corpus_typed_check.py DIR
``TURING_ROOT=<worktree>`` points it at another tree; ``TURING_NODE`` names
the node executable (default: ``node`` on PATH, then /opt/node22/bin/node).
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import tempfile
import warnings
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")
ROOT = os.environ.get(
    "TURING_ROOT", str(pathlib.Path(__file__).resolve().parents[1])
)
sys.path.insert(0, ROOT)


def node_executable() -> str | None:
    explicit = os.environ.get("TURING_NODE")
    if explicit:
        return explicit
    found = shutil.which("node")
    if found:
        return found
    fallback = "/opt/node22/bin/node"
    return fallback if os.path.exists(fallback) else None


# Runs ``root`` on every row and reports each argument and result by kind and
# exact bits.  Anything but a number or boolean result is reported as its
# typeof, which the judge counts as a TYPE finding.
_NODE_ORACLE = r"""
const fs = require("fs");
const vm = require("vm");
const request = JSON.parse(fs.readFileSync(0, "utf8"));
function encode(value) {
  if (typeof value === "number") {
    const view = new DataView(new ArrayBuffer(8));
    view.setFloat64(0, value);
    return {kind: "number", bits: view.getBigUint64(0).toString(16).padStart(16, "0")};
  }
  if (typeof value === "boolean") return {kind: "boolean", value};
  return {kind: typeof value, text: String(value)};
}
const context = vm.createContext({});
// Pinned transcendentals: Math.<fn> answers from the table (keyed by the
// argument's bits) and records every argument it has no answer for.
const pins = request.pins || null;
const misses = [];
if (pins) {
  const view = new DataView(new ArrayBuffer(8));
  const bits = (x) => { view.setFloat64(0, x); return view.getBigUint64(0).toString(16).padStart(16, "0"); };
  const value = (h) => { view.setBigUint64(0, BigInt("0x" + h)); return view.getFloat64(0); };
  const math = vm.runInContext("Math", context);
  for (const name of request.pinned) {
    const original = math[name];
    math[name] = function (x) {
      const key = name + ":" + bits(+x);
      if (key in pins) return value(pins[key]);
      misses.push(key);
      return original(x);
    };
  }
}
vm.runInContext(request.source, context);
const rows = vm.runInContext(request.inputs, context);
const out = [];
for (const row of rows) {
  const args = Array.isArray(row) ? row : [row];
  let result;
  try {
    result = encode(context[request.entry](...args));
  } catch (error) {
    result = {kind: "throw", text: String(error)};
  }
  out.push({args: args.map(encode), result});
}
process.stdout.write(JSON.stringify(pins ? {rows: out, misses} : out));
"""


def node_rows(
    source: str, inputs: str, entry: str = "root",
    pins: dict[str, str] | None = None,
) -> Any:
    """Evaluate ``entry`` on each JavaScript-spelled input row in node.

    With ``pins`` the unary approximated Math functions answer from that
    table and the result is ``{"rows": ..., "misses": [...]}``.
    """

    executable = node_executable()
    if executable is None:
        raise RuntimeError("no node executable (set TURING_NODE)")
    # A function declaration binds on the context's global object; a
    # ``const root = ...`` would not, so expose the entry explicitly.
    exposed = f"{source}\n;globalThis[{json.dumps(entry)}] = {entry};\n"
    result = subprocess.run(
        [executable, "-e", _NODE_ORACLE],
        input=json.dumps({
            "source": exposed, "inputs": inputs, "entry": entry,
            "pins": pins, "pinned": sorted(PINNED_MATH),
        }).encode("utf-8"),
        capture_output=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", "replace").strip())
    return json.loads(result.stdout.decode("utf-8"))


#: Unary implementation-approximated Math functions the oracle can pin to
#: the C library's answers.  ``Math.pow`` and ``**`` are not pinnable
#: (``**`` is an operator), so programs using them keep the ulp tier.
PINNED_MATH = frozenset({
    "acos", "acosh", "asin", "asinh", "atan", "atanh", "cos", "cosh",
    "exp", "log", "sin", "sinh", "tan", "tanh",
})


_PINNED_LIBRARY: dict | None = None


def _libm():
    """The pinned functions as the native code resolves them.

    Not a guessed system libm (``libm.so.6`` exists only on glibc): a
    one-line shim per function, ``double pin_exp(double x) { return
    exp(x); }``, built by the same toolchain, command and contract flags
    as the C lane's artifacts, so each call binds to whatever ``exp`` the
    compiled programs bind to on this platform.  The argument is a
    runtime value, so the compiler cannot fold the call away (a folded
    literal call is a separate hazard, docs/hazards/20261027T230236Z_*).
    A toolchain that cannot build the shim refuses by name.
    """

    global _PINNED_LIBRARY
    if _PINNED_LIBRARY is not None:
        return _PINNED_LIBRARY
    import ctypes

    from src.compiler.work_contract import active_contract

    directory = pathlib.Path(tempfile.mkdtemp())
    source = directory / "turing_pinned_math.c"
    library = directory / "turing_pinned_math.dll"
    source.write_text(
        "#include <math.h>\n"
        "#if defined(_WIN32)\n#define PIN __declspec(dllexport)\n"
        "#else\n#define PIN __attribute__((visibility(\"default\")))\n#endif\n"
        + "".join(
            f"PIN double pin_{name}(double x) {{ return {name}(x); }}\n"
            for name in sorted(PINNED_MATH)
        ),
        encoding="utf-8",
    )
    command = [
        sys.executable, "-m", "ziglang", "cc", "-shared", "-O3",
        *map(str, active_contract().compiler_flags),
        "-o", str(library), str(source),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0 or not library.is_file():
        raise RuntimeError(
            "pinned-math shim did not build with the C lane's toolchain: "
            + (completed.stderr or completed.stdout).strip()[:300]
        )
    loaded = ctypes.CDLL(str(library))
    functions = {}
    for name in PINNED_MATH:
        function = getattr(loaded, f"pin_{name}")
        function.restype = ctypes.c_double
        function.argtypes = (ctypes.c_double,)
        functions[name] = function
    _PINNED_LIBRARY = functions
    return functions


def pinned_node_rows(source: str, inputs: str, entry: str = "root") -> list[dict]:
    """node's answers with the transcendentals pinned to the C library's.

    ECMAScript leaves Math.sin, Math.log, ... implementation-approximated,
    and the compiled program calls the C library's.  An approximation
    differing by one ulp can be amplified without bound downstream
    (``Math.exp(10) % Math.log(3)``: ~2e4 ulp; a branch on it: anything),
    so a ulp tier on the final value is no standard there.  Instead node
    replays the program with every pinned call answered by libm (the same
    functions the native code calls), and the result is compared bit for
    bit: everything that is not the approximation itself -- the program's
    JavaScript semantics -- must agree exactly.  Arguments a replay
    reaches for the first time are answered on the next replay.
    """

    libm = _libm()
    pins: dict[str, str] = {}
    for _ in range(16):
        result = node_rows(source, inputs, entry, pins=pins)
        if not result["misses"]:
            return result["rows"]
        for key in result["misses"]:
            name, bits = key.split(":")
            argument = struct.unpack(">d", bytes.fromhex(bits))[0]
            pins[key] = number_bits(libm[name](argument))
    raise RuntimeError("pinned replay did not settle in 16 rounds")


def decode(item: dict):
    if item["kind"] == "number":
        return struct.unpack(">d", bytes.fromhex(item["bits"]))[0]
    if item["kind"] == "boolean":
        return bool(item["value"])
    return item


def number_bits(value: float) -> str:
    return struct.pack(">d", float(value)).hex()


def same_number(got: float, want_bits: str) -> bool:
    want = struct.unpack(">d", bytes.fromhex(want_bits))[0]
    if want != want:
        return got != got
    return number_bits(got) == want_bits


def ulp_distance(got: float, want: float) -> float:
    """Distance in ulps of ``want``; special values must match exactly.

    NaN matches only NaN, an infinity only itself, and a zero only the zero
    of the same sign.  Anything else is ``inf``.
    """

    if want != want or got != got:
        return 0.0 if (want != want and got != got) else math.inf
    if math.isinf(want) or math.isinf(got) or want == 0.0 or got == 0.0:
        return 0.0 if number_bits(got) == number_bits(want) else math.inf
    if (got < 0.0) != (want < 0.0):
        return math.inf
    return abs(got - want) / float(np.spacing(abs(want)))


def oracle_ulps() -> tuple[str, float]:
    """The standard for implementation-approximated programs.

    ECMAScript leaves Math.sin, Math.exp, ``**`` and the rest
    implementation-approximated, and V8's answers are not more accurate than
    the repository's compiled ones (measured against a 200-bit reference,
    V8 misrounds more often).  So a program that uses one is judged against
    node within a named tier of ``precision_policy.PRECISION_TIERS``
    (``JS_ORACLE_TIER``, default ``tight``); programs without one stay
    bit-exact.
    """

    from src.compiler.precision_policy import PRECISION_TIERS

    tier = os.environ.get("JS_ORACLE_TIER", "tight")
    return tier, float(PRECISION_TIERS[tier])


def judge(row: dict, got, got_kind: str, ulps: float | None = None) -> str:
    """``ok``, or a WRONG/TYPE line for one row.

    ``ulps`` None compares bits; a number compares within that many ulps
    of node's value (special values still exactly).
    """

    want = row["result"]
    args = [decode(item) for item in row["args"]]
    if want["kind"] == "number":
        if got_kind != "f":
            return f"TYPE {args} got kind {got_kind!r} want number (float)"
        if ulps is not None:
            distance = ulp_distance(float(got), decode(want))
            if distance > ulps:
                return (
                    f"WRONG {args} got {float(got)!r} want {decode(want)!r} "
                    f"({distance:.3g} ulp > {ulps:g})"
                )
            return "ok"
        if not same_number(float(got), want["bits"]):
            return (
                f"WRONG {args} got {float(got)!r} [{number_bits(got)}] "
                f"want {decode(want)!r} [{want['bits']}]"
            )
        return "ok"
    if want["kind"] == "boolean":
        if got_kind != "b":
            return f"TYPE {args} got kind {got_kind!r} want boolean"
        if bool(got) != bool(want["value"]):
            return f"WRONG {args} got {bool(got)} want {want['value']}"
        return "ok"
    return f"TYPE {args} node returned {want['kind']} {want.get('text')!r}"


def check_program(source: str, inputs: str, *, entry: str = "root") -> str:
    """One status line (without the program name) for one program."""

    from src.compiler.javascript_source_ingestion import (
        lower_javascript_source_to_ssa,
    )
    from src.compiler.ssa_c_backend import emit_ssa_to_c

    try:
        rows = node_rows(source, inputs, entry)
    except Exception as error:
        return f"NODE-FAIL {type(error).__name__}: {str(error)[:300]}"
    try:
        lowered = lower_javascript_source_to_ssa(source, function_name=entry)
    except Exception as error:
        return f"LOWER-FAIL {type(error).__name__}: {str(error)[:300]}"
    if not lowered.complete:
        return "REFUSED " + "; ".join(
            item.format()[:160] for item in lowered.shortfalls[:3]
        )
    function = lowered.module.functions[lowered.function_name]
    returned = [
        instruction.args
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret"
    ][0]
    try:
        artifact = emit_ssa_to_c(lowered.module, lowered.function_name)
        if not artifact.complete:
            return f"EMIT-INCOMPLETE {[s.reason[:150] for s in artifact.shortfalls[:2]]}"
        artifact.compile(pathlib.Path(tempfile.mkdtemp()) / "native")
    except Exception as error:
        return f"EMIT-FAIL {type(error).__name__}: {str(error)[:300]}"
    approximated_ops = {
        str(instruction.op)
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("ecmascript_approximated")
    }
    approximated = bool(approximated_ops)
    pinned = approximated and approximated_ops <= PINNED_MATH
    if pinned:
        # Every approximation is a pinnable call: judge bit for bit
        # against node replaying with the C library's answers.
        try:
            rows = pinned_node_rows(source, inputs, entry)
        except Exception as error:
            return f"NODE-FAIL {type(error).__name__}: {str(error)[:300]}"
        approximated = False
    tier, ulps = oracle_ulps() if approximated else (None, None)
    worst = 0.0
    verdicts = []
    for row in rows:
        args = [decode(item) for item in row["args"]]
        if len(args) != len(function.args):
            verdicts.append(f"ARITY {args}")
            continue
        try:
            execution = artifact.prepare_execution({
                value.id: np.array([arg], dtype=(
                    np.bool_ if value.dtype == "bool" else np.float64
                ))
                for value, arg in zip(function.args, args)
            }).run()
            buffer = np.asarray(execution.buffers[returned[0].id]).reshape(-1)
            got = buffer[0].item()
            verdicts.append(judge(row, got, buffer.dtype.kind, ulps))
            if ulps is not None and row["result"]["kind"] == "number":
                distance = ulp_distance(float(got), decode(row["result"]))
                if math.isfinite(distance):
                    worst = max(worst, distance)
        except Exception as error:
            verdicts.append(f"RUN-FAIL {args} {type(error).__name__}: {str(error)[:150]}")
    bad = [item for item in verdicts if item != "ok"]
    standard = (
        f" (approximated: max {worst:.3g} ulp vs node, tier {tier} = {ulps:g})"
        if approximated
        else f" (libm-pinned: {', '.join(sorted(approximated_ops))})"
        if pinned else ""
    )
    return (
        f"{'PASS' if not bad else 'FAIL'}{standard} {bad[:3] if bad else ''}"
    ).rstrip()


def split_program(text: str) -> tuple[str, str]:
    marker = "// INPUTS"
    if marker not in text:
        raise ValueError("program has no // INPUTS line")
    source, inputs = text.rsplit(marker, 1)
    return source, inputs.strip()


def main(argv: list[str]) -> int:
    directory = pathlib.Path(argv[1] if len(argv) > 1 else os.path.join(
        ROOT, "tests", "javascript_semantics_corpus"
    ))
    statuses = {}
    for path in sorted(directory.glob("c*.js")):
        source, inputs = split_program(path.read_text(encoding="utf-8"))
        status = check_program(source, inputs)
        statuses[path.stem] = status.split(" ", 1)[0]
        print(f"{path.stem}: {status}", flush=True)
    counts: dict[str, int] = {}
    for status in statuses.values():
        counts[status] = counts.get(status, 0) + 1
    print("summary:", " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 0 if set(statuses.values()) <= {"PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
