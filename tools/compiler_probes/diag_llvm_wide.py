"""Isolate the width-2 disagreement between the C and LLVM lanes.

Same source, same feeds, two lanes: whichever differs from exact
arithmetic is the one with the defect. The kernel is stripped to the
suspects -- a narrow value derived from a wide one by floor, and a
narrow-times-wide product -- so a disagreement names its own cause.
"""
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.common.tensors.signal_symbolic import constant_limbs
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.compiler.ssa_llvm_backend import (
    compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
)

WIDTH = 2

CASES = {
    "wide_mul_only": """
def probe(x: Precision[2], y: Precision[2], n, w: Precision[2]):
    for i in range(n):
        y[i] = x[i] * w
    return y
""",
    "narrow_times_wide": """
def probe(x: Precision[2], y: Precision[2], n, w: Precision[2], k):
    for i in range(n):
        y[i] = x[i] - k * w
    return y
""",
    "long_horner": "\n".join([
        "",
        "def probe(x: Precision[2], y: Precision[2], n, w: Precision[2], "
        + ", ".join(f"a{i}: Precision[2]" for i in range(14)) + "):",
        "    for i in range(n):",
        "        s = x[i]",
        "        acc = a13",
        *[f"        acc = acc * s + a{i}" for i in range(12, -1, -1)],
        "        y[i] = acc",
        "    return y",
        "",
    ]),
    "mask_blend": """
def probe(x: Precision[2], y: Precision[2], n, w: Precision[2], iv):
    for i in range(n):
        z = x[i]
        k = (z * iv + 0.5).floor()
        r = z - k * w
        f = k * 0.25
        q = k - f.floor() * 4.0
        m0 = (q == 0.0) * 1.0
        m1 = (q == 1.0) * 1.0
        m2 = (q == 2.0) * 1.0
        m3 = (q == 3.0) * 1.0
        y[i] = r * m0 + r * m1 - r * m2 - r * m3
    return y
""",
    "floor_then_wide": """
def probe(x: Precision[2], y: Precision[2], n, w: Precision[2], iv):
    for i in range(n):
        z = x[i]
        k = (z * iv + 0.5).floor()
        y[i] = z - k * w
    return y
""",
}

POINTS = [0.3, 3.0, 100.0]
quarter = constant_limbs("tau", WIDTH, scale=Fraction(1, 4))
quarter_exact = sum(Fraction(part) for part in quarter)


def feeds_for(ids, extra):
    count = len(POINTS)
    buffer = np.zeros(count * WIDTH)
    buffer[::WIDTH] = POINTS
    feeds = {
        int(ids["x"]): buffer,
        int(ids["y"]): np.zeros(count * WIDTH),
        int(ids["n"]): np.int32(count),
    }
    if "w" in ids:
        feeds[int(ids["w"])] = np.float64(quarter[0])
    if "w__limb1" in ids:
        feeds[int(ids["w__limb1"])] = np.float64(quarter[1])
    for name, value in extra.items():
        if name in ids:
            feeds[int(ids[name])] = np.float64(value)
    for identifier in ids.values():
        feeds.setdefault(int(identifier), np.float64(0.0))
    return feeds


def read(execution, ids):
    produced = np.asarray(execution.buffers[int(ids["y"])])
    return [
        sum(
            Fraction(float(produced[position * WIDTH + limb]))
            for limb in range(WIDTH)
        )
        for position in range(len(POINTS))
    ]


print(f"{'case':>20} {'lane':>8}  {'worst |lane - exact|':>22}")
for label, source in CASES.items():
    module, _outputs, _exports = lower_ast_source_to_ssa(
        source, "probe", name=f"d_{label}"
    )
    entry = f"d_{label}__probe"
    function = module.functions[entry]
    ids = dict(function.metadata["parameter_names"])
    rows = dict(function.metadata.get("precision_lowered_values") or ())
    for name, identifier in tuple(ids.items()):
        row = rows.get(int(identifier))
        if row:
            for position, limb in enumerate(tuple(row)[1:], start=1):
                ids.setdefault(f"{name}__limb{position}", int(limb))

    extra = {"k": 3.0, "iv": 1.0 / float(quarter_exact)}
    for i in range(14):
        parts = [float(Fraction(i + 1, 7)), float(Fraction(i + 1, 7) - Fraction(float(Fraction(i + 1, 7))))]
        extra[f"a{i}"] = parts[0]
        extra[f"a{i}__limb1"] = parts[1]
    expected = []
    for point in POINTS:
        value = Fraction(point)
        if label == "wide_mul_only":
            expected.append(value * quarter_exact)
        elif label == "narrow_times_wide":
            expected.append(value - 3 * quarter_exact)
        elif label == "long_horner":
            acc = Fraction(0)
            for i in range(13, -1, -1):
                acc = acc * value + Fraction(i + 1, 7)
            expected.append(acc)
        elif label == "mask_blend":
            index = int(np.floor(point * extra["iv"] + 0.5))
            residual = value - index * quarter_exact
            quadrant = index - int(np.floor(index * 0.25)) * 4
            sign = {0: 1, 1: 1, 2: -1, 3: -1}[quadrant]
            expected.append(residual * sign)
        else:
            index = int(np.floor(point * extra["iv"] + 0.5))
            expected.append(value - index * quarter_exact)

    for lane in ("c", "llvm"):
        try:
            if lane == "c":
                artifact = emit_ssa_module_to_c(module, entry)
                if not artifact.complete:
                    raise RuntimeError(artifact.shortfalls[0].reason)
                native = artifact.compile(Path(f"build/diag/{label}_c"))
                execution = native.prepare_execution(feeds_for(ids, extra))
            else:
                artifact = emit_ssa_function_to_llvm(module, entry)
                if artifact.shortfalls:
                    raise RuntimeError(artifact.shortfalls[0].reason)
                native = compile_artifact(
                    artifact, directory=Path(f"build/diag/{label}_llvm")
                )
                execution = prepare_artifact_execution(
                    native, feeds_for(ids, extra)
                )
            execution.run()
            got = read(execution, ids)
            worst = max(
                abs(float(a - b)) for a, b in zip(got, expected)
            )
            print(f"{label:>20} {lane:>8}  {worst:22.3e}")
        except Exception as error:
            print(f"{label:>20} {lane:>8}  {str(error)[:22]:>22}")
