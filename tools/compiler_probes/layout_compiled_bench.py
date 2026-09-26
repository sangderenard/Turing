"""Interleaved versus blocked limb layout, measured on a COMPILED kernel.

The eager measurement is irrelevant here: what matters is how a compiled
per-element kernel reads its limbs. Interleaved puts one element's limbs
adjacent, which is one cache line per element; blocked puts each limb in
its own contiguous run, which is one line per limb but perfectly
sequential across elements.

Both kernels below do the same arithmetic on the same data and differ only
in how they address it, so the difference is the layout and nothing else.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_module_to_c

WIDTH = 2

INTERLEAVED = """
def limb_interleaved(x, y, n):
    for i in range(n):
        a0 = x[i * 2]
        a1 = x[i * 2 + 1]
        s = a0 + a1
        y[i * 2] = s
        y[i * 2 + 1] = a0 - s + a1
    return y
"""

BLOCKED = """
def limb_blocked(x, y, n):
    for i in range(n):
        a0 = x[i]
        a1 = x[n + i]
        s = a0 + a1
        y[i] = s
        y[n + i] = a0 - s + a1
    return y
"""


def build(source: str, entry: str, directory: Path):
    module, _outputs, _exports = lower_ast_source_to_ssa(
        source, entry, name=f"layout_{entry}"
    )
    wrapper = f"layout_{entry}__{entry}"
    artifact = emit_ssa_module_to_c(module, wrapper)
    if not artifact.complete:
        raise SystemExit(
            f"{entry}: " + "; ".join(
                f"{item.operation}: {item.reason}"
                for item in artifact.shortfalls[:3]
            )
        )
    artifact.compile(directory)
    function = module.functions[wrapper]
    ids = dict(function.metadata["parameter_names"])
    return artifact, ids


def timed(artifact, ids, count):
    feeds = {
        int(ids["n"]): np.int32(count),
        int(ids["x"]): np.random.default_rng(0).random(count * WIDTH),
        int(ids["y"]): np.zeros(count * WIDTH),
    }
    execution = artifact.prepare_execution(feeds)
    execution.run()
    best = float("inf")
    for _ in range(9):
        started = time.perf_counter()
        execution.run()
        best = min(best, time.perf_counter() - started)
    return best


root = Path("build/layout_compiled")
interleaved_artifact, interleaved_ids = build(
    INTERLEAVED, "limb_interleaved", root / "interleaved"
)
blocked_artifact, blocked_ids = build(BLOCKED, "limb_blocked", root / "blocked")

print(f"{'elements':>10}  {'interleaved':>13}  {'blocked':>13}  {'blocked is':>12}")
for count in (4_096, 65_536, 1_048_576):
    a = timed(interleaved_artifact, interleaved_ids, count)
    b = timed(blocked_artifact, blocked_ids, count)
    print(
        f"{count:10,d}  {a * 1e6:11.1f}us  {b * 1e6:11.1f}us  "
        f"{a / max(b, 1e-12):10.2f}x"
    )
