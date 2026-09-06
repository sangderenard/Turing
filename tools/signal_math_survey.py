"""Survey every signal-math core across families and error targets.

Produces the accuracy/cost map the design decisions are actually made from:
one panel per core, one point per (family, epsilon), positioned by how many
constants it costs and how accurate it measures.

Accuracy is reported as a DISTRIBUTION, not a single worst case. Relative
error near a function's zero is unbounded no matter how good the core is --
four separate times during this work a lone sample landing on a zero produced
a headline number in the millions of ULP while the median was 0. The 95th
percentile is what the panels plot; the max is kept in the data for anyone who
wants the tail.

    python -m tools.signal_math_survey
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from src.common.tensors import signal_math as sm
from src.common.tensors.abstraction import AbstractTensor

EPSILON_LADDER = (1.0e-5, 1.0e-9, 1.0e-12, 1.0e-15)
DOUBLE_EPSILON = 2.220446049250313e-16


def _score(core) -> dict:
    """Measure one baked core against a high-precision reference."""

    import mpmath

    reference = sm._reference(core.core)
    positions = np.linspace(core.low, core.high, 2001)
    produced = np.asarray(
        sm.evaluate_core(AbstractTensor.get_tensor(positions), core).tolist(),
        dtype=np.float64,
    ).ravel()
    with mpmath.workdps(40):
        expected = np.asarray(
            [float(reference(float(item))) for item in positions],
            dtype=np.float64,
        )
    absolute = np.abs(produced - expected)
    magnitude = np.abs(expected)
    # A sample sitting on a genuine zero contributes its absolute error; the
    # relative figure there is not a property of the core.
    relative = np.where(
        magnitude > 0.0, absolute / np.where(magnitude > 0.0, magnitude, 1.0),
        absolute,
    )
    ulp = relative / DOUBLE_EPSILON
    return {
        "max_abs": float(np.max(absolute)),
        "ulp_p50": float(np.percentile(ulp, 50)),
        "ulp_p95": float(np.percentile(ulp, 95)),
        "ulp_max": float(np.max(ulp)),
    }


def survey(cores, families, epsilons) -> list[dict]:
    rows: list[dict] = []
    started = time.perf_counter()
    for core in cores:
        for family in families:
            for epsilon in epsilons:
                mark = time.perf_counter()
                try:
                    baked = sm.fit_core(core, family, epsilon)
                except Exception as error:
                    print(f"  {core:7s} {family:11s} {epsilon:.0e}  "
                          f"{type(error).__name__}", flush=True)
                    continue
                row = {
                    "core": core, "family": baked.family, "epsilon": epsilon,
                    "constants": len(baked.values),
                    "segments": baked.segments,
                    "structure": baked.structure,
                    "admitted": bool(baked.admitted),
                    "bake_seconds": time.perf_counter() - mark,
                    **_score(baked),
                }
                rows.append(row)
                print(f"[{time.perf_counter()-started:7.1f}s] {core:7s} "
                      f"{baked.family:11s} {epsilon:.0e}  "
                      f"{row['constants']:7d} consts  "
                      f"p95 {row['ulp_p95']:10.1f} ulp  "
                      f"{'adm' if row['admitted'] else '---'}", flush=True)
    return rows


def plot(rows: list[dict], destination: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"structured": "#d1495b", "series": "#2e86ab",
               "polyspline": "#f0a202", "lut": "#5b8c5a"}
    cores = sorted({row["core"] for row in rows},
                   key=lambda name: list(sm.CORE_RANGES).index(name))
    columns = 5
    lines = -(-len(cores) // columns)
    figure, axes = plt.subplots(lines, columns,
                                figsize=(3.5 * columns, 3.0 * lines),
                                squeeze=False)
    for position, core in enumerate(cores):
        axis = axes[position // columns][position % columns]
        for family, colour in colours.items():
            points = [r for r in rows
                      if r["core"] == core and r["family"] == family]
            if not points:
                continue
            points.sort(key=lambda r: r["constants"])
            axis.plot([r["constants"] for r in points],
                      [max(r["ulp_p95"], 0.05) for r in points],
                      color=colour, alpha=0.35, zorder=2)
            for row in points:
                axis.scatter(
                    row["constants"], max(row["ulp_p95"], 0.05),
                    s=54, color=colour, zorder=3,
                    marker="o" if row["admitted"] else "x",
                    edgecolor="white" if row["admitted"] else colour,
                    linewidth=0.8,
                )
        axis.axhline(1.0, color="#333", ls="--", lw=1.0, zorder=1)
        axis.set_xscale("log")
        axis.set_yscale("log")
        structure = sm.CORE_RANGES[core].structure or "plain"
        axis.set_title(f"{core}  ({structure})", fontsize=10)
        axis.grid(alpha=0.22, which="both")
        if position % columns == 0:
            axis.set_ylabel("ULP, 95th pct")
        if position // columns == lines - 1:
            axis.set_xlabel("constants")
    for spare in range(len(cores), lines * columns):
        axes[spare // columns][spare % columns].axis("off")
    handles = [
        plt.Line2D([], [], color=colour, marker="o", ls="", label=family)
        for family, colour in colours.items()
        if any(r["family"] == family for r in rows)
    ]
    handles += [
        plt.Line2D([], [], color="#333", marker="o", ls="", label="admitted"),
        plt.Line2D([], [], color="#333", marker="x", ls="", label="missed target"),
        plt.Line2D([], [], color="#333", ls="--", label="libm, 1 ulp"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=7, frameon=False,
                  fontsize=10, bbox_to_anchor=(0.5, -0.012))
    figure.suptitle(
        "signal-math cores: relative accuracy against cost, by family and "
        "error target\n(down and left is better; dashed line is libm)",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.96))
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=140)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="+",
                        default=["structured", "series", "polyspline"])
    parser.add_argument("--cores", nargs="+", default=list(sm.CORE_RANGES))
    parser.add_argument("--output", type=Path,
                        default=ROOT / "build" / "signal-math-survey")
    arguments = parser.parse_args(argv)

    rows = survey(arguments.cores, arguments.families, EPSILON_LADDER)
    arguments.output.mkdir(parents=True, exist_ok=True)
    (arguments.output / "survey.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8",
    )
    image = plot(rows, arguments.output / "signal_math_survey.png")
    print(f"\nrows: {len(rows)}")
    print(f"data: {arguments.output / 'survey.json'}")
    print(f"plot: {image}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
