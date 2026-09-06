"""The noise floor of every operator in the signal-math surface, on one chart.

``demo_twiddle_floor`` measured one thing (the twiddles of a DFT) by the only
honest trick available there: a tone exactly on a bin has a KNOWN spectrum, so
whatever else appears is arithmetic error. That trick does not generalise --
``sqrt`` of a tone is not a tone -- so this uses the general form of the same
idea.

Drive each operator with a sinusoid that stays inside its domain, and subtract
a 40-digit reference. The difference IS the error signal. Its spectrum says
more than a single ulp figure does:

* a FLAT spectrum means the error is uncorrelated with the signal -- rounding
  noise, the irreducible kind, which dither and averaging can push around;
* SPIKES at harmonics of the drive mean the error is a systematic function of
  the argument. That is a distortion product. It does not average away, it
  moves with the signal, and in a feedback path it can be mistaken for signal.

Two implementations are measured identically: this surface, and libm through
numpy. The comparison is the point -- an ulp count cannot tell those two error
characters apart, and a spectrum can.

Run::

    python -m tools.demo_noise_floor_all --size 4096
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from src.common.tensors import signal_math as sm
from src.common.tensors.abstraction import AbstractTensor as AT

TWO_PI = 2.0 * math.pi


class Operator:
    """One function, with a drive that keeps it inside its own domain.

    ``centre`` and ``amplitude`` are chosen so the sweep never touches a pole
    or a branch point. That is not tuning the result -- an operator evaluated
    outside its domain has no error to measure, only a NaN.
    """

    def __init__(self, name, centre, amplitude, reference, libm, group):
        self.name = name
        self.centre = centre
        self.amplitude = amplitude
        self.reference = reference
        self.libm = libm
        self.group = group

    def drive(self, size, tone):
        sample = np.arange(size)
        return self.centre + self.amplitude * np.cos(TWO_PI * tone * sample / size)


def operators():
    """The surface, grouped by family so the chart can colour by kind."""

    import mpmath

    def inverse(function):
        return lambda v: 1.0 / function(v)

    return [
        # circular
        Operator("sin", 0.0, 1.2, mpmath.sin, np.sin, "circular"),
        Operator("cos", 0.0, 1.2, mpmath.cos, np.cos, "circular"),
        Operator("tan", 0.0, 1.2, mpmath.tan, np.tan, "circular"),
        Operator("sec", 0.0, 1.2, inverse(mpmath.cos),
                 lambda x: 1.0 / np.cos(x), "circular"),
        Operator("csc", 1.5, 0.8, inverse(mpmath.sin),
                 lambda x: 1.0 / np.sin(x), "circular"),
        Operator("cot", 1.5, 0.8, inverse(mpmath.tan),
                 lambda x: 1.0 / np.tan(x), "circular"),
        # turn-native: the argument is cycles, never radians
        Operator("sin_turns", 0.0, 0.2, lambda v: mpmath.sin(TWO_PI * v),
                 lambda x: np.sin(TWO_PI * x), "turns"),
        Operator("cos_turns", 0.0, 0.2, lambda v: mpmath.cos(TWO_PI * v),
                 lambda x: np.cos(TWO_PI * x), "turns"),
        Operator("tan_turns", 0.0, 0.2, lambda v: mpmath.tan(TWO_PI * v),
                 lambda x: np.tan(TWO_PI * x), "turns"),
        # inverse circular
        Operator("asin", 0.0, 0.85, mpmath.asin, np.arcsin, "inverse"),
        Operator("acos", 0.0, 0.85, mpmath.acos, np.arccos, "inverse"),
        Operator("atan", 0.0, 3.0, mpmath.atan, np.arctan, "inverse"),
        # exponential
        Operator("exp", 0.0, 1.5, mpmath.exp, np.exp, "exponential"),
        Operator("expm1", 0.0, 1.5, lambda v: mpmath.exp(v) - 1,
                 np.expm1, "exponential"),
        Operator("log", 2.5, 2.0, mpmath.log, np.log, "exponential"),
        Operator("log2", 2.5, 2.0, lambda v: mpmath.log(v, 2),
                 np.log2, "exponential"),
        Operator("log10", 2.5, 2.0, lambda v: mpmath.log10(v),
                 np.log10, "exponential"),
        Operator("log1p", 1.0, 0.9, lambda v: mpmath.log(1 + v),
                 np.log1p, "exponential"),
        Operator("sqrt", 2.5, 2.0, mpmath.sqrt, np.sqrt, "exponential"),
        # hyperbolic
        Operator("sinh", 0.0, 1.5, mpmath.sinh, np.sinh, "hyperbolic"),
        Operator("cosh", 0.0, 1.5, mpmath.cosh, np.cosh, "hyperbolic"),
        Operator("tanh", 0.0, 1.5, mpmath.tanh, np.tanh, "hyperbolic"),
        Operator("sech", 0.0, 1.5, inverse(mpmath.cosh),
                 lambda x: 1.0 / np.cosh(x), "hyperbolic"),
        Operator("csch", 1.5, 0.8, inverse(mpmath.sinh),
                 lambda x: 1.0 / np.sinh(x), "hyperbolic"),
        Operator("coth", 1.5, 0.8, inverse(mpmath.tanh),
                 lambda x: 1.0 / np.tanh(x), "hyperbolic"),
        Operator("asinh", 0.0, 1.5, mpmath.asinh, np.arcsinh, "hyperbolic"),
        Operator("acosh", 3.0, 1.5, mpmath.acosh, np.arccosh, "hyperbolic"),
        Operator("atanh", 0.0, 0.85, mpmath.atanh, np.arctanh, "hyperbolic"),
        # the cancelling one
        Operator("sinc", 0.0, 3.0, lambda v: mpmath.sin(v) / v if v != 0 else 1,
                 lambda x: np.sinc(x / np.pi), "special"),
    ]


def spectrum_db(error, reference):
    """Error spectrum in dB relative to the driven signal's own peak.

    A Hann window keeps the finite record from smearing the drive across every
    bin, which would otherwise bury the very floor being measured.
    """

    window = np.hanning(error.size)
    error_bins = np.abs(np.fft.rfft(error * window))
    reference_bins = np.abs(np.fft.rfft(reference * window))
    peak = float(np.max(reference_bins))
    if peak <= 0.0:
        peak = 1.0
    return 20.0 * np.log10(np.maximum(error_bins / peak, 1e-24))


def flatness(curve):
    """Crest of the error spectrum: peak minus median, in dB.

    Small means flat -- rounding noise. Large means the error is concentrated
    at particular frequencies, which is distortion.
    """

    return float(np.max(curve) - np.median(curve))


def measure(operator, surface, size, tone):
    import mpmath

    x = operator.drive(size, tone)
    with mpmath.workdps(40):
        exact = np.array([float(operator.reference(mpmath.mpf(float(v))))
                          for v in x])

    tensor = AT.get_tensor(x)
    started = time.perf_counter()
    ours = np.asarray(getattr(surface, operator.name)(tensor).tolist(),
                      dtype=float).ravel()
    elapsed = time.perf_counter() - started
    theirs = np.asarray(operator.libm(x), dtype=float).ravel()

    scale = float(np.sqrt(np.mean(exact * exact)))
    result = {"name": operator.name, "group": operator.group,
              "seconds": elapsed}
    for label, produced in (("signal_math", ours), ("libm", theirs)):
        error = produced - exact
        curve = spectrum_db(error, exact)
        rms = float(np.sqrt(np.mean(error * error)))
        result[label] = {
            "curve": curve,
            "snr": 20.0 * math.log10(max(rms, 1e-300) / max(scale, 1e-300)),
            "flatness": flatness(curve),
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--tone", type=int, default=17)
    parser.add_argument("--quality", default="double")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "build" / "signal-field" / "noise_floor_all.png",
    )
    arguments = parser.parse_args(argv)

    started = time.perf_counter()
    surface = sm.signal_math(arguments.quality)
    print(f"baked the {arguments.quality!r} set in "
          f"{time.perf_counter() - started:.1f}s")
    print(f"drive: {arguments.size} samples, tone on bin {arguments.tone}, "
          f"reference at 40 digits")

    every = operators()
    results = []
    print(f"\n{'operator':11s} {'signal_math':>22s} {'libm':>22s}")
    print(f"{'':11s} {'noise':>10s} {'crest':>11s} "
          f"{'noise':>10s} {'crest':>11s}")
    for operator in every:
        if not hasattr(surface, operator.name):
            continue
        try:
            row = measure(operator, surface, arguments.size, arguments.tone)
        except Exception as error:  # a domain slip should name itself, not die
            print(f"{operator.name:11s} skipped: {type(error).__name__}: {error}")
            continue
        results.append(row)
        print(f"{row['name']:11s} "
              f"{row['signal_math']['snr']:9.1f}dB "
              f"{row['signal_math']['flatness']:9.1f}dB "
              f"{row['libm']['snr']:9.1f}dB "
              f"{row['libm']['flatness']:9.1f}dB")

    ours = np.array([r["signal_math"]["snr"] for r in results])
    theirs = np.array([r["libm"]["snr"] for r in results])
    ours_crest = np.array([r["signal_math"]["flatness"] for r in results])
    theirs_crest = np.array([r["libm"]["flatness"] for r in results])
    print(f"\n{len(results)} operators measured")
    print(f"  median noise   signal_math {np.median(ours):8.1f} dB   "
          f"libm {np.median(theirs):8.1f} dB")
    print(f"  worst  noise   signal_math {np.max(ours):8.1f} dB   "
          f"libm {np.max(theirs):8.1f} dB")
    print(f"  median crest   signal_math {np.median(ours_crest):8.1f} dB   "
          f"libm {np.median(theirs_crest):8.1f} dB")
    ahead = int(np.sum(ours < theirs))
    print(f"  quieter than libm on {ahead} of {len(results)} operators")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"circular": "#2e86ab", "turns": "#1b998b",
               "inverse": "#a06cd5", "exponential": "#f0a202",
               "hyperbolic": "#d64550", "special": "#5c6672"}
    figure = plt.figure(figsize=(16, 11))
    grid = figure.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.28,
                               wspace=0.16)

    for column, label in enumerate(("signal_math", "libm")):
        axis = figure.add_subplot(grid[0, column])
        for row in results:
            axis.plot(row[label]["curve"], lw=0.7, alpha=0.75,
                      color=colours.get(row["group"], "#888888"))
        axis.set_title(f"{label}: error spectrum of every operator",
                       fontsize=12)
        axis.set_xlabel("bin")
        axis.set_ylabel("dB relative to the driven signal")
        axis.set_ylim(-420, -180)
        axis.grid(alpha=0.22)
        handles = [plt.Line2D([], [], color=colour, lw=2, label=group)
                   for group, colour in colours.items()]
        axis.legend(handles=handles, loc="upper right", fontsize=8, ncol=2)

    # libm's sqrt is bit-exact (IEEE-754 mandates correct rounding for it
    # alone), so its measured error is not small but ZERO, and plotting the
    # clamp value would flatten every other bar to invisibility.
    FLOOR = -400.0
    shown_ours, shown_theirs = np.maximum(ours, FLOOR), np.maximum(theirs, FLOOR)

    axis = figure.add_subplot(grid[1, 0])
    order = np.argsort(ours)
    names = [results[i]["name"] for i in order]
    position = np.arange(len(order))
    axis.barh(position + 0.2, shown_ours[order], height=0.38,
              color="#2e86ab", label="signal_math")
    axis.barh(position - 0.2, shown_theirs[order], height=0.38,
              color="#f0a202", label="libm")
    for slot, index in enumerate(order):
        if theirs[index] <= FLOOR:
            axis.text(FLOOR + 3, slot - 0.2, "exact", va="center",
                      fontsize=7, color="#8a5a00")
    axis.set_yticks(position)
    axis.set_yticklabels(names, fontsize=8)
    axis.set_xlabel("noise, dB relative to the signal (lower is quieter)")
    axis.set_title("noise floor: libm wins by a few dB", fontsize=12)
    axis.grid(alpha=0.22, axis="x")
    axis.legend(loc="lower left", fontsize=8)
    axis.set_xlim(FLOOR, -290)

    # The level is not the interesting axis. THIS one is.
    axis = figure.add_subplot(grid[1, 1])
    axis.barh(position + 0.2, ours_crest[order], height=0.38,
              color="#2e86ab", label="signal_math")
    axis.barh(position - 0.2, theirs_crest[order], height=0.38,
              color="#f0a202", label="libm")
    axis.axvline(10.5, color="#444444", ls="--", lw=1.2)
    axis.text(10.9, len(order) - 1.5,
              "crest of white noise -- to the left of this line the error is"
              " uncorrelated with the signal",
              fontsize=8, color="#444444", va="top", rotation=90,
              ha="left")
    axis.set_yticks(position)
    axis.set_yticklabels([])
    axis.set_xlabel("crest of the error spectrum, dB (peak minus median)")
    axis.set_title("error CHARACTER: flat noise, or distortion?", fontsize=12)
    axis.grid(alpha=0.22, axis="x")
    axis.legend(loc="lower right", fontsize=8)

    figure.suptitle(
        f"signal-math noise floor across {len(results)} operators   "
        f"(N={arguments.size}, tone bin {arguments.tone}, "
        f"{arguments.quality!r} bake)", fontsize=14)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=140, bbox_inches="tight")
    print(f"\nwrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
