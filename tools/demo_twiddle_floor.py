"""What an exact twiddle table buys: the spectral noise floor of a DFT.

A DFT's twiddles are ``exp(-2*pi*i*k/N)`` -- a DECLARED angle set, which is
exactly what ``signal_math.AnglePalette`` bakes. This compares two DFTs of the
same signal that differ in nothing but where their twiddles came from:

* ``libm``   -- ``cos(2*pi*k/N)`` and ``sin(...)`` computed the usual way,
  which first forms ``2*pi*k/N`` (not the exact angle) and then computes an
  accurate function of a slightly wrong argument.
* ``palette`` -- the correctly-rounded value of each angle, one quadrant
  stored, cardinal values placed so the symmetries are exact.

For a tone sitting exactly on a bin, the true spectrum is one nonzero bin and
silence everywhere else. Whatever appears in the other bins is arithmetic
error, so the floor IS a measurement of the twiddles -- no reference spectrum
needed, which is why this makes a good test rather than just a good picture.

Needs no compiler: everything here is the eager path plus the baked palette.

Run::

    python -m tools.demo_twiddle_floor --size 1024
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from src.common.tensors import signal_math as sm


def palette_twiddles(palette: sm.AnglePalette, size: int) -> np.ndarray:
    """The N x N DFT matrix from exactly-baked angles.

    ``k*j mod N`` is integer arithmetic, so every twiddle is a LOOKUP -- the
    angle is never formed as a float at all. That is the whole point: the
    usual route's error enters when ``2*pi*k*j/N`` is rounded, before any
    trigonometry happens.
    """

    sine = np.asarray(palette.sine, dtype=np.float64)
    cosine = np.asarray(palette.cosine, dtype=np.float64)
    index = np.outer(np.arange(size), np.arange(size)) % size
    return cosine[index] - 1j * sine[index]


def libm_twiddles(size: int) -> np.ndarray:
    """The same matrix, with the angle formed in floating point first."""

    index = np.outer(np.arange(size), np.arange(size))
    angle = 2.0 * np.pi * index / size
    return np.cos(angle) - 1j * np.sin(angle)


def floor_of(spectrum: np.ndarray, bin_index: int) -> tuple[float, float]:
    """Peak-to-floor in dB, and the worst spur, ignoring the true bin."""

    magnitude = np.abs(spectrum)
    peak = magnitude[bin_index]
    mask = np.ones(magnitude.size, dtype=bool)
    for offset in (-1, 0, 1):
        mask[(bin_index + offset) % magnitude.size] = False
    # The conjugate bin carries the same tone for a real input.
    for offset in (-1, 0, 1):
        mask[(-bin_index + offset) % magnitude.size] = False
    spur = float(np.max(magnitude[mask]))
    return 20.0 * np.log10(spur / peak), spur


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--bin", type=int, default=57)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "build" / "signal-field" / "twiddle_floor.png",
    )
    arguments = parser.parse_args(argv)
    size, bin_index = arguments.size, arguments.bin

    palette = sm.bake_angle_palette(size)
    print(f"palette: {size} divisions, {size // 4 + 1} stored, "
          f"{palette.measured_error:.1e} ulp(full scale), "
          f"admitted={palette.admitted}")

    # A tone exactly on a bin: the true spectrum is one line and silence.
    sample = np.arange(size)
    signal = np.asarray(palette.cosine, dtype=np.float64)[
        (bin_index * sample) % size
    ]

    results = {}
    for name, matrix in (("palette", palette_twiddles(palette, size)),
                         ("libm", libm_twiddles(size))):
        spectrum = matrix @ signal
        decibels, spur = floor_of(spectrum, bin_index)
        results[name] = (spectrum, decibels, spur)
        print(f"  {name:8s} worst spur {decibels:8.2f} dB below the tone "
              f"(absolute {spur:.3e})")

    quieter = results["palette"][1] - results["libm"][1]
    print(f"\nthe exact table is {abs(quieter):.1f} dB quieter"
          if quieter < 0 else
          f"\nno advantage measured ({quieter:+.1f} dB)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(11, 5.5))
    for name, colour in (("libm", "#f0a202"), ("palette", "#2e86ab")):
        spectrum = results[name][0]
        magnitude = np.abs(spectrum) / np.max(np.abs(spectrum))
        axis.plot(20.0 * np.log10(np.maximum(magnitude, 1e-20)),
                  color=colour, lw=0.9, label=f"{name} twiddles", alpha=0.85)
    axis.set_xlabel("bin")
    axis.set_ylabel("dB relative to the tone")
    axis.set_title(
        f"DFT of a tone exactly on bin {bin_index}, N={size}\n"
        "everything below the tone is arithmetic error, so the floor "
        "measures the twiddles",
        fontsize=11,
    )
    axis.set_ylim(-400, 10)
    axis.grid(alpha=0.25)
    axis.legend(loc="upper right")
    figure.tight_layout()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=140)
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
