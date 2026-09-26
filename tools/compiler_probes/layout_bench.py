"""Interleaved channels versus planar limbs, measured on the eager path.

The claim under test: channel storage is what makes extended precision
fast. The alternative is planar -- one contiguous tensor per limb, kept in
a tuple -- which tells the truth about shape but is assumed slower.

Both forms run the same work: the limb-wise arithmetic an expansion add
actually performs, which touches every limb of every element.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

WIDTH = 2
SIZES = (1_024, 65_536, 1_048_576)
REPEATS = 20


def timed(function, *arguments):
    function(*arguments)
    best = float("inf")
    for _ in range(REPEATS):
        started = time.perf_counter()
        function(*arguments)
        best = min(best, time.perf_counter() - started)
    return best


def channel_two_sum(left, right, width):
    """Knuth two_sum over interleaved storage: every limb is a strided view."""

    out = np.empty_like(left)
    for index in range(width):
        a = left[index::width]
        b = right[index::width]
        s = a + b
        shifted = s - a
        out[index::width] = (a - (s - shifted)) + (b - shifted)
    return out


def planar_two_sum(left, right, width):
    """The same, over one contiguous array per limb."""

    out = []
    for index in range(width):
        a, b = left[index], right[index]
        s = a + b
        shifted = s - a
        out.append((a - (s - shifted)) + (b - shifted))
    return out


print(f"{'elements':>10}  {'interleaved':>12}  {'planar':>12}  {'planar is':>12}")
for count in SIZES:
    flat_left = np.random.default_rng(0).random(count * WIDTH)
    flat_right = np.random.default_rng(1).random(count * WIDTH)
    planar_left = [np.ascontiguousarray(flat_left[i::WIDTH]) for i in range(WIDTH)]
    planar_right = [np.ascontiguousarray(flat_right[i::WIDTH]) for i in range(WIDTH)]

    channel_seconds = timed(channel_two_sum, flat_left, flat_right, WIDTH)
    planar_seconds = timed(planar_two_sum, planar_left, planar_right, WIDTH)
    ratio = channel_seconds / max(planar_seconds, 1e-12)
    print(
        f"{count:10,d}  {channel_seconds * 1e6:10.1f}us  "
        f"{planar_seconds * 1e6:10.1f}us  {ratio:10.2f}x"
    )
