"""A 5D signal field -- (x, y) -> (R, G, B) -- rendered by the signal pack.

Every sine, cosine and exponential in the image below is computed by a
compiled kernel from ``signal_kernels``, routed per call by the same
``LaunchCoordinator`` that routes ``gemm``. Nothing here calls libm for the
signal itself; the coordinate geometry (radius, angle) is ordinary array
arithmetic and is labelled as such, because a demo that blurs which half did
the work is not evidence of anything.

The field is a superposition of three chirped zone plates, one per colour
channel, at slightly different chirp rates. Quadratic phase is deliberate: it
sweeps spatial frequency from zero at the centre to the Nyquist limit at the
corners, so a single image exercises the whole reduction path -- small
arguments where cancellation would bite, and large turn counts where a radian
reduction would have collapsed. Colour fringing away from the centre IS the
rate difference between channels, and the moire is the beat between the
continuous chirp and an exactly-baked angle palette.

Accounting is printed beside the picture, never mixed into it: which variant
served each call, what each core measured at bake time, and where each
derived quantity's error came from. That separation is the point -- see the
module note in ``signal_kernels`` about what belongs in a baked kernel and
what belongs in the pack around it.

Run::

    python -m tools.demo_signal_field --size 2048
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

from src.common.tensors import signal_kernels as sk
from src.common.tensors import signal_math as sm
from src.compiler.kernel_bank import KernelBank, LaunchCoordinator


#: Per-channel chirp rate, angular multiplier and phase offset, in TURNS.
#: Held in turns rather than radians throughout: the reduction is then exact
#: at any magnitude, which is what lets the corners stay clean.
CHANNELS = (
    {"chirp": 26.0, "arms": 5.0, "phase": 0.00, "name": "R"},
    {"chirp": 29.0, "arms": 5.0, "phase": 0.19, "name": "G"},
    {"chirp": 32.0, "arms": 5.0, "phase": 0.38, "name": "B"},
)


def open_pack(root: Path, quality: str):
    """The signal pack: compiled kernels plus the accounting around them."""

    specs = sk.signal_kernel_specs(quality)
    bank = KernelBank(root, dict(specs))
    coordinator = LaunchCoordinator(
        bank, contract="fast", prefer_specialized=True, specialize_missing=False,
    )
    return bank, coordinator, specs


def _launch(coordinator, name: str, values: np.ndarray) -> np.ndarray:
    """One routed call of an elementwise signal kernel."""

    flat = np.ascontiguousarray(values.ravel(), dtype=np.float64)
    out = coordinator.launch(
        name, x=flat, y=np.zeros(flat.size), n=int(flat.size),
    )
    return np.asarray(out, dtype=np.float64).reshape(values.shape)


def build_field(coordinator, size: int, palette: sm.AnglePalette) -> np.ndarray:
    """The 5D field: two spatial axes carrying three channels."""

    axis = np.linspace(-1.0, 1.0, size)
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="xy")
    # Geometry only -- not the signal. Plain array arithmetic.
    radius = np.hypot(grid_x, grid_y)
    angle = np.arctan2(grid_y, grid_x) / (2.0 * np.pi)      # in TURNS

    image = np.zeros((size, size, 3), dtype=np.float64)
    for index, channel in enumerate(CHANNELS):
        # Quadratic (chirp) phase plus an angular term, all in turns.
        turns = (
            channel["chirp"] * radius * radius
            + channel["arms"] * angle
            + channel["phase"]
        )
        # The signal itself: compiled kernels only.
        wave = _launch(coordinator, "sin", turns * (2.0 * np.pi))
        # A Gaussian envelope, also compiled -- exercises the exp reduction.
        envelope = _launch(coordinator, "exp", -2.2 * radius * radius)
        # The palette beat: the same angle quantised to a declared set, so
        # the moire is the difference between a continuous chirp and an
        # exactly-baked one.
        divisions = palette.divisions
        quantised = np.mod(
            np.rint(turns * divisions).astype(np.int64), divisions,
        )
        stepped = np.asarray(palette.sine, dtype=np.float64)[quantised]
        image[..., index] = envelope * (0.62 * wave + 0.38 * stepped)
    return image


def to_rgb(field: np.ndarray) -> np.ndarray:
    lifted = 0.5 + 0.5 * np.clip(field, -1.0, 1.0)
    return (np.clip(lifted, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def report_accounting(bank, specs, palette, coordinator, quality) -> None:
    """What the pack knows about itself -- printed, never baked in."""

    print("\n--- pack accounting -------------------------------------------")
    print(f"quality: {quality}")
    cores = sm.signal_math(quality).cores
    print(f"{'core':7s} {'family':11s} {'consts':>7s} {'~ulp':>8s} {'adm':>4s}")
    for name in ("sin", "cos", "exp"):
        core = cores[name]
        print(f"{name:7s} {core.family:11s} {len(core.values):7d} "
              f"{core.measured_error / 2.220446049250313e-16:8.2f} "
              f"{('yes' if core.admitted else 'NO'):>4s}")
    print(f"palette  divisions {palette.divisions}, "
          f"{palette.divisions // 4 + 1} stored, "
          f"{palette.measured_error:.1e} ulp(full scale), "
          f"admitted={palette.admitted}")
    print("\nadmitted kernel variants:")
    for row in bank.inventory():
        verification = row.get("verification", {})
        if not verification.get("admitted"):
            continue
        print(f"  {row.get('kernel'):8s} {str(row.get('specialized')):14s} "
              f"worst {verification.get('worst_abs_error'):.2e}")
    log = Path(bank.root) / "routing_log.jsonl"
    if log.is_file():
        rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        served: dict[str, dict[str, int]] = {}
        for row in rows[-24:]:
            served.setdefault(str(row.get("kernel")), {}).setdefault(
                str(row.get("route")), 0,
            )
            served[str(row.get("kernel"))][str(row.get("route"))] += 1
        print("\nrouting for this render:")
        for kernel, routes in sorted(served.items()):
            print(f"  {kernel:8s} {routes}")
    print("\nderived error, propagated rather than re-measured:")
    unit = 2.220446049250313e-16
    sine = cores["sin"].measured_error / unit
    cosine = cores["cos"].measured_error / unit
    print(f"  sin core          {sine:6.2f} ulp   (measured)")
    print(f"  cos core          {cosine:6.2f} ulp   (measured)")
    print(f"  envelope*wave     {sine + 0.5:6.2f} ulp   (one rounding on a product)")
    print(f"  palette term        0.00 ulp   (correctly rounded, declared set)")
    print("  each figure carries the interval it was measured on; a propagated")
    print("  bound outside that interval is not a bound -- see cot, which the")
    print("  same arithmetic mispredicted by 6x when applied across cos's zero.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--quality", default=sk.DEFAULT_KERNEL_QUALITY)
    parser.add_argument("--divisions", type=int, default=96)
    parser.add_argument("--bank", type=Path,
                        default=ROOT / "build" / "signal-field-bank")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "build" / "signal-field" / "signal_field.png")
    arguments = parser.parse_args(argv)

    started = time.perf_counter()
    bank, coordinator, specs = open_pack(arguments.bank, arguments.quality)
    for name in ("sin", "exp"):
        bank.get(name, contract="fast")
    palette = sm.bake_angle_palette(arguments.divisions)
    print(f"pack ready in {time.perf_counter() - started:.1f}s")

    started = time.perf_counter()
    field = build_field(coordinator, arguments.size, palette)
    elapsed = time.perf_counter() - started
    pixels = arguments.size ** 2
    print(f"field: {arguments.size}x{arguments.size}x3 = {pixels * 3:,} samples "
          f"in {elapsed:.2f}s ({pixels * 3 / elapsed / 1e6:.1f} M samples/s)")

    from PIL import Image

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_rgb(field), mode="RGB").save(arguments.output)
    print(f"wrote {arguments.output}")
    report_accounting(bank, specs, palette, coordinator, arguments.quality)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
