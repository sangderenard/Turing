"""Per-participant exchange_time against the blended energy/power pin.

The claim the whole port rests on is that these are the same law.
``_energy_time_limit`` pins the next step at ``fraction * energy / power`` for
one blended participant; ``exchange_time_bound`` pins it at ``fraction * exchange_time`` per
participant and takes the minimum.  exchange_time IS energy/power, so with ONE
participant the two must agree -- and if they do, replacing the blended pin is
not a behaviour change for the single-participant case that everything shipping
today relies on.

With SEVERAL participants they deliberately disagree, and that disagreement is
the point rather than an error: the blended form sums energy and power across
participants before dividing, so a small stiff participant is averaged away.
This reports both -- the agreement where it is required, and the size of the
intended divergence where it is not -- so the second is never mistaken for the
first.

Method follows ``tools/frame_parity.py`` rather than inventing a third one:

* relative error is scaled by the larger of the two values, floored at 1e-12 of
  the sample's own scale, so a reference passing through zero does not report an
  infinite relative error;
* it is reported against the program's own ONE-ULP SENSITIVITY -- how far the
  answer moves when an input is nudged by a single ULP -- because an absolute
  ULP count means nothing for a quantity derived through a division;
* the distribution is reported, not a lone worst case.  ``signal_math_survey``
  records that a single sample landing on a zero produced a headline in the
  millions of ULP four separate times while the median was 0.

    python -u tools/dt_parity.py [samples]
"""

from __future__ import annotations


import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common.tensors import AbstractTensor

from src.common.dt_system.dt_controller import Targets, _energy_time_limit  # noqa: E402
from src.common.dt_system.dt_scaler import Metrics  # noqa: E402
from src.common.dt_system.participants import (  # noqa: E402
    Publication,
    StepSpans,
    exchange_time_bound,
)
from src.common.dt_system.time_contracts import BIND, ParticipantRegistry  # noqa: E402


def _registry(*names):
    registry = ParticipantRegistry()
    for name in names:
        registry.declare(name)
    return registry


def blended(energy, power, fraction, dt_proposed):
    """Today's pin: one participant's energy and power, through Targets."""
    metrics = Metrics(max_vel=0.0, max_flux=0.0, div_inf=0.0, mass_err=0.0,
                      error_channels=AbstractTensor.tensor([energy, power, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                      error_present=AbstractTensor.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    targets = Targets(cfl=0.5, div_max=1e9, mass_max=1e-3,
                      energy_exchange_fraction=fraction)
    limit = _energy_time_limit(metrics, targets)
    return dt_proposed if limit is None else min(dt_proposed, limit)


def per_participant(exchange_times, fraction, dt_proposed):
    """The port's pin: each participant's own exchange_time, masked minimum over them."""
    names = tuple(f"p{index}" for index in range(len(exchange_times)))
    spans = StepSpans.of(
        _registry(*names),
        {name: Publication(exchange_time_s=exchange_time, contract=BIND)
         for name, exchange_time in zip(names, exchange_times)},
    )
    return float(exchange_time_bound(spans, fraction, dt_proposed).item())


def relative(reference, value, scale_floor):
    """frame_parity's scaling: never infinite just because a value is zero."""
    if not (math.isfinite(reference) and math.isfinite(value)):
        return float("inf")
    scale = max(abs(reference), abs(value), scale_floor)
    return abs(value - reference) / scale if scale > 0.0 else 0.0


def quartiles(values):
    if not values:
        return [float("nan")] * 5
    ordered = sorted(values)
    def at(fraction):
        if len(ordered) == 1:
            return ordered[0]
        position = fraction * (len(ordered) - 1)
        low = int(math.floor(position))
        high = min(low + 1, len(ordered) - 1)
        weight = position - low
        return ordered[low] * (1.0 - weight) + ordered[high] * weight
    return [at(q) for q in (0.0, 0.25, 0.5, 0.75, 1.0)]


def one_ulp_sensitivity(samples, rng):
    """How far the answer moves when energy is nudged by a single ULP.

    The denominator every reported error is measured against.  Without it a
    division's error has no scale to be judged on.
    """
    moves = []
    for energy, power, fraction, dt_proposed in samples:
        base = blended(energy, power, fraction, dt_proposed)
        nudged = blended(math.nextafter(energy, math.inf), power, fraction,
                         dt_proposed)
        moves.append(relative(base, nudged, 1.0e-12 * max(abs(base), 1e-300)))
    return max(moves) if moves else 0.0


def main(argv) -> int:
    count = int(argv[0]) if argv else 4000
    rng = random.Random(20260919)

    # A spread of magnitudes, because the disagreement this is looking for is a
    # cancellation effect and a narrow range would hide it.
    samples = []
    for _ in range(count):
        energy = 10.0 ** rng.uniform(-6, 6)
        power = 10.0 ** rng.uniform(-6, 6)
        fraction = rng.uniform(0.01, 1.0)
        dt_proposed = 10.0 ** rng.uniform(-6, -1)
        samples.append((energy, power, fraction, dt_proposed))

    sensitivity = one_ulp_sensitivity(samples, rng)
    print(f"samples                      : {count}")
    print(f"one-ULP sensitivity (rel)    : {sensitivity:.3e}")
    print()

    # ---- the equivalence that must hold -------------------------------------
    errors = []
    worst = None
    for energy, power, fraction, dt_proposed in samples:
        reference = blended(energy, power, fraction, dt_proposed)
        ported = per_participant([energy / power], fraction, dt_proposed)
        error = relative(reference, ported, 1.0e-12 * max(abs(reference), 1e-300))
        errors.append(error)
        if worst is None or error > worst[0]:
            worst = (error, energy, power, fraction, dt_proposed, reference, ported)

    exact = sum(1 for error in errors if error == 0.0)
    print("ONE participant: exchange_time_bound against the blended pin")
    print(f"  exactly equal              : {exact}/{count}"
          f"  ({100.0 * exact / count:.2f}%)")
    print(f"  relative error quartiles   : "
          + "  ".join(f"{q:.2e}" for q in quartiles(errors)))
    if sensitivity > 0.0:
        print(f"  worst as multiple of 1 ULP : "
              f"{max(errors) / sensitivity:.3g}")
    if worst and worst[0] > 0.0:
        _err, energy, power, fraction, dt_proposed, reference, ported = worst
        print(f"  worst sample               : energy={energy:.6e} "
              f"power={power:.6e} fraction={fraction:.4f}")
        print(f"     blended={reference!r}")
        print(f"     ported ={ported!r}")
    print()

    # ---- the divergence that is intended -----------------------------------
    print("SEVEN participants: the blended sum-then-divide against per participant")
    print("  (these SHOULD differ; the blended form averages a stiff one away)")
    ratios = []
    for _ in range(min(count, 2000)):
        energies = [10.0 ** rng.uniform(-3, 3) for _ in range(7)]
        powers = [10.0 ** rng.uniform(-3, 3) for _ in range(7)]
        fraction = rng.uniform(0.05, 0.5)
        dt_proposed = 10.0 ** rng.uniform(-4, -1)
        # today: sum both, then divide once
        blended_limit = blended(sum(energies), sum(powers), fraction, dt_proposed)
        # ported: each participant's own exchange_time, minimum over them
        ported_limit = per_participant(
            [e / p for e, p in zip(energies, powers)], fraction, dt_proposed)
        if blended_limit > 0.0:
            ratios.append(ported_limit / blended_limit)
    print(f"  ported / blended quartiles : "
          + "  ".join(f"{q:.3g}" for q in quartiles(ratios)))
    stricter = sum(1 for r in ratios if r < 1.0)
    print(f"  ported is stricter in       : {stricter}/{len(ratios)}"
          f"  ({100.0 * stricter / max(len(ratios), 1):.1f}%)")
    print()
    print("A ratio below 1 is the stiff participant becoming visible: the")
    print("blended pin allowed a step its own stiffest member would not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
