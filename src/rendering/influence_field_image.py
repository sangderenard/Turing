"""Static images of an influence field, for looking at before animating one.

The live surface has many ways to look wrong that have nothing to do with
whether the colour is right: physics that never settles, an event relay that
drops, a camera that frames nothing. A written image isolates the one question
worth answering first -- given known power sums, does the collapse produce a
colour that means what it claims. It is also diffable, which the live surface
is not.

Colour is produced through OkLCh rather than HSV. That is not a refinement:
``InfluenceReading.hue`` is the *mean* of a distribution over the source arc,
so it is read quantitatively, and on a perceptually non-uniform ramp equal
steps in the mean are unequal steps to the eye -- the centroid would lie. HSV
also couples hue to lightness badly enough that yellow and blue at the same
``value`` differ in apparent brightness by more than the magnitude channel
spans, which would put hue crosstalk directly into the readout that is
supposed to carry weight.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

from ..compiler.influence_field import (
    BAKED,
    DYNAMIC,
    RESERVED_BAND,
    SPECTRUM_END,
    InfluenceContract,
    InfluenceField,
    InfluenceReading,
    Moments,
)

# OkLCh hue angles walking the spectrum from red to violet. The dye arc maps
# onto this span; the reserved band continues into the non-spectral magentas,
# which is why an annotation can never be confused with a measurement.
SPECTRAL_START_DEGREES = 29.0
SPECTRAL_END_DEGREES = 300.0
RESERVED_END_DEGREES = 360.0

BASE_LIGHTNESS = 0.72

# sRGB holds far more chroma at yellow and green than at blue and violet, so a
# single constant chroma across the arc is pinned to the worst hue and starves
# every other one -- measurably: at this lightness even 0.125 clips in violet
# while yellow still has room to spare. Chroma is therefore normalised per hue
# against the in-gamut ceiling, so saturation 1.0 means "as vivid as this hue
# can be" and reads with equal force everywhere along the arc.
#
# That makes equal saturation mean unequal chroma, which is the right trade
# here: saturation carries dispersion, a relative quantity in [0, 1], not a
# colorimetric one. Uniform legibility is what the readout needs.
_CHROMA_CEILING_STEPS = 256
_CHROMA_CEILING: list[float] = []
# Held just inside the boundary; landing exactly on it clips under rounding.
_GAMUT_MARGIN = 0.97


def _linear_to_srgb(channel: float) -> float:
    if channel <= 0.0031308:
        return 12.92 * channel
    return 1.055 * (channel ** (1.0 / 2.4)) - 0.055


def _linear_rgb(lightness: float, chroma: float, hue_degrees: float):
    hue = math.radians(hue_degrees)
    a = chroma * math.cos(hue)
    b = chroma * math.sin(hue)
    l_ = lightness + 0.3963377774 * a + 0.2158037573 * b
    m_ = lightness - 0.1055613458 * a - 0.0638541728 * b
    s_ = lightness - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    return (
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    )


def oklch_to_rgb(lightness: float, chroma: float, hue_degrees: float):
    """Convert OkLCh to clipped 8-bit sRGB."""

    return tuple(
        max(0, min(255, round(255.0 * _linear_to_srgb(max(0.0, channel)))))
        for channel in _linear_rgb(lightness, chroma, hue_degrees)
    )


def max_chroma(lightness: float, hue_degrees: float) -> float:
    """Largest chroma that stays inside sRGB at this lightness and hue."""

    low, high = 0.0, 0.4
    for _ in range(24):
        mid = (low + high) / 2.0
        channels = _linear_rgb(lightness, mid, hue_degrees)
        inside = all(-1e-9 <= channel <= 1.0 + 1e-9 for channel in channels)
        low, high = (mid, high) if inside else (low, mid)
    return low * _GAMUT_MARGIN


def _chroma_ceiling(hue_degrees: float) -> float:
    """Interpolated, smoothed in-gamut chroma ceiling at this hue.

    Both corrections exist because the raw ceiling is a landmark generator.
    Sampling it at 256 steps and taking the nearest leaves a visible seam where
    it moves fastest, and the ceiling peaks sharply at green -- so an
    unsmoothed curve draws a bright band there that a reader would take for
    structure in the data. Neither is cosmetic: this channel is a measurement,
    and an artefact in it is indistinguishable from a finding.
    """

    if not _CHROMA_CEILING:
        raw = [
            max_chroma(BASE_LIGHTNESS, 360.0 * index / _CHROMA_CEILING_STEPS)
            for index in range(_CHROMA_CEILING_STEPS)
        ]
        window = _CHROMA_CEILING_STEPS // 12
        _CHROMA_CEILING.extend(
            min(
                raw[index],
                sum(
                    raw[(index + offset) % _CHROMA_CEILING_STEPS]
                    for offset in range(-window, window + 1)
                ) / (2 * window + 1),
            )
            for index in range(_CHROMA_CEILING_STEPS)
        )
    position = (hue_degrees / 360.0 * _CHROMA_CEILING_STEPS) % _CHROMA_CEILING_STEPS
    low = int(position)
    fraction = position - low
    return (
        _CHROMA_CEILING[low] * (1.0 - fraction)
        + _CHROMA_CEILING[(low + 1) % _CHROMA_CEILING_STEPS] * fraction
    )


def dye_rgb(hue: float, saturation: float, value: float = 1.0):
    """Colour for one collapsed reading.

    ``hue`` is the arc position, ``saturation`` carries dispersion, and
    ``value`` carries arriving weight. They map to OkLCh hue, chroma, and
    lightness respectively, which keeps them perceptually independent -- the
    property the whole encoding rests on.
    """

    position = 0.0 if SPECTRUM_END <= 0 else max(0.0, min(1.0, hue / SPECTRUM_END))
    degrees = SPECTRAL_START_DEGREES + position * (
        SPECTRAL_END_DEGREES - SPECTRAL_START_DEGREES
    )
    # A source carries only its own unit deposit, so it genuinely sits low on
    # this channel -- that is honest, but the origins are also the legend, and
    # they have to stay readable. The floor compresses the bottom of the range
    # rather than lifting sources dishonestly above what reached them.
    lightness = BASE_LIGHTNESS * (0.55 + 0.45 * max(0.0, min(1.0, value)))
    return oklch_to_rgb(
        lightness,
        _chroma_ceiling(degrees) * max(0.0, min(1.0, saturation)),
        degrees,
    )


def marker_rgb(hue: float):
    """Colour for a reserved-band semantic annotation."""

    low, high = RESERVED_BAND
    position = 0.0 if high <= low else max(0.0, min(1.0, (hue - low) / (high - low)))
    degrees = SPECTRAL_END_DEGREES + position * (
        RESERVED_END_DEGREES - SPECTRAL_END_DEGREES
    )
    return oklch_to_rgb(BASE_LIGHTNESS, _chroma_ceiling(degrees), degrees)


def bake_palette(width: int = 256, height: int = 64):
    """Bake the collapse into a sampler: u = hue on the arc, v = dispersion.

    This is the same surface ``render_colour_space`` draws, which is the point
    -- the diagnostic chart and the GPU asset are one artifact, so what a
    reviewer inspected is literally what the shader samples. It also moves the
    entire OkLCh conversion and gamut-ceiling search to bake time, leaving the
    fragment stage a single texture fetch.

    Returns ``(rgb, reserved)``: the dye surface, and the annotation strip that
    no dye colour can reach.
    """

    import numpy as np

    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for column in range(width):
        hue = SPECTRUM_END * column / max(1, width - 1)
        for row in range(height):
            dispersion = row / max(1, height - 1)
            rgb[row, column] = dye_rgb(hue, 1.0 - dispersion, 1.0)
    low, high = RESERVED_BAND
    reserved = np.zeros((1, width, 3), dtype=np.uint8)
    for column in range(width):
        reserved[0, column] = marker_rgb(
            low + (high - low) * column / max(1, width - 1)
        )
    return rgb, reserved


def render_colour_space(
    path: str | Path,
    contract: InfluenceContract | None = None,
    *,
    width: int = 900,
    height: int = 520,
) -> Path:
    """Write the entire space the encoding can produce, with no graph involved.

    Horizontal is mean hue across the arc; vertical is dispersion. Every colour
    any node can ever take appears somewhere in the field, so a dead zone or a
    muddy band is visible here before it is ever blamed on a program.
    """

    contract = contract or InfluenceContract(enabled=True)
    image = Image.new("RGB", (width, height), (18, 18, 22))
    draw = ImageDraw.Draw(image)

    margin = 60
    field_width = width - 2 * margin
    field_height = height - 2 * margin - 70

    for column in range(field_width):
        hue = SPECTRUM_END * column / max(1, field_width - 1)
        for row in range(field_height):
            dispersion = row / max(1, field_height - 1)
            shaped = (1.0 - dispersion) ** contract.saturation_gamma
            saturation = (
                contract.saturation_floor
                + (1.0 - contract.saturation_floor) * shaped
            )
            draw.point(
                (margin + column, margin + row), fill=dye_rgb(hue, saturation)
            )

    # The reserved band, drawn adjacent so the disjointness is checkable by eye
    # rather than only by assertion.
    band_top = margin + field_height + 24
    low, high = RESERVED_BAND
    for column in range(field_width):
        hue = low + (high - low) * column / max(1, field_width - 1)
        for row in range(28):
            draw.point((margin + column, band_top + row), fill=marker_rgb(hue))

    draw.text((margin, 20), "influence field colour space", fill=(235, 235, 240))
    draw.text(
        (margin, 36),
        f"x: mean hue 0 -> {SPECTRUM_END} (arc)    "
        f"y: dispersion 0 (top) -> 1 (bottom)    "
        f"gamma={contract.saturation_gamma} floor={contract.saturation_floor}",
        fill=(150, 150, 160),
    )
    draw.text(
        (margin, band_top + 34),
        f"reserved band {low} -> {high}: semantic annotation, "
        "unreachable by transport",
        fill=(150, 150, 160),
    )
    draw.text((margin, margin - 14), "concentrated", fill=(150, 150, 160))
    draw.text(
        (margin, margin + field_height + 4), "mixed", fill=(150, 150, 160)
    )

    target = Path(path)
    image.save(target)
    return target


def render_field(
    path: str | Path,
    field: InfluenceField,
    positions: Mapping[Any, tuple[float, float]],
    *,
    width: int = 900,
    height: int = 700,
    radius: int = 16,
    labels: Mapping[Any, str] | None = None,
) -> Path:
    """Draw a field over a supplied layout, as core (dynamic) and rim (baked).

    The two categories are drawn as concentric regions rather than blended,
    because blending them would average two hues that describe different
    binding times and report a dispersion that describes nothing. Rim thickness
    is the staging ratio, so a constant-foldable node is visibly all rim.
    """

    image = Image.new("RGB", (width, height), (18, 18, 22))
    draw = ImageDraw.Draw(image)
    readings = {reading.key: reading for reading in field.table()}
    if not positions:
        return _save(image, path)

    xs = [point[0] for point in positions.values()]
    ys = [point[1] for point in positions.values()]
    span_x = max(1e-6, max(xs) - min(xs))
    span_y = max(1e-6, max(ys) - min(ys))
    margin = 70

    def place(key: Any) -> tuple[float, float]:
        x, y = positions[key]
        return (
            margin + (x - min(xs)) / span_x * (width - 2 * margin),
            margin + (y - min(ys)) / span_y * (height - 2 * margin),
        )

    # Edges first, tinted by the heaviest transport that crossed them, so the
    # picture shows what actually flowed rather than what merely connects.
    heaviest: dict[tuple[Any, Any], Any] = {}
    for transport in field.trace():
        edge = (transport.source_key, transport.target_key)
        current = heaviest.get(edge)
        if current is None or transport.weight > current.weight:
            heaviest[edge] = transport
    for (source, target), transport in heaviest.items():
        if source not in positions or target not in positions:
            continue
        colour = dye_rgb(transport.hue, 0.9, min(1.0, transport.weight * 3.0))
        draw.line([place(source), place(target)], fill=colour, width=2)

    for key, _ in positions.items():
        x, y = place(key)
        reading = readings.get(key)
        if reading is None:
            draw.ellipse(
                [x - 4, y - 4, x + 4, y + 4], fill=(60, 60, 68)
            )
            continue
        baked = reading.categories.get(BAKED)
        dynamic = reading.categories.get(DYNAMIC)
        outer = radius
        if baked is not None and baked.weight > 0.0:
            # Each category spans the whole arc independently, so the n-th
            # dynamic and n-th baked source share a hue. Without a second
            # difference the rim and core render as one flat disc and the
            # staging split disappears precisely where it is most worth
            # reading, so the rim is additionally darkened. Hue still carries
            # origin; lightness separates binding time.
            draw.ellipse(
                [x - outer, y - outer, x + outer, y + outer],
                fill=dye_rgb(
                    baked.hue, baked.saturation, reading.value * 0.45
                ),
            )
        inner = outer * (1.0 - 0.7 * reading.staging)
        if dynamic is not None and dynamic.weight > 0.0:
            draw.ellipse(
                [x - inner, y - inner, x + inner, y + inner],
                fill=dye_rgb(dynamic.hue, dynamic.saturation, reading.value),
            )
        if labels and key in labels:
            draw.text((x + outer + 4, y - 6), labels[key], fill=(210, 210, 218))

    return _save(image, path)


def _save(image: Image.Image, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target)
    return target


__all__ = [
    "oklch_to_rgb", "dye_rgb", "marker_rgb",
    "bake_palette", "render_colour_space", "render_field",
]
