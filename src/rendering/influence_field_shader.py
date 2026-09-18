"""Shader stage for an influence field: palette, manifests, and dye optics.

The flat diagram and the eventual 3D surface want the same three things, so
they are built once here and consumed twice.

**Palette.** A baked 2D sampler: ``u`` is hue on the spectral arc, ``v`` is
dispersion. It is the exact surface ``render_colour_space`` draws, so the chart
a reviewer inspected is literally the texture the shader samples, and the whole
OkLCh conversion plus gamut-ceiling search happens at bake time. The fragment
stage costs one fetch. In 3D nothing changes: same texture, sampled as a
palette rather than blitted as a diagram.

**Manifests.** Per-edge and per-node attributes packed into data textures, two
texels each, indexed by element id. Everything the fragment stage needs travels
in 8 floats and nothing has to be re-derived on the GPU. A 3D pass binds the
same manifests against different geometry.

**Optics.** Dye in the pipes is rendered as an optically active solution.

That choice is not decoration bolted onto a measurement. A sucrose solution
between crossed polarizers rotates the plane of polarization by an angle
proportional to concentration times path length, and shorter wavelengths rotate
further -- optical rotatory dispersion. Transmitted intensity per wavelength
goes as ``sin^2`` of that angle, which is why a concentration gradient reads as
travelling rainbow bands. We already hold both terms the physics wants:
``weight`` is concentration, and distance along an edge is path length. So the
shimmer is a second, independent readout of a quantity the field already
measured, rather than a new claim about the program.

The discipline that has held everywhere else holds here too, and it is the
reason the layers are kept apart:

* the **data layer** -- hue, dispersion, weight, staging -- is measured, and
  nothing in the optics is permitted to move it;
* the **optical layer** is expressive, but every parameter it is driven by is
  already in the data layer.

Concretely: the dye's hue never drifts to look pretty. Bands travel because
``uTime`` advances the path-length term, exactly as a real gradient would, and
their spacing is set by concentration. A viewer reading band density is reading
weight. If the optics were allowed to invent hue, a shimmer would become
indistinguishable from a finding -- the same failure the smoothed chroma
ceiling exists to prevent.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from ..compiler.influence_field import (
    BAKED,
    DYNAMIC,
    RECURRENT,
    SPECTRUM_END,
    InfluenceField,
)

# Representative wavelengths in nanometres, normalised against green. Three
# samples is enough: rotation varies smoothly with wavelength, so the banding
# is reproduced by its low-order behaviour and a full spectral integration buys
# nothing a viewer can see.
WAVELENGTHS_NM = (610.0, 550.0, 460.0)
REFERENCE_NM = 550.0

# Biot's law gives rotation proportional to concentration and path; the
# dispersion term is empirically near inverse-square in wavelength away from
# absorption bands, which is the regime a sugar solution sits in.
ORD_EXPONENT = 2.0

# Calibration, measured rather than chosen. Rotation accumulates as
# concentration x path, so a gain high enough to be dramatic sweeps tens of
# cycles across a long pipe and every edge renders as a full rainbow -- band
# density still tracks concentration, but the dye's measured hue is destroyed,
# which is precisely the failure this module's layering exists to prevent.
# Gain is therefore set so the longest pipe in a normalised layout shows a
# small number of bands, and amplitude is held well below saturation so the
# optics read as iridescence over the dye rather than in place of it.
SHIMMER_TARGET_BANDS = 3.0
SHIMMER_AMOUNT = 0.40


def shimmer_gain(longest_path: float, peak_concentration: float = 1.0) -> float:
    """Gain that renders ``SHIMMER_TARGET_BANDS`` across the longest pipe."""

    reach = max(1e-6, longest_path * max(1e-6, peak_concentration))
    return SHIMMER_TARGET_BANDS * 2.0 * math.pi / reach



def _rotation_scale() -> np.ndarray:
    """Per-channel rotation multiplier from the wavelength dependence."""

    return np.asarray(
        [(REFERENCE_NM / nm) ** ORD_EXPONENT for nm in WAVELENGTHS_NM],
        dtype=np.float32,
    )


PALETTE_GLSL = """
// Shared by every pass. The palette carries the whole collapse, so a consumer
// never reimplements OkLCh and can never disagree with the baked chart.
vec3 dyeColour(sampler2D palette, float hue, float dispersion) {
    return texture(palette, vec2(clamp(hue / SPECTRUM_END, 0.0, 1.0),
                                 clamp(dispersion, 0.0, 1.0))).rgb;
}

// Optical rotation through an active solution: angle scales with
// concentration and path, and with wavelength through uOrdScale. Between
// crossed polarizers the transmitted fraction is sin^2 of that angle, so a
// gradient becomes bands and a stronger solution packs them tighter.
vec3 rotatoryTransmission(float concentration, float path, vec3 ordScale,
                          float gain) {
    vec3 angle = ordScale * (gain * concentration * path);
    vec3 transmitted = sin(angle);
    return transmitted * transmitted;
}
"""

EDGE_VERTEX_GLSL = """#version 330 core
layout(location=0) in vec2 aCorner;      // -1..1 across the pipe, 0..1 along
layout(location=1) in vec2 aFrom;
layout(location=2) in vec2 aTo;
layout(location=3) in float aEdgeIndex;

uniform mat4 uMVP;
uniform float uPipeWidth;

out vec2 vPipe;          // x: along 0..1, y: across -1..1
out float vEdgeIndex;
out float vPathLength;

void main() {
    vec2 axis = aTo - aFrom;
    float span = max(length(axis), 1e-6);
    vec2 forward = axis / span;
    vec2 side = vec2(-forward.y, forward.x);
    vec2 world = aFrom + forward * (span * aCorner.x)
               + side * (uPipeWidth * aCorner.y);
    vPipe = vec2(aCorner.x, aCorner.y);
    vEdgeIndex = aEdgeIndex;
    vPathLength = span;
    gl_Position = uMVP * vec4(world, 0.0, 1.0);
}
"""

EDGE_FRAGMENT_GLSL = """#version 330 core
#define SPECTRUM_END %(spectrum_end)s

in vec2 vPipe;
in float vEdgeIndex;
in float vPathLength;

uniform sampler2D uPalette;
uniform sampler2D uEdgeManifest;   // two texels per edge
uniform float uEdgeCount;
uniform float uTime;
uniform vec3 uOrdScale;
uniform float uShimmerGain;
uniform float uShimmerAmount;
uniform float uFlowSpeed;

out vec4 fragColour;

%(palette)s

vec4 manifest(float index, float slot) {
    float u = (index * 2.0 + slot + 0.5) / (uEdgeCount * 2.0);
    return texture(uEdgeManifest, vec2(u, 0.5));
}

void main() {
    vec4 tone = manifest(vEdgeIndex, 0.0);   // hueDyn, dispDyn, hueBake, dispBake
    vec4 flux = manifest(vEdgeIndex, 1.0);   // weight, staging, recurrence, phase

    // Round pipe cross-section. The profile is geometry, not data: it makes a
    // pipe read as a pipe without consuming a channel that carries meaning.
    float across = clamp(abs(vPipe.y), 0.0, 1.0);
    float profile = sqrt(max(0.0, 1.0 - across * across));
    if (profile <= 0.01) discard;

    vec3 dynamic = dyeColour(uPalette, tone.x, tone.y);
    vec3 baked = dyeColour(uPalette, tone.z, tone.w);
    // Binding time separates by lightness as well as hue, because two
    // categories can hold neighbouring hues and would otherwise merge.
    vec3 base = mix(dynamic, baked * 0.55, clamp(flux.y, 0.0, 1.0));

    // Path length advances with time, which is what makes the bands travel.
    // Only the path term moves; concentration and hue stay as measured.
    float path = vPipe.x * vPathLength - uTime * uFlowSpeed + flux.w;
    vec3 shimmer = rotatoryTransmission(flux.x, path, uOrdScale, uShimmerGain);

    // Quiet edges stay calm: shimmer strength is itself concentration, so a
    // barely-used pipe cannot flicker as though it were busy.
    float amount = uShimmerAmount * clamp(flux.x, 0.0, 1.0);
    vec3 colour = base * mix(vec3(1.0), shimmer, amount);

    float alpha = clamp(0.25 + 0.75 * flux.x, 0.0, 1.0) * profile;
    fragColour = vec4(colour * profile, alpha);
}
"""

NODE_VERTEX_GLSL = """#version 330 core
layout(location=0) in vec2 aCentre;
layout(location=1) in float aRadius;
layout(location=2) in float aNodeIndex;

uniform mat4 uMVP;

out float vNodeIndex;

void main() {
    vNodeIndex = aNodeIndex;
    gl_PointSize = aRadius;
    gl_Position = uMVP * vec4(aCentre, 0.0, 1.0);
}
"""

NODE_FRAGMENT_GLSL = """#version 330 core
#define SPECTRUM_END %(spectrum_end)s

in float vNodeIndex;

uniform sampler2D uPalette;
uniform sampler2D uNodeManifest;
uniform float uNodeCount;
uniform float uTime;
uniform vec3 uOrdScale;
uniform float uShimmerGain;
uniform float uShimmerAmount;

out vec4 fragColour;

%(palette)s

vec4 manifest(float index, float slot) {
    float u = (index * 2.0 + slot + 0.5) / (uNodeCount * 2.0);
    return texture(uNodeManifest, vec2(u, 0.5));
}

void main() {
    vec2 offset = gl_PointCoord * 2.0 - 1.0;
    float radius = length(offset);
    if (radius > 1.0) discard;

    vec4 tone = manifest(vNodeIndex, 0.0);
    vec4 flux = manifest(vNodeIndex, 1.0);

    // Core is runtime influence, rim is baked, rim thickness is the staging
    // ratio -- so a constant-foldable node is visibly all rim. They are drawn
    // as regions rather than blended because averaging two binding times
    // reports a dispersion that describes nothing.
    float coreEdge = 1.0 - 0.7 * clamp(flux.y, 0.0, 1.0);
    bool core = radius <= coreEdge;
    vec3 colour = core
        ? dyeColour(uPalette, tone.x, tone.y)
        : dyeColour(uPalette, tone.z, tone.w) * 0.55;

    // A node is a mixing chamber: the solution is at rest, so its bands do not
    // travel with the pipe flow, only with its own concentration.
    vec3 shimmer = rotatoryTransmission(flux.x, radius + uTime * 0.05,
                                        uOrdScale, uShimmerGain);
    colour *= mix(vec3(1.0), shimmer, uShimmerAmount * clamp(flux.x, 0.0, 1.0));

    float edge = smoothstep(1.0, 1.0 - fwidth(radius) * 2.0, radius);
    fragColour = vec4(colour, edge);
}
"""


def edge_shader_sources() -> tuple[str, str]:
    substitution = {
        "spectrum_end": repr(float(SPECTRUM_END)),
        "palette": PALETTE_GLSL,
    }
    return EDGE_VERTEX_GLSL, EDGE_FRAGMENT_GLSL % substitution


def node_shader_sources() -> tuple[str, str]:
    substitution = {
        "spectrum_end": repr(float(SPECTRUM_END)),
        "palette": PALETTE_GLSL,
    }
    return NODE_VERTEX_GLSL, NODE_FRAGMENT_GLSL % substitution


def pack_node_manifest(
    field: InfluenceField, order: Sequence[Any]
) -> np.ndarray:
    """Pack per-node attributes into a (1, 2N, 4) float32 data texture."""

    readings = {reading.key: reading for reading in field.table()}
    manifest = np.zeros((1, len(order) * 2, 4), dtype=np.float32)
    for index, key in enumerate(order):
        reading = readings.get(key)
        if reading is None:
            continue
        dynamic = reading.categories.get(DYNAMIC)
        baked = reading.categories.get(BAKED)
        manifest[0, index * 2] = (
            0.0 if dynamic is None else dynamic.hue,
            0.0 if dynamic is None else dynamic.dispersion,
            0.0 if baked is None else baked.hue,
            0.0 if baked is None else baked.dispersion,
        )
        manifest[0, index * 2 + 1] = (
            reading.value, reading.staging, reading.recurrence, 0.0,
        )
    return manifest


def pack_edge_manifest(
    field: InfluenceField,
    order: Sequence[tuple[Any, Any]],
) -> np.ndarray:
    """Pack per-edge attributes into a (1, 2E, 4) float32 data texture.

    An edge takes the heaviest transport that crossed it, so the picture shows
    what actually flowed rather than what merely connects. The phase slot is
    seeded from the edge's own identity: without it every pipe in the diagram
    would band in lockstep, which reads as a global pulse the program does not
    have.
    """

    heaviest: dict[tuple[Any, Any], Any] = {}
    for transport in field.trace():
        pair = (transport.source_key, transport.target_key)
        current = heaviest.get(pair)
        if current is None or transport.weight > current.weight:
            heaviest[pair] = transport
    readings = {reading.key: reading for reading in field.table()}

    manifest = np.zeros((1, len(order) * 2, 4), dtype=np.float32)
    for index, pair in enumerate(order):
        transport = heaviest.get(pair)
        if transport is None:
            continue
        target = readings.get(pair[1])
        staging = 0.0 if target is None else target.staging
        baked = None if target is None else target.categories.get(BAKED)
        manifest[0, index * 2] = (
            transport.hue,
            0.0 if target is None else max(
                0.0, min(1.0, 1.0 - transport.weight)
            ) * 0.35,
            transport.hue if baked is None else baked.hue,
            0.0 if baked is None else baked.dispersion,
        )
        manifest[0, index * 2 + 1] = (
            min(1.0, transport.weight),
            staging,
            float(transport.iteration),
            (hash(str(pair)) % 1000) / 1000.0 * 6.2831853,
        )
    return manifest


def rotatory_transmission(
    concentration: np.ndarray,
    path: np.ndarray,
    *,
    gain: float,
) -> np.ndarray:
    """CPU reference for the shader's ``rotatoryTransmission``.

    Kept identical on purpose: a still rendered through this is a faithful
    preview of the fragment stage, so the optics can be judged before a GL
    context exists. Any divergence between the two would make the preview a
    different program that merely resembles the one being shipped.
    """

    angle = (
        _rotation_scale()[None, :]
        * (gain * concentration * path)[:, None]
    )
    return np.sin(angle) ** 2


__all__ = [
    "WAVELENGTHS_NM", "REFERENCE_NM", "ORD_EXPONENT",
    "SHIMMER_TARGET_BANDS", "SHIMMER_AMOUNT", "shimmer_gain",
    "PALETTE_GLSL",
    "EDGE_VERTEX_GLSL", "EDGE_FRAGMENT_GLSL",
    "NODE_VERTEX_GLSL", "NODE_FRAGMENT_GLSL",
    "edge_shader_sources", "node_shader_sources",
    "pack_node_manifest", "pack_edge_manifest",
    "rotatory_transmission",
]


def reduce_crossings(
    layers: Mapping[int, Sequence[Any]],
    edges: Sequence[tuple[Any, Any]],
    *,
    sweeps: int = 8,
) -> dict[int, list[Any]]:
    """Order each layer by the barycentre of its neighbours, sweeping both ways.

    True planarity is not available -- these graphs are generally non-planar,
    and asking for zero crossings would be asking for something that does not
    exist. Barycentre ordering is the standard layered-drawing answer: it does
    not eliminate crossings, it reduces them enough that the diagram reads as
    routed pipes rather than a hairball, which is the property actually wanted.
    """

    ordered = {level: list(members) for level, members in layers.items()}
    successors: dict[Any, list[Any]] = {}
    predecessors: dict[Any, list[Any]] = {}
    for source, target in edges:
        successors.setdefault(source, []).append(target)
        predecessors.setdefault(target, []).append(source)

    levels = sorted(ordered)
    for sweep in range(sweeps):
        downward = sweep % 2 == 0
        walk = levels if downward else list(reversed(levels))
        for level in walk:
            positions = {
                key: index
                for neighbour in ((level - 1,) if downward else (level + 1,))
                for index, key in enumerate(ordered.get(neighbour, ()))
            }
            relations = predecessors if downward else successors
            current = {key: index for index, key in enumerate(ordered[level])}

            def barycentre(key: Any) -> float:
                related = [
                    positions[item] for item in relations.get(key, ())
                    if item in positions
                ]
                # A node with no neighbour on that side has no opinion, so it
                # keeps its place instead of being dragged to the top.
                return sum(related) / len(related) if related else current[key]

            ordered[level].sort(key=lambda key: (barycentre(key), current[key]))
    return ordered


def count_crossings(
    ordered: Mapping[int, Sequence[Any]],
    edges: Sequence[tuple[Any, Any]],
    levels: Mapping[Any, int],
) -> int:
    """Count pairwise crossings between adjacent layers, for measuring."""

    rank = {
        key: index
        for members in ordered.values()
        for index, key in enumerate(members)
    }
    spans: dict[int, list[tuple[int, int]]] = {}
    for source, target in edges:
        if source not in rank or target not in rank:
            continue
        low = levels[source]
        if levels[target] != low + 1:
            continue
        spans.setdefault(low, []).append((rank[source], rank[target]))
    total = 0
    for pairs in spans.values():
        for index, (a_from, a_to) in enumerate(pairs):
            for b_from, b_to in pairs[index + 1:]:
                if (a_from - b_from) * (a_to - b_to) < 0:
                    total += 1
    return total
