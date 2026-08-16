"""Dye solver and its optics as compute and fragment shaders, GLSL 460.

``InfluenceFlow`` stays the reference. This runs the same arithmetic on the
GPU and ``max_state_difference`` compares the two after identical step counts,
so agreement is measured rather than asserted.

Compute over storage buffers
----------------------------
Three compute dispatches over flat SSBOs. An invocation is indexed directly by
the thing it computes, so no texel arithmetic sits between the code and the
model, and state lives in buffers shaped the way the reference shapes arrays.

Arrivals are gathered rather than scattered, and that is a choice, not a
limitation carried over from anywhere. ``atomicAdd`` on floats is available
here, but float addition is not associative and the racy summation order would
change the answer run to run. The reference is exact and this is checked
against it, so determinism is worth more than the convenience of scatter.

Junctions are volumes
---------------------
A bead does not sit on top of its tubes; the tube ends melt into it. Its
surface is a smooth-minimum union of a sphere with every capsule that meets
there, evaluated per fragment, so the joint is a single continuous body.

Its colour is the fluid genuinely present at that junction -- the tail cells of
arriving pipes and the head cells of leaving ones, mixed by adding power sums.
An exponential average of the per-step flux, which is what a first attempt used,
shows almost no dye: the flux is a small spiky quantity and averaging it
smears a passing drop away to nothing. Reading the adjacent fluid instead means
a drop reaching a junction actually colours it.

Transparency
------------
Per-pixel linked lists, sorted and composited in the resolve shader. Every
translucent fragment -- tube and bead alike -- appends itself to its pixel's
list; the resolve walks it, sorts by depth, and composites back to front. That
is exact, and it is what lets liquid behind show through liquid in front.
Weighted blending would average by alpha and wash out precisely where many
translucent tubes overlap, which is the case that matters here.
"""

from __future__ import annotations

import ctypes

import numpy as np
from OpenGL.GL import *  # noqa: F403 -- GL name surface is intentionally flat

from ..compiler.influence_field import BACK_EDGE_ROLES, RECURRENT
from .influence_flow import MOMENTS, InfluenceFlow

GLSL_VERSION = "#version 460 core\n"

B_PIPE_IN, B_PIPE_OUT, B_ARRIVALS = 0, 1, 2
B_EDGES, B_CSR_OFFSET, B_CSR_ENTRY, B_EMITTER = 3, 4, 5, 6
B_FRAGMENTS, B_INCIDENT_OFFSET, B_INCIDENT_ENTRY, B_GEOMETRY = 7, 8, 9, 10

MAX_CATEGORIES = 4

SOLVER_STORAGE = """
struct EdgeRecord { uint source; uint target; float factor; uint isBack; };
struct CsrEntry   { uint edge; float factor; uint isBack; uint pad; };
struct Emitter    { float hue; float phase; float duty; float pad; };

layout(std430, binding = 0) readonly buffer PipeIn  { float pipeIn[]; };
layout(std430, binding = 1) writeonly buffer PipeOut { float pipeOut[]; };
layout(std430, binding = 2) buffer Arrivals { float arrivals[]; };
layout(std430, binding = 3) readonly buffer Edges { EdgeRecord edges[]; };
layout(std430, binding = 4) readonly buffer CsrOffset { uint csrOffset[]; };
layout(std430, binding = 5) readonly buffer CsrEntries { CsrEntry csrEntry[]; };
layout(std430, binding = 6) readonly buffer Emitters { Emitter emitters[]; };

uniform uint uNodes;
uniform uint uEdges;
uniform uint uCells;
uniform uint uCategories;
uniform int  uRecurrent;
uniform float uShift;
uniform float uTime;
uniform float uPeriod;
uniform float uDt;

uint pipeIndex(uint edge, uint category, uint cell) {
    return (((edge * uCategories) + category) * uCells + cell) * 3u;
}

vec3 readPipe(uint edge, uint category, uint cell) {
    uint base = pipeIndex(edge, category, cell);
    return vec3(pipeIn[base], pipeIn[base + 1u], pipeIn[base + 2u]);
}
"""

ARRIVALS_COMPUTE = GLSL_VERSION + "layout(local_size_x = 64) in;\n" + \
    SOLVER_STORAGE + """
void main() {
    uint slot = gl_GlobalInvocationID.x;
    if (slot >= uNodes * uCategories) return;
    uint node = slot / uCategories;
    uint category = slot - node * uCategories;

    vec3 total = vec3(0.0);
    for (uint i = csrOffset[node]; i < csrOffset[node + 1u]; ++i) {
        CsrEntry entry = csrEntry[i];
        vec3 outflow = vec3(0.0);
        if (entry.isBack == 1u) {
            // Influence that crossed a back edge is loop-carried from here on:
            // the recurrent slot absorbs every incoming category and the rest
            // take nothing from this edge.
            if (uRecurrent < 0 || category != uint(uRecurrent)) continue;
            for (uint c = 0u; c < uCategories; ++c) {
                outflow += readPipe(entry.edge, c, uCells - 1u);
            }
        } else {
            outflow = readPipe(entry.edge, category, uCells - 1u);
        }
        total += outflow * uShift * entry.factor;
    }

    Emitter emitter = emitters[slot];
    if (emitter.duty > 0.0) {
        float position = fract(uTime / uPeriod + emitter.phase);
        if (position < emitter.duty) {
            // Raised-cosine valve. A hard gate is a discontinuity, and the
            // least difference between this clock and the reference's would
            // put them either side of it, differing by a whole quantum.
            float envelope =
                (1.0 - cos(6.28318530718 * position / emitter.duty))
                / emitter.duty;
            float amount = envelope * uDt / uPeriod;
            total += vec3(amount, amount * emitter.hue,
                          amount * emitter.hue * emitter.hue);
        }
    }

    arrivals[slot * 3u]      = total.x;
    arrivals[slot * 3u + 1u] = total.y;
    arrivals[slot * 3u + 2u] = total.z;
}
"""

ADVECT_COMPUTE = GLSL_VERSION + "layout(local_size_x = 64) in;\n" + \
    SOLVER_STORAGE + """
void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= uEdges * uCategories * uCells) return;

    uint cell = gid % uCells;
    uint row = gid / uCells;
    uint category = row % uCategories;
    uint edge = row / uCategories;

    vec3 next = readPipe(edge, category, cell) * (1.0 - uShift);
    if (cell == 0u) {
        // Arrivals enter undiluted: advection already removed exactly the
        // outflow, so scaling the injection would leak mass every hop.
        uint slot = (edges[edge].source * uCategories + category) * 3u;
        next += vec3(arrivals[slot], arrivals[slot + 1u], arrivals[slot + 2u]);
    } else {
        next += readPipe(edge, category, cell - 1u) * uShift;
    }

    uint base = pipeIndex(edge, category, cell);
    pipeOut[base]      = next.x;
    pipeOut[base + 1u] = next.y;
    pipeOut[base + 2u] = next.z;
}
"""

# ------------------------------------------------------------------ drawing

SCENE_STORAGE = """
struct Capsule { vec2 from; vec2 to; float radius; uint edge; };

layout(std430, binding = 0) readonly buffer PipeState { float pipeState[]; };
layout(std430, binding = 8) readonly buffer IncidentOffset { uint incidentOffset[]; };
layout(std430, binding = 9) readonly buffer IncidentEntry { uvec2 incidentEntry[]; };
layout(std430, binding = 10) readonly buffer Geometry { Capsule capsules[]; };

uniform uint uCells;
uniform int uCategories;

vec3 cellMoments(uint edge, int category, uint cell) {
    uint base = (((edge * uint(uCategories)) + uint(category)) * uCells + cell) * 3u;
    return vec3(pipeState[base], pipeState[base + 1u], pipeState[base + 2u]);
}
"""

DYE_GLSL = """
#define SPECTRUM_END 0.75
#define MAX_DISPERSION 0.375

uniform sampler2D uPalette;
uniform vec3 uWaterColour;
uniform float uDyeStrength;
uniform float uPeakWeight;
uniform int uBakedCategory;

// Collapse power sums to colour exactly as the CPU readout does. Nothing here
// is a stored colour; it is solver state, read live.
vec3 collapseDye(vec3 sums[4], out float density) {
    float weight = 0.0, baked = 0.0, meanSum = 0.0, varSum = 0.0;
    for (int c = 0; c < uCategories; ++c) {
        vec3 m = sums[c];
        if (m.x <= 0.0) continue;
        float mean = m.y / m.x;
        weight += m.x;
        meanSum += mean * m.x;
        varSum += max(0.0, m.z / m.x - mean * mean) * m.x;
        if (c == uBakedCategory) baked += m.x;
    }
    float hue = weight > 0.0 ? meanSum / weight : 0.0;
    float dispersion = weight > 0.0
        ? min(1.0, sqrt(varSum / weight) / MAX_DISPERSION) : 0.0;
    float staging = weight > 0.0 ? baked / weight : 0.0;

    // Cube root of the concentration ratio: the luminance-to-lightness law
    // CIE L* uses, so equal steps on screen are equal perceptual steps in
    // concentration. Cell weights run a median near a tenth of the peak, where
    // a linear ratio reads as black and log1p is barely different from linear.
    density = clamp(pow(weight / max(uPeakWeight, 1e-6), 1.0 / 3.0), 0.0, 1.0);

    vec3 dye = texture(uPalette, vec2(clamp(hue / SPECTRUM_END, 0.0, 1.0),
                                      clamp(dispersion, 0.0, 1.0))).rgb;
    dye = mix(dye, dye * 0.55, clamp(staging, 0.0, 1.0));
    // Beer-Lambert: a faint trace tints the carrier, a dense slug goes fully
    // to dye colour and never clips past it.
    return mix(uWaterColour, dye, 1.0 - exp(-uDyeStrength * density));
}
"""

OIT_APPEND_GLSL = """
struct ListFragment { vec4 colour; float depth; uint next; uvec2 pad; };

layout(std430, binding = 7) buffer FragmentPool { ListFragment pool[]; };
layout(binding = 0, r32ui) uniform coherent uimage2D uHeads;
layout(binding = 0, offset = 0) uniform atomic_uint uAllocator;
uniform uint uMaxFragments;

void appendFragment(vec3 colour, float alpha, float depth) {
    uint index = atomicCounterIncrement(uAllocator);
    if (index >= uMaxFragments) return;
    uint previous = imageAtomicExchange(uHeads, ivec2(gl_FragCoord.xy), index);
    pool[index].colour = vec4(colour * alpha, alpha);
    pool[index].depth = depth;
    pool[index].next = previous;
}
"""

PIPE_VERTEX = GLSL_VERSION + """
layout(location=0) in vec2 aCorner;
layout(location=1) in vec2 aFrom;
layout(location=2) in vec2 aTo;
layout(location=3) in float aEdgeIndex;

uniform mat4 uMVP;
uniform float uPipeWidth;

out vec2 vPipe;
flat out uint vEdge;

void main() {
    vec2 axis = aTo - aFrom;
    float span = max(length(axis), 1e-6);
    vec2 forward = axis / span;
    vec2 side = vec2(-forward.y, forward.x);
    vec2 world = aFrom + forward * (span * aCorner.x)
               + side * (uPipeWidth * aCorner.y);
    vPipe = aCorner;
    vEdge = uint(aEdgeIndex + 0.5);
    gl_Position = uMVP * vec4(world, 0.0, 1.0);
}
"""

PIPE_FRAGMENT = GLSL_VERSION + SCENE_STORAGE + DYE_GLSL + OIT_APPEND_GLSL + """
in vec2 vPipe;
flat in uint vEdge;
uniform float uPipeDepth;

void main() {
    float across = clamp(abs(vPipe.y), 0.0, 1.0);
    float profile = sqrt(max(0.0, 1.0 - across * across));
    if (profile <= 0.02) discard;

    uint cell = uint(clamp(vPipe.x * float(uCells), 0.0, float(uCells) - 1.0));
    vec3 sums[4];
    for (int c = 0; c < MAX_CATEGORIES_C; ++c) sums[c] = vec3(0.0);
    for (int c = 0; c < uCategories; ++c) sums[c] = cellMoments(vEdge, c, cell);

    float density;
    vec3 body = collapseDye(sums, density);

    // Tube shading: the round section darkens toward the walls and a narrow
    // highlight sits off-centre. Geometry only; carries no measurement.
    float shade = 0.44 + 0.56 * profile;
    float highlight = pow(max(0.0, 1.0 - abs(vPipe.y + 0.42) * 2.6), 8.0);
    vec3 colour = body * shade + vec3(highlight * 0.30);

    // Translucent, and more so where the tube is thin, so liquid behind shows
    // through liquid in front rather than being occluded by it.
    float alpha = clamp(0.20 + 0.34 * profile, 0.0, 1.0);
    appendFragment(colour, alpha, uPipeDepth + 0.0005 * float(vEdge));
}
""".replace("MAX_CATEGORIES_C", str(MAX_CATEGORIES))

BULB_VERTEX = GLSL_VERSION + """
layout(location=0) in vec2 aCorner;
layout(location=1) in vec2 aCentre;
layout(location=2) in float aRadius;
layout(location=3) in float aNodeIndex;

uniform mat4 uMVP;

out vec2 vLocal;      // world-space offset from the bead centre
out vec2 vWorld;
flat out uint vNode;
flat out float vRadius;

void main() {
    vLocal = aCorner * aRadius;
    vWorld = aCentre + vLocal;
    vNode = uint(aNodeIndex + 0.5);
    vRadius = aRadius;
    gl_Position = uMVP * vec4(vWorld, 0.0, 1.0);
}
"""

BULB_FRAGMENT = GLSL_VERSION + SCENE_STORAGE + DYE_GLSL + OIT_APPEND_GLSL + """
in vec2 vLocal;
in vec2 vWorld;
flat in uint vNode;
flat in float vRadius;

uniform float uBeadRadius;
uniform float uBlend;      // smooth-union radius, in world units
uniform float uBulbDepth;

float capsuleDistance(vec2 point, vec2 a, vec2 b, float radius) {
    vec2 pa = point - a, ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * h) - radius;
}

// Polynomial smooth minimum. This is what makes the joint one body instead of
// a sphere parked on top of some tubes.
float smoothUnion(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / max(k, 1e-6), 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

void main() {
    // Surface: a sphere at the junction, smoothly unioned with every tube that
    // meets it, so the tube ends blend outward into a single volume.
    float field = length(vLocal) - uBeadRadius;
    for (uint i = incidentOffset[vNode]; i < incidentOffset[vNode + 1u]; ++i) {
        Capsule capsule = capsules[incidentEntry[i].x];
        field = smoothUnion(
            field,
            capsuleDistance(vWorld, capsule.from, capsule.to, capsule.radius),
            uBlend
        );
    }
    if (field > 0.0) discard;

    // Thickness through the body at this pixel, which is what gives the bead
    // volume and makes its centre read as deeper liquid than its rim.
    float thickness = clamp(-field / max(uBeadRadius, 1e-6), 0.0, 1.0);
    float depth = sqrt(max(0.0, thickness));

    // The fluid actually present at the junction: the near cell of every tube
    // that meets here. Power sums add, so this mixes by the field's own merge
    // operator rather than by a blend chosen to look right.
    vec3 sums[4];
    for (int c = 0; c < MAX_CATEGORIES_C; ++c) sums[c] = vec3(0.0);
    for (uint i = incidentOffset[vNode]; i < incidentOffset[vNode + 1u]; ++i) {
        uvec2 entry = incidentEntry[i];
        uint edge = capsules[entry.x].edge;
        uint cell = entry.y == 0u ? 0u : uCells - 1u;
        for (int c = 0; c < uCategories; ++c) {
            sums[c] += cellMoments(edge, c, cell);
        }
    }

    float density;
    vec3 body = collapseDye(sums, density);

    // Glass bead: lambert off a fixed key, a tight specular, and a fresnel rim
    // that brightens at the silhouette where a real bead is thickest.
    vec3 normal = normalize(vec3(vLocal / max(vRadius, 1e-6), depth + 0.35));
    vec3 key = normalize(vec3(-0.42, -0.55, 0.72));
    float lambert = 0.44 + 0.56 * max(0.0, dot(normal, key));
    float specular = pow(max(0.0, dot(normal, key)), 48.0) * 0.45;
    float rim = pow(1.0 - depth, 3.0) * 0.22;
    vec3 colour = body * lambert + vec3(specular + rim);

    float alpha = clamp(0.22 + 0.40 * depth, 0.0, 1.0);
    appendFragment(colour, alpha, uBulbDepth);
}
""".replace("MAX_CATEGORIES_C", str(MAX_CATEGORIES))

RESOLVE_VERTEX = GLSL_VERSION + """
layout(location=0) in vec2 aPos;
void main() { gl_Position = vec4(aPos, 0.0, 1.0); }
"""

RESOLVE_FRAGMENT = GLSL_VERSION + """
struct ListFragment { vec4 colour; float depth; uint next; uvec2 pad; };

layout(std430, binding = 7) readonly buffer FragmentPool { ListFragment pool[]; };
layout(binding = 0, r32ui) uniform readonly uimage2D uHeads;
uniform vec3 uBackground;

out vec4 fragColour;
#define MAX_LAYERS 24

void main() {
    uint index = imageLoad(uHeads, ivec2(gl_FragCoord.xy)).r;
    if (index == 0xFFFFFFFFu) { fragColour = vec4(uBackground, 1.0); return; }

    vec4 colours[MAX_LAYERS];
    float depths[MAX_LAYERS];
    int count = 0;
    while (index != 0xFFFFFFFFu && count < MAX_LAYERS) {
        colours[count] = pool[index].colour;
        depths[count] = pool[index].depth;
        index = pool[index].next;
        ++count;
    }

    // Insertion sort, far to near. This is the whole point of the list: an
    // exact depth order, rather than an alpha-weighted average that washes out
    // precisely where many translucent tubes overlap.
    for (int i = 1; i < count; ++i) {
        vec4 colour = colours[i];
        float depth = depths[i];
        int j = i - 1;
        while (j >= 0 && depths[j] < depth) {
            colours[j + 1] = colours[j];
            depths[j + 1] = depths[j];
            --j;
        }
        colours[j + 1] = colour;
        depths[j + 1] = depth;
    }

    vec3 accumulated = uBackground;
    for (int i = 0; i < count; ++i) {
        accumulated = colours[i].rgb + accumulated * (1.0 - colours[i].a);
    }
    fragColour = vec4(accumulated, 1.0);
}
"""


def _compile(source: str, stage: int) -> int:
    shader = glCreateShader(stage)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        log = glGetShaderInfoLog(shader)
        text = log.decode() if isinstance(log, bytes) else str(log)
        numbered = "\n".join(
            f"{n + 1:4d}| {line}" for n, line in enumerate(source.splitlines())
        )
        raise RuntimeError(text + "\n--- source ---\n" + numbered)
    return shader


def _link(*shaders: int) -> int:
    program = glCreateProgram()
    for shader in shaders:
        glAttachShader(program, shader)
        glDeleteShader(shader)
    glLinkProgram(program)
    if not glGetProgramiv(program, GL_LINK_STATUS):
        log = glGetProgramInfoLog(program)
        raise RuntimeError(log.decode() if isinstance(log, bytes) else str(log))
    return program


def compute_program(source: str) -> int:
    return _link(_compile(source, GL_COMPUTE_SHADER))


def draw_program(vertex: str, fragment: str) -> int:
    return _link(
        _compile(vertex, GL_VERTEX_SHADER),
        _compile(fragment, GL_FRAGMENT_SHADER),
    )


def storage_buffer(binding: int, data) -> int:
    buffer = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer)
    if isinstance(data, int):
        glBufferData(GL_SHADER_STORAGE_BUFFER, data, None, GL_DYNAMIC_COPY)
    else:
        payload = np.ascontiguousarray(data)
        glBufferData(
            GL_SHADER_STORAGE_BUFFER, payload.nbytes, payload, GL_DYNAMIC_COPY
        )
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, buffer)
    return buffer


class GPUInfluenceFlow:
    """Compute-shader dye transport, stepping the reference solver's arithmetic."""

    def __init__(self, reference: InfluenceFlow) -> None:
        self.reference = reference
        self.categories = len(reference.categories)
        self.edges = len(reference.edges)
        self.nodes = len(reference.nodes)
        self.cells = reference.cell_count
        self.time = 0.0
        if self.categories > MAX_CATEGORIES:
            raise RuntimeError(
                f"the dye shaders carry at most {MAX_CATEGORIES} categories"
            )
        self.recurrent = (
            reference.categories.index(RECURRENT)
            if RECURRENT in reference.categories else -1
        )

        edge_records = np.zeros(self.edges, dtype=np.dtype([
            ("source", np.uint32), ("target", np.uint32),
            ("factor", np.float32), ("isBack", np.uint32),
        ]))
        incoming: dict[int, list[tuple[int, float, int]]] = {}
        for index, (source, target, role) in enumerate(reference.edges):
            back = 1 if role in BACK_EDGE_ROLES else 0
            factor = float(reference.edge_factor[index])
            edge_records[index] = (
                reference._node_index[source], reference._node_index[target],
                factor, back,
            )
            incoming.setdefault(reference._node_index[target], []).append(
                (index, factor, back)
            )

        offsets = np.zeros(self.nodes + 1, dtype=np.uint32)
        entries: list[tuple[int, float, int]] = []
        for node in range(self.nodes):
            offsets[node] = len(entries)
            entries.extend(incoming.get(node, ()))
        offsets[self.nodes] = len(entries)
        csr = np.zeros(max(1, len(entries)), dtype=np.dtype([
            ("edge", np.uint32), ("factor", np.float32),
            ("isBack", np.uint32), ("pad", np.uint32),
        ]))
        for slot, (edge, factor, back) in enumerate(entries):
            csr[slot] = (edge, factor, back, 0)

        settings = reference.settings
        emitters = np.zeros(self.nodes * self.categories, dtype=np.dtype([
            ("hue", np.float32), ("phase", np.float32),
            ("duty", np.float32), ("pad", np.float32),
        ]))
        for node_index, category, hue, phase in reference.emitters:
            emitters[node_index * self.categories + category] = (
                hue, phase, settings.emission_duty, 0.0
            )

        pipe_floats = self.edges * self.categories * self.cells * MOMENTS
        self.pipe_buffers = [
            storage_buffer(B_PIPE_IN, np.zeros(pipe_floats, dtype=np.float32)),
            storage_buffer(B_PIPE_OUT, np.zeros(pipe_floats, dtype=np.float32)),
        ]
        self.arrivals_buffer = storage_buffer(
            B_ARRIVALS,
            np.zeros(self.nodes * self.categories * MOMENTS, dtype=np.float32),
        )
        storage_buffer(B_EDGES, edge_records)
        storage_buffer(B_CSR_OFFSET, offsets)
        storage_buffer(B_CSR_ENTRY, csr)
        storage_buffer(B_EMITTER, emitters)

        self.arrivals_program = compute_program(ARRIVALS_COMPUTE)
        self.advect_program = compute_program(ADVECT_COMPUTE)
        self._uniforms = {
            program: {
                name: glGetUniformLocation(program, name)
                for name in (
                    "uNodes", "uEdges", "uCells", "uCategories", "uRecurrent",
                    "uShift", "uTime", "uPeriod", "uDt",
                )
            }
            for program in (self.arrivals_program, self.advect_program)
        }

    @property
    def state_buffer(self) -> int:
        return self.pipe_buffers[0]

    def _set_common(self, program: int, shift: float, dt: float) -> None:
        slots = self._uniforms[program]
        glUniform1ui(slots["uNodes"], self.nodes)
        glUniform1ui(slots["uEdges"], self.edges)
        glUniform1ui(slots["uCells"], self.cells)
        glUniform1ui(slots["uCategories"], self.categories)
        glUniform1i(slots["uRecurrent"], self.recurrent)
        glUniform1f(slots["uShift"], shift)
        glUniform1f(slots["uTime"], self.time)
        glUniform1f(slots["uPeriod"], self.reference.settings.emission_period)
        glUniform1f(slots["uDt"], dt)

    def step(self, dt: float) -> None:
        shift = min(1.0, self.reference.settings.flow_speed * dt)

        def groups(count: int) -> int:
            return (int(count) + 63) // 64

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, B_PIPE_IN, self.pipe_buffers[0])
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, B_PIPE_OUT, self.pipe_buffers[1])

        glUseProgram(self.arrivals_program)
        self._set_common(self.arrivals_program, shift, dt)
        glDispatchCompute(groups(self.nodes * self.categories), 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        glUseProgram(self.advect_program)
        self._set_common(self.advect_program, shift, dt)
        glDispatchCompute(groups(self.edges * self.categories * self.cells), 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        self.pipe_buffers.reverse()
        self.time += dt

    def read_state(self) -> np.ndarray:
        count = self.edges * self.categories * self.cells * MOMENTS
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.pipe_buffers[0])
        raw = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, count * 4)
        flat = np.frombuffer(raw, dtype=np.float32, count=count)
        shaped = flat.reshape(self.edges, self.categories, self.cells, MOMENTS)
        return np.transpose(shaped, (0, 2, 1, 3))

    def max_state_difference(self) -> tuple[float, float]:
        gpu = self.read_state().astype(np.float64)
        cpu = self.reference.pipes
        gap = np.abs(gpu - cpu)
        scale = max(1e-12, float(np.abs(cpu).max()))
        return float(gap.max()), float(gap.max() / scale)


class TransparencyLists:
    """Per-pixel linked lists: append while drawing, sort and composite on resolve."""

    def __init__(self, width: int, height: int, layers: int = 12) -> None:
        self.width, self.height = width, height
        self.capacity = width * height * layers
        # 8 words per record: vec4 colour, float depth, uint next, uvec2 pad.
        self.fragment_buffer = storage_buffer(B_FRAGMENTS, self.capacity * 8 * 4)

        self.heads = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.heads)
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, width, height)
        self.clear_pattern = np.full(width * height, 0xFFFFFFFF, dtype=np.uint32)

        self.allocator = glGenBuffers(1)
        glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, self.allocator)
        glBufferData(GL_ATOMIC_COUNTER_BUFFER, 4,
                     np.zeros(1, dtype=np.uint32), GL_DYNAMIC_DRAW)

        self.resolve_program = draw_program(RESOLVE_VERTEX, RESOLVE_FRAGMENT)
        self.background_slot = glGetUniformLocation(
            self.resolve_program, "uBackground"
        )
        quad = np.asarray([-1, -1, 3, -1, -1, 3], dtype=np.float32)
        self.vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glBindVertexArray(0)

    def begin(self) -> None:
        glBindTexture(GL_TEXTURE_2D, self.heads)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height,
                        GL_RED_INTEGER, GL_UNSIGNED_INT, self.clear_pattern)
        glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, self.allocator)
        glBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, 4,
                        np.zeros(1, dtype=np.uint32))
        glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, self.allocator)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, B_FRAGMENTS,
                         self.fragment_buffer)
        glBindImageTexture(0, self.heads, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI)
        # Appending is the only output of the geometry pass, so no colour is
        # written and blending and depth testing are irrelevant to it.
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)

    def resolve(self, background: tuple[float, float, float]) -> None:
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)
        glMemoryBarrier(
            GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT
        )
        glUseProgram(self.resolve_program)
        glBindImageTexture(0, self.heads, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI)
        glUniform3f(self.background_slot, *background)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 3)
        glBindVertexArray(0)

    def used_fragments(self) -> int:
        glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, self.allocator)
        raw = glGetBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, 4)
        return int(np.frombuffer(raw, dtype=np.uint32)[0])


__all__ = [
    "GLSL_VERSION", "MAX_CATEGORIES",
    "ARRIVALS_COMPUTE", "ADVECT_COMPUTE",
    "PIPE_VERTEX", "PIPE_FRAGMENT", "BULB_VERTEX", "BULB_FRAGMENT",
    "RESOLVE_VERTEX", "RESOLVE_FRAGMENT",
    "compute_program", "draw_program", "storage_buffer",
    "GPUInfluenceFlow", "TransparencyLists",
    "B_PIPE_IN", "B_PIPE_OUT", "B_ARRIVALS", "B_FRAGMENTS",
    "B_INCIDENT_OFFSET", "B_INCIDENT_ENTRY", "B_GEOMETRY",
]
