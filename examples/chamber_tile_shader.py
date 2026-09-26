"""The chamber as a grid of tiles, each cycling its own state indicators.

WHAT A TILE IS

One tile per cell of the chamber.  A cell is rarely doing one thing -- it
can be humid, hold cloud, and have rain falling through it, all at once --
so the tile does not pick a winner, and it does not blend them into a
colour that means nothing.  It CYCLES: every indicator present above a
threshold takes the tile in turn, in place, with a crossfade, so watching
one tile for a couple of seconds tells you everything that cell is doing.
A row of pips along the bottom edge shows how many indicators are in the
cycle, so the rotation never hides how much is going on.

WHY THE ICONS ARE PROCEDURAL

Each icon is a signed distance field in tile-local coordinates: a few
lines of maths, no atlas, no image asset, resolution independent, and
animated without a sprite sheet.  Rain falls and wraps, vapour ripples,
heat rises, the crystal turns.  Adding an indicator is one function and
one colour, not an art pipeline.

WHAT THIS MODULE DOES NOT DO

It reads no simulation state and owns no timing.  It is handed two float
textures -- one for the condensable channels, one for temperature and the
surface films -- and draws them.  The caller decides what a channel means
and how it is normalised.
"""

from __future__ import annotations

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER, GL_BLEND, GL_CLAMP_TO_EDGE, GL_DEPTH_TEST, GL_FALSE,
    GL_FLOAT, GL_NEAREST, GL_RGBA, GL_RGBA32F, GL_STATIC_DRAW, GL_TEXTURE0,
    GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TRIANGLE_STRIP,
    glActiveTexture, glBindBuffer, glBindTexture, glBindVertexArray,
    glBufferData, glDisable, glDrawArrays, glEnableVertexAttribArray,
    glGenBuffers, glGenTextures, glGenVertexArrays, glGetUniformLocation,
    glTexImage2D, glTexParameteri, glUniform1f, glUniform1i, glUniform2f,
    glUseProgram, glVertexAttribPointer,
)
from OpenGL.GL import shaders as gl_shaders

_VERTEX_SHADER = 0x8B31
_FRAGMENT_SHADER = 0x8B30


TILE_VERT = """
#version 330 core
layout(location=0) in vec2 pos;
out vec2 v_uv;
void main() {
    v_uv = pos * 0.5 + 0.5;
    gl_Position = vec4(pos, 0.0, 1.0);
}
"""

TILE_FRAG = """
#version 330 core
in vec2 v_uv;
out vec4 frag;

uniform sampler2D uState;   // r=saturation  g=cloud  b=ice  a=rain
uniform sampler2D uAux;     // r=temperature g=film   b=frost a=unused
uniform vec2  uGrid;        // cells across, cells up
uniform float uTime;        // seconds
uniform float uCycle;       // seconds each indicator holds its tile

const float PRESENT = 0.045;   // below this an indicator is not in the cycle

float hash21(vec2 c) {
    return fract(sin(dot(c, vec2(12.9898, 78.233))) * 43758.5453);
}

// -- the indicators, each a coverage in 0..1 over tile-local p ----------
// p is centred on its tile and spans about -0.5 .. 0.5.

float icon_vapour(vec2 p, float t) {
    // ripples drifting sideways: humid air before it is anything else
    float cover = 0.0;
    for (int i = 0; i < 3; i++) {
        float row = (float(i) - 1.0) * 0.17;
        float wave = 0.055 * sin(p.x * 16.0 + t * 2.2 + float(i) * 1.7);
        float d = abs(p.y - row - wave) - 0.021;
        cover = max(cover, 1.0 - smoothstep(0.0, 0.02, d));
    }
    return cover * (1.0 - smoothstep(0.30, 0.42, abs(p.x)));
}

float icon_cloud(vec2 p, float t) {
    // three lobes over a flat base, breathing
    vec2 q = p * (1.0 + 0.04 * sin(t * 1.3)) + vec2(0.0, 0.05);
    float d = length(q - vec2(-0.14, 0.0)) - 0.135;
    d = min(d, length(q - vec2(0.13, -0.01)) - 0.12);
    d = min(d, length(q - vec2(0.0, 0.09)) - 0.16);
    d = min(d, max(-q.y - 0.10, abs(q.x) - 0.27));
    return 1.0 - smoothstep(0.0, 0.025, d);
}

float icon_ice(vec2 p, float t) {
    // a six-spoke crystal, turning slowly
    float a = atan(p.y, p.x) + t * 0.35;
    float r = length(p);
    float arm = r - 0.30 * (0.45 + 0.55 * pow(abs(cos(a * 3.0)), 6.0));
    return 1.0 - smoothstep(0.0, 0.03, min(arm, r - 0.06));
}

float icon_rain(vec2 p, float t) {
    // drops falling through the tile and wrapping
    float cover = 0.0;
    for (int i = 0; i < 3; i++) {
        float lane = (float(i) - 1.0) * 0.20;
        float fall = fract(t * 0.9 + float(i) * 0.37);
        vec2 q = p - vec2(lane, 0.42 - fall * 0.84);
        float body = length(q * vec2(1.0, 0.62)) - 0.055;
        float tip = max(length(q - vec2(0.0, 0.055)) - 0.03, -q.y + 0.03);
        cover = max(cover, 1.0 - smoothstep(0.0, 0.02, min(body, tip)));
    }
    return cover;
}

float icon_heat(vec2 p, float t) {
    // warmth rising: chevrons climbing out of the tile
    float cover = 0.0;
    for (int i = 0; i < 3; i++) {
        float rise = fract(t * 0.55 + float(i) * 0.33);
        float chev = abs(abs(p.x) * 0.9 - (p.y - (-0.38 + rise * 0.76))) - 0.028;
        float span = 1.0 - smoothstep(0.20, 0.30, abs(p.x));
        cover = max(cover, (1.0 - smoothstep(0.0, 0.022, chev)) * span
                           * sin(rise * 3.14159));
    }
    return cover;
}

float icon_frost(vec2 p, float t) {
    // a crusted edge: what a surface tile wears
    float cover = 0.0;
    for (int i = 0; i < 5; i++) {
        float x = (float(i) - 2.0) * 0.16;
        float h = 0.10 + 0.07 * hash21(vec2(x, 3.0));
        float d = max(abs(p.x - x) - 0.052, abs(p.y + 0.34) - h);
        cover = max(cover, 1.0 - smoothstep(0.0, 0.02, d));
    }
    return cover;
}

float icon_of(int which, vec2 p, float t) {
    if (which == 0) return icon_vapour(p, t);
    if (which == 1) return icon_cloud(p, t);
    if (which == 2) return icon_ice(p, t);
    if (which == 3) return icon_rain(p, t);
    if (which == 4) return icon_heat(p, t);
    return icon_frost(p, t);
}

vec3 colour_of(int which) {
    if (which == 0) return vec3(0.46, 0.82, 0.62);   // saturation
    if (which == 1) return vec3(0.96, 0.97, 1.00);   // cloud
    if (which == 2) return vec3(0.67, 0.88, 1.00);   // ice
    if (which == 3) return vec3(0.46, 0.66, 0.98);   // rain
    if (which == 4) return vec3(1.00, 0.66, 0.34);   // heat
    return vec3(0.88, 0.94, 1.00);                   // frost
}

void main() {
    vec2 g = v_uv * uGrid;
    vec2 cell = floor(g);
    vec2 p = fract(g) - 0.5;
    vec2 st = (cell + 0.5) / uGrid;

    vec4 s = texture(uState, st);
    vec4 aux = texture(uAux, st);

    float weight[6];
    weight[0] = s.r;
    weight[1] = s.g;
    weight[2] = s.b;
    weight[3] = s.a;
    weight[4] = aux.r;
    weight[5] = max(aux.g, aux.b);

    int active[6];
    int count = 0;
    for (int i = 0; i < 6; i++) {
        if (weight[i] > PRESENT) { active[count] = i; count++; }
    }

    // the tile itself: cool slate warming with temperature, with a seam
    vec3 col = mix(vec3(0.055, 0.075, 0.115), vec3(0.24, 0.13, 0.09),
                   clamp(aux.r, 0.0, 1.0));
    col = mix(col, vec3(0.16, 0.19, 0.25),
              smoothstep(0.455, 0.5, max(abs(p.x), abs(p.y))));

    if (count > 0) {
        // Offsetting the phase by a hash of the cell keeps neighbours from
        // pulsing in lockstep, which would read as a screen-wide flicker
        // instead of each cell having its own story.
        float phase = uTime / max(uCycle, 0.05) + hash21(cell) * 4.0;
        float held = floor(phase);
        float fade = smoothstep(0.78, 1.0, fract(phase));

        int k0 = active[int(mod(held, float(count)))];
        int k1 = active[int(mod(held + 1.0, float(count)))];

        float a0 = icon_of(k0, p, uTime) * (1.0 - fade)
                   * clamp(weight[k0] * 2.2, 0.25, 1.0);
        float a1 = icon_of(k1, p, uTime) * fade
                   * clamp(weight[k1] * 2.2, 0.25, 1.0);

        col = mix(col, colour_of(k0), clamp(a0, 0.0, 1.0));
        col = mix(col, colour_of(k1), clamp(a1, 0.0, 1.0));

        // one pip per indicator in the cycle, along the bottom edge
        float slot = (p.x + 0.5) * 6.0;
        int which = int(floor(slot));
        if (p.y < -0.44 && which < count) {
            if (abs(fract(slot) - 0.5) - 0.16 < 0.0) {
                col = mix(col, colour_of(active[which]), 0.75);
            }
        }
    }

    frag = vec4(col, 1.0);
}
"""


class TileField:
    """One float RGBA texel per cell, re-uploaded each frame."""

    def __init__(self, nx: int, nz: int, unit: int):
        self.nx, self.nz, self.unit = int(nx), int(nz), int(unit)
        self.tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        for name in (GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER):
            glTexParameteri(GL_TEXTURE_2D, name, GL_NEAREST)
        for name in (GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T):
            glTexParameteri(GL_TEXTURE_2D, name, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, self.nx, self.nz, 0,
                     GL_RGBA, GL_FLOAT, None)

    def upload(self, rgba_zx4: np.ndarray) -> None:
        data = np.ascontiguousarray(rgba_zx4, dtype=np.float32)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, self.nx, self.nz, 0,
                     GL_RGBA, GL_FLOAT, data)

    def bind(self) -> None:
        glActiveTexture(GL_TEXTURE0 + self.unit)
        glBindTexture(GL_TEXTURE_2D, self.tex)


class TileGrid:
    """The whole grid, drawn as one quad by the tile shader."""

    def __init__(self, nx: int, nz: int, cycle_s: float = 1.6):
        self.nx, self.nz = int(nx), int(nz)
        self.cycle_s = float(cycle_s)
        self.program = gl_shaders.compileProgram(
            gl_shaders.compileShader(TILE_VERT, _VERTEX_SHADER),
            gl_shaders.compileShader(TILE_FRAG, _FRAGMENT_SHADER),
        )
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, None)
        glBindVertexArray(0)
        self.state = TileField(self.nx, self.nz, 0)
        self.aux = TileField(self.nx, self.nz, 1)

    def draw(self, t: float) -> None:
        glUseProgram(self.program)
        self.state.bind()
        glUniform1i(glGetUniformLocation(self.program, "uState"), 0)
        self.aux.bind()
        glUniform1i(glGetUniformLocation(self.program, "uAux"), 1)
        glUniform2f(glGetUniformLocation(self.program, "uGrid"),
                    float(self.nx), float(self.nz))
        glUniform1f(glGetUniformLocation(self.program, "uTime"), float(t))
        glUniform1f(glGetUniformLocation(self.program, "uCycle"), self.cycle_s)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)


class Autoscale:
    """Rises at once, falls slowly.

    Cloud water is ~1e-4 kg/m3 and rain an order below it, so a fixed range
    shows either a black screen or a white one.  The decay stops one burst
    from flattening everything that follows it."""

    def __init__(self, floor: float = 1e-12, decay: float = 0.985):
        self.peak, self.floor, self.decay = floor, floor, decay

    def __call__(self, values: np.ndarray) -> np.ndarray:
        high = float(np.max(values)) if values.size else 0.0
        self.peak = max(high, self.peak * self.decay, self.floor)
        return np.clip(values / self.peak, 0.0, 1.0)


def plane(flat, nx: int, ny: int, nz: int) -> np.ndarray:
    """A per-cell column as the (nz, nx) plane the tiles draw.

    The flat index is ``x + nx * (y + ny * z)``, so the reshape is
    ``[z][y][x]`` and the depth axis averages away -- which is what the slab
    means: the front and back are walls, and what lies between them is one
    room seen side on."""
    a = np.asarray(flat, dtype=np.float64).reshape(-1)
    if a.size != nx * ny * nz:
        return np.zeros((nz, nx), dtype=np.float64)
    return a.reshape(nz, ny, nx).mean(axis=1)
