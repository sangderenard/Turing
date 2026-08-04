"""Single-Python-function Elastic DOM page compiled by the normal bundler.

The ingested source owns event response, persistent state transition,
BoundSpring motion, managed-time admission/rollback, and render attributes.
The browser liaison only measures the DOM, forwards input, and presents the
compiled state with the one shader carried in ``TURING_PAGE``.
"""

from __future__ import annotations

import argparse
from pathlib import Path


PRESENTATION_SHADER = r'''#version 300 es
precision highp float;
precision highp sampler2D;

uniform sampler2D turing_dom_state;
uniform int turing_dom_count;
uniform vec2 turing_resolution;
uniform float turing_time;
layout(location = 0) out vec4 turing_output_0;

bool rayBox(
    vec3 rayOrigin, vec3 rayDirection, vec3 center, vec3 halfExtent,
    out float hitDistance, out vec3 hitNormal
) {
    vec3 inverseDirection = 1.0 / rayDirection;
    vec3 first = (center - halfExtent - rayOrigin) * inverseDirection;
    vec3 second = (center + halfExtent - rayOrigin) * inverseDirection;
    vec3 nearAxis = min(first, second);
    vec3 farAxis = max(first, second);
    float nearDistance = max(max(nearAxis.x, nearAxis.y), nearAxis.z);
    float farDistance = min(min(farAxis.x, farAxis.y), farAxis.z);
    if (farDistance < max(nearDistance, 0.0)) return false;
    hitDistance = nearDistance > 0.0 ? nearDistance : farDistance;
    vec3 hit = rayOrigin + rayDirection * hitDistance - center;
    vec3 face = abs(abs(hit) - halfExtent);
    hitNormal = face.x < face.y && face.x < face.z
        ? vec3(sign(hit.x), 0.0, 0.0)
        : (face.y < face.z
            ? vec3(0.0, sign(hit.y), 0.0)
            : vec3(0.0, 0.0, sign(hit.z)));
    return true;
}

void main() {
    vec2 ndc = (gl_FragCoord.xy / turing_resolution) * 2.0 - 1.0;
    float aspect = turing_resolution.x / max(turing_resolution.y, 1.0);
    vec3 rayOrigin = vec3(0.0, 0.0, 8.5);
    vec3 rayDirection = normalize(vec3(ndc.x * aspect, ndc.y, -2.15));
    vec3 lightPosition = vec3(
        3.8 * cos(turing_time * 0.19),
        4.6,
        5.5 + 0.7 * sin(turing_time * 0.13)
    );
    vec3 color = vec3(0.008, 0.012, 0.025);
    float nearest = 1.0e20;

    for (int index = 0; index < 256; ++index) {
        if (index >= turing_dom_count) break;
        vec4 geometry = texelFetch(turing_dom_state, ivec2(0, index), 0);
        vec4 material = texelFetch(turing_dom_state, ivec2(1, index), 0);
        vec4 tint = texelFetch(turing_dom_state, ivec2(2, index), 0);
        vec2 normalizedCenter = geometry.xy / turing_resolution * 2.0 - 1.0;
        vec2 normalizedHalf = geometry.zw / turing_resolution * 2.0;
        vec3 center = vec3(
            normalizedCenter.x * aspect * 4.0,
            normalizedCenter.y * 4.0,
            -0.12 * material.x - float(index) * 0.03
        );
        vec3 halfExtent = vec3(
            max(normalizedHalf.x * aspect * 4.0, 0.018),
            max(normalizedHalf.y * 4.0, 0.018),
            0.10 + 0.025 * min(material.y, 1.0)
        );
        float distance;
        vec3 normal;
        if (rayBox(rayOrigin, rayDirection, center, halfExtent, distance, normal)
            && distance < nearest) {
            vec3 point = rayOrigin + rayDirection * distance;
            vec3 lightVector = lightPosition - point;
            float lightDistance2 = dot(lightVector, lightVector);
            vec3 lightDirection = normalize(lightVector);
            vec3 viewDirection = normalize(rayOrigin - point);
            vec3 halfVector = normalize(lightDirection + viewDirection);
            float diffuse = max(dot(normal, lightDirection), 0.0);
            float specular = pow(max(dot(normal, halfVector), 0.0), 48.0);
            float attenuation = 1.0 / (1.0 + 0.035 * lightDistance2);
            vec3 surface = tint.rgb * (0.13 + 1.15 * diffuse * attenuation);
            surface += vec3(1.0, 0.91, 0.75) * specular * attenuation * 0.9;
            surface += tint.rgb * min(material.y, 1.0) * 0.08;
            color = surface;
            nearest = distance;
        }
    }
    turing_output_0 = vec4(pow(max(color, 0.0), vec3(1.0 / 2.2)), 1.0);
}
'''


DEMO_DOCUMENT = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root { color-scheme: dark; font-family: Inter, system-ui, sans-serif; }
* { box-sizing: border-box; }
body { margin: 0; min-height: 100vh; padding: 5vw; color: #dfeaff;
  background: #070b1b; }
header { max-width: 900px; padding: 26px 30px; border-radius: 28px;
  background: rgb(25,52,96); }
h1 { margin: 0 0 10px; font-size: clamp(34px,6vw,76px); }
p { margin: 0; color: #b9c9ec; font-size: 18px; line-height: 1.55; }
nav { display: flex; gap: 12px; margin: 22px 0 30px; }
button { border: 0; border-radius: 999px; padding: 12px 20px;
  background: rgb(112,224,255); font-weight: 800; }
main { display: grid; grid-template-columns: repeat(3,minmax(170px,1fr)); gap: 20px; }
article { min-height: 180px; padding: 22px; border-radius: 24px; background: rgb(45,73,131); }
article:nth-child(2) { background: rgb(94,61,142); }
article:nth-child(3) { background: rgb(30,111,119); }
footer { margin-top: 24px; padding: 18px 22px; border-radius: 18px; background: rgb(25,42,76); }
@media (max-width:720px) { main { grid-template-columns:1fr; } body { padding:24px; } }
</style></head><body>
<header data-turing-identity="hero"><h1>Elastic Document</h1>
<p>DOM layout becomes physical geometry. Click an element to strike its spring.</p></header>
<nav data-turing-identity="actions"><button>Impulse the surface</button><button>Let it ring</button></nav>
<main>
<article data-turing-identity="layout-card"><h2>Measured layout</h2><p>The browser supplies only final rectangles and events.</p></article>
<article data-turing-identity="wasm-card"><h2>Resident physics</h2><p>Bound springs and managed time execute in WebAssembly.</p></article>
<article data-turing-identity="shader-card"><h2>Ray-cast depth</h2><p>One shader intersects and lights projected boxes.</p></article>
</main><footer data-turing-identity="footer">Drop an HTML file to replace the hidden document.</footer>
</body></html>'''


# One candidate transaction in the authored Python function.  Repetition is
# bounded at bake time because the current flat Wasm backend deliberately has
# no unbounded execution.  Every card computes provisional state and commits
# it only when the DT admission predicate succeeds; rejection therefore has
# true rollback semantics without a second mutable state area.
_MANAGED_CANDIDATE = '''
    spring_dx = anchor_x - position_x
    spring_dy = anchor_y - position_y
    speed2 = velocity_x * velocity_x + velocity_y * velocity_y
    scale2 = extent_x * extent_x + extent_y * extent_y + 1.0
    stable_dt = 0.35 / (28.0 + speed2 / scale2).sqrt()
    candidate_dt = step_dt.minimum(remaining_time)
    admitted = (candidate_dt <= stable_dt) * (remaining_time > 0.000000001)
    acceleration_x = spring_dx * 28.0 - velocity_x * 0.16
    acceleration_y = spring_dy * 28.0 - velocity_y * 0.16
    provisional_x = position_x + velocity_x * candidate_dt + acceleration_x * candidate_dt * candidate_dt * 0.5
    provisional_y = position_y + velocity_y * candidate_dt + acceleration_y * candidate_dt * candidate_dt * 0.5
    next_acceleration_x = (anchor_x - provisional_x) * 28.0 - velocity_x * 0.16
    next_acceleration_y = (anchor_y - provisional_y) * 28.0 - velocity_y * 0.16
    provisional_vx = velocity_x + (acceleration_x + next_acceleration_x) * candidate_dt * 0.5
    provisional_vy = velocity_y + (acceleration_y + next_acceleration_y) * candidate_dt * 0.5
    position_x = admitted * provisional_x + (1.0 - admitted) * position_x
    position_y = admitted * provisional_y + (1.0 - admitted) * position_y
    velocity_x = admitted * provisional_vx + (1.0 - admitted) * velocity_x
    velocity_y = admitted * provisional_vy + (1.0 - admitted) * velocity_y
    remaining_time = remaining_time - admitted * candidate_dt
    rejected_steps = rejected_steps + (1.0 - admitted) * (remaining_time > 0.000000001)
    step_dt = admitted * remaining_time.minimum(stable_dt) + (1.0 - admitted) * candidate_dt * 0.5
'''


SOURCE = (
    "TURING_PAGE = "
    + repr({
        "entrypoint": "elastic_dom_page",
        "title": "Elastic DOM Surface",
        "slug": "elastic-dom-surface",
        "width": 960,
        "height": 640,
        "probe_size": 4,
        "feeds": {
            "position_x": {"values": [90.0, 260.0, 430.0, 600.0]},
            "position_y": {"values": [90.0, 150.0, 230.0, 320.0]},
            "velocity_x": 0.0, "velocity_y": 0.0,
            "anchor_x": {"values": [90.0, 260.0, 430.0, 600.0]},
            "anchor_y": {"values": [90.0, 150.0, 230.0, 320.0]},
            "extent_x": {"values": [140.0, 150.0, 160.0, 180.0]},
            "extent_y": {"values": [48.0, 64.0, 72.0, 84.0]},
            "pointer_x": 0.0, "pointer_y": 0.0,
            "pointer_buttons": 0.0, "button_latch": 0.0,
            "score": 0.0, "window_dt": 1.0 / 60.0,
        },
        "backend": "c", "remove_loops": True,
        "state_feedback": {
            "position_x": "next_position_x", "position_y": "next_position_y",
            "velocity_x": "next_velocity_x", "velocity_y": "next_velocity_y",
            "button_latch": "next_button_latch", "score": "next_score",
        },
        "render_fps": 60.0, "autostart": True,
        "presentation_shader": PRESENTATION_SHADER,
        "presentation_document": DEMO_DOCUMENT,
        "shader_configuration": {
            "dom_surface": True,
            "max_elements": 256,
            "dom_io": {
                "inputs": {
                    "position_x": "position_x", "position_y": "position_y",
                    "velocity_x": "velocity_x", "velocity_y": "velocity_y",
                    "anchor_x": "anchor_x", "anchor_y": "anchor_y",
                    "extent_x": "extent_x", "extent_y": "extent_y",
                    "pointer_x": "pointer_x", "pointer_y": "pointer_y",
                    "pointer_buttons": "pointer_buttons",
                    "button_latch": "button_latch", "score": "score",
                    "window_dt": "window_dt",
                },
                "outputs": {
                    "position_x": 0, "position_y": 1,
                    "velocity_x": 2, "velocity_y": 3,
                    "button_latch": 4, "score": 5, "activity": 6,
                    "rejected_steps": 7, "advanced_time": 8,
                    "remaining_time": 9,
                },
            },
        },
    })
    + '''

def elastic_dom_page(
    position_x, position_y, velocity_x, velocity_y,
    anchor_x, anchor_y, extent_x, extent_y,
    pointer_x, pointer_y, pointer_buttons, button_latch, score, window_dt,
):
    half_x = extent_x * 0.5
    half_y = extent_y * 0.5
    hovered = (
        (pointer_x >= position_x - half_x)
        * (pointer_x <= position_x + half_x)
        * (pointer_y >= position_y - half_y)
        * (pointer_y <= position_y + half_y)
    )
    pressed = hovered * (pointer_buttons > 0.0) * (button_latch <= 0.0)
    pointer_dx = position_x - pointer_x
    pointer_dy = position_y - pointer_y
    velocity_x = velocity_x + pressed * (150.0 + pointer_dx * 1.8)
    velocity_y = velocity_y + pressed * (-95.0 + pointer_dy * 1.8)
    remaining_time = window_dt
    step_dt = window_dt
    rejected_steps = window_dt * 0.0
'''
    + _MANAGED_CANDIDATE * 16
    + '''
    next_position_x = position_x
    next_position_y = position_y
    next_velocity_x = velocity_x
    next_velocity_y = velocity_y
    next_button_latch = pointer_buttons > 0.0
    next_score = score + pressed
    activity = (hovered * 0.2 + pressed * 0.65 + (velocity_x * velocity_x + velocity_y * velocity_y) * 0.00002).minimum(1.0)
    advanced_time = window_dt - remaining_time
    return (
        next_position_x, next_position_y, next_velocity_x, next_velocity_y,
        next_button_latch, next_score, activity, rejected_steps,
        advanced_time, remaining_time,
    )
'''
)


def build_demo(destination: Path):
    """Compile the single authored Python page through the ordinary bundler."""

    from .site_bundle import build_program_bundle

    return build_program_bundle(
        SOURCE,
        destination,
        source_filename="elastic_dom_page.py",
        include_backends=False,
        include_mathematics=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile the Elastic DOM page.")
    parser.add_argument(
        "--destination", type=Path,
        default=Path(__file__).resolve().parents[2].parent,
        help="static-site publication root",
    )
    arguments = parser.parse_args(argv)
    bundle = build_demo(arguments.destination.resolve())
    print(bundle.page_path)
    return 0


__all__ = ["SOURCE", "build_demo", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
