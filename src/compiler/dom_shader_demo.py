"""Single-Python-function Elastic DOM page compiled by the normal bundler.

The ingested source owns event response, persistent state transition,
BoundSpring motion, managed-time admission/rollback, render attributes, and
the ray-box presentation math -- all in one WASM-compiled function.
Presentation is a full per-pixel image computed with ordinary tensor
broadcasting and reductions, which are unrestricted in WASM; the page's
displayed shader is the standing default passthrough (see
``_PASSTHROUGH_SOURCE`` in ``site_bundle.py``), which the bundler attaches
automatically once this function names outputs ``red``/``green``/``blue``.
Nothing here is hand-authored GLSL, and nothing here is a second,
hand-split "presentation entrypoint" either -- one function, one compile.
"""

from __future__ import annotations

import argparse
from pathlib import Path


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
<article data-turing-identity="shader-card"><h2>Ray-cast depth</h2><p>The same compiled program ray-marches its own state.</p></article>
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

_CANVAS_WIDTH = 480.0
_CANVAS_HEIGHT = 320.0
_CANVAS_PIXELS = int(_CANVAS_WIDTH * _CANVAS_HEIGHT)
# A representative-shaped probe: the compiler bakes array length in as part
# of the program's shape, so this needs to be the real pixel count, not a
# placeholder -- exactly like the fluid-sim demo's column_x/column_y probes.
_PIXEL_ZEROS = [0.0] * _CANVAS_PIXELS


SOURCE = (
    "TURING_PAGE = "
    + repr({
        "entrypoint": "elastic_dom_page",
        "title": "Elastic DOM Surface",
        "slug": "elastic-dom-surface",
        "width": int(_CANVAS_WIDTH),
        "height": int(_CANVAS_HEIGHT),
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
            # Per-pixel feeds (one value per output pixel, not per element)
            # -- the same feed_expressions mechanism the fluid-sim demo uses
            # for its column grid, evaluated by JS from the implicit
            # per-pixel x/y/w/h/t.
            "ndc_x": {"values": list(_PIXEL_ZEROS)},
            "ndc_y": {"values": list(_PIXEL_ZEROS)},
            "aspect": {"values": list(_PIXEL_ZEROS)},
            "pixel_time": {"values": list(_PIXEL_ZEROS)},
            # Per-element constants: one tint per tracked DOM element and a
            # fixed z-separation so elements don't shear into each other
            # (widened from the original hand-written shader's 0.006/index
            # to 0.03/index worth of separation).
            "tint_r": {"values": [0.44, 0.37, 0.18, 0.44]},
            "tint_g": {"values": [0.62, 0.24, 0.43, 0.62]},
            "tint_b": {"values": [0.86, 0.56, 0.47, 0.86]},
            "depth_bias": {"values": [0.0, 0.03, 0.06, 0.09]},
        },
        "backend": "c", "remove_loops": True,
        "state_feedback": {
            "position_x": "next_position_x", "position_y": "next_position_y",
            "velocity_x": "next_velocity_x", "velocity_y": "next_velocity_y",
            "button_latch": "next_button_latch", "score": "next_score",
        },
        "render_fps": 60.0, "autostart": True,
        "presentation_document": DEMO_DOCUMENT,
        "feed_expressions": {
            "ndc_x": "(x / Math.max(w - 1.0, 1.0)) * 2.0 - 1.0",
            "ndc_y": "(y / Math.max(h - 1.0, 1.0)) * 2.0 - 1.0",
            "aspect": "w / Math.max(h, 1.0)",
            "pixel_time": "t",
        },
    })
    + '''

def elastic_dom_page(
    position_x, position_y, velocity_x, velocity_y,
    anchor_x, anchor_y, extent_x, extent_y,
    pointer_x, pointer_y, pointer_buttons, button_latch, score, window_dt,
    ndc_x, ndc_y, aspect, pixel_time,
    tint_r, tint_g, tint_b, depth_bias,
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

    # --- Presentation: ray-box intersection and Blinn-Phong shading,
    # folded into the same compiled program. This is a genuine reduction
    # (nearest-of-N-elements per pixel), which only WASM tensor math can
    # express -- a WebGL fragment shader compiled straight from Python
    # cannot do a loop or a reduction, only flat per-fragment scalar
    # arithmetic. So the whole image is computed here, once, as ordinary
    # array broadcasting: per-pixel values (ndc_x et al, one value per
    # output pixel) against per-element values (position_x et al, one
    # value per tracked DOM element) via reshape-to-broadcast, exactly the
    # same (-1, 1) / (1, -1) pattern already proven in
    # ``advance_columnar_surface_spring_local`` above. The result is
    # reduced back to one row per pixel with a masked minimum (nearest
    # valid hit) and a masked sum (that hit's own color) instead of a
    # loop with an early ``break``.
    pixel_ndc_x = ndc_x.reshape((-1, 1))
    pixel_ndc_y = ndc_y.reshape((-1, 1))
    pixel_aspect = aspect.reshape((-1, 1))
    pixel_time_col = pixel_time.reshape((-1, 1))

    elem_position_x = next_position_x.reshape((1, -1))
    elem_position_y = next_position_y.reshape((1, -1))
    elem_half_x = half_x.reshape((1, -1))
    elem_half_y = half_y.reshape((1, -1))
    elem_tint_r = tint_r.reshape((1, -1))
    elem_tint_g = tint_g.reshape((1, -1))
    elem_tint_b = tint_b.reshape((1, -1))
    elem_depth_bias = depth_bias.reshape((1, -1))

    ray_origin_x = 0.0
    ray_origin_y = 0.0
    ray_origin_z = 8.5

    ray_dir_x = pixel_ndc_x * pixel_aspect
    ray_dir_y = pixel_ndc_y
    ray_dir_z = pixel_ndc_x * 0.0 - 2.15
    ray_len = (ray_dir_x * ray_dir_x + ray_dir_y * ray_dir_y + ray_dir_z * ray_dir_z + 0.000001).sqrt()
    ray_dir_x = ray_dir_x / ray_len
    ray_dir_y = ray_dir_y / ray_len
    ray_dir_z = ray_dir_z / ray_len

    light_x = 3.8 * (pixel_time_col * 0.19).cos()
    light_y = pixel_time_col * 0.0 + 4.6
    light_z = 5.5 + 0.7 * (pixel_time_col * 0.13).sin()

    norm_center_x = elem_position_x / __WIDTH__ * 2.0 - 1.0
    norm_center_y = elem_position_y / __HEIGHT__ * 2.0 - 1.0
    norm_half_x = elem_half_x / __WIDTH__ * 2.0
    norm_half_y = elem_half_y / __HEIGHT__ * 2.0

    center_x = pixel_ndc_x * 0.0 + norm_center_x * pixel_aspect * 4.0
    center_y = pixel_ndc_x * 0.0 + norm_center_y * 4.0
    center_z = pixel_ndc_x * 0.0 - elem_depth_bias
    box_half_x = (norm_half_x * pixel_aspect * 4.0).maximum(0.018) + pixel_ndc_x * 0.0
    box_half_y = (norm_half_y * 4.0).maximum(0.018) + pixel_ndc_x * 0.0
    box_half_z = 0.12 + pixel_ndc_x * 0.0

    inv_dir_x = 1.0 / ray_dir_x
    inv_dir_y = 1.0 / ray_dir_y
    inv_dir_z = 1.0 / ray_dir_z

    first_x = (center_x - box_half_x - ray_origin_x) * inv_dir_x
    second_x = (center_x + box_half_x - ray_origin_x) * inv_dir_x
    near_x = first_x.minimum(second_x)
    far_x = first_x.maximum(second_x)

    first_y = (center_y - box_half_y - ray_origin_y) * inv_dir_y
    second_y = (center_y + box_half_y - ray_origin_y) * inv_dir_y
    near_y = first_y.minimum(second_y)
    far_y = first_y.maximum(second_y)

    first_z = (center_z - box_half_z - ray_origin_z) * inv_dir_z
    second_z = (center_z + box_half_z - ray_origin_z) * inv_dir_z
    near_z = first_z.minimum(second_z)
    far_z = first_z.maximum(second_z)

    near_distance = near_x.maximum(near_y).maximum(near_z)
    far_distance = far_x.minimum(far_y).minimum(far_z)

    hit_ok = (far_distance >= near_distance.maximum(0.0))
    hit_distance = (near_distance > 0.0) * near_distance + (near_distance <= 0.0) * far_distance

    big = 1.0e20
    effective_distance = hit_ok * hit_distance + (1.0 - hit_ok) * big
    nearest = effective_distance.min(dim=-1, keepdim=True)
    winner = (effective_distance <= nearest) * hit_ok

    point_x = ray_origin_x + ray_dir_x * hit_distance
    point_y = ray_origin_y + ray_dir_y * hit_distance
    point_z = ray_origin_z + ray_dir_z * hit_distance

    surface_x = point_x - center_x
    surface_y = point_y - center_y
    surface_z = point_z - center_z

    face_x = (surface_x.abs() - box_half_x).abs()
    face_y = (surface_y.abs() - box_half_y).abs()
    face_z = (surface_z.abs() - box_half_z).abs()

    is_x = (face_x < face_y) * (face_x < face_z)
    is_y = (face_y < face_z) * (1.0 - is_x)
    is_z = 1.0 - is_x - is_y

    normal_x = is_x * surface_x.sign()
    normal_y = is_y * surface_y.sign()
    normal_z = is_z * surface_z.sign()

    light_vec_x = light_x - point_x
    light_vec_y = light_y - point_y
    light_vec_z = light_z - point_z
    light_dist2 = light_vec_x * light_vec_x + light_vec_y * light_vec_y + light_vec_z * light_vec_z
    light_len = (light_dist2 + 0.000001).sqrt()
    light_dir_x = light_vec_x / light_len
    light_dir_y = light_vec_y / light_len
    light_dir_z = light_vec_z / light_len

    view_vec_x = ray_origin_x - point_x
    view_vec_y = ray_origin_y - point_y
    view_vec_z = ray_origin_z - point_z
    view_len = (view_vec_x * view_vec_x + view_vec_y * view_vec_y + view_vec_z * view_vec_z + 0.000001).sqrt()
    view_dir_x = view_vec_x / view_len
    view_dir_y = view_vec_y / view_len
    view_dir_z = view_vec_z / view_len

    half_dir_x = light_dir_x + view_dir_x
    half_dir_y = light_dir_y + view_dir_y
    half_dir_z = light_dir_z + view_dir_z
    half_dir_len = (half_dir_x * half_dir_x + half_dir_y * half_dir_y + half_dir_z * half_dir_z + 0.000001).sqrt()
    half_dir_x = half_dir_x / half_dir_len
    half_dir_y = half_dir_y / half_dir_len
    half_dir_z = half_dir_z / half_dir_len

    diffuse = (normal_x * light_dir_x + normal_y * light_dir_y + normal_z * light_dir_z).maximum(0.0)
    spec_dot = (normal_x * half_dir_x + normal_y * half_dir_y + normal_z * half_dir_z).maximum(0.0)
    spec2 = spec_dot * spec_dot
    spec4 = spec2 * spec2
    spec8 = spec4 * spec4
    spec16 = spec8 * spec8
    specular = spec16 * spec16 * spec16

    attenuation = 1.0 / (1.0 + 0.035 * light_dist2)

    surface_r = elem_tint_r * (0.13 + 1.15 * diffuse * attenuation) + 1.00 * specular * attenuation * 0.9
    surface_g = elem_tint_g * (0.13 + 1.15 * diffuse * attenuation) + 0.91 * specular * attenuation * 0.9
    surface_b = elem_tint_b * (0.13 + 1.15 * diffuse * attenuation) + 0.75 * specular * attenuation * 0.9

    winner_count = winner.sum(dim=-1) + 0.000001
    any_hit = (winner.sum(dim=-1) > 0.0)
    picked_r = (winner * surface_r).sum(dim=-1) / winner_count
    picked_g = (winner * surface_g).sum(dim=-1) / winner_count
    picked_b = (winner * surface_b).sum(dim=-1) / winner_count

    background_r = ndc_x * 0.0 + 0.008
    background_g = ndc_x * 0.0 + 0.012
    background_b = ndc_x * 0.0 + 0.025

    color_r = any_hit * picked_r + (1.0 - any_hit) * background_r
    color_g = any_hit * picked_g + (1.0 - any_hit) * background_g
    color_b = any_hit * picked_b + (1.0 - any_hit) * background_b

    red = color_r.maximum(0.0).reshape((-1,)) ** (1.0 / 2.2)
    green = color_g.maximum(0.0).reshape((-1,)) ** (1.0 / 2.2)
    blue = color_b.maximum(0.0).reshape((-1,)) ** (1.0 / 2.2)

    return (
        next_position_x, next_position_y, next_velocity_x, next_velocity_y,
        next_button_latch, next_score, activity, rejected_steps,
        advanced_time, remaining_time, red, green, blue,
    )
'''.replace("__WIDTH__", repr(_CANVAS_WIDTH)).replace("__HEIGHT__", repr(_CANVAS_HEIGHT))
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
