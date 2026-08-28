"""General solid contact-surface contracts shared by every physics body."""

from __future__ import annotations

import math
from typing import Any


ABSTRACT_UI_SURFACE_VERSION = "abstract-ui-contact-surfaces-v1"


def support_surface_model(root: str) -> dict[str, Any]:
    return {
        "schema": ABSTRACT_UI_SURFACE_VERSION,
        "identity": f"{root}/physics/contact-surfaces",
        "consumers": ["platformer-body", "projectile-body", "vehicle-contact-patch",
                      "rigid-body"],
        "sample_abi": {
            "inputs": ["world_x", "world_z", "previous_base_y", "candidate_base_y",
                       "vertical_velocity", "reach"],
            "outputs": ["supported", "height", "gradient_x", "gradient_z",
                        "normal_x", "normal_y", "normal_z", "surface_identity"],
        },
        "selection": {
            "rule": "nearest-non-separating-contact-manifold-candidate",
            "reachability": "within-shape-reach-and-not-separated-along-contact-normal",
            "tie_break": "stable-runtime-part-id",
            "edge_rule": "sampled-half-space-latch-persists-until-body-leaves-local-xz-mask",
        },
        "contact_constraint": {
            "normal": "normalize((-gradient_x, 1, -gradient_z))",
            "normal_response": "weight-plus-penetration-and-damping-determine-normal-impulse",
            "tangent_response": "static-or-kinetic-friction-from-contact-material",
            "motion": "velocity-is-projected-only-by-the-resolved-contact-impulse",
        },
        "world_bottom": {
            "kind": "thick-rejection-volume",
            "top_y": 0.0,
            "thickness": 8.0,
            "minimum_y": -8.0,
            "sampled_surface_guard_depth": 0.75,
            "activation": "swept-entry-or-contained-below-local-sampled-surface-minus-guard-depth",
            "response": "project-to-local-sampled-support-and-remove-inward-normal-velocity",
            "scope": "platformer-projectile-vehicle-and-rigid-body-safety-manifold",
            "role": "emergency-containment-not-ordinary-floor-contact",
            "sampled_surface_latch": {
                "inequality": "body-support-base >= local-height - maximum-material-sink-depth",
                "acquire": "swept-crossing-or-any-contained-below-surface-state",
                "release": "outside-authored-surface-domain-only",
                "response": "atomic-local-height-rejection-plus-material-normal-restitution",
            },
        },
        "authority": "portable-physics-contract",
        "current_adapter": "analytic-plane-sampler-in-browser-host",
    }


def linear_gradient_solid(
    identity: str,
    parent: str,
    *,
    center_x: float,
    center_z: float,
    half_width: float,
    half_run: float,
    low_height: float,
    high_height: float,
    rise_axis: str = "x",
    rise_sign: int = 1,
    palette_role: str = "artifact-source",
) -> dict[str, Any]:
    if rise_axis not in {"x", "z"} or rise_sign not in {-1, 1}:
        raise ValueError("a linear gradient rises along signed x or z")
    if min(half_width, half_run, low_height) <= 0 or high_height <= low_height:
        raise ValueError("a gradient solid needs positive dimensions and a higher side")
    half_x, half_z = ((half_run, half_width) if rise_axis == "x" else (half_width, half_run))
    gradient = rise_sign * (high_height - low_height) / (2 * half_run)
    return {
        "identity": identity, "kind": "static-gradient-solid", "label": "Gradient test solid",
        "parent_identity": parent, "center": [center_x, center_z],
        "half_extent": [half_x, half_z], "height": high_height,
        "floor_height": low_height, "wall_thickness": 0.04,
        "palette_role": palette_role, "wall_palette_role": palette_role,
        "geometry_mode": "height-field-prism", "openings": [],
        "placement": {"custody": "placed", "elevation": 0.0, "yaw_degrees": 0.0},
        "physics": {"body": "static", "collider": "solid-contact-surface", "enabled": True,
                    "contact_mode": "normal-constraint-with-coulomb-friction"},
        "surface": {
            "schema": ABSTRACT_UI_SURFACE_VERSION, "kind": "linear-height-field",
            "origin": [center_x, (low_height + high_height) * 0.5, center_z],
            "gradient": [gradient if rise_axis == "x" else 0.0,
                         gradient if rise_axis == "z" else 0.0],
            "domain": {"minimum_x": center_x - half_x, "maximum_x": center_x + half_x,
                       "minimum_z": center_z - half_z, "maximum_z": center_z + half_z},
            "one_sided": True,
        },
    }


def sampled_ramp_slab(
    identity: str,
    parent: str,
    *,
    center_x: float,
    center_z: float,
    half_width: float,
    half_run: float,
    low_height: float,
    high_height: float,
    rise_axis: str = "x",
    rise_sign: int = 1,
    columns: int = 33,
    rows: int = 5,
    palette_role: str = "artifact-source",
) -> dict[str, Any]:
    """Return a smooth sampled ramp top backed by its full-depth solid prism."""

    if rise_axis not in {"x", "z"} or rise_sign not in {-1, 1}:
        raise ValueError("a sampled ramp rises along signed x or z")
    if min(half_width, half_run, low_height) <= 0 or high_height <= low_height:
        raise ValueError("a sampled ramp slab needs positive dimensions and a higher side")
    if columns < 2 or rows < 2:
        raise ValueError("a sampled ramp slab needs at least a 2x2 contact map")
    half_x, half_z = ((half_run, half_width) if rise_axis == "x" else (half_width, half_run))
    minimum_x, minimum_z = center_x - half_x, center_z - half_z
    cell_x, cell_z = 2 * half_x / (columns - 1), 2 * half_z / (rows - 1)
    heights: list[float] = []
    for row in range(rows):
        z = minimum_z + row * cell_z
        for column in range(columns):
            x = minimum_x + column * cell_x
            along = ((x - minimum_x) / (2 * half_x) if rise_axis == "x"
                     else (z - minimum_z) / (2 * half_z))
            if rise_sign < 0:
                along = 1.0 - along
            # C2 end easing gives the tire a flat tangent at both joins while
            # preserving a constant, modest grade through most of the ramp.
            eased = along * along * along * (10 + along * (-15 + 6 * along))
            heights.append(round(low_height + (high_height - low_height) * eased, 6))
    return {
        "identity": identity, "kind": "static-sampled-ramp-slab", "label": "Sampled ramp slab",
        "parent_identity": parent, "center": [center_x, center_z],
        "half_extent": [half_x, half_z], "height": high_height,
        "floor_height": low_height, "wall_thickness": .04,
        "palette_role": palette_role, "wall_palette_role": palette_role,
        "geometry_mode": "sampled-height-field-prism", "openings": [],
        "placement": {"custody": "placed", "elevation": 0.0, "yaw_degrees": 0.0},
        "physics": {"body": "static", "collider": "solid-contact-surface", "enabled": True,
                    "contact_mode": "normal-constraint-with-coulomb-friction"},
        "surface": {
            "schema": ABSTRACT_UI_SURFACE_VERSION, "kind": "sampled-height-field",
            "resolution": [columns, rows], "heights": heights,
            "origin": [minimum_x, 0.0, minimum_z], "cell_size": [cell_x, cell_z],
            "domain": {"minimum_x": minimum_x, "maximum_x": center_x + half_x,
                       "minimum_z": minimum_z, "maximum_z": center_z + half_z},
            "interpolation": "piecewise-planar-two-triangles-per-cell",
            "gradient": "triangle-plane-gradient", "one_sided": True,
            "contact_material": {"identity": "compacted-ramp-fill", "maximum_sink_depth_m": .018,
                                 "equilibrium_sink_depth_m": .004, "restitution": .12},
            "features": {"course": "smooth-c2-ramp", "support": "full-depth-slab",
                         "authority": "local-contact-depth-map"},
        },
    }


def sampled_mud_oval_height_field(
    identity: str,
    parent: str,
    *,
    center_x: float,
    center_z: float,
    half_x: float = 8.0,
    half_z: float = 5.5,
    columns: int = 49,
    rows: int = 33,
    palette_role: str = "artifact-scratch",
) -> dict[str, Any]:
    """Bake crawler terrain with a smooth oval corridor into a sampled depth map."""

    if min(half_x, half_z) <= 0 or columns < 3 or rows < 3:
        raise ValueError("a sampled mud oval needs positive extents and at least a 3x3 grid")
    minimum_x, minimum_z = center_x - half_x, center_z - half_z
    cell_x, cell_z = 2 * half_x / (columns - 1), 2 * half_z / (rows - 1)
    ellipse_x, ellipse_z = half_x * .72, half_z * .68
    heights: list[float] = []
    for row in range(rows):
        z = minimum_z + row * cell_z
        for column in range(columns):
            x = minimum_x + column * cell_x
            dx, dz = (x - center_x) / ellipse_x, (z - center_z) / ellipse_z
            radius = math.hypot(dx, dz)
            angle = math.atan2(dz, dx)
            local_x, local_z = x - center_x, z - center_z
            broad_hills = (.27 * math.sin(local_x * .19)
                           + .22 * math.cos(local_z * .23)
                           + .16 * math.sin((local_x + local_z) * .31)
                           + .10 * math.cos((local_x - 1.7 * local_z) * .43))
            crawler_texture = .13 * abs(math.sin(local_x * .71) * math.cos(local_z * .63))
            landscape = .53 + broad_hills + crawler_texture
            # A quartic Gaussian produces a broad flat tire corridor with smooth
            # first derivatives into the surrounding crawler landscape.
            track_blend = math.exp(-((radius - .72) / .082) ** 4)
            track_height = .105 + .018 * math.sin(angle * 2)
            shoulder = .09 * math.exp(-((abs(radius - .72) - .105) / .045) ** 2)
            height = landscape * (1 - track_blend) + track_height * track_blend + shoulder
            heights.append(round(max(.035, height), 6))
    minimum_height, maximum_height = min(heights), max(heights)
    return {
        "identity": identity, "kind": "static-sampled-terrain", "label": "Crawler landscape oval",
        "parent_identity": parent, "center": [center_x, center_z],
        "half_extent": [half_x, half_z], "height": maximum_height,
        "floor_height": minimum_height, "wall_thickness": .04,
        "palette_role": palette_role, "wall_palette_role": palette_role,
        "geometry_mode": "sampled-height-field-prism", "openings": [],
        "placement": {"custody": "placed", "elevation": 0.0, "yaw_degrees": 0.0},
        "physics": {"body": "static", "collider": "solid-contact-surface", "enabled": True,
                    "contact_mode": "normal-constraint-with-coulomb-friction"},
        "surface": {
            "schema": ABSTRACT_UI_SURFACE_VERSION, "kind": "sampled-height-field",
            "resolution": [columns, rows], "heights": heights,
            "middle_height": .105,
            "origin": [minimum_x, 0.0, minimum_z], "cell_size": [cell_x, cell_z],
            "domain": {"minimum_x": minimum_x, "maximum_x": center_x + half_x,
                       "minimum_z": minimum_z, "maximum_z": center_z + half_z},
            "interpolation": "piecewise-planar-two-triangles-per-cell",
            "gradient": "triangle-plane-gradient", "one_sided": True,
            "contact_material": {"identity": "soft-mud", "maximum_sink_depth_m": .16,
                                 "equilibrium_sink_depth_m": .075, "restitution": .025},
            "features": {"landscape": "procedural-crawler-hills",
                         "course": "smooth-cut-oval", "authority": "outer-courtyard-depth-map"},
        },
    }


def _smoothstep(value: float) -> float:
    clamped = min(1.0, max(0.0, value))
    return clamped * clamped * (3.0 - 2.0 * clamped)


def _soft_box(value_x: float, value_z: float, half_x: float, half_z: float,
              softness: float) -> float:
    """Return a smooth mask for a centered axis-aligned feature zone."""

    edge_x = 1.0 - _smoothstep((abs(value_x) - half_x + softness) / softness)
    edge_z = 1.0 - _smoothstep((abs(value_z) - half_z + softness) / softness)
    return max(0.0, min(edge_x, edge_z))


def sampled_offroad_playground_height_field(
    identity: str,
    parent: str,
    *,
    center_x: float,
    center_z: float,
    half_x: float,
    half_z: float,
    baseline_height: float = 0.08,
    columns: int = 81,
    rows: int = 81,
    palette_role: str = "artifact-scratch",
) -> dict[str, Any]:
    """Bake several legible truck obstacles into one bounded play surface."""

    if min(half_x, half_z) <= 0 or columns < 3 or rows < 3:
        raise ValueError("an off-road playground needs positive extents and at least a 3x3 grid")
    minimum_x, minimum_z = center_x - half_x, center_z - half_z
    cell_x, cell_z = 2 * half_x / (columns - 1), 2 * half_z / (rows - 1)
    heights: list[float] = []
    for row in range(rows):
        z = minimum_z + row * cell_z
        for column in range(columns):
            x = minimum_x + column * cell_x
            local_x, local_z = x - center_x, z - center_z
            height = baseline_height

            hill_x, hill_z = local_x + half_x * .48, local_z - half_z * .38
            hill_mask = _soft_box(hill_x, hill_z, half_x * .27, half_z * .36, 2.2)
            hill_progress = _smoothstep((hill_z + half_z * .30) / (half_z * .52))
            hill = 2.65 * hill_progress + .22 * math.sin(hill_x * 1.15) ** 2
            height += hill_mask * hill

            rocks_x, rocks_z = local_x - half_x * .43, local_z - half_z * .37
            rock_mask = _soft_box(rocks_x, rocks_z, half_x * .30, half_z * .34, 1.8)
            rock = 0.0
            for offset_x, offset_z, radius, amplitude in (
                (-5.8, -5.0, 2.4, .72), (-1.8, -3.6, 1.5, 1.05),
                (2.1, -5.2, 2.0, .62), (5.4, -2.0, 1.4, 1.18),
                (-4.3, .2, 1.2, .84), (.3, .7, 2.5, .58),
                (4.2, 2.6, 1.8, .92), (-1.9, 4.6, 1.6, .75),
            ):
                distance = math.hypot(rocks_x - offset_x, rocks_z - offset_z) / radius
                rock = max(rock, amplitude * math.exp(-(distance ** 4)))
            height += rock_mask * rock

            whoop_x, whoop_z = local_x + half_x * .48, local_z + half_z * .43
            whoop_mask = _soft_box(whoop_x, whoop_z, half_x * .29, half_z * .29, 1.8)
            whoops = .18 + .62 * math.sin((whoop_z + half_z * .22) * .72) ** 2
            height += whoop_mask * (whoops + .10 * math.cos(whoop_x * .65))

            creek_z = local_z + half_z * .42
            creek_center = half_x * .43 + 2.2 * math.sin(creek_z * .16)
            creek_distance = local_x - creek_center
            creek_zone = _soft_box(local_x - half_x * .42, creek_z,
                                   half_x * .32, half_z * .32, 2.0)
            creek_banks = .78 * math.exp(-((abs(creek_distance) - 2.5) / 1.15) ** 2)
            creek_bed = .32 * math.exp(-(creek_distance / 1.55) ** 4)
            height += creek_zone * (.30 + creek_banks - creek_bed
                                     + .08 * math.sin(creek_z * .9))

            for ramp_x, direction in ((-4.0, 1.0), (4.0, -1.0)):
                rx, rz = local_x - ramp_x, local_z
                ramp_mask = _soft_box(rx, rz, 2.4, 6.0, 1.0)
                ramp_profile = max(0.0, 1.0 - abs(rz - direction * 1.2) / 5.0)
                height += ramp_mask * 1.15 * ramp_profile

            heights.append(round(max(.02, height), 6))
    minimum_height, maximum_height = min(heights), max(heights)
    return {
        "identity": identity, "kind": "static-sampled-terrain", "label": "Off-road play area",
        "parent_identity": parent, "center": [center_x, center_z],
        "half_extent": [half_x, half_z], "height": maximum_height,
        "floor_height": minimum_height, "wall_thickness": .04,
        "palette_role": palette_role, "wall_palette_role": palette_role,
        "geometry_mode": "sampled-height-field-prism", "openings": [],
        "placement": {"custody": "placed", "elevation": 0.0, "yaw_degrees": 0.0},
        "physics": {"body": "static", "collider": "solid-contact-surface", "enabled": True,
                    "contact_mode": "normal-constraint-with-coulomb-friction"},
        "surface": {
            "schema": ABSTRACT_UI_SURFACE_VERSION, "kind": "sampled-height-field",
            "resolution": [columns, rows], "heights": heights,
            "middle_height": baseline_height,
            "origin": [minimum_x, 0.0, minimum_z], "cell_size": [cell_x, cell_z],
            "domain": {"minimum_x": minimum_x, "maximum_x": center_x + half_x,
                       "minimum_z": minimum_z, "maximum_z": center_z + half_z},
            "interpolation": "piecewise-planar-two-triangles-per-cell",
            "gradient": "triangle-plane-gradient", "one_sided": True,
            "tracking_scope": "bounded-play-area-only",
            "contact_material": {"identity": "dry-packed-earth", "maximum_sink_depth_m": .065,
                                 "equilibrium_sink_depth_m": .022, "restitution": .08},
            "features": {
                "landscape": "multi-zone-off-road-playground",
                "zones": ["hill-climb", "rock-crawl", "whoops", "dry-creek-bed",
                          "sampled-opposing-ramps"],
                "staging": "flat-apron-outside-height-field",
                "authority": "outer-yard-local-depth-map",
            },
        },
    }


# Compatibility name for callers that still describe the wedge-shaped mesh by
# its appearance. It has no ramp-specific physics semantics.
linear_ramp_box = linear_gradient_solid


__all__ = ["ABSTRACT_UI_SURFACE_VERSION", "linear_gradient_solid", "linear_ramp_box",
           "sampled_mud_oval_height_field", "sampled_offroad_playground_height_field",
           "support_surface_model"]
