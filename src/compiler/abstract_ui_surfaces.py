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
            "edge_rule": "loss-of-contact-applies-no-impulse-and-preserves-momentum",
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
            "features": {"landscape": "procedural-crawler-hills",
                         "course": "smooth-cut-oval", "authority": "outer-courtyard-depth-map"},
        },
    }


def _smoothstep(value: float) -> float:
    clamped = min(1.0, max(0.0, value))
    return clamped * clamped * (3.0 - 2.0 * clamped)


def sampled_hoop_height_field(
    identity: str,
    parent: str,
    *,
    center_x: float,
    center_z: float,
    half_x: float,
    half_z: float,
    door_x: float | None = None,
    door_z: float | None = None,
    door_clear_radius: float = 9.0,
    inner_radius: float = 0.50,
    outer_radius: float = 0.86,
    band_softness: float = 0.10,
    baseline_height: float = 0.12,
    amplitude: float = 0.42,
    columns: int = 97,
    rows: int = 97,
    palette_role: str = "artifact-scratch",
) -> dict[str, Any]:
    """Bake a ring of crawler terrain around the wall of a courtyard.

    The plaza in the middle and the ground within reach of a doorway stay flat
    -- open space for driving -- while a band near the wall carries real
    bumps.  Unlike ``sampled_mud_oval_height_field``, whose landscape fills
    the whole interior and cuts a single smooth lane through it, this leaves
    the interior clear and puts the terrain only where a wall-hugging loop
    would cross it.
    """

    if min(half_x, half_z) <= 0 or columns < 3 or rows < 3:
        raise ValueError("a sampled hoop needs positive extents and at least a 3x3 grid")
    if not 0 <= inner_radius < outer_radius <= 1:
        raise ValueError("the hoop band must sit strictly between the plaza and the wall")
    minimum_x, minimum_z = center_x - half_x, center_z - half_z
    cell_x, cell_z = 2 * half_x / (columns - 1), 2 * half_z / (rows - 1)
    ellipse_x, ellipse_z = half_x * .94, half_z * .94
    door_soften = max(1.0, door_clear_radius * .35)
    heights: list[float] = []
    for row in range(rows):
        z = minimum_z + row * cell_z
        for column in range(columns):
            x = minimum_x + column * cell_x
            dx, dz = (x - center_x) / ellipse_x, (z - center_z) / ellipse_z
            radius = math.hypot(dx, dz)
            local_x, local_z = x - center_x, z - center_z
            inner_edge = _smoothstep((radius - inner_radius) / band_softness)
            outer_edge = 1.0 - _smoothstep((radius - outer_radius) / band_softness)
            band = max(0.0, min(inner_edge, outer_edge))
            if door_x is not None and door_z is not None:
                door_distance = math.hypot(x - door_x, z - door_z)
                band *= _smoothstep((door_distance - door_clear_radius) / door_soften)
            broad_hills = (.31 * math.sin(local_x * .21)
                           + .26 * math.cos(local_z * .19)
                           + .17 * math.sin((local_x - local_z) * .27)
                           + .11 * math.cos((local_x + 1.6 * local_z) * .37))
            whoops = .12 * abs(math.sin(local_x * .58) * math.cos(local_z * .64))
            height = baseline_height + amplitude * band * (0.55 + broad_hills + whoops)
            heights.append(round(max(.02, height), 6))
    minimum_height, maximum_height = min(heights), max(heights)
    return {
        "identity": identity, "kind": "static-sampled-terrain", "label": "Hoop crawler ring",
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
            "features": {"landscape": "procedural-crawler-hoop",
                         "course": "clear-plaza-ring-band", "authority": "outer-yard-depth-map"},
        },
    }


# Compatibility name for callers that still describe the wedge-shaped mesh by
# its appearance. It has no ramp-specific physics semantics.
linear_ramp_box = linear_gradient_solid


__all__ = ["ABSTRACT_UI_SURFACE_VERSION", "linear_gradient_solid", "linear_ramp_box",
           "sampled_mud_oval_height_field", "sampled_hoop_height_field",
           "support_surface_model"]
