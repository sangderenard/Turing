"""Geometric projections of identified Abstract UI mechanical objects.

The source graph object owns shape and relationships. Realization adds a
geometric view of that object; render buffers are a later backend projection.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .abstract_ui import AbstractUI
from .abstract_ui_world import WorldObject


def fixture_world_objects(realizations, root: str) -> tuple[WorldObject, ...]:
    """Expose fixture solids as canonical game objects with physical ownership."""
    objects = []
    for _pillar, roller in realizations:
        model = roller.model
        objects.append(WorldObject(
            identity=roller.identity, kind=model["kind"], parent=root,
            label="Roller carriage", transform={"position": [0.0, 0.0, 0.0]},
            form={"recipe": "assembly"}, capabilities=("inspect",),
            physics={"authority": "vehicle-coupled-graph", "state": model["mechanical_state"]},
            extensions={"mechanical_connections": {
                "wheels": model["wheel_identities"], "hubs": model["hub_identities"],
                "articulation": model["articulation"], "operation": model["operation"],
            }},
        ))
        for solid in model["geometry_projection"]["solids"]:
            identity = solid["identity"]
            surface_identity = f"{identity}/surface"
            objects.append(WorldObject(
                identity=identity, kind="fixture-roller", parent=roller.identity,
                label="Cylindrical roller", transform={"position": solid["position"]},
                form={"recipe": "indexed-surface", "shape": solid["form"],
                      "surface": solid["surface"], "surface_identity": surface_identity,
                      "color_rgb": solid["material"]["base_color_rgb"]},
                capabilities=("inspect", "select", "publish-mesh"),
                semantic_parts=({"identity": surface_identity, "role": "roller-surface",
                                 "material_role": "body"},),
                physics={"authority": "vehicle-coupled-graph", "body": "kinematic",
                         "collider": solid["form"], "axle": solid["axis"]},
                extensions={"shading": "phong", "mechanical_owner": roller.identity},
            ))
    return tuple(objects)


def cylinder_surface(radius: float, length: float, segments: int = 32) -> dict:
    """Closed local-z cylinder with separate cap normals for Phong rendering."""
    if not (math.isfinite(radius) and math.isfinite(length) and radius > 0 and length > 0):
        raise ValueError("cylinder dimensions must be finite positive metres")
    if isinstance(segments, bool) or not isinstance(segments, int) or segments < 3:
        raise ValueError("cylinder requires at least three angular segments")
    positions, normals, triangles = [], [], []
    for index in range(segments):
        angle = 2 * math.pi * index / segments
        x, y = math.cos(angle), math.sin(angle)
        for z in (-length / 2, length / 2):
            positions.append([radius * x, radius * y, z])
            normals.append([x, y, 0.0])
    for index in range(segments):
        a, b = 2 * index, 2 * ((index + 1) % segments)
        triangles.extend([[a, b, b + 1], [a, b + 1, a + 1]])
    for sign in (-1, 1):
        center = len(positions)
        positions.append([0.0, 0.0, sign * length / 2])
        normals.append([0.0, 0.0, float(sign)])
        for index in range(segments):
            angle = 2 * math.pi * index / segments
            positions.append([radius * math.cos(angle), radius * math.sin(angle), sign * length / 2])
            normals.append([0.0, 0.0, float(sign)])
        for index in range(segments):
            a, b = center + 1 + index, center + 1 + (index + 1) % segments
            triangles.append([center, a, b] if sign > 0 else [center, b, a])
    return {"positions": positions, "normals": normals, "triangles": triangles}


def realize_edge_geometry(edge: Mapping[str, Any], endpoints: np.ndarray,
                          alpha: float) -> AbstractUI:
    """Keep graph identity/connectivity even when its geometric view is hidden."""
    identity = edge["identity"]
    if not isinstance(identity, str) or not identity:
        raise ValueError("geometric realization requires a source object identity")
    edge_class = edge.get("edge_class")
    color = ((104, 185, 235) if edge_class == "drivetrain" else
             (94, 206, 193) if edge_class == "pneumatic" else
             (190, 116, 224) if edge_class == "contact-seal" else
             (225, 154, 72) if edge_class == "load-bearing-structure" else
             (132, 143, 151))
    color = [int(channel * (0.35 + 0.65 * alpha)) for channel in color]
    polylines = []
    if alpha > 0.0:
        polylines.append({
            "identity": f"{identity}/geometry/polyline/0", "owner": identity,
            "positions": np.asarray(endpoints, dtype=np.float64).tolist(),
            "color_rgb": color, "width_px": 3, "closed": False,
        })
    return AbstractUI(identity=identity, model={
        **edge,
        "geometry_projection": {
            "schema": "abstract-ui-object-geometry-v1",
            "coordinate_space": "world", "visible": alpha > 0.0,
            "polylines": polylines,
        },
    })


def part_geometry_lines(center: np.ndarray,
                        node: Mapping[str, Any], color,
                        alpha: float) -> list:
    """Build world-space part polylines as (points, RGB, width, closed)."""

    geometry = node.get("geometry") or {}
    primitive = str(geometry.get("primitive", ""))
    if not primitive or primitive.startswith("solver-membrane"):
        return []
    lines = []
    axis = np.asarray(geometry.get("axis", (0.0, 0.0, 1.0)),
                      dtype=np.float64)
    axis /= max(1.0e-12, float(np.linalg.norm(axis)))
    seed = (np.asarray((0.0, 1.0, 0.0)) if abs(axis[1]) < 0.9 else
            np.asarray((1.0, 0.0, 0.0)))
    radial_a = np.cross(axis, seed)
    radial_a /= max(1.0e-12, float(np.linalg.norm(radial_a)))
    radial_b = np.cross(axis, radial_a)
    shade = tuple(max(0, min(255, int(channel * (0.35 + 0.65 * alpha))))
                  for channel in color)

    def ring(radius: float, offset: float = 0.0, width: int = 2):
        angles = np.linspace(0.0, 2.0 * math.pi, 41)
        points = (center + offset * axis + radius *
                  (np.cos(angles)[:, None] * radial_a
                   + np.sin(angles)[:, None] * radial_b))
        lines.append((points, shade, width, True))
        return points

    if primitive == "wheel-center-disc":
        radius = float(geometry["radius_m"])
        ring(radius, 0.0, 3)
        spoke_angles = np.linspace(0.0, 2.0 * math.pi, 7)[:-1]
        endpoints = np.asarray([
            center + radius * (math.cos(angle) * radial_a
                               + math.sin(angle) * radial_b)
            for angle in spoke_angles])
        for endpoint in endpoints:
            lines.append((np.vstack((center, endpoint)), shade, 2, False))
    elif primitive == "drop-center-rim":
        radius = float(geometry["radius_m"])
        bead_radius = float(geometry["bead_seat_radius_m"])
        half = 0.5 * float(geometry["width_m"])
        ring(bead_radius, -half, 3)
        ring(bead_radius, half, 3)
        ring(radius * 0.88, 0.0, 2)
    elif primitive == "bead-ring":
        ring(float(geometry["radius_m"]),
             float(geometry.get("axial_offset_m", 0.0)), 4)
    elif primitive == "bearing-races":
        ring(0.5 * float(geometry["outer_diameter_m"]), 0.0, 4)
        ring(0.5 * float(geometry["bore_m"]), 0.0, 2)
    elif primitive == "wheel-mounting-hub":
        ring(float(geometry["flange_radius_m"]), 0.0, 4)
        ring(float(geometry["barrel_radius_m"]), 0.0, 3)
    elif primitive == "brake-drum":
        half = 0.5 * float(geometry["width_m"])
        ring(float(geometry["radius_m"]), -half, 3)
        ring(float(geometry["radius_m"]), half, 3)
    elif primitive == "axial-structural-casing":
        half = 0.5 * float(geometry["length_m"])
        tube_radius = float(geometry["tube_radius_m"])
        left = ring(tube_radius, -half, 3)
        right = ring(tube_radius, half, 3)
        for angle in np.linspace(0.0, 2.0 * math.pi, 7)[:-1]:
            left_point = (center - half * axis + tube_radius *
                          (math.cos(angle) * radial_a
                           + math.sin(angle) * radial_b))
            right_point = (center + half * axis + tube_radius *
                           (math.cos(angle) * radial_a
                            + math.sin(angle) * radial_b))
            lines.append((np.vstack((left_point, right_point)), shade, 2, False))
        ring(float(geometry["center_radius_m"]), 0.0, 5)
    return lines


def realize_part_geometry(node: Mapping[str, Any], position: np.ndarray,
                          color, alpha: float) -> AbstractUI:
    """Project an existing graph node without inventing another object identity."""
    identity = node["identity"]
    if not isinstance(identity, str) or not identity:
        raise ValueError("geometric realization requires a source object identity")
    polylines = []
    for index, (points, shade, width, closed) in enumerate(
            part_geometry_lines(position, node, color, alpha)):
        polylines.append({
            "identity": f"{identity}/geometry/polyline/{index}",
            "owner": identity,
            "positions": points.tolist(),
            "color_rgb": list(shade),
            "width_px": width,
            "closed": closed,
        })
    return AbstractUI(identity=identity, model={
        **node,
        "geometry_projection": {
            "schema": "abstract-ui-object-geometry-v1",
            "coordinate_space": "world",
            "position": np.asarray(position, dtype=np.float64).tolist(),
            "polylines": polylines,
        },
    })


def fixture_geometry(snapshot: Mapping[str, Any]) -> list:
    """Return each support segment and its pair of world-space roller markers."""
    pillar_pose = np.asarray(snapshot["pillar_pose"], dtype=np.float64)
    pillar_alpha = np.asarray(snapshot["pillar_alpha"], dtype=np.float64)
    fixture = np.asarray(snapshot["fixture_wheel"], dtype=np.float64)
    anchor = np.asarray(snapshot["roller_anchor"], dtype=np.float64)
    geometry = []
    for wheel in range(len(pillar_pose)):
        top = pillar_pose[wheel]
        bottom = np.asarray((top[0], -0.75, top[2]))
        color = (220, 173, 61) if pillar_alpha[wheel] > 0.01 else (85, 91, 96)
        carriage_y = fixture[wheel, 0]
        roller_points = np.asarray([
            (anchor[wheel, 0] - 0.18, carriage_y, anchor[wheel, 1]),
            (anchor[wheel, 0] + 0.18, carriage_y, anchor[wheel, 1]),
        ])
        geometry.append((np.stack((bottom, top)), color, 5,
                         roller_points, (196, 202, 207), 8, 2))
    return geometry


def realize_fixture_geometry(plan, snapshot: Mapping[str, Any]) -> list:
    """Project the negotiated fixture objects, retaining their wheel/hub links."""
    geometry = fixture_geometry(snapshot)
    if not (len(geometry) == len(plan.wheel_identities) == len(plan.pillars)
            == len(plan.tire_mounting_rollers)):
        raise ValueError("fixture geometry and negotiated object counts differ")
    objects = []
    for index, (support, color, width, rollers, marker_color, radius, stroke) in enumerate(geometry):
        pillar = plan.pillars[index]
        roller = plan.tire_mounting_rollers[index]
        if roller.roller_length_m is None:
            raise ValueError("roller solid requires its negotiated axial length")
        surface = cylinder_surface(roller.roller_radius_m, roller.roller_length_m)
        wheel = plan.wheel_identities[index]
        if pillar.wheel_identity != wheel or roller.wheel_identities != (wheel,):
            raise ValueError("fixture buffer order differs from negotiated wheel identities")
        pillar_object = AbstractUI(identity=pillar.identity, model={
            "identity": pillar.identity, "kind": "articulated-wheel-pillar",
            "wheel_identity": pillar.wheel_identity, "hub_identity": pillar.hub_identity,
            "gravity_parallel_axis": pillar.gravity_parallel_axis,
            "synchronized_wheels": pillar.synchronized_wheels,
            "initialization_position": pillar.initialization_position,
            "eventual_installation_position": pillar.eventual_installation_position,
            "mechanical_state": {
                "reaction_force_y_n": float(snapshot["pillar_reaction_force"][index]),
                "source": "vehicle-coupled-graph.pillar-reaction",
            },
            "geometry_projection": {
                "schema": "abstract-ui-object-geometry-v1", "coordinate_space": "world",
                "polylines": [{
                    "identity": f"{pillar.identity}/geometry/support", "owner": pillar.identity,
                    "positions": support.tolist(), "color_rgb": list(color),
                    "width_px": width, "closed": False,
                }],
            },
        })
        roller_object = AbstractUI(identity=roller.identity, model={
            "identity": roller.identity, "kind": roller.kind, "operation": roller.operation,
            "wheel_identities": roller.wheel_identities, "hub_identities": roller.hub_identities,
            "articulation": roller.articulation, "reason": roller.reason,
            "mechanical_state": {
                "carriage_y": float(snapshot["fixture_wheel"][index][0]),
                "carriage_velocity_y": float(snapshot["fixture_wheel"][index][1]),
                "actuator_force": float(snapshot["fixture_wheel"][index][2]),
                "passive_force": float(snapshot["fixture_wheel"][index][3]),
                "compensation_force": float(snapshot["fixture_wheel"][index][4]),
                "command_y": float(snapshot["fixture_command"][index][4]),
                "command_velocity_y": float(snapshot["fixture_command"][index][5]),
                "corner_mode": float(snapshot["fixture_command"][index][7]),
                "source": "vehicle-coupled-graph.roller-fixture",
            },
            "geometry_projection": {
                "schema": "abstract-ui-object-geometry-v1", "coordinate_space": "world",
                "solids": [{
                    "identity": f"{roller.identity}/geometry/roller/{side}", "owner": roller.identity,
                    "position": position.tolist(), "axis": [0.0, 0.0, 1.0],
                    "form": {"primitive": "cylinder", "radius_m": roller.roller_radius_m,
                             "length_m": roller.roller_length_m},
                    "surface": surface,
                    "material": {"shading": "phong", "base_color_rgb": list(marker_color)},
                } for side, position in zip(("left", "right"), rollers)],
            },
        })
        objects.append((pillar_object, roller_object))
    return objects


