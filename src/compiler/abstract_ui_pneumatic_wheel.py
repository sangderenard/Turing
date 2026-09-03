"""Reusable Abstract UI contracts for pneumatic wheel assemblies.

Vocabulary in this module is deliberately strict:

* the axle ``hub`` rotates on a ``bearing``;
* the metal ``wheel center`` fastens to that hub;
* the ``rim`` is the annular bead-seat structure joined to the wheel center;
* each ``bead`` is part of the tire casing and seats on the rim;
* ``sidewall`` and ``tread`` are casing regions;
* the ``pneumatic boundary`` owns pressure containment and is either a tube or
  the tubeless combination of inner liner, casing, bead seals, and rim well.

The returned records are renderer-neutral Abstract UI data and mechanical-
graph objects.  Validators and world renderers consume the same identities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


PNEUMATIC_WHEEL_SCHEMA = "abstract-ui-pneumatic-wheel-assembly-v1"


@dataclass(frozen=True, slots=True)
class PneumaticWheelAssembly:
    """One wheel/rim/casing/boundary assembly, excluding axle hub/bearing."""

    identity: str
    pneumatic_mode: str
    parts: Mapping[str, Mapping[str, Any]]
    graph_nodes: tuple[Mapping[str, Any], ...]
    graph_edges: tuple[Mapping[str, Any], ...]

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema": PNEUMATIC_WHEEL_SCHEMA,
            "identity": self.identity,
            "pneumatic_mode": self.pneumatic_mode,
            "parts": {name: dict(part) for name, part in self.parts.items()},
            "graph_node_identities": [node["identity"]
                                      for node in self.graph_nodes],
            "graph_edge_identities": [edge["identity"]
                                      for edge in self.graph_edges],
        }


def _node(identity: str, kind: str, center: tuple[float, float, float],
          mass_kg: float, role: str, stage: str,
          geometry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "abstract-ui-mechanical-part-node-v1",
        "identity": identity,
        "kind": kind,
        "abstract_ui_role": role,
        "mass_kg": float(mass_kg),
        "reference_position": list(center),
        "assembly_stage": stage,
        "geometry": dict(geometry),
        "presentation": {
            "source": "abstract-ui-part-geometry",
            "primitive": geometry["primitive"],
        },
    }


def pneumatic_wheel_assembly(
    identity: str,
    *,
    center: tuple[float, float, float],
    axis: tuple[float, float, float],
    tire_radius_m: float,
    section_width_m: float,
    rim_radius_m: float | None = None,
    wheel_mass_kg: float,
    casing_mass_kg: float,
    rated_pressure_pa: float,
    pneumatic_mode: str = "tubeless",
    material_profile: str = "new-commercial-casing",
    pneumatic_feed: str = "conventional-rim-valve-only",
) -> PneumaticWheelAssembly:
    """Build separate Abstract UI parts for a serviceable pneumatic wheel.

    ``pneumatic_mode`` changes pressure ownership, not the meaning of wheel,
    rim, bead, or casing.  Both modes use the same casing and rim interfaces.
    """

    if pneumatic_mode not in {"tubeless", "tube"}:
        raise ValueError("pneumatic mode must be 'tubeless' or 'tube'")
    if pneumatic_feed not in {
            "conventional-rim-valve-only", "hub-fed-plus-conventional"}:
        raise ValueError("unknown pneumatic feed arrangement")
    if tire_radius_m <= 0.0 or section_width_m <= 0.0:
        raise ValueError("wheel dimensions must be positive")
    rim_radius = (tire_radius_m - 0.72 * section_width_m
                  if rim_radius_m is None else float(rim_radius_m))
    bead_radius = rim_radius + 0.018
    center_width = section_width_m * 0.82
    wheel_center = identity
    rim = f"{identity}/rim"
    bead_inboard = f"{identity}/tire-casing/bead-inboard"
    bead_outboard = f"{identity}/tire-casing/bead-outboard"
    sidewall = f"{identity}/tire-casing/sidewall"
    tread = f"{identity}/tire-casing/tread"
    boundary = f"{identity}/pneumatic-boundary"
    valve = f"{identity}/pneumatics/outer-service-valve"
    rim_port_a = f"{identity}/pneumatics/rim-seat-a"
    rim_port_b = f"{identity}/pneumatics/rim-seat-b"

    thermal_layers = [
        {
            "role": "inner-liner", "material": "halobutyl-rubber",
            "thickness_m": 0.0022, "orientation": "isotropic",
            "thermal_conductivity_w_m_k": 0.13,
            "specific_heat_j_kg_k": 1850.0,
        },
        {
            "role": "radial-carcass", "material": "rubberized-polyester-cord",
            "thickness_m": 0.0028, "orientation": "radial-cord-field",
            "thermal_conductivity_w_m_k": 0.24,
            "specific_heat_j_kg_k": 1450.0,
        },
        {
            "role": "cross-belt-a", "material": "rubberized-steel-cord",
            "thickness_m": 0.0015, "orientation_degrees": 22.0,
            "thermal_conductivity_w_m_k": 6.0,
            "specific_heat_j_kg_k": 760.0,
        },
        {
            "role": "cross-belt-b", "material": "rubberized-steel-cord",
            "thickness_m": 0.0015, "orientation_degrees": -22.0,
            "thermal_conductivity_w_m_k": 6.0,
            "specific_heat_j_kg_k": 760.0,
        },
        {
            "role": "tread-and-sidewall-skin", "material": "truck-tire-rubber",
            "thickness_m": 0.018, "orientation": "isotropic",
            "thermal_conductivity_w_m_k": 0.19,
            "specific_heat_j_kg_k": 1750.0,
        },
    ]
    if material_profile == "cheap-commercial-retread":
        thermal_layers[-1] = {
            "role": "economy-retread-cap", "material": "retread-tread-rubber",
            "thickness_m": 0.020, "orientation": "isotropic",
            "thermal_conductivity_w_m_k": 0.18,
            "specific_heat_j_kg_k": 1780.0,
            "bond": "buff-cement-and-vulcanized-cushion-gum",
            "carcass_history": "reused-inspected-commercial-radial-casing",
        }
    if pneumatic_mode == "tube":
        thermal_layers.insert(0, {
            "role": "tube-membrane", "material": "butyl-rubber",
            "thickness_m": 0.0018, "orientation": "isotropic",
            "thermal_conductivity_w_m_k": 0.12,
            "specific_heat_j_kg_k": 1900.0,
        })

    parts: dict[str, dict[str, Any]] = {
        "wheel_center": {
            "identity": wheel_center, "kind": "pressed-steel-wheel-center-disc",
            "owner": identity, "fastens_to": "axle-hub-not-bearing",
        },
        "rim": {
            "identity": rim, "kind": "heavy-drop-center-steel-rim",
            "owner": identity, "joined_to": wheel_center,
            "bead_seats": ["inboard", "outboard"],
            "rim_well_is_pressure_boundary": pneumatic_mode == "tubeless",
        },
        "bead_inboard": {
            "identity": bead_inboard, "kind": "pressure-seated-wire-casing-bead",
            "owner": sidewall, "seats_on": f"{rim}:inboard-seat",
            "retention": "inflation-pressure-rim-flange-and-bead-seat-friction",
            "beadlock": False, "fasteners": [],
        },
        "bead_outboard": {
            "identity": bead_outboard, "kind": "pressure-seated-wire-casing-bead",
            "owner": sidewall, "seats_on": f"{rim}:outboard-seat",
            "retention": "inflation-pressure-rim-flange-and-bead-seat-friction",
            "beadlock": False, "fasteners": [],
        },
        "sidewall": {
            "identity": sidewall, "kind": "oriented-composite-tire-sidewall",
            "owner": identity, "layers": thermal_layers,
            "material_profile": material_profile,
            "solver_surface": "single-invariant-center-surface",
        },
        "tread": {
            "identity": tread, "kind": "truck-tire-tread-crown",
            "owner": sidewall, "contact_boundary": True,
        },
        "pneumatic_boundary": {
            "identity": boundary,
            "kind": ("tube-pressure-membrane" if pneumatic_mode == "tube"
                     else "tubeless-casing-rim-bead-seal-boundary"),
            "owner": identity, "rated_pressure_pa": float(rated_pressure_pa),
            "volume_owner": (boundary if pneumatic_mode == "tube" else sidewall),
            "sealed_by": ([boundary, valve] if pneumatic_mode == "tube" else
                          [sidewall, bead_inboard, bead_outboard, rim, valve]),
            "thermal_layers": thermal_layers,
        },
        "outer_service_valve": {
            "identity": valve, "kind": "user-accessible-rotating-rim-valve",
            "owner": rim,
        },
    }
    positioned = "position-four-wheel-units-on-eventual-installation-pillars"
    mounted = "mount-four-tire-casings"
    inflated = "inflate-four-tires-through-wheel-ports"
    nodes = [
        _node(wheel_center, parts["wheel_center"]["kind"], center,
              wheel_mass_kg * 0.52, "wheel-center", positioned,
              {"primitive": "wheel-center-disc", "radius_m": rim_radius * 0.72,
               "width_m": center_width, "axis": list(axis)}),
        _node(rim, parts["rim"]["kind"], center, wheel_mass_kg * 0.48,
              "rim", positioned,
              {"primitive": "drop-center-rim", "radius_m": rim_radius,
               "bead_seat_radius_m": bead_radius, "width_m": center_width,
               "axis": list(axis)}),
        _node(bead_inboard, parts["bead_inboard"]["kind"], center,
              casing_mass_kg * 0.035, "casing-bead", mounted,
              {"primitive": "bead-ring", "radius_m": bead_radius,
               "axial_offset_m": -center_width * 0.5, "axis": list(axis)}),
        _node(bead_outboard, parts["bead_outboard"]["kind"], center,
              casing_mass_kg * 0.035, "casing-bead", mounted,
              {"primitive": "bead-ring", "radius_m": bead_radius,
               "axial_offset_m": center_width * 0.5, "axis": list(axis)}),
        _node(sidewall, parts["sidewall"]["kind"], center,
              casing_mass_kg * 0.78, "tire-casing-sidewall", mounted,
              {"primitive": "solver-membrane-owner", "radius_m": tire_radius_m,
               "width_m": section_width_m, "axis": list(axis)}),
        _node(tread, parts["tread"]["kind"], center,
              casing_mass_kg * 0.15, "tire-casing-tread", mounted,
              {"primitive": "solver-membrane-contact-region",
               "radius_m": tire_radius_m, "width_m": section_width_m,
               "axis": list(axis)}),
        _node(boundary, parts["pneumatic_boundary"]["kind"], center,
              0.0, "pneumatic-boundary", inflated,
              {"primitive": "pressure-boundary-state",
               "radius_m": tire_radius_m, "axis": list(axis)}),
        _node(valve, parts["outer_service_valve"]["kind"], center,
              0.12, "pneumatic-service-port", positioned,
              {"primitive": "pneumatic-port", "radius_m": 0.008,
               "axis": list(axis)}),
    ]
    if pneumatic_feed == "hub-fed-plus-conventional":
        parts["rim_seat_port_a"] = {
            "identity": rim_port_a, "kind": "rim-interior-pneumatic-port",
            "owner": rim,
        }
        parts["rim_seat_port_b"] = {
            "identity": rim_port_b, "kind": "rim-interior-pneumatic-port",
            "owner": rim,
        }
        nodes.extend((
            _node(rim_port_a, parts["rim_seat_port_a"]["kind"], center,
                  0.0, "pneumatic-rim-port", positioned,
                  {"primitive": "pneumatic-port", "radius_m": 0.005,
                   "axis": list(axis)}),
            _node(rim_port_b, parts["rim_seat_port_b"]["kind"], center,
                  0.0, "pneumatic-rim-port", positioned,
                  {"primitive": "pneumatic-port", "radius_m": 0.005,
                   "axis": list(axis)}),
        ))

    def edge(name: str, kind: str, left: str, right: str, stage: str,
             edge_class: str) -> dict[str, Any]:
        return {
            "identity": f"{identity}/edges/{name}", "kind": kind,
            "nodes": [left, right], "assembly_stage": stage,
            "edge_class": edge_class,
        }

    edges = [
        edge("wheel-center-to-rim", "formed-or-welded-wheel-joint",
             wheel_center, rim, positioned, "load-bearing-structure"),
        edge("rim-to-inboard-bead", "rim-seat-bead-contact",
             rim, bead_inboard, mounted, "contact-seal"),
        edge("rim-to-outboard-bead", "rim-seat-bead-contact",
             rim, bead_outboard, mounted, "contact-seal"),
        edge("inboard-bead-to-sidewall", "composite-casing-continuity",
             bead_inboard, sidewall, mounted, "membrane-structure"),
        edge("outboard-bead-to-sidewall", "composite-casing-continuity",
             bead_outboard, sidewall, mounted, "membrane-structure"),
        edge("sidewall-to-tread", "vulcanized-casing-continuity",
             sidewall, tread, mounted, "membrane-structure"),
        edge("pressure-to-casing", "pneumatic-boundary-load",
             boundary, sidewall, inflated, "pneumatic"),
        edge("service-valve-to-pressure", "pneumatic-material-flow",
             valve, boundary, inflated, "pneumatic"),
    ]
    if pneumatic_feed == "hub-fed-plus-conventional":
        edges.extend((
            edge("rim-port-a-to-pressure", "pneumatic-material-flow",
                 rim_port_a, boundary, inflated, "pneumatic"),
            edge("rim-port-b-to-pressure", "pneumatic-material-flow",
                 rim_port_b, boundary, inflated, "pneumatic"),
        ))
    return PneumaticWheelAssembly(
        identity, pneumatic_mode, parts, tuple(nodes), tuple(edges))


__all__ = ["PNEUMATIC_WHEEL_SCHEMA", "PneumaticWheelAssembly",
           "pneumatic_wheel_assembly"]
