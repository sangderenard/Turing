"""General portal record: an edge of a generic graph with two boundary-attached ends.

The Living Data Map is the source of this shape.  Its browser template mints
a portal end ("splat": identity, port_role, object_identity, centre, normal,
tangent, bitangent, radius, aperture_class, aperture_scale) and a directed
edge between an ``in`` and an ``out`` end (identity, source, target, weight,
distance, aperture_class, path_model); Python publishes only the policy
(``PlacementPolicy.portal_contract``).  This module lifts those fields into a
neutral record.  New relative to the living map: ``PortalTransmission``,
``PortalProvenance`` (kind/specifier/source line, which the repository scan
needs) and ``PortalFrameMap`` as an explicit record (the living map holds the
same numbers as the WASM parameters ``portal_source_*``, ``portal_target_*``,
``portal_cos``, ``portal_sin``).  See docs/GENERAL_PORTAL_RECORD.md.

Stdlib only.  No geometry policy, no backend names.  Placement is filled by a
world builder; transmission by the particle simulation, shaders or ray
tracers.  Owners are graph node identities; any graph with nodes, containment
and directed reference edges can build a record via ``from_graph_edge``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

SCHEMA = "portal-record.v1"
ROLES = ("in", "out")
DIRECTION = "source-to-target"

Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]


def _vec3(value: Any, name: str) -> Vec3:
    items = tuple(float(x) for x in value)
    if len(items) != 3 or not all(math.isfinite(x) for x in items):
        raise ValueError(f"{name} must be three finite numbers")
    return items  # type: ignore[return-value]


def _fraction(value: Any, name: str) -> float:
    number = float(value)
    if not (0.0 <= number <= 1.0):  # also rejects NaN
        raise ValueError(f"{name} must be in [0, 1], not {value!r}")
    return number


@dataclass(frozen=True, slots=True)
class PortalPlacement:
    """Where an end sits on its owner's boundary, in the OWNER'S LOCAL frame.

    Living-map counterparts: ``center``, ``normal``, ``tangent``,
    ``bitangent``, ``radius``.  Give ``face`` (a face name such as "north")
    or ``normal`` (outward, unit); at least one.  Units: metres.
    ``extent`` is the (u, v) half-extent along (tangent, bitangent) of the
    aperture; ``radius`` is the living map's round aperture radius.
    """

    face: str | None = None
    normal: Vec3 | None = None
    tangent: Vec3 | None = None
    bitangent: Vec3 | None = None
    center: Vec3 | None = None
    extent: tuple[float, float] | None = None
    radius: float | None = None

    def __post_init__(self) -> None:
        if self.face is None and self.normal is None:
            raise ValueError("a placement needs a face name or an outward normal")
        for name in ("normal", "tangent", "bitangent", "center"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _vec3(value, name))
        if self.normal is not None and abs(math.hypot(*self.normal) - 1.0) > 1e-6:
            raise ValueError("placement normal must be a unit vector")
        if self.extent is not None:
            ext = tuple(float(x) for x in self.extent)
            if len(ext) != 2 or not all(math.isfinite(x) and x > 0 for x in ext):
                raise ValueError("placement extent must be two positive numbers")
            object.__setattr__(self, "extent", ext)
        if self.radius is not None and not (math.isfinite(self.radius) and self.radius > 0):
            raise ValueError("placement radius must be positive")

    def to_data(self) -> dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in (
            ("face", self.face), ("normal", self.normal), ("tangent", self.tangent),
            ("bitangent", self.bitangent), ("center", self.center),
            ("extent", self.extent), ("radius", self.radius))}

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "PortalPlacement":
        return cls(**{k: data.get(k) for k in (
            "face", "normal", "tangent", "bitangent", "center", "extent", "radius")})


@dataclass(frozen=True, slots=True)
class PortalFrameMap:
    """Rigid transform taking the OUT-face frame to the IN-face frame.

    ``p_to = target_anchor + rotation @ (p_from - source_anchor)`` and
    ``v_to = rotation @ v_from``.  Same law as the living map's traversal
    (abstract_ui_physics.py, source-relative rotation then target translation).
    Metres; ``rotation`` is row-major, orthonormal, determinant +1.
    """

    source_anchor: Vec3
    target_anchor: Vec3
    rotation: Mat3

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_anchor", _vec3(self.source_anchor, "source_anchor"))
        object.__setattr__(self, "target_anchor", _vec3(self.target_anchor, "target_anchor"))
        rows = tuple(_vec3(r, "rotation row") for r in self.rotation)
        if len(rows) != 3:
            raise ValueError("rotation must be 3x3")
        for i in range(3):
            for j in range(3):
                dot = sum(rows[i][k] * rows[j][k] for k in range(3))
                if abs(dot - (1.0 if i == j else 0.0)) > 1e-9:
                    raise ValueError("rotation must be orthonormal")
        det = (rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
               - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
               + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0]))
        if abs(det - 1.0) > 1e-9:
            raise ValueError("rotation must be proper (determinant +1)")
        object.__setattr__(self, "rotation", rows)

    @classmethod
    def from_yaw(cls, source_anchor, target_anchor, cos: float, sin: float) -> "PortalFrameMap":
        """The living map's parameterisation: yaw about +y (portal_cos, portal_sin)."""
        return cls(source_anchor, target_anchor,
                   ((cos, 0.0, -sin), (0.0, 1.0, 0.0), (sin, 0.0, cos)))

    def to_data(self) -> dict[str, Any]:
        return {"source_anchor": list(self.source_anchor),
                "target_anchor": list(self.target_anchor),
                "rotation": [list(r) for r in self.rotation]}

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "PortalFrameMap":
        return cls(data["source_anchor"], data["target_anchor"], data["rotation"])


@dataclass(frozen=True, slots=True)
class PortalTransmission:
    """Declared slot for what crosses a portal end.  Every field is optional.

    spectral:        tuple of (wavelength_m, fraction) samples; wavelengths in
                     metres, strictly increasing and positive; fraction is
                     power transmittance in [0, 1].
    scalar:          grey power transmittance in [0, 1]; exclusive with spectral.
    refractive_index: (n_before, n_after), dimensionless, > 0: the medium on the
                     owner side of the end and the medium beyond it.
    pass_probability: probability in [0, 1] that a particle presented to the
                     end crosses it.
    maps_velocity:   True if a crossing particle's velocity is carried through
                     the pair's frame map (rotation only); False = position
                     only; None = undecided.
    thermal, acoustic: named, left None; their units are not decided here.
    """

    spectral: tuple[tuple[float, float], ...] | None = None
    scalar: float | None = None
    refractive_index: tuple[float, float] | None = None
    pass_probability: float | None = None
    maps_velocity: bool | None = None
    thermal: Mapping[str, Any] | None = None
    acoustic: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.spectral is not None and self.scalar is not None:
            raise ValueError("give spectral or scalar transmittance, not both")
        if self.spectral is not None:
            samples = tuple((float(w), _fraction(f, "spectral fraction")) for w, f in self.spectral)
            if not samples:
                raise ValueError("spectral samples must not be empty")
            if any(not (math.isfinite(w) and w > 0) for w, _ in samples):
                raise ValueError("wavelengths must be positive metres")
            if any(b[0] <= a[0] for a, b in zip(samples, samples[1:])):
                raise ValueError("wavelengths must be strictly increasing")
            object.__setattr__(self, "spectral", samples)
        if self.scalar is not None:
            object.__setattr__(self, "scalar", _fraction(self.scalar, "scalar transmittance"))
        if self.refractive_index is not None:
            pair = tuple(float(n) for n in self.refractive_index)
            if len(pair) != 2 or not all(math.isfinite(n) and n > 0 for n in pair):
                raise ValueError("refractive_index must be two positive numbers")
            object.__setattr__(self, "refractive_index", pair)
        if self.pass_probability is not None:
            object.__setattr__(self, "pass_probability",
                               _fraction(self.pass_probability, "pass_probability"))

    def to_data(self) -> dict[str, Any]:
        return {
            "spectral": None if self.spectral is None else [list(s) for s in self.spectral],
            "scalar": self.scalar,
            "refractive_index": None if self.refractive_index is None else list(self.refractive_index),
            "pass_probability": self.pass_probability,
            "maps_velocity": self.maps_velocity,
            "thermal": None if self.thermal is None else dict(self.thermal),
            "acoustic": None if self.acoustic is None else dict(self.acoustic),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "PortalTransmission":
        spectral = data.get("spectral")
        return cls(
            spectral=None if spectral is None else tuple(tuple(s) for s in spectral),
            scalar=data.get("scalar"),
            refractive_index=data.get("refractive_index"),
            pass_probability=data.get("pass_probability"),
            maps_velocity=data.get("maps_velocity"),
            thermal=data.get("thermal"), acoustic=data.get("acoustic"))


@dataclass(frozen=True, slots=True)
class PortalProvenance:
    """Where the pair came from.  New relative to the living map (the scan needs it)."""

    kind: str                      # e.g. "python-import", "es-module-import", "call", "placed"
    specifier: str | None = None   # the text the source used to name the target
    source_line: int | None = None
    origin: str | None = None      # any other origin: tool, recipe, graph identity

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("provenance needs a kind")
        if self.source_line is not None and self.source_line < 1:
            raise ValueError("source_line is 1-based")

    def to_data(self) -> dict[str, Any]:
        return {"kind": self.kind, "specifier": self.specifier,
                "source_line": self.source_line, "origin": self.origin}

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "PortalProvenance":
        return cls(data["kind"], data.get("specifier"), data.get("source_line"), data.get("origin"))


@dataclass(frozen=True, slots=True)
class PortalEnd:
    """One end of a portal pair, attached to the boundary of one graph node.

    Living-map counterparts: identity<-splat.identity, role<-port_role,
    owner<-object_identity, aperture_class/aperture_scale<-same names,
    placement<-centre/normal/tangent/bitangent/radius.  ``pair`` is new on
    the end (the living map joins ends only through graph edges).
    """

    identity: str
    owner: str
    role: str
    pair: str
    placement: PortalPlacement | None = None
    transmission: PortalTransmission | None = None
    aperture_class: str | None = None
    aperture_scale: float | None = None

    def __post_init__(self) -> None:
        if not self.identity or not self.owner or not self.pair:
            raise ValueError("a portal end needs identity, owner and pair")
        if self.role not in ROLES:
            raise ValueError(f"portal end role must be 'in' or 'out', not {self.role!r}")
        if self.aperture_scale is not None and not (self.aperture_scale > 0):
            raise ValueError("aperture_scale must be positive")

    def to_data(self) -> dict[str, Any]:
        return {
            "identity": self.identity, "owner": self.owner, "role": self.role,
            "pair": self.pair,
            "placement": None if self.placement is None else self.placement.to_data(),
            "transmission": None if self.transmission is None else self.transmission.to_data(),
            "aperture_class": self.aperture_class, "aperture_scale": self.aperture_scale,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "PortalEnd":
        placement, transmission = data.get("placement"), data.get("transmission")
        return cls(
            data["identity"], data["owner"], data["role"], data["pair"],
            None if placement is None else PortalPlacement.from_data(placement),
            None if transmission is None else PortalTransmission.from_data(transmission),
            data.get("aperture_class"), data.get("aperture_scale"))


@dataclass(frozen=True, slots=True)
class PortalPair:
    """A directed edge of a graph, realised as two ends.

    Living-map counterparts: identity<-edge.identity, weight<-edge.weight
    (probability among the in-end's out-edges), path_model<-edge.path_model.
    ``frame_map`` stays None until both ends are placed.
    """

    identity: str
    source: PortalEnd
    target: PortalEnd
    provenance: PortalProvenance
    direction: str = DIRECTION
    frame_map: PortalFrameMap | None = None
    weight: float | None = None
    path_model: str | None = None

    def __post_init__(self) -> None:
        if not self.identity:
            raise ValueError("a portal pair needs an identity")
        if self.direction != DIRECTION:
            raise ValueError(f"direction must be {DIRECTION!r}")
        if self.source.role == self.target.role:
            raise ValueError("a pair needs one 'in' end and one 'out' end")
        if self.source.pair != self.identity or self.target.pair != self.identity:
            raise ValueError("both ends must name this pair")
        if self.source.owner == self.target.owner:
            raise ValueError("a portal needs two distinct owners")
        if self.frame_map is not None and (self.source.placement is None
                                           or self.target.placement is None):
            raise ValueError("a frame map needs both ends placed")
        if self.weight is not None:
            object.__setattr__(self, "weight", _fraction(self.weight, "weight"))

    @property
    def ends(self) -> tuple[PortalEnd, PortalEnd]:
        return (self.source, self.target)

    def to_data(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA, "identity": self.identity, "direction": self.direction,
            "provenance": self.provenance.to_data(),
            "source": self.source.to_data(), "target": self.target.to_data(),
            "frame_map": None if self.frame_map is None else self.frame_map.to_data(),
            "weight": self.weight, "path_model": self.path_model,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "PortalPair":
        if data.get("schema") != SCHEMA:
            raise ValueError(f"expected schema {SCHEMA!r}, got {data.get('schema')!r}")
        frame_map = data.get("frame_map")
        return cls(
            data["identity"], PortalEnd.from_data(data["source"]),
            PortalEnd.from_data(data["target"]),
            PortalProvenance.from_data(data["provenance"]),
            data.get("direction", DIRECTION),
            None if frame_map is None else PortalFrameMap.from_data(frame_map),
            data.get("weight"), data.get("path_model"))


def from_graph_edge(source_id: str, target_id: str, *, kind: str,
                    provenance: PortalProvenance | Mapping[str, Any] | None = None,
                    identity: str | None = None) -> PortalPair:
    """Build a pair from one directed reference edge of any graph.

    ``source_id`` and ``target_id`` are node identities (the owners).  The
    source node gets the ``out`` end, the target node the ``in`` end, as the
    repository scan labels them.  Placement and transmission stay None.
    """
    if isinstance(provenance, PortalProvenance):
        prov = provenance
    else:
        extra = dict(provenance or {})
        prov = PortalProvenance(kind, extra.get("specifier"), extra.get("source_line"),
                                extra.get("origin"))
    if source_id == target_id:
        raise ValueError("a portal needs two distinct nodes")
    if identity is None:
        line = "" if prov.source_line is None else f":{prov.source_line}"
        identity = f"portal:{prov.kind}:{source_id}{line}->{target_id}"
    return PortalPair(
        identity,
        PortalEnd(f"{identity}/end:out", source_id, "out", identity),
        PortalEnd(f"{identity}/end:in", target_id, "in", identity),
        prov)


def from_repo_scan_pair(pair: Any) -> PortalPair:
    """Convert a ``RepoPortalPair`` (duck-typed; the scanner is not imported)."""
    def end(raw: Any) -> PortalEnd:
        tx = raw.transmission
        return PortalEnd(raw.identity, raw.room, raw.role, raw.pair, None,
                         None if tx is None else PortalTransmission.from_data(tx))
    return PortalPair(
        pair.identity, end(pair.out_end), end(pair.in_end),
        PortalProvenance(pair.reference_kind, pair.specifier, pair.source_line))


def ends_by_owner(pairs: Iterable[PortalPair]) -> dict[str, tuple[PortalEnd, ...]]:
    """Group ends by owner; owners and ends sorted so input order never matters."""
    grouped: dict[str, list[PortalEnd]] = {}
    for pair in pairs:
        for item in pair.ends:
            grouped.setdefault(item.owner, []).append(item)
    return {owner: tuple(sorted(grouped[owner], key=lambda e: e.identity))
            for owner in sorted(grouped)}


__all__ = [
    "SCHEMA", "PortalPlacement", "PortalFrameMap", "PortalTransmission",
    "PortalProvenance", "PortalEnd", "PortalPair", "from_graph_edge",
    "from_repo_scan_pair", "ends_by_owner",
]
