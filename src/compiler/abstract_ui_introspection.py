"""Turn Python or SSA-described classes into an introspective AbstractUI map.

The prototype is deliberately whimsical at the presentation edge and strict at
the identity edge.  A module becomes a region, a class becomes an enterable
building, and its declared members become rooms.  Room metaphors are selected
from configurable collections by a stable hash, so a different palette may
tell a different spatial story without changing program identity or authored
member order.

No class is instantiated and no descriptor is invoked.  Python inspection is
limited to class dictionaries, annotations, signatures, bases, and nested type
objects.  SSA-shaped :class:`ClassEmission` records are accepted by structure
and produce the same map records plus explicit SSA-intent code receipts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import inspect
import math
from typing import Any, Iterable, Mapping, Sequence

from . import abstract_ui_vocabulary as words
from .abstract_ui_vocabulary import UIIntention


ABSTRACT_UI_INTROSPECTION_VERSION = "abstract-ui-introspection-v0"


DEFAULT_ROOM_PALETTE: Mapping[str, tuple[str, ...]] = {
    "field": (
        "cabinet", "reservoir", "archive shelf", "specimen room",
        "store room", "memory garden",
    ),
    "property": (
        "observation deck", "gallery window", "reading room",
        "instrument alcove", "mirror chamber",
    ),
    "method": (
        "workshop", "control room", "forge", "laboratory",
        "conversation chamber", "machine room",
    ),
    "constructor": (
        "arrival hall", "assembly room", "foundry gate",
        "reception chamber", "birth room",
    ),
    "nested_class": (
        "interior building", "annex entrance", "model room",
        "courtyard portal", "nested archive",
    ),
}

DEFAULT_BUILDING_PALETTE: tuple[str, ...] = (
    "archive", "workshop", "observatory", "library", "station",
    "laboratory", "guild hall", "machine house", "garden house",
)


@dataclass(frozen=True, slots=True)
class GridPosition:
    column: int
    row: int
    width: int = 1
    height: int = 1


@dataclass(frozen=True, slots=True)
class ImpliedCode:
    """Honest code-shaped meaning behind one spatial affordance."""

    operation: str
    dialect: str
    source: str
    executable: bool
    explanation: str


@dataclass(frozen=True, slots=True)
class AbstractUIRoom:
    identity: str
    name: str
    member_kind: str
    metaphor: str
    position: GridPosition
    type_name: str | None = None
    parameters: tuple[str, ...] = ()
    intentions: tuple[UIIntention, ...] = ()
    implied_code: tuple[ImpliedCode, ...] = ()


@dataclass(frozen=True, slots=True)
class AbstractUIBuilding:
    identity: str
    name: str
    module: str
    source_kind: str
    metaphor: str
    position: GridPosition
    rooms: tuple[AbstractUIRoom, ...]
    intention: UIIntention

    def room(self, name: str) -> AbstractUIRoom:
        for candidate in self.rooms:
            if candidate.name == name:
                return candidate
        raise KeyError(name)


@dataclass(frozen=True, slots=True)
class AbstractUIRegion:
    identity: str
    name: str
    position: GridPosition
    buildings: tuple[AbstractUIBuilding, ...]
    intention: UIIntention


@dataclass(frozen=True, slots=True)
class AbstractUITrack:
    source: str
    target: str
    relationship: str
    label: str
    intention: UIIntention


@dataclass(frozen=True, slots=True)
class AbstractUIWorld:
    identity: str
    regions: tuple[AbstractUIRegion, ...]
    tracks: tuple[AbstractUITrack, ...]
    intention: UIIntention
    schema_version: str = ABSTRACT_UI_INTROSPECTION_VERSION

    def buildings(self) -> tuple[AbstractUIBuilding, ...]:
        return tuple(
            building
            for region in self.regions
            for building in region.buildings
        )

    def building(self, identity_or_name: str) -> AbstractUIBuilding:
        for candidate in self.buildings():
            if candidate.identity == identity_or_name or candidate.name == identity_or_name:
                return candidate
        raise KeyError(identity_or_name)

    def objects(self) -> tuple[Any, ...]:
        """Return every instantiated map record in containment order."""

        return (
            self,
            *(
                item
                for region in self.regions
                for item in (
                    region,
                    *(
                        nested
                        for building in region.buildings
                        for nested in (building, *building.rooms)
                    ),
                )
            ),
            *self.tracks,
        )


@dataclass(frozen=True, slots=True)
class _MemberDescription:
    name: str
    kind: str
    type_name: str | None = None
    parameters: tuple[str, ...] = ()
    python_expression: str | None = None
    ssa_expression: str | None = None
    nested_type: type | None = None


@dataclass(frozen=True, slots=True)
class _ClassDescription:
    identity: str
    name: str
    module: str
    source_kind: str
    members: tuple[_MemberDescription, ...]
    source: Any


@dataclass(frozen=True, slots=True)
class _ClassRelationship:
    source: str
    target: str
    kind: str
    label: str


def _stable_choice(
    choices: Sequence[str],
    *,
    seed: str,
    identity: str,
) -> str:
    if not choices:
        raise ValueError("metaphor collection cannot be empty")
    digest = hashlib.sha256(f"{seed}\0{identity}".encode("utf-8")).digest()
    return str(choices[int.from_bytes(digest[:8], "big") % len(choices)])


def _grid_positions(count: int, *, aspect: float = 1.35) -> tuple[GridPosition, ...]:
    """Return a compact deterministic row-major grid."""

    if count < 0:
        raise ValueError("grid count must be non-negative")
    if count == 0:
        return ()
    if aspect <= 0:
        raise ValueError("grid aspect must be positive")
    columns = max(1, math.ceil(math.sqrt(count * aspect)))
    return tuple(
        GridPosition(index % columns, index // columns)
        for index in range(count)
    )


def _annotation_name(annotation: Any) -> str | None:
    if annotation is None:
        return None
    if isinstance(annotation, str):
        return annotation
    return getattr(annotation, "__qualname__", None) or str(annotation)


def _signature_parameters(value: Any) -> tuple[str, ...]:
    try:
        parameters = tuple(inspect.signature(value).parameters.values())
    except (TypeError, ValueError):
        return ()
    return tuple(
        parameter.name
        for parameter in parameters
        if parameter.name not in {"self", "cls"}
    )


def _python_class_identity(cls: type) -> str:
    return f"python:{cls.__module__}.{cls.__qualname__}"


def _describe_python_class(cls: type) -> _ClassDescription:
    identity = _python_class_identity(cls)
    namespace = vars(cls)
    annotations = dict(namespace.get("__annotations__", {}) or {})
    members: list[_MemberDescription] = []
    seen: set[str] = set()

    for name, annotation in annotations.items():
        raw = namespace.get(name)
        if isinstance(raw, (staticmethod, classmethod, property)) or inspect.isfunction(raw):
            continue
        nested = annotation if isinstance(annotation, type) else None
        members.append(_MemberDescription(
            str(name),
            "nested_class" if nested is not None and nested.__module__ != "builtins" else "field",
            _annotation_name(annotation),
            python_expression=f"instance.{name}",
            nested_type=(
                nested if nested is not None and nested.__module__ != "builtins" else None
            ),
        ))
        seen.add(str(name))

    for name, raw in namespace.items():
        if name in seen:
            continue
        if name in {"__init__", "__new__"}:
            function = raw.__func__ if isinstance(raw, (staticmethod, classmethod)) else raw
            members.append(_MemberDescription(
                name,
                "constructor",
                parameters=_signature_parameters(function),
                python_expression=(
                    f"{cls.__qualname__}("
                    f"{', '.join(_signature_parameters(function))})"
                ),
            ))
            seen.add(name)
            continue
        if name.startswith("_"):
            continue
        if isinstance(raw, type):
            members.append(_MemberDescription(
                name, "nested_class", raw.__qualname__,
                python_expression=f"{cls.__qualname__}.{name}",
                nested_type=raw,
            ))
        elif isinstance(raw, property):
            members.append(_MemberDescription(
                name, "property", _annotation_name(
                    None if raw.fget is None else raw.fget.__annotations__.get("return")
                ), python_expression=f"instance.{name}",
            ))
        elif isinstance(raw, (staticmethod, classmethod)):
            function = raw.__func__
            parameters = _signature_parameters(function)
            receiver = cls.__qualname__ if isinstance(raw, classmethod) else cls.__qualname__
            members.append(_MemberDescription(
                name, "method", parameters=parameters,
                python_expression=f"{receiver}.{name}({', '.join(parameters)})",
            ))
        elif inspect.isfunction(raw) or inspect.ismethoddescriptor(raw):
            parameters = _signature_parameters(raw)
            members.append(_MemberDescription(
                name, "method", parameters=parameters,
                python_expression=f"instance.{name}({', '.join(parameters)})",
            ))
        else:
            members.append(_MemberDescription(
                name, "field", _annotation_name(type(raw)),
                python_expression=f"instance.{name}",
            ))
        seen.add(name)

    return _ClassDescription(
        identity, cls.__name__, cls.__module__, "python", tuple(members), cls,
    )


def _looks_like_class_emission(value: Any) -> bool:
    return (
        not isinstance(value, type)
        and isinstance(getattr(value, "identity", None), str)
        and isinstance(getattr(value, "fields", None), tuple)
        and isinstance(getattr(value, "methods", None), tuple)
    )


def _describe_ssa_class(value: Any) -> _ClassDescription:
    identity = f"ssa:{value.identity}"
    members = [
        _MemberDescription(
            str(field.name), "field", str(getattr(field, "type_name", "unknown")),
            ssa_expression=(
                f"%value = load_field %receiver, slot {int(field.slot)}"
            ),
        )
        for field in value.fields
    ]
    for method in value.methods:
        parameters = tuple(str(item.name) for item in method.parameters)
        arguments = [f"%{name}" for name in parameters]
        if not bool(method.is_static):
            arguments.insert(0, "%receiver")
        members.append(_MemberDescription(
            str(method.name),
            "constructor" if str(method.kind) in {"allocator", "initializer"} else "method",
            parameters=parameters,
            ssa_expression=(
                f"%result = call @{int(method.function_reference)}("
                f"{', '.join(arguments)})"
            ),
        ))
    module = str(value.identity).rpartition(".")[0] or "ssa"
    name = str(value.identity).rpartition(".")[2]
    return _ClassDescription(
        identity, name, module, "ssa", tuple(members), value,
    )


def _member_intentions(
    class_identity: str,
    member: _MemberDescription,
    metaphor: str,
    adornments: Mapping[str, Sequence[UIIntention]],
) -> tuple[UIIntention, ...]:
    identity = f"{class_identity}/member:{member.kind}:{member.name}"
    base = words.room(
        member.name,
        words.world_role(metaphor),
        words.identity(identity),
        words.describe(
            words.program_construct(member.kind),
            words.label(member.name),
        ),
    )
    if member.kind in {"method", "constructor"}:
        affordance = words.affordance(
            words.invoke(member.name, *member.parameters),
            words.action_binding(identity),
        )
    elif member.kind == "nested_class":
        affordance = words.portal(
            words.navigate(member.name),
            words.identity_binding(identity),
        )
    else:
        affordance = words.inspect_(
            words.value_binding(identity),
            words.describe_state(member.name),
        )
    extras = tuple(adornments.get(member.name, ()))
    return (base, affordance, *extras)


def _implied_code(member: _MemberDescription) -> tuple[ImpliedCode, ...]:
    records = []
    if member.python_expression is not None:
        records.append(ImpliedCode(
            "invoke" if member.kind in {"method", "constructor"} else "inspect",
            "python-expression",
            member.python_expression,
            True,
            "Direct source-level expression implied by the room affordance.",
        ))
    if member.ssa_expression is not None:
        records.append(ImpliedCode(
            "invoke" if member.kind in {"method", "constructor"} else "inspect",
            "repository-ssa-intent",
            member.ssa_expression,
            False,
            "Readable SSA-shaped intent; final lowering still owns concrete values and types.",
        ))
    return tuple(records)


def _building_from_description(
    description: _ClassDescription,
    *,
    position: GridPosition,
    seed: str,
    room_palette: Mapping[str, Sequence[str]],
    building_palette: Sequence[str],
    adornments: Mapping[str, Sequence[UIIntention]],
) -> AbstractUIBuilding:
    room_positions = _grid_positions(len(description.members))
    rooms = []
    for member, room_position in zip(description.members, room_positions):
        room_identity = (
            f"{description.identity}/member:{member.kind}:{member.name}"
        )
        choices = room_palette.get(member.kind) or room_palette.get("field") or ()
        metaphor = _stable_choice(
            choices, seed=seed, identity=room_identity,
        )
        rooms.append(AbstractUIRoom(
            room_identity,
            member.name,
            member.kind,
            metaphor,
            room_position,
            member.type_name,
            member.parameters,
            _member_intentions(
                description.identity, member, metaphor, adornments,
            ),
            _implied_code(member),
        ))
    building_metaphor = _stable_choice(
        building_palette, seed=seed, identity=description.identity,
    )
    building_intention = words.building(
        description.name,
        *(room.intentions[0] for room in rooms),
        words.identity(description.identity),
        source_kind=description.source_kind,
        metaphor=building_metaphor,
    )
    return AbstractUIBuilding(
        description.identity,
        description.name,
        description.module,
        description.source_kind,
        building_metaphor,
        position,
        tuple(rooms),
        building_intention,
    )


def _python_descriptions(
    root: type,
    *,
    depth_up: int,
    depth_down: int,
) -> tuple[tuple[_ClassDescription, ...], tuple[_ClassRelationship, ...]]:
    descriptions: list[_ClassDescription] = []
    relationships: list[_ClassRelationship] = []
    seen: set[str] = set()

    def visit(cls: type, up: int, down: int) -> str:
        identity = _python_class_identity(cls)
        if identity not in seen:
            seen.add(identity)
            description = _describe_python_class(cls)
            descriptions.append(description)
        else:
            description = next(item for item in descriptions if item.identity == identity)

        if up > 0:
            for base in cls.__bases__:
                if base is object:
                    continue
                target = _python_class_identity(base)
                relationships.append(_ClassRelationship(
                    identity, target, "inherits", f"inherits from {base.__name__}",
                ))
                visit(base, up - 1, 0)
        if down > 0:
            for member in description.members:
                nested = member.nested_type
                if nested is None or nested is cls:
                    continue
                target = _python_class_identity(nested)
                relationships.append(_ClassRelationship(
                    identity, target, "contains-type",
                    f"{member.name} leads to {nested.__name__}",
                ))
                visit(nested, 0, down - 1)
        return identity

    visit(root, depth_up, depth_down)
    unique_relationships = tuple(dict.fromkeys(relationships))
    return tuple(descriptions), unique_relationships


def build_introspective_world(
    subject: type | Any,
    *,
    depth_up: int = 0,
    depth_down: int = 0,
    seed: str = "abstract-ui",
    room_palette: Mapping[str, Sequence[str]] | None = None,
    building_palette: Sequence[str] = DEFAULT_BUILDING_PALETTE,
    adornments: Mapping[str, Sequence[UIIntention]] | None = None,
) -> AbstractUIWorld:
    """Build a deterministic spatial story for a Python or SSA class.

    ``depth_up`` follows Python base classes. ``depth_down`` follows nested
    classes and non-builtin class-valued annotations. SSA class-emission
    records currently describe one class at a time and therefore reject
    non-zero recursion rather than inventing unavailable relationships.
    """

    if depth_up < 0 or depth_down < 0:
        raise ValueError("introspection depths must be non-negative")
    active_room_palette = room_palette or DEFAULT_ROOM_PALETTE
    active_adornments = adornments or {}

    if isinstance(subject, type):
        descriptions, relationships = _python_descriptions(
            subject, depth_up=depth_up, depth_down=depth_down,
        )
    elif _looks_like_class_emission(subject):
        if depth_up or depth_down:
            raise ValueError(
                "SSA ClassEmission recursion requires a correlated class plan"
            )
        descriptions = (_describe_ssa_class(subject),)
        relationships = ()
    else:
        raise TypeError(
            "subject must be a Python class or SSA-shaped ClassEmission"
        )

    descriptions_by_module: dict[str, list[_ClassDescription]] = {}
    for description in descriptions:
        descriptions_by_module.setdefault(description.module, []).append(description)

    region_positions = _grid_positions(len(descriptions_by_module), aspect=1.6)
    regions = []
    building_by_identity: dict[str, AbstractUIBuilding] = {}
    for (module, module_descriptions), region_position in zip(
        descriptions_by_module.items(), region_positions,
    ):
        building_positions = _grid_positions(len(module_descriptions))
        buildings = tuple(
            _building_from_description(
                description,
                position=building_position,
                seed=seed,
                room_palette=active_room_palette,
                building_palette=building_palette,
                adornments=active_adornments,
            )
            for description, building_position in zip(
                module_descriptions, building_positions,
            )
        )
        building_by_identity.update(
            (building.identity, building) for building in buildings
        )
        region_identity = f"region:{module}"
        regions.append(AbstractUIRegion(
            region_identity,
            module,
            region_position,
            buildings,
            words.region(
                module,
                *(building.intention for building in buildings),
                words.identity(region_identity),
            ),
        ))

    tracks = []
    for relationship in relationships:
        if relationship.source not in building_by_identity or relationship.target not in building_by_identity:
            continue
        tracks.append(AbstractUITrack(
            relationship.source,
            relationship.target,
            relationship.kind,
            relationship.label,
            words.track(
                words.identity(relationship.source),
                words.identity(relationship.target),
                relationship=relationship.kind,
                label=relationship.label,
            ),
        ))

    root_name = descriptions[0].name
    world_identity = f"introspection-world:{descriptions[0].identity}"
    world_intention = words.world(
        *(region.intention for region in regions),
        words.identity(world_identity),
        words.describe(
            f"An introspective world for {root_name}",
            words.accessibility_route("prosaic class map"),
        ),
    )
    return AbstractUIWorld(
        world_identity, tuple(regions), tuple(tracks), world_intention,
    )


__all__ = [
    "ABSTRACT_UI_INTROSPECTION_VERSION",
    "DEFAULT_BUILDING_PALETTE",
    "DEFAULT_ROOM_PALETTE",
    "AbstractUIBuilding",
    "AbstractUIRegion",
    "AbstractUIRoom",
    "AbstractUITrack",
    "AbstractUIWorld",
    "GridPosition",
    "ImpliedCode",
    "build_introspective_world",
]
