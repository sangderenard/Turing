"""AbstractUI actors, NPCs, organizations, and the isolated entity cycle.

Entities occupy a mezzanine below the system root and above backend-specific
timekeeping, presentation, and action execution.  An entity archetype supplies
embodiment and capabilities; a controller binding supplies agency.  A native
pointer and an NPC follower can therefore be spawned identically while reading
different control sources.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any, Mapping, Sequence


ABSTRACT_UI_ENTITY_VERSION = "abstract-ui-entity-v0"
ENTITY_CYCLE_PHASES = ("control-input", "integration", "interaction", "presentation")


def _freeze_mapping(value: Mapping[str, Any] | None) -> tuple[tuple[str, Any], ...]:
    return tuple((str(key), item) for key, item in (value or {}).items())


@dataclass(frozen=True, slots=True)
class EntityGeometry:
    kind: str
    parameters: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class EntityTexture:
    kind: str
    reference: str
    parameters: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class EntityArchetype:
    name: str
    geometry: EntityGeometry
    texture: EntityTexture
    capabilities: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EntityController:
    """Data-only binding between an entity and a control source."""

    kind: str
    source: str
    parameters: tuple[tuple[str, Any], ...] = ()

    def parameter(self, name: str, default: Any = None) -> Any:
        return dict(self.parameters).get(name, default)


@dataclass(frozen=True, slots=True)
class EntityPose:
    coordinate_space: str
    position: tuple[float, float, float]
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    acceleration: tuple[float, float, float] = (0.0, 0.0, 0.0)
    jerk: tuple[float, float, float] = (0.0, 0.0, 0.0)
    facing: tuple[float, float, float] = (0.0, -1.0, 0.0)


@dataclass(frozen=True, slots=True)
class AbstractUIEntity:
    identity: str
    archetype: EntityArchetype
    controller: EntityController
    pose: EntityPose
    principal: str | None = None
    traits: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class EntityOrganization:
    """A corral administered as one conceptual collection."""

    identity: str
    name: str
    members: tuple[str, ...] = ()
    cycle: str = "entities.default-cycle"

    def include(self, entity_identity: str) -> "EntityOrganization":
        if entity_identity in self.members:
            raise ValueError(f"entity already belongs to {self.identity}: {entity_identity}")
        return replace(self, members=(*self.members, entity_identity))


@dataclass(frozen=True, slots=True)
class EntitySpawnEdit:
    action: str
    entity: str
    organization: str
    before_revision: int
    after_revision: int


@dataclass(frozen=True, slots=True)
class EntitySpawn:
    mezzanine: "EntityMezzanine"
    entity: AbstractUIEntity
    edit: EntitySpawnEdit


@dataclass(frozen=True, slots=True)
class EntityMezzanine:
    """Entity namespace placed directly beneath one system root."""

    identity: str
    system_root: str
    revision: int = 0
    entities: tuple[AbstractUIEntity, ...] = ()
    organizations: tuple[EntityOrganization, ...] = ()

    @staticmethod
    def under(system_root: str = "system-root") -> "EntityMezzanine":
        root = str(system_root)
        return EntityMezzanine(f"{root}/entities", root)

    def entity(self, identity: str) -> AbstractUIEntity:
        for entity in self.entities:
            if entity.identity == identity:
                return entity
        raise KeyError(identity)

    def organization(self, identity: str) -> EntityOrganization:
        for organization in self.organizations:
            if organization.identity == identity:
                return organization
        raise KeyError(identity)

    def with_organization(self, name: str, *, cycle: str = "entities.default-cycle") -> "EntityMezzanine":
        identity = f"{self.identity}/organizations/{name}"
        if any(item.identity == identity for item in self.organizations):
            raise ValueError(f"entity organization already exists: {identity}")
        return replace(
            self,
            revision=self.revision + 1,
            organizations=(*self.organizations, EntityOrganization(identity, name, (), cycle)),
        )

    def spawn(
        self,
        archetype: EntityArchetype,
        *,
        name: str,
        controller: EntityController,
        organization: str,
        pose: EntityPose | None = None,
        principal: str | None = None,
        traits: Mapping[str, Any] | None = None,
    ) -> EntitySpawn:
        identity = f"{self.identity}/{name}"
        if any(entity.identity == identity for entity in self.entities):
            raise ValueError(f"entity already exists: {identity}")
        group = self.organization(organization)
        entity = AbstractUIEntity(
            identity,
            archetype,
            controller,
            pose or EntityPose("viewport", (0.0, 0.0, 0.0)),
            principal,
            _freeze_mapping(traits),
        )
        updated_group = group.include(identity)
        organizations = tuple(
            updated_group if item.identity == organization else item
            for item in self.organizations
        )
        after = self.revision + 1
        mezzanine = replace(
            self,
            revision=after,
            entities=(*self.entities, entity),
            organizations=organizations,
        )
        return EntitySpawn(
            mezzanine,
            entity,
            EntitySpawnEdit("spawn-entity", identity, organization, self.revision, after),
        )


@dataclass(frozen=True, slots=True)
class ControlInputFrame:
    controller: str
    sequence: int
    time: float
    position: tuple[float, float, float]
    coordinate_space: str = "viewport"
    buttons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EntityInteraction:
    actor: str
    type: str
    destination: str
    payload: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class EntityTick:
    sequence: int
    time: float
    delta: float


@dataclass(frozen=True, slots=True)
class EntityPresentation:
    entity: str
    geometry: EntityGeometry
    texture: EntityTexture
    pose: EntityPose


@dataclass(frozen=True, slots=True)
class EntityCyclePolicy:
    """Placement is a deployment choice; phase semantics remain identical."""

    execution: str = "inline"
    phases: tuple[str, ...] = ENTITY_CYCLE_PHASES

    def __post_init__(self) -> None:
        if self.execution not in {"inline", "worker"}:
            raise ValueError("entity cycle execution must be 'inline' or 'worker'")
        if self.phases != ENTITY_CYCLE_PHASES:
            raise ValueError("the initial entity cycle phase order is fixed")


@dataclass(frozen=True, slots=True)
class EntityCycleResult:
    mezzanine: EntityMezzanine
    tick: EntityTick
    interactions: tuple[EntityInteraction, ...]
    presentation: tuple[EntityPresentation, ...]
    policy: EntityCyclePolicy


def _latest_inputs(frames: Sequence[ControlInputFrame]) -> dict[str, ControlInputFrame]:
    latest: dict[str, ControlInputFrame] = {}
    for frame in frames:
        current = latest.get(frame.controller)
        if current is None or (frame.sequence, frame.time) > (current.sequence, current.time):
            latest[frame.controller] = frame
    return latest


def run_entity_cycle(
    mezzanine: EntityMezzanine,
    tick: EntityTick,
    *,
    inputs: Sequence[ControlInputFrame] = (),
    interactions: Sequence[EntityInteraction] = (),
    policy: EntityCyclePolicy = EntityCyclePolicy(),
) -> EntityCycleResult:
    """Run the deterministic reference cycle for one isolated mezzanine.

    ``worker`` is currently a placement contract: a host may run this entire
    pure cycle on another thread. It intentionally does not change ordering or
    numerical results relative to the inline reference implementation.
    """

    if tick.delta < 0:
        raise ValueError("entity tick delta must be non-negative")
    by_identity = {entity.identity: entity for entity in mezzanine.entities}
    latest = _latest_inputs(inputs)

    # Phase 1: native controls publish poses.
    controlled: dict[str, AbstractUIEntity] = {}
    for entity in mezzanine.entities:
        if entity.controller.kind == "native-input":
            frame = latest.get(entity.controller.source)
            if frame is not None:
                displacement = tuple(
                    current - previous
                    for current, previous in zip(frame.position, entity.pose.position)
                )
                length = math.hypot(displacement[0], displacement[1])
                facing = (
                    (displacement[0] / length, displacement[1] / length, 0.0)
                    if length > 1e-9 else entity.pose.facing
                )
                velocity = (
                    tuple(value / tick.delta for value in displacement)
                    if tick.delta > 0 else entity.pose.velocity
                )
                controlled[entity.identity] = replace(
                    entity,
                    pose=EntityPose(
                        frame.coordinate_space, frame.position, velocity,
                        entity.pose.acceleration, entity.pose.jerk, facing,
                    ),
                )
                continue
        controlled[entity.identity] = entity

    # Phase 2: follower controllers read the completed control phase. Their
    # derivative chains implement (D + omega)^order x = omega^order target.
    integrated = dict(controlled)
    for entity in mezzanine.entities:
        orders = {
            "first-order-follow": 1,
            "second-order-follow": 2,
            "third-order-follow": 3,
            "fourth-order-follow": 4,
        }
        order = orders.get(entity.controller.kind)
        if order is None:
            continue
        target_identity = str(entity.controller.parameter("target", ""))
        if target_identity not in controlled:
            raise KeyError(f"follower controller target is missing: {target_identity}")
        target = controlled[target_identity]
        dt = tick.delta
        derivatives = [
            list(entity.pose.position), list(entity.pose.velocity),
            list(entity.pose.acceleration), list(entity.pose.jerk),
            [0.0, 0.0, 0.0],
        ]
        if order == 2 and entity.controller.parameter("stiffness") is not None:
            stiffness = float(entity.controller.parameter("stiffness"))
            damping = float(entity.controller.parameter("damping", 7.0))
            highest = [
                stiffness * (goal - position) - damping * velocity
                for goal, position, velocity in zip(
                    target.pose.position, entity.pose.position, entity.pose.velocity,
                )
            ]
        else:
            omega = float(entity.controller.parameter("frequency", 4.0))
            highest = []
            for axis in range(3):
                value = omega ** order * (
                    target.pose.position[axis] - derivatives[0][axis]
                )
                for derivative_order in range(1, order):
                    value -= (
                        math.comb(order, derivative_order)
                        * omega ** (order - derivative_order)
                        * derivatives[derivative_order][axis]
                    )
                highest.append(value)
        derivatives[order] = highest
        for derivative_order in range(order - 1, -1, -1):
            derivatives[derivative_order] = [
                value + rate * dt
                for value, rate in zip(
                    derivatives[derivative_order], derivatives[derivative_order + 1],
                )
            ]
        planar_speed = math.hypot(derivatives[1][0], derivatives[1][1])
        facing = (
            (
                derivatives[1][0] / planar_speed,
                derivatives[1][1] / planar_speed,
                0.0,
            )
            if planar_speed > 1e-9 else entity.pose.facing
        )
        integrated[entity.identity] = replace(
            entity,
            pose=EntityPose(
                target.pose.coordinate_space,
                tuple(derivatives[0]), tuple(derivatives[1]),
                tuple(derivatives[2]), tuple(derivatives[3]), facing,
            ),
        )

    ordered = tuple(integrated[entity.identity] for entity in mezzanine.entities)
    next_mezzanine = replace(mezzanine, revision=mezzanine.revision + 1, entities=ordered)

    # Phases 3 and 4: interactions remain conceptual records; presentation is
    # an immutable snapshot which graphics backends may consume asynchronously.
    for interaction in interactions:
        if interaction.actor not in by_identity:
            raise KeyError(f"interaction actor is missing: {interaction.actor}")
    presentation = tuple(
        EntityPresentation(entity.identity, entity.archetype.geometry, entity.archetype.texture, entity.pose)
        for entity in ordered
    )
    return EntityCycleResult(
        next_mezzanine, tick, tuple(interactions), presentation, policy,
    )


POINTER_ARCHETYPE = EntityArchetype(
    "pointer-being",
    EntityGeometry("point", (("hotspot", (0.0, 0.0)), ("radius", 1.75),
                             ("embodiment_scale", 0.25))),
    EntityTexture("sprite", "abstract-ui:pointer-orb"),
    ("point", "select", "inspect", "interact"),
)

PLAYER_ARCHETYPE = EntityArchetype(
    "player-being",
    EntityGeometry("point", (("hotspot", (0.0, 0.0)), ("radius", 1.75),
                             ("embodiment_scale", 0.25))),
    EntityTexture("sprite", "abstract-ui:player-orb"),
    ("navigate", "select", "inspect", "interact"),
)


def spawn_world_player(*, system_root: str = "system-root") -> EntityMezzanine:
    """Spawn one user actor whose pose belongs to the game world, not the mouse."""

    mezzanine = EntityMezzanine.under(system_root).with_organization("players")
    organization = f"{mezzanine.identity}/organizations/players"
    return mezzanine.spawn(
        PLAYER_ARCHETYPE,
        name="player.local",
        controller=EntityController("world-player", "game.controls"),
        organization=organization,
        pose=EntityPose("data-world", (0.0, 0.2875, 0.0)),
        principal="user.local",
        traits={"color": "#f4f1de", "player": True},
    ).mezzanine


def spawn_pointer_and_follower(
    *, system_root: str = "system-root",
) -> EntityMezzanine:
    """Spawn identical pointer beings with native and NPC controllers."""

    mezzanine = EntityMezzanine.under(system_root).with_organization("pointer-beings")
    organization = f"{mezzanine.identity}/organizations/pointer-beings"
    pointer = mezzanine.spawn(
        POINTER_ARCHETYPE,
        name="pointer.primary",
        controller=EntityController("native-input", "mouse.primary"),
        organization=organization,
        principal="user.local",
        traits={"color": "#f4f1de", "order": 0},
    )
    follower = pointer.mezzanine.spawn(
        POINTER_ARCHETYPE,
        name="pointer.echo",
        controller=EntityController(
            "second-order-follow",
            pointer.entity.identity,
            (("target", pointer.entity.identity), ("stiffness", 18.0), ("damping", 7.0)),
        ),
        organization=organization,
        pose=EntityPose("viewport", (24.0, 24.0, 0.0)),
        principal="system.local",
        traits={"npc": True, "color": "#ffd166", "order": 2},
    )
    return follower.mezzanine


def spawn_pointer_and_followers(
    *, system_root: str = "system-root",
) -> EntityMezzanine:
    """Spawn one native pointer and first-through-fourth-order echoes."""

    mezzanine = EntityMezzanine.under(system_root).with_organization("pointer-beings")
    organization = f"{mezzanine.identity}/organizations/pointer-beings"
    native = mezzanine.spawn(
        POINTER_ARCHETYPE,
        name="pointer.primary",
        controller=EntityController("native-input", "mouse.primary"),
        organization=organization,
        principal="user.local",
        traits={"color": "#f4f1de", "order": 0},
    )
    mezzanine = native.mezzanine
    target = native.entity.identity
    definitions = (
        ("pointer.first", "first-order-follow", 1, "#ff6b6b", 7.0, (36.0, 36.0, 0.0)),
        ("pointer.second", "second-order-follow", 2, "#ffd166", 5.5, (48.0, 48.0, 0.0)),
        ("pointer.third", "third-order-follow", 3, "#4cc9f0", 4.5, (60.0, 60.0, 0.0)),
        ("pointer.fourth", "fourth-order-follow", 4, "#c77dff", 4.0, (72.0, 72.0, 0.0)),
    )
    for name, kind, order, color, frequency, position in definitions:
        spawned = mezzanine.spawn(
            POINTER_ARCHETYPE,
            name=name,
            controller=EntityController(
                kind,
                target,
                (("target", target), ("order", order), ("frequency", frequency)),
            ),
            organization=organization,
            pose=EntityPose("viewport", position),
            principal="system.local",
            traits={"npc": True, "color": color, "order": order},
        )
        mezzanine = spawned.mezzanine
    return mezzanine


def entity_mezzanine_model(mezzanine: EntityMezzanine) -> dict[str, Any]:
    """Return the transport form used by AbstractUI backend packets."""

    return {
        "schema": ABSTRACT_UI_ENTITY_VERSION,
        "identity": mezzanine.identity,
        "system_root": mezzanine.system_root,
        "revision": mezzanine.revision,
        "organizations": [
            {
                "identity": group.identity,
                "name": group.name,
                "members": list(group.members),
                "cycle": group.cycle,
            }
            for group in mezzanine.organizations
        ],
        "entities": [
            {
                "identity": entity.identity,
                "kind": "entity",
                "name": entity.identity.rsplit("/", 1)[-1],
                "archetype": entity.archetype.name,
                "geometry": {
                    "kind": entity.archetype.geometry.kind,
                    "parameters": dict(entity.archetype.geometry.parameters),
                },
                "texture": {
                    "kind": entity.archetype.texture.kind,
                    "reference": entity.archetype.texture.reference,
                    "parameters": dict(entity.archetype.texture.parameters),
                },
                "capabilities": list(entity.archetype.capabilities),
                "controller": {
                    "kind": entity.controller.kind,
                    "source": entity.controller.source,
                    "parameters": dict(entity.controller.parameters),
                },
                "pose": {
                    "coordinate_space": entity.pose.coordinate_space,
                    "position": list(entity.pose.position),
                    "velocity": list(entity.pose.velocity),
                    "acceleration": list(entity.pose.acceleration),
                    "jerk": list(entity.pose.jerk),
                    "facing": list(entity.pose.facing),
                },
                "principal": entity.principal,
                "traits": dict(entity.traits),
                "color": dict(entity.traits).get("color"),
                "interaction": {"type": "inspect", "destination": entity.identity},
            }
            for entity in mezzanine.entities
        ],
        "cycle": {
            "identity": "entities.default-cycle",
            "phases": list(ENTITY_CYCLE_PHASES),
            "execution": ["inline", "worker"],
        },
    }


__all__ = [
    "ABSTRACT_UI_ENTITY_VERSION",
    "ENTITY_CYCLE_PHASES",
    "POINTER_ARCHETYPE",
    "PLAYER_ARCHETYPE",
    "AbstractUIEntity",
    "ControlInputFrame",
    "EntityArchetype",
    "EntityController",
    "EntityCyclePolicy",
    "EntityCycleResult",
    "EntityGeometry",
    "EntityInteraction",
    "EntityMezzanine",
    "EntityOrganization",
    "EntityPose",
    "EntityPresentation",
    "EntitySpawn",
    "EntitySpawnEdit",
    "EntityTexture",
    "EntityTick",
    "entity_mezzanine_model",
    "run_entity_cycle",
    "spawn_pointer_and_follower",
    "spawn_pointer_and_followers",
    "spawn_world_player",
]
