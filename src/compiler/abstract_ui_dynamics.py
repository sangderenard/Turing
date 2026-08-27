"""Neutral device telemetry and dynamics spaces for controllable viewports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


ABSTRACT_UI_DYNAMICS_VERSION = "abstract-ui-dynamics-v0"


def _signal_label(source: str) -> str:
    labels = {
        "pointer:relative-motion": "move",
        "pointer:button-0": "M1",
        "pointer:button-2": "M2",
        "gamepad:left-y-negative": "LS↑",
        "gamepad:left-y-positive": "LS↓",
        "gamepad:left-x-negative": "LS←",
        "gamepad:left-x-positive": "LS→",
        "gamepad:right-stick": "RS",
        "gamepad:button-0": "B0",
        "gamepad:button-1": "B1",
        "keyboard:ShiftLeft": "Shift",
        "keyboard:ShiftRight": "Shift",
        "keyboard:Space": "Space",
    }
    if source.startswith("keyboard:Key"):
        return source.removeprefix("keyboard:Key")
    return labels.get(source, source.split(":", 1)[-1])


@dataclass(frozen=True, slots=True)
class DeviceSignal:
    identity: str
    device: str
    source: str
    action: str
    label: str

    def to_data(self) -> dict[str, str]:
        return {
            "identity": self.identity,
            "device": self.device,
            "source": self.source,
            "action": self.action,
            "label": self.label,
        }


@dataclass(frozen=True, slots=True)
class DeviceMonitor:
    identity: str
    actor: str | None
    signals: tuple[DeviceSignal, ...]

    @classmethod
    def from_bindings(
        cls,
        identity: str,
        actor: str | None,
        bindings: Iterable[Any],
    ) -> "DeviceMonitor":
        signals: list[DeviceSignal] = []
        seen: set[str] = set()
        for binding in bindings:
            for source in binding.inputs:
                if source in seen:
                    continue
                seen.add(source)
                device = source.split(":", 1)[0]
                signals.append(DeviceSignal(
                    f"{identity}/signals/{device}/{source.split(':', 1)[-1]}",
                    device,
                    source,
                    binding.action,
                    _signal_label(source),
                ))
        return cls(str(identity), actor, tuple(signals))

    def to_data(self) -> dict[str, Any]:
        groups = []
        for device in ("pointer", "keyboard", "gamepad"):
            members = [signal.to_data() for signal in self.signals if signal.device == device]
            if members:
                groups.append({"device": device, "signals": members})
        return {
            "schema": "abstract-ui-device-monitor-v0",
            "identity": self.identity,
            "kind": "device-monitor",
            "actor": self.actor,
            "groups": groups,
        }


@dataclass(frozen=True, slots=True)
class PhysicsStage:
    identity: str
    operation: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    selection: str
    dispatch: str = "compute-shader"
    status: str = "selected-unbound"

    def to_data(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "operation": self.operation,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "selection": self.selection,
            "dispatch": self.dispatch,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class DynamicsLane:
    identity: str
    kind: str
    phase: str
    channels: tuple[str, ...]
    bound_channels: tuple[str, ...] = ()
    stages: tuple[PhysicsStage, ...] = ()

    def to_data(self) -> dict[str, Any]:
        bound = set(self.bound_channels)
        result = {
            "identity": self.identity,
            "kind": self.kind,
            "phase": self.phase,
            "channels": [
                {"name": channel, "status": "bound" if channel in bound else "unbound"}
                for channel in self.channels
            ],
        }
        if self.stages:
            result["stages"] = [stage.to_data() for stage in self.stages]
            result["dispatch_policy"] = {
                "ordering": "selected-stage-order",
                "identity_specialization": "stable-identity-to-dense-u32",
                "backend_candidates": ["webgpu-compute", "wasm-simd", "cpu"],
                "welded_world": True,
            }
            result["equation_program"] = {
                "source_language": "sympy-equation-set",
                "selection": "stage-selection-defines-active-equations",
                "lowering": [
                    "sympy-expressions", "canonical-process-graph",
                    "compiler-ssa", "webassembly",
                ],
                "state_layout": "dense-runtime-identities-plus-typed-arrays",
                "semantic_identity_authority": "world.identity_specialization",
                "status": "contract-only-unbound",
            }
        return result


@dataclass(frozen=True, slots=True)
class DynamicsSpace:
    identity: str
    actor: str | None
    world: str
    timer: str
    user_dynamics: DynamicsLane
    world_physics: DynamicsLane

    def to_data(self) -> dict[str, Any]:
        dependencies = [
            {"relationship": "solves", "target": self.world},
            {"relationship": "clocked-by", "target": self.timer},
        ]
        if self.actor is not None:
            dependencies.insert(0, {"relationship": "integrates", "target": self.actor})
        return {
            "schema": ABSTRACT_UI_DYNAMICS_VERSION,
            "identity": self.identity,
            "kind": "dynamics-space",
            "actor": self.actor,
            "world": self.world,
            "timer": self.timer,
            "lanes": [self.user_dynamics.to_data(), self.world_physics.to_data()],
            "dependencies": dependencies,
        }


def viewport_dynamics_space(
    system_root: str,
    actor: str | None,
) -> DynamicsSpace:
    identity = f"{system_root}/dynamics"
    return DynamicsSpace(
        identity,
        actor,
        system_root,
        f"{system_root}/timer",
        DynamicsLane(
            f"{identity}/user", "user-dynamics", "integrate-user",
            ("intent", "position", "velocity", "facing"),
            ("intent", "position", "velocity", "facing"),
        ),
        DynamicsLane(
            f"{identity}/world", "world-physics", "solve-world",
            ("geometry", "contacts", "collision", "gravity"),
            ("geometry",),
            (
                PhysicsStage(
                    f"{identity}/world/stages/specialize",
                    "specialize-world-identities",
                    ("world.objects", "mesh.semantic-parts"),
                    ("dense-object-table", "dense-part-table"),
                    "identity-hot-path",
                ),
                PhysicsStage(
                    f"{identity}/world/stages/weld",
                    "weld-static-collider-batches",
                    ("dense-part-table", "mesh.vertices"),
                    ("welded-static-colliders",),
                    "world-declares-welded-objects",
                ),
                PhysicsStage(
                    f"{identity}/world/stages/broad-phase",
                    "broad-phase-player-world",
                    ("player-bounds", "welded-static-colliders"),
                    ("candidate-pairs",),
                    "player-experiences-static-world",
                ),
                PhysicsStage(
                    f"{identity}/world/stages/contacts",
                    "narrow-phase-contacts",
                    ("candidate-pairs", "semantic-part-spans"),
                    ("contacts",),
                    "semantic-surfaces-retain-identity",
                ),
                PhysicsStage(
                    f"{identity}/world/stages/resolve",
                    "resolve-player-contacts",
                    ("player-pose", "contacts", "movement-intent"),
                    ("resolved-player-pose",),
                    "selected-player-embodiment",
                ),
                PhysicsStage(
                    f"{identity}/world/stages/publish",
                    "publish-physics-pose",
                    ("resolved-player-pose",),
                    ("presentation-pose", "action-events"),
                    "world-to-presentation-boundary",
                ),
            ),
        ),
    )


__all__ = [
    "ABSTRACT_UI_DYNAMICS_VERSION", "DeviceMonitor", "DeviceSignal",
    "DynamicsLane", "DynamicsSpace", "PhysicsStage", "viewport_dynamics_space",
]
