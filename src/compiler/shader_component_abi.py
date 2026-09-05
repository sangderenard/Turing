"""Canonical ABI for local and online shader components.

Backend spelling is decoration. Logical ports, sentinel words, link scope, and
component scheduling are shared by WGSL, desktop GLSL compute, and GLSL
fragment modules so shells never infer compatibility by parsing shader text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .shader_stages import ShaderIOLayout, STAGES


SCHEMA = "turing.shader-component.v1"
SENTINEL_MAGIC = 0x54555247  # "TURG"
SENTINEL_VERSION = 1
SENTINEL_ENDIAN = 0x01020304


class LinkScope(str, Enum):
    LOCAL = "system-local"
    ONLINE = "online-cross-program"


class LinkTransport(str, Enum):
    SHARED_ARENA = "shared-arena"
    COMPILED_ARTIFACT = "compiled-artifact"
    ONLINE_MESSAGE = "online-message"


@dataclass(frozen=True, slots=True)
class ComponentSentinels:
    """Fixed u32 header read before and after every component dispatch."""

    magic: int = SENTINEL_MAGIC
    version: int = SENTINEL_VERSION
    endian: int = SENTINEL_ENDIAN
    generation: int = 0
    ready: int = 0
    error: int = 0
    port_count: int = 0
    checksum: int = 0

    def words(self) -> tuple[int, ...]:
        return (
            self.magic, self.version, self.endian, self.generation,
            self.ready, self.error, self.port_count, self.checksum,
        )

    @classmethod
    def for_ports(cls, count: int) -> "ComponentSentinels":
        if count < 0:
            raise ValueError("sentinel port count cannot be negative")
        checksum = (SENTINEL_MAGIC ^ SENTINEL_VERSION ^ SENTINEL_ENDIAN ^ count) & 0xFFFFFFFF
        return cls(port_count=count, checksum=checksum)

    def validate(self, expected_ports: int) -> None:
        expected = self.for_ports(expected_ports)
        if (self.magic, self.version, self.endian, self.port_count, self.checksum) != (
            expected.magic, expected.version, expected.endian,
            expected.port_count, expected.checksum,
        ):
            raise ValueError("shader component sentinel header is incompatible or corrupt")


@dataclass(frozen=True, slots=True)
class ComponentPort:
    slot: int
    name: str
    role: str
    dtype: str
    value_id: int | None
    backend_binding: int
    transport: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "name": self.name,
            "role": self.role,
            "dtype": self.dtype,
            "value_id": self.value_id,
            "backend_binding": self.backend_binding,
            "transport": self.transport,
        }


@dataclass(frozen=True, slots=True)
class ShaderComponentABI:
    component_id: str
    language: str
    stage: str
    entrypoint: str
    ports: tuple[ComponentPort, ...]
    sentinels: ComponentSentinels
    decorations: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"unknown shader stage {self.stage!r}")
        if not self.component_id or not self.entrypoint:
            raise ValueError("shader components require identities and entrypoints")
        slots = [port.slot for port in self.ports]
        if slots != list(range(len(self.ports))):
            raise ValueError("component port slots must be unique and contiguous from zero")
        if any(port.role not in {"feed", "output", "uniform"} for port in self.ports):
            raise ValueError("component port has an invalid role")
        self.sentinels.validate(len(self.ports))

    def port(self, slot: int) -> ComponentPort:
        try:
            return self.ports[int(slot)]
        except IndexError as error:
            raise KeyError(f"component {self.component_id!r} has no port {slot}") from error

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "component_id": self.component_id,
            "language": self.language,
            "stage": self.stage,
            "entrypoint": self.entrypoint,
            "sentinels": {
                "word_layout": [
                    "magic", "version", "endian", "generation",
                    "ready", "error", "port_count", "checksum",
                ],
                "words": list(self.sentinels.words()),
            },
            "decorations": dict(self.decorations),
            "ports": [port.to_mapping() for port in self.ports],
        }


def component_abi_from_layout(
    component_id: str,
    language: str,
    layout: ShaderIOLayout,
    *,
    entrypoint: str = "main",
    decorations: Mapping[str, Any] | None = None,
) -> ShaderComponentABI:
    ordered = (*layout.feeds, *layout.outputs, *layout.uniforms)
    ports = tuple(
        ComponentPort(
            slot=slot,
            name=binding.name,
            role=binding.role,
            dtype=binding.dtype,
            value_id=binding.value_id,
            backend_binding=binding.index,
            transport=(
                "texture" if layout.stage == "fragment" and binding.role == "feed"
                else "draw-buffer" if layout.stage == "fragment" and binding.role == "output"
                else "storage-buffer"
            ),
        )
        for slot, binding in enumerate(ordered)
    )
    return ShaderComponentABI(
        component_id=str(component_id),
        language=str(language),
        stage=layout.stage,
        entrypoint=str(entrypoint),
        ports=ports,
        sentinels=ComponentSentinels.for_ports(len(ports)),
        decorations=MappingProxyType(dict(decorations or {})),
    )


@dataclass(frozen=True, slots=True)
class ExternalComponentLink:
    link_id: str
    source_component: str
    source_slot: int
    target_component: str
    target_slot: int
    scope: LinkScope
    transport: LinkTransport
    alias: bool = True
    endpoint: str | None = None
    feedback: bool = False

    def __post_init__(self) -> None:
        if self.scope is LinkScope.ONLINE and self.transport is not LinkTransport.ONLINE_MESSAGE:
            raise ValueError("online links require online-message transport")
        if self.scope is LinkScope.LOCAL and self.transport is LinkTransport.ONLINE_MESSAGE:
            raise ValueError("local links cannot use online-message transport")
        if self.scope is LinkScope.ONLINE and not self.endpoint:
            raise ValueError("online links require an endpoint identity")
        if self.feedback and self.alias:
            raise ValueError("feedback links must cross a versioned, non-aliasing sentinel boundary")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "link_id": self.link_id,
            "source": {"component": self.source_component, "slot": self.source_slot},
            "target": {"component": self.target_component, "slot": self.target_slot},
            "scope": self.scope.value,
            "transport": self.transport.value,
            "alias": self.alias,
            "endpoint": self.endpoint,
            "feedback": self.feedback,
            "sentinel_policy": "required",
        }


@dataclass(frozen=True, slots=True)
class ExternalLinkSSABoundary:
    link_id: str
    producer_value_id: int | None
    consumer_value_id: int | None
    dtype: str
    scope: str
    transport: str
    sentinel_required: bool = True


@dataclass(frozen=True, slots=True)
class ComponentAssemblyPlan:
    components: tuple[ShaderComponentABI, ...]
    links: tuple[ExternalComponentLink, ...] = ()

    def __post_init__(self) -> None:
        by_id = {component.component_id: component for component in self.components}
        if len(by_id) != len(self.components):
            raise ValueError("component identities must be unique")
        for link in self.links:
            if link.source_component not in by_id or link.target_component not in by_id:
                raise ValueError(f"link {link.link_id!r} references an unknown component")
            source = by_id[link.source_component].port(link.source_slot)
            target = by_id[link.target_component].port(link.target_slot)
            if source.role != "output" or target.role != "feed":
                raise ValueError(f"link {link.link_id!r} must connect output to feed")
            if source.dtype != target.dtype:
                raise ValueError(f"link {link.link_id!r} has incompatible dtypes")

    def shell_waves(self) -> tuple[tuple[str, ...], ...]:
        """Topologically group components; each wave may dispatch concurrently."""

        identities = {component.component_id for component in self.components}
        incoming = {identity: set() for identity in identities}
        outgoing = {identity: set() for identity in identities}
        for link in self.links:
            if not link.feedback:
                incoming[link.target_component].add(link.source_component)
                outgoing[link.source_component].add(link.target_component)
        waves: list[tuple[str, ...]] = []
        remaining = set(identities)
        while remaining:
            ready = tuple(sorted(identity for identity in remaining if not incoming[identity] & remaining))
            if not ready:
                raise ValueError("component link graph contains a cycle requiring an explicit feedback sentinel")
            waves.append(ready)
            remaining.difference_update(ready)
        return tuple(waves)

    def ssa_boundaries(self) -> tuple[ExternalLinkSSABoundary, ...]:
        by_id = {component.component_id: component for component in self.components}
        return tuple(
            ExternalLinkSSABoundary(
                link.link_id,
                by_id[link.source_component].port(link.source_slot).value_id,
                by_id[link.target_component].port(link.target_slot).value_id,
                by_id[link.source_component].port(link.source_slot).dtype,
                link.scope.value,
                link.transport.value,
            )
            for link in self.links
        )

    def lower_to_ssa(self) -> "ComponentAssemblySSALowering":
        """Lower every external seam to an explicit repository SSA function."""

        from ..transmogrifier.ssa import BasicBlock, Function, IRModule, Instr, SSAValue

        boundaries = self.ssa_boundaries()
        functions = {}
        links_by_id = {link.link_id: link for link in self.links}
        for index, boundary in enumerate(boundaries):
            link = links_by_id[boundary.link_id]
            argument = SSAValue(0, boundary.dtype)
            result = SSAValue(1, boundary.dtype)
            function_name = f"external_link_{index}"
            attributes = {
                "callee": "__turing_external_component_link__",
                "external": True,
                "link_id": boundary.link_id,
                "scope": boundary.scope,
                "transport": boundary.transport,
                "endpoint": link.endpoint,
                "alias": link.alias,
                "feedback": link.feedback,
                "sentinel_required": True,
                "sentinel_generation_rule": "consumer-generation=producer-generation",
                "source_component": link.source_component,
                "source_slot": link.source_slot,
                "target_component": link.target_component,
                "target_slot": link.target_slot,
            }
            functions[function_name] = Function(
                function_name,
                [argument],
                {"entry": BasicBlock("entry", [
                    Instr("Call", [argument], result, attributes=attributes),
                    Instr("Ret", [result], None),
                ])},
                metadata={
                    "kind": "external-component-link",
                    "component_abi_schema": SCHEMA,
                    "link": attributes,
                },
            )
        return ComponentAssemblySSALowering(
            IRModule(functions), boundaries, self.shell_waves(),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Serialize the complete local/online assembly for site manifests."""

        return {
            "schema": "turing.shader-component-assembly.v1",
            "components": [component.to_mapping() for component in self.components],
            "links": [link.to_mapping() for link in self.links],
            "shell_waves": [list(wave) for wave in self.shell_waves()],
            "ssa_boundaries": [
                {
                    "link_id": boundary.link_id,
                    "producer_value_id": boundary.producer_value_id,
                    "consumer_value_id": boundary.consumer_value_id,
                    "dtype": boundary.dtype,
                    "scope": boundary.scope,
                    "transport": boundary.transport,
                    "sentinel_required": boundary.sentinel_required,
                }
                for boundary in self.ssa_boundaries()
            ],
        }


@dataclass(frozen=True, slots=True)
class ComponentAssemblySSALowering:
    module: Any
    boundaries: tuple[ExternalLinkSSABoundary, ...]
    shell_waves: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class HierarchicalComponentBinding:
    callsite_id: int
    caller_component: str
    callee_component: str
    argument_slots: tuple[tuple[int, int], ...]
    result_slots: tuple[tuple[int, int], ...]


def validate_hierarchical_component_plan(
    root: Any,
    components: Iterable[ShaderComponentABI],
    closure_components: Mapping[int, str],
) -> tuple[HierarchicalComponentBinding, ...]:
    """Verify planner-authored multi-shell calls against canonical ABI ports.

    ``root`` is the existing :class:`hierarchical_plan.PlanClosure`; this
    adapter intentionally consumes its explicit call bindings rather than
    rediscovering calls from source or shader text.
    """

    from .hierarchical_plan import PlanCall, PlanClosure

    by_id = {component.component_id: component for component in components}

    def component_for(closure: Any) -> ShaderComponentABI:
        component_id = closure_components.get(int(closure.closure_id))
        if component_id is None or component_id not in by_id:
            raise ValueError(
                f"planned closure {closure.closure_id} has no shader component ABI"
            )
        return by_id[component_id]

    def slot_for(component: ShaderComponentABI, value_id: int, role: str) -> int:
        matches = [
            port.slot for port in component.ports
            if port.value_id == int(value_id) and port.role == role
        ]
        if len(matches) != 1:
            raise ValueError(
                f"component {component.component_id!r} needs exactly one "
                f"{role} port for value {value_id}; found {matches}"
            )
        return matches[0]

    bindings: list[HierarchicalComponentBinding] = []

    def visit(closure: Any) -> None:
        caller = component_for(closure)
        for item in closure.items:
            if isinstance(item, PlanCall):
                callee = component_for(item.callee)
                arguments = tuple(
                    (
                        slot_for(caller, caller_value, "feed"),
                        slot_for(callee, callee_value, "feed"),
                    )
                    for caller_value, callee_value in item.argument_bindings
                )
                results = tuple(
                    (
                        slot_for(callee, callee_value, "output"),
                        slot_for(caller, caller_value, "output"),
                    )
                    for callee_value, caller_value in item.result_bindings
                )
                bindings.append(HierarchicalComponentBinding(
                    int(item.callsite_id), caller.component_id,
                    callee.component_id, arguments, results,
                ))
                visit(item.callee)
            elif isinstance(item, PlanClosure):
                visit(item)

    visit(root)
    return tuple(bindings)


__all__ = [
    "ComponentAssemblyPlan", "ComponentAssemblySSALowering", "ComponentPort", "ComponentSentinels",
    "ExternalComponentLink", "ExternalLinkSSABoundary", "HierarchicalComponentBinding", "LinkScope",
    "LinkTransport", "SCHEMA", "ShaderComponentABI",
    "component_abi_from_layout", "validate_hierarchical_component_plan",
]
