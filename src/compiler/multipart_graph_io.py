"""Typed IO contracts for hierarchical/multipart graph plans.

``PlanClosure`` parts already own local value IDs and ``PlanCall`` already
owns the exact argument/result correlations between parts.  This module gives
those correlations a stable IO vocabulary without selecting a transport,
allocator, serializer, or runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping

from .hierarchical_plan import PlanCall, PlanClosure
from .shell_io import ShellIOManifest


class PortDirection(str, Enum):
    INPUT = "input"
    OUTPUT = "output"


class IOMode(str, Enum):
    VALUE = "value"
    STREAM = "stream"
    CONTROL = "control"
    STATE = "state"


class BackpressurePolicy(str, Enum):
    """Producer behavior when a fixed-capacity output wheel is full."""

    BLOCK = "block"
    YIELD = "yield"
    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"


@dataclass(frozen=True)
class OutputWheel:
    """A bounded ring for irregular output; storage is supplied by the caller."""

    capacity: int
    token_bytes: int | None = None
    backpressure: BackpressurePolicy = BackpressurePolicy.BLOCK
    low_watermark: int = 0
    high_watermark: int | None = None

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("output wheel capacity must be positive")
        if self.token_bytes is not None and self.token_bytes <= 0:
            raise ValueError("output wheel token size must be positive")
        high = self.capacity if self.high_watermark is None else self.high_watermark
        if not 0 <= self.low_watermark < high <= self.capacity:
            raise ValueError(
                "output wheel watermarks must satisfy 0 <= low < high <= capacity"
            )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "token_bytes": self.token_bytes,
            "backpressure": self.backpressure.value,
            "low_watermark": self.low_watermark,
            "high_watermark": (
                self.capacity if self.high_watermark is None else self.high_watermark
            ),
            "counter_fields": [
                "read_sequence", "write_sequence", "closed", "dropped",
            ],
        }


@dataclass(frozen=True)
class PortSchema:
    """Optional ABI facts; absent fields remain backend decisions."""

    dtype: str | None = None
    shape: tuple[int | None, ...] = ()
    token_bytes: int | None = None
    encoding: str | None = None

    def compatible_with(self, other: "PortSchema") -> bool:
        return all((
            self.dtype is None or other.dtype is None or self.dtype == other.dtype,
            not self.shape or not other.shape or self.shape == other.shape,
            self.token_bytes is None
            or other.token_bytes is None
            or self.token_bytes == other.token_bytes,
            self.encoding is None
            or other.encoding is None
            or self.encoding == other.encoding,
        ))


@dataclass(frozen=True)
class MultipartPort:
    port_id: str
    part_id: int
    direction: PortDirection
    value_id: int
    role: str
    schema: PortSchema = PortSchema()
    external: bool = False


@dataclass(frozen=True)
class MultipartChannel:
    channel_id: str
    source_port_id: str
    target_port_ids: tuple[str, ...]
    mode: IOMode = IOMode.VALUE
    capacity: int | None = None
    ordered: bool = True
    attributes: tuple[tuple[str, Any], ...] = ()
    output_wheel: OutputWheel | None = None


@dataclass(frozen=True)
class MultipartGraphIO:
    """Serializable IO table shared by planners, backends, and runtimes."""

    part_ids: tuple[int, ...]
    ports: tuple[MultipartPort, ...]
    channels: tuple[MultipartChannel, ...]
    schema_version: int = 1
    shell_io: ShellIOManifest = ShellIOManifest()

    def validate(self) -> None:
        parts = set(self.part_ids)
        if len(parts) != len(self.part_ids):
            raise ValueError("multipart IO contains duplicate part IDs")
        port_by_id = {port.port_id: port for port in self.ports}
        if len(port_by_id) != len(self.ports):
            raise ValueError("multipart IO contains duplicate port IDs")
        if any(port.part_id not in parts for port in self.ports):
            raise ValueError("multipart IO port refers to an unknown part")
        channel_ids = {channel.channel_id for channel in self.channels}
        if len(channel_ids) != len(self.channels):
            raise ValueError("multipart IO contains duplicate channel IDs")
        driven_inputs: set[str] = set()
        for channel in self.channels:
            source = port_by_id.get(channel.source_port_id)
            if source is None or source.direction is not PortDirection.OUTPUT:
                raise ValueError(
                    f"channel {channel.channel_id!r} needs an output source"
                )
            if not channel.target_port_ids:
                raise ValueError(
                    f"channel {channel.channel_id!r} has no targets"
                )
            if channel.output_wheel is not None:
                if channel.mode is not IOMode.STREAM:
                    raise ValueError(
                        f"channel {channel.channel_id!r} has an output wheel "
                        "but is not a stream"
                    )
                if (
                    channel.capacity is not None
                    and channel.capacity != channel.output_wheel.capacity
                ):
                    raise ValueError(
                        f"channel {channel.channel_id!r} capacity disagrees "
                        "with its output wheel"
                    )
            for target_id in channel.target_port_ids:
                target = port_by_id.get(target_id)
                if target is None or target.direction is not PortDirection.INPUT:
                    raise ValueError(
                        f"channel {channel.channel_id!r} target {target_id!r} "
                        "is not an input port"
                    )
                if target_id in driven_inputs:
                    raise ValueError(
                        f"input port {target_id!r} has multiple producers"
                    )
                if not source.schema.compatible_with(target.schema):
                    raise ValueError(
                        f"channel {channel.channel_id!r} has incompatible schemas"
                    )
                driven_inputs.add(target_id)
        undriven = {
            port.port_id
            for port in self.ports
            if port.direction is PortDirection.INPUT
            and not port.external
            and port.port_id not in driven_inputs
        }
        if undriven:
            raise ValueError(
                "multipart IO has undriven internal inputs: "
                + ", ".join(sorted(undriven))
            )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": "turing-multipart-graph-io",
            "version": self.schema_version,
            "shell_io": self.shell_io.to_mapping(),
            "parts": list(self.part_ids),
            "ports": [
                {
                    "id": port.port_id,
                    "part": port.part_id,
                    "direction": port.direction.value,
                    "value": port.value_id,
                    "role": port.role,
                    "external": port.external,
                    "schema": {
                        "dtype": port.schema.dtype,
                        "shape": list(port.schema.shape),
                        "token_bytes": port.schema.token_bytes,
                        "encoding": port.schema.encoding,
                    },
                }
                for port in self.ports
            ],
            "channels": [
                {
                    "id": channel.channel_id,
                    "source": channel.source_port_id,
                    "targets": list(channel.target_port_ids),
                    "mode": channel.mode.value,
                    "capacity": channel.capacity,
                    "ordered": channel.ordered,
                    "attributes": dict(channel.attributes),
                    "output_wheel": (
                        None if channel.output_wheel is None
                        else channel.output_wheel.to_mapping()
                    ),
                }
                for channel in self.channels
            ],
        }


@dataclass
class MultipartIOBuilder:
    """Mutable construction surface; ``finish`` returns immutable IR."""

    part_ids: set[int] = field(default_factory=set)
    ports: dict[str, MultipartPort] = field(default_factory=dict)
    channels: dict[str, MultipartChannel] = field(default_factory=dict)
    shell_io: ShellIOManifest = ShellIOManifest()

    def add_part(self, part_id: int) -> None:
        self.part_ids.add(int(part_id))

    def add_port(self, port: MultipartPort) -> None:
        if port.port_id in self.ports:
            raise ValueError(f"duplicate multipart port {port.port_id!r}")
        self.add_part(port.part_id)
        self.ports[port.port_id] = port

    def add_channel(self, channel: MultipartChannel) -> None:
        if channel.channel_id in self.channels:
            raise ValueError(f"duplicate multipart channel {channel.channel_id!r}")
        self.channels[channel.channel_id] = channel

    def finish(self) -> MultipartGraphIO:
        result = MultipartGraphIO(
            tuple(sorted(self.part_ids)),
            tuple(self.ports[key] for key in sorted(self.ports)),
            tuple(self.channels[key] for key in sorted(self.channels)),
            shell_io=self.shell_io,
        )
        result.validate()
        return result


def multipart_io_from_hierarchy(
    root: PlanClosure,
    *,
    root_output_value_ids: Iterable[int] = (),
    schemas: Mapping[tuple[int, int], PortSchema] | None = None,
    shell_io: ShellIOManifest = ShellIOManifest(),
) -> MultipartGraphIO:
    """Derive cross-part ports/channels from authoritative call bindings."""

    schemas = dict(schemas or {})
    builder = MultipartIOBuilder(shell_io=shell_io)

    def port(
        port_id: str,
        part_id: int,
        direction: PortDirection,
        value_id: int,
        role: str,
        *,
        external: bool = False,
    ) -> None:
        builder.add_port(MultipartPort(
            port_id,
            int(part_id),
            direction,
            int(value_id),
            role,
            schemas.get((int(part_id), int(value_id)), PortSchema()),
            external,
        ))

    def walk(closure: PlanClosure) -> None:
        owner = int(closure.closure_id)
        if owner < 0:
            raise ValueError("multipart IO requires assigned closure IDs")
        builder.add_part(owner)
        for item in closure.items:
            if not isinstance(item, PlanCall):
                if isinstance(item, PlanClosure):
                    walk(item)
                continue
            child = int(item.callee.closure_id)
            walk(item.callee)
            for index, (caller_value, callee_value) in enumerate(
                item.argument_bindings
            ):
                source_id = f"p{owner}.call{item.callsite_id}.arg{index}.out"
                target_id = f"p{child}.call{item.callsite_id}.arg{index}.in"
                port(source_id, owner, PortDirection.OUTPUT, caller_value, "argument")
                port(target_id, child, PortDirection.INPUT, callee_value, "argument")
                builder.add_channel(MultipartChannel(
                    f"call{item.callsite_id}.argument{index}",
                    source_id,
                    (target_id,),
                ))
            for index, (callee_value, caller_value) in enumerate(
                item.result_bindings
            ):
                source_id = f"p{child}.call{item.callsite_id}.result{index}.out"
                target_id = f"p{owner}.call{item.callsite_id}.result{index}.in"
                port(source_id, child, PortDirection.OUTPUT, callee_value, "result")
                port(target_id, owner, PortDirection.INPUT, caller_value, "result")
                builder.add_channel(MultipartChannel(
                    f"call{item.callsite_id}.result{index}",
                    source_id,
                    (target_id,),
                ))

    walk(root)
    root_id = int(root.closure_id)
    for index, value_id in enumerate(root.captures):
        port(
            f"root.input{index}", root_id, PortDirection.INPUT, value_id,
            "root_input", external=True,
        )
    for index, value_id in enumerate(root_output_value_ids):
        port(
            f"root.output{index}", root_id, PortDirection.OUTPUT, value_id,
            "root_output", external=True,
        )
    return builder.finish()


__all__ = [
    "BackpressurePolicy",
    "IOMode",
    "MultipartChannel",
    "MultipartGraphIO",
    "MultipartIOBuilder",
    "MultipartPort",
    "OutputWheel",
    "PortDirection",
    "PortSchema",
    "multipart_io_from_hierarchy",
]
