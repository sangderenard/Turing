"""Resident SPSC GLChunk lanes connecting vertical fused programs."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, Mapping, Sequence

import numpy as np

from ..fused_ir import FusedProgram, Meta
from .glsl_backend import (
    GLChunk,
    _compute_limits,
    execute_multi_output_program,
)


@dataclass(frozen=True)
class GLChunkLaneSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    slots: int = 2


class GLChunkFIFOLane:
    """One preallocated single-producer/single-consumer resident lane."""

    def __init__(self, spec: GLChunkLaneSpec, slots: Sequence[GLChunk]):
        self.spec = spec
        self._slots = tuple(slots)
        self._write_sequence = 0
        self._read_sequence = 0

    @property
    def unread(self) -> int:
        return self._write_sequence - self._read_sequence

    def empty_output(self) -> GLChunk:
        if self.unread >= len(self._slots):
            raise BufferError(f"GLChunk FIFO lane {self.spec.name!r} is full")
        return self._slots[self._write_sequence % len(self._slots)]

    def publish(self) -> None:
        if self.unread >= len(self._slots):
            raise BufferError(f"GLChunk FIFO lane {self.spec.name!r} is full")
        self._write_sequence += 1

    def input(self) -> GLChunk:
        if self.unread <= 0:
            raise BufferError(f"GLChunk FIFO lane {self.spec.name!r} is empty")
        return self._slots[self._read_sequence % len(self._slots)]

    def consume(self) -> None:
        if self.unread <= 0:
            raise BufferError(f"GLChunk FIFO lane {self.spec.name!r} is empty")
        self._read_sequence += 1

    def write_host(self, value: Any) -> None:
        slot = self.empty_output()
        array = np.asarray(value, dtype=slot.dtype).reshape(self.spec.shape)
        slot.upload_numpy(array)
        self.publish()


class GLChunkFIFOArena:
    """Typed GPU arenas partitioned into persistent FIFO lane slots."""

    def __init__(self, specs: Sequence[GLChunkLaneSpec]):
        alignment_bytes = _compute_limits().ssbo_offset_alignment
        alignment_elements = max(1, alignment_bytes // 4)
        grouped: dict[str, list[GLChunkLaneSpec]] = {}
        for spec in specs:
            if spec.slots < 1:
                raise ValueError("GLChunk FIFO lanes need at least one slot")
            grouped.setdefault(np.dtype(spec.dtype).name, []).append(spec)

        self._roots: list[GLChunk] = []
        self._views: list[GLChunk] = []
        self.lanes: dict[str, GLChunkFIFOLane] = {}
        for dtype, dtype_specs in grouped.items():
            offsets: dict[str, tuple[int, int]] = {}
            cursor = 0
            for spec in dtype_specs:
                cursor = (
                    (cursor + alignment_elements - 1)
                    // alignment_elements
                    * alignment_elements
                )
                count = prod(spec.shape) if spec.shape else 1
                offsets[spec.name] = (cursor, count)
                cursor += count * spec.slots
            root = GLChunk((cursor,), dtype=dtype).to_gpu()
            root.discard_host()
            self._roots.append(root)
            for spec in dtype_specs:
                offset, count = offsets[spec.name]
                slots = []
                for slot_index in range(spec.slots):
                    view = root.range_view(
                        spec.shape,
                        offset=offset + slot_index * count,
                    )
                    slots.append(view)
                    self._views.append(view)
                self.lanes[spec.name] = GLChunkFIFOLane(spec, slots)

    def release(self) -> None:
        for view in self._views:
            view.release()
        self._views.clear()
        for root in self._roots:
            root.release()
        self._roots.clear()


@dataclass(frozen=True)
class FusedProgramLane:
    name: str
    producer: int | None
    consumer: int | None
    value_id: int
    output_name: str | None = None


def _meta_for(program: FusedProgram, value_id: int) -> Meta:
    meta = (program.meta or {}).get(value_id)
    if meta is None or meta.shape is None or meta.dtype is None:
        raise ValueError(
            f"fused value {value_id} needs shape and dtype for a GLChunk lane"
        )
    return meta


class GLSLFusedProgramNetwork:
    """Run ordered fused programs through arena-backed GLChunk FIFO lanes."""

    def __init__(
        self,
        programs: Sequence[FusedProgram],
        *,
        fifo_slots: int = 2,
    ) -> None:
        self.programs = tuple(programs)
        producers: dict[int, int] = {}
        for stage_index, program in enumerate(self.programs):
            for value_id in program.outputs.values():
                if value_id in producers:
                    raise ValueError(
                        f"fused value {value_id} has multiple producers"
                    )
                producers[value_id] = stage_index

        lanes: list[FusedProgramLane] = []
        for consumer, program in enumerate(self.programs):
            for value_id in sorted(program.feeds):
                producer = producers.get(value_id)
                lanes.append(
                    FusedProgramLane(
                        name=f"p{producer if producer is not None else 'in'}"
                        f"_v{value_id}_c{consumer}",
                        producer=producer,
                        consumer=consumer,
                        value_id=value_id,
                    )
                )

        consumed_values = {
            lane.value_id for lane in lanes if lane.producer is not None
        }
        for producer, program in enumerate(self.programs):
            for output_name, value_id in program.outputs.items():
                if value_id in consumed_values:
                    continue
                lanes.append(
                    FusedProgramLane(
                        name=f"p{producer}_v{value_id}_out_{output_name}",
                        producer=producer,
                        consumer=None,
                        value_id=value_id,
                        output_name=output_name,
                    )
                )
        self.routes = tuple(lanes)

        specs = []
        for lane in self.routes:
            owner = (
                self.programs[lane.producer]
                if lane.producer is not None
                else self.programs[lane.consumer]
            )
            meta = _meta_for(owner, lane.value_id)
            specs.append(
                GLChunkLaneSpec(
                    lane.name,
                    tuple(int(size) for size in meta.shape),
                    str(meta.dtype),
                    fifo_slots,
                )
            )
        self.arena = GLChunkFIFOArena(specs)

    def execute(self, external_feeds: Mapping[int, Any]) -> dict[str, GLChunk]:
        for route in self.routes:
            if route.producer is None:
                try:
                    value = external_feeds[route.value_id]
                except KeyError as exc:
                    raise KeyError(
                        f"missing external fused feed {route.value_id}"
                    ) from exc
                self.arena.lanes[route.name].write_host(value)

        terminal: dict[str, GLChunk] = {}
        for stage_index, original in enumerate(self.programs):
            incoming = [
                route
                for route in self.routes
                if route.consumer == stage_index
            ]
            outgoing = [
                route
                for route in self.routes
                if route.producer == stage_index
            ]
            feeds = {
                route.value_id: self.arena.lanes[route.name].input()
                for route in incoming
            }
            outputs = {
                route.name: route.value_id
                for route in outgoing
            }
            program = FusedProgram(
                version=original.version,
                feeds=set(original.feeds),
                steps=list(original.steps),
                outputs=outputs,
                state_in=original.state_in,
                meta=original.meta,
                extras=original.extras,
            )
            outs = {
                route.name: self.arena.lanes[route.name].empty_output()
                for route in outgoing
            }
            execute_multi_output_program(program, feeds, outs=outs)
            for route in outgoing:
                self.arena.lanes[route.name].publish()
            for route in incoming:
                self.arena.lanes[route.name].consume()

        for route in self.routes:
            if route.consumer is not None:
                continue
            lane = self.arena.lanes[route.name]
            terminal[route.output_name or route.name] = lane.input()
            lane.consume()
        return terminal

    def release(self) -> None:
        self.arena.release()


__all__ = [
    "FusedProgramLane",
    "GLChunkFIFOArena",
    "GLChunkFIFOLane",
    "GLChunkLaneSpec",
    "GLSLFusedProgramNetwork",
]
