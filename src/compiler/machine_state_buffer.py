"""Lock-free observation buffers for a free-running reversible machine.

Execution history belongs to :mod:`machine_execution`.  The buffers here are
disposable observations of that history: one runner writes a complete back
buffer and atomically publishes its ``(generation, slot)`` tuple, while one
display reader leases the newest completed slot.  The runner never waits for
the display and the display is free to skip generations.

The Python implementation is deliberately single-writer/single-reader.  It
uses atomic object-reference publication under CPython's GIL; the serialized
header and offsets are the language-neutral ABI to reproduce with native or
WebAssembly atomics when those hosts own the runner.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum
from math import floor
import struct
from threading import Event, Thread
from types import MappingProxyType
from typing import Callable, Iterator, Mapping, Sequence

from .machine_chip_layout import RegisterBankLayout, pack_register_banks
from .machine_execution import (
    MachineExecutionResult,
    MachineExecutionStatus,
    MachineVirtualMulticore,
)


SNAPSHOT_MAGIC = b"TMSNAP01"
SNAPSHOT_VERSION = 1
SNAPSHOT_HEADER_BYTES = 256
CORE_STATUS_BYTES = 32
OUTPUT_DESCRIPTOR_BYTES = 64

_HEADER = struct.Struct("<8sIIQiiQIIIIIIIII")
_CORE_STATUS = struct.Struct("<QQIIII")
_OUTPUT_DESCRIPTOR = struct.Struct("<IIIIIIQQQQQ")


def _align(value: int, alignment: int) -> int:
    return (int(value) + alignment - 1) & -alignment


class MachineRunDirection(IntEnum):
    BACKWARD = -1
    PAUSED = 0
    FORWARD = 1


@dataclass(slots=True)
class ExternalMachineClock:
    """Convert shell-owned elapsed ticks into an exact transition budget."""

    transitions_per_second: float = 60.0
    time_scale: float = 1.0
    maximum_transitions_per_tick: int = 1_000_000
    fractional_transitions: float = 0.0

    def __post_init__(self) -> None:
        if self.transitions_per_second <= 0 or self.time_scale < 0:
            raise ValueError("machine-clock rates must be positive")
        if self.maximum_transitions_per_tick <= 0:
            raise ValueError("machine-clock tick capacity must be positive")

    def set_speed(self, transitions_per_second: float) -> None:
        if transitions_per_second <= 0:
            raise ValueError("machine-clock speed must be positive")
        self.transitions_per_second = float(transitions_per_second)

    def budget(self, elapsed_seconds: float) -> int:
        if elapsed_seconds < 0:
            raise ValueError("machine-clock elapsed time cannot be negative")
        requested = (
            elapsed_seconds * self.transitions_per_second * self.time_scale
            + self.fractional_transitions
        )
        whole = floor(requested)
        self.fractional_transitions = requested - whole
        return min(whole, self.maximum_transitions_per_tick)


class SubjectOutputKind(IntEnum):
    NONE = 0
    BYTES = 1
    TERMINAL = 2
    FRAMEBUFFER = 3
    AUDIO = 4


class SubjectOutputFormat(IntEnum):
    RAW_U8 = 1
    UTF8 = 2
    RGBA8 = 3
    F32 = 4


@dataclass(frozen=True, slots=True)
class SubjectOutputBuffer:
    """One subject-owned device buffer copied into an observation slot."""

    kind: SubjectOutputKind
    format: SubjectOutputFormat
    data: bytes
    width: int = 0
    height: int = 0
    channels: int = 1
    row_stride: int = 0
    generation: int = 0

    def __post_init__(self) -> None:
        if min(self.width, self.height, self.channels, self.row_stride, self.generation) < 0:
            raise ValueError("subject-output dimensions and generation cannot be negative")


@dataclass(frozen=True, slots=True)
class MachineSnapshotLayout:
    """Fixed byte layout shared by all three observation slots."""

    core_count: int
    register_count: int
    register_stride_bytes: int
    register_offset: int
    core_status_offset: int
    output_descriptor_offset: int
    maximum_outputs: int
    output_payload_offset: int
    maximum_output_bytes: int
    byte_size: int

    @classmethod
    def build(
        cls,
        registers: RegisterBankLayout,
        *,
        core_count: int,
        maximum_outputs: int = 4,
        maximum_output_bytes: int = 4 * 1024 * 1024,
    ) -> "MachineSnapshotLayout":
        if core_count <= 0 or maximum_outputs < 0 or maximum_output_bytes < 0:
            raise ValueError("snapshot capacities must be non-negative and include a core")
        if len(registers.cells) != core_count * len(registers.register_names):
            raise ValueError("register layout does not match snapshot core count")
        register_offset = SNAPSHOT_HEADER_BYTES
        status_offset = _align(register_offset + registers.byte_size, 64)
        descriptor_offset = _align(status_offset + core_count * CORE_STATUS_BYTES, 64)
        payload_offset = _align(
            descriptor_offset + maximum_outputs * OUTPUT_DESCRIPTOR_BYTES, 256,
        )
        return cls(
            core_count=core_count,
            register_count=len(registers.register_names),
            register_stride_bytes=registers.core_stride,
            register_offset=register_offset,
            core_status_offset=status_offset,
            output_descriptor_offset=descriptor_offset,
            maximum_outputs=maximum_outputs,
            output_payload_offset=payload_offset,
            maximum_output_bytes=maximum_output_bytes,
            byte_size=payload_offset + maximum_output_bytes,
        )


@dataclass(frozen=True, slots=True)
class MachineSnapshotHeader:
    generation: int
    direction: MachineRunDirection
    slot_index: int
    transitions: int
    core_count: int
    register_count: int
    register_stride_bytes: int
    register_offset: int
    core_status_offset: int
    output_descriptor_offset: int
    output_count: int
    output_payload_offset: int
    output_payload_bytes: int
    byte_size: int


@dataclass(frozen=True, slots=True)
class SubjectOutputDescriptor:
    kind: SubjectOutputKind
    format: SubjectOutputFormat
    width: int
    height: int
    channels: int
    row_stride: int
    byte_offset: int
    byte_length: int
    generation: int


class MachineSnapshotView:
    """A read-only view valid for the duration of its slot lease."""

    def __init__(self, data: memoryview) -> None:
        self.data = data.toreadonly()
        values = _HEADER.unpack_from(self.data)
        if values[0] != SNAPSHOT_MAGIC or values[1] != SNAPSHOT_VERSION:
            raise ValueError("machine snapshot has an unknown header")
        self.header = MachineSnapshotHeader(
            byte_size=values[2], generation=values[3],
            direction=MachineRunDirection(values[4]), slot_index=values[5],
            transitions=values[6], core_count=values[7], register_count=values[8],
            register_stride_bytes=values[9], register_offset=values[10],
            core_status_offset=values[11], output_descriptor_offset=values[12],
            output_count=values[13], output_payload_offset=values[14],
            output_payload_bytes=values[15],
        )

    def register_words(self, core: int, register: int) -> tuple[int, int]:
        if not 0 <= core < self.header.core_count:
            raise IndexError("snapshot core index is out of range")
        if not 0 <= register < self.header.register_count:
            raise IndexError("snapshot register index is out of range")
        offset = (
            self.header.register_offset
            + core * self.header.register_stride_bytes
            + register * 8
        )
        return struct.unpack_from("<II", self.data, offset)

    def core_status(self, core: int) -> Mapping[str, int]:
        if not 0 <= core < self.header.core_count:
            raise IndexError("snapshot core index is out of range")
        values = _CORE_STATUS.unpack_from(
            self.data, self.header.core_status_offset + core * CORE_STATUS_BYTES,
        )
        return MappingProxyType({
            "pc": values[0], "history_position": values[1],
            "history_length": values[2], "status": values[3],
            "steps": values[4],
        })

    def output_descriptor(self, index: int) -> SubjectOutputDescriptor:
        if not 0 <= index < self.header.output_count:
            raise IndexError("snapshot output index is out of range")
        values = _OUTPUT_DESCRIPTOR.unpack_from(
            self.data,
            self.header.output_descriptor_offset + index * OUTPUT_DESCRIPTOR_BYTES,
        )
        return SubjectOutputDescriptor(
            kind=SubjectOutputKind(values[0]), format=SubjectOutputFormat(values[1]),
            width=values[2], height=values[3], channels=values[4],
            row_stride=values[5], byte_offset=values[6], byte_length=values[7],
            generation=values[8],
        )

    def output_bytes(self, index: int) -> memoryview:
        descriptor = self.output_descriptor(index)
        return self.data[
            descriptor.byte_offset:descriptor.byte_offset + descriptor.byte_length
        ]


class MachineSnapshotTripleBuffer:
    """Three preallocated snapshot slots with a non-blocking SPSC flip."""

    def __init__(self, layout: MachineSnapshotLayout, registers: RegisterBankLayout) -> None:
        self.layout = layout
        self.registers = registers
        self._slots = tuple(bytearray(layout.byte_size) for _ in range(3))
        self._publication: tuple[int, int] = (0, -1)
        self._reader_index = -1
        self._next_slot = 0

    @property
    def publication(self) -> tuple[int, int]:
        return self._publication

    def _back_slot(self) -> int:
        published = self._publication[1]
        for delta in range(3):
            candidate = (self._next_slot + delta) % 3
            if candidate != published and candidate != self._reader_index:
                self._next_slot = (candidate + 1) % 3
                return candidate
        raise RuntimeError("triple-buffer ownership invariant was violated")

    def publish(
        self,
        machine: MachineVirtualMulticore,
        *,
        direction: MachineRunDirection,
        transitions: int,
        results: Sequence[MachineExecutionResult] = (),
        outputs: Sequence[SubjectOutputBuffer] = (),
    ) -> int:
        """Write a complete back slot, then expose it with one publication store."""

        if len(machine.cores) != self.layout.core_count:
            raise ValueError("machine core count does not match snapshot layout")
        if len(outputs) > self.layout.maximum_outputs:
            raise ValueError("subject output count exceeds snapshot capacity")
        payload_bytes = sum(len(output.data) for output in outputs)
        if payload_bytes > self.layout.maximum_output_bytes:
            raise ValueError("subject output bytes exceed snapshot capacity")

        slot_index = self._back_slot()
        slot = self._slots[slot_index]
        generation = self._publication[0] + 1
        register_words = pack_register_banks(machine, self.registers)
        for index, word in enumerate(register_words):
            struct.pack_into("<I", slot, self.layout.register_offset + index * 4, word)

        statuses = {
            index: result.status for index, result in enumerate(results)
        }
        for index, core in enumerate(machine.cores):
            _CORE_STATUS.pack_into(
                slot, self.layout.core_status_offset + index * CORE_STATUS_BYTES,
                core.state.pc & 0xFFFFFFFFFFFFFFFF,
                core.position,
                core.history_length,
                int(statuses.get(index, MachineExecutionStatus.RUNNING)),
                core.state.steps,
                0,
            )

        payload_cursor = self.layout.output_payload_offset
        for index, output in enumerate(outputs):
            data = bytes(output.data)
            slot[payload_cursor:payload_cursor + len(data)] = data
            _OUTPUT_DESCRIPTOR.pack_into(
                slot,
                self.layout.output_descriptor_offset + index * OUTPUT_DESCRIPTOR_BYTES,
                int(output.kind), int(output.format), output.width, output.height,
                output.channels, output.row_stride, payload_cursor, len(data),
                output.generation, 0, 0,
            )
            payload_cursor += len(data)

        _HEADER.pack_into(
            slot, 0, SNAPSHOT_MAGIC, SNAPSHOT_VERSION, self.layout.byte_size,
            generation, int(direction), slot_index, int(transitions),
            self.layout.core_count, self.layout.register_count,
            self.layout.register_stride_bytes, self.layout.register_offset,
            self.layout.core_status_offset, self.layout.output_descriptor_offset,
            len(outputs), self.layout.output_payload_offset, payload_bytes,
        )
        self._publication = (generation, slot_index)
        return generation

    @contextmanager
    def lease_latest(self) -> Iterator[MachineSnapshotView | None]:
        """Pin and expose the newest completed slot without blocking its writer."""

        while True:
            publication = self._publication
            index = publication[1]
            if index < 0:
                yield None
                return
            self._reader_index = index
            if self._publication == publication:
                break
            self._reader_index = -1
        try:
            yield MachineSnapshotView(memoryview(self._slots[index]))
        finally:
            self._reader_index = -1

    def copy_latest(self) -> bytes | None:
        with self.lease_latest() as snapshot:
            return bytes(snapshot.data) if snapshot is not None else None


SubjectOutputProvider = Callable[[], Sequence[SubjectOutputBuffer]]


class FreeRunningMachineRunner:
    """Run reversible heads from shell ticks or an optional maximum-speed loop."""

    def __init__(
        self,
        machine: MachineVirtualMulticore,
        snapshots: MachineSnapshotTripleBuffer,
        *,
        transitions_per_publication: int = 256,
        output_provider: SubjectOutputProvider | None = None,
    ) -> None:
        if transitions_per_publication <= 0:
            raise ValueError("transitions_per_publication must be positive")
        self.machine = machine
        self.snapshots = snapshots
        self.transitions_per_publication = int(transitions_per_publication)
        self.output_provider = output_provider or (lambda: ())
        self._direction = MachineRunDirection.PAUSED
        self._stop = Event()
        self._thread: Thread | None = None
        self._transitions = 0
        self._last_results: tuple[MachineExecutionResult, ...] = ()
        self.failure: BaseException | None = None

    @property
    def direction(self) -> MachineRunDirection:
        return self._direction

    @property
    def transitions(self) -> int:
        return self._transitions

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def set_direction(self, direction: MachineRunDirection) -> None:
        self._direction = MachineRunDirection(direction)

    def start(self, direction: MachineRunDirection = MachineRunDirection.FORWARD) -> None:
        """Start the optional maximum-speed clock mode on a worker thread."""
        if self.running:
            raise RuntimeError("machine runner is already active")
        self._stop.clear()
        self.failure = None
        self._direction = MachineRunDirection(direction)
        self.snapshots.publish(
            self.machine, direction=self._direction, transitions=self._transitions,
            results=self._last_results, outputs=self.output_provider(),
        )
        self._thread = Thread(target=self._run, name="turing-machine-runner", daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = 2.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
            if thread.is_alive():
                raise TimeoutError("machine runner did not stop")

    def _publish(self) -> None:
        self.snapshots.publish(
            self.machine, direction=self._direction, transitions=self._transitions,
            results=self._last_results, outputs=self.output_provider(),
        )

    def _advance(self, limit: int) -> int:
        direction = self._direction
        if direction is MachineRunDirection.PAUSED:
            return 0
        completed = 0
        for _ in range(limit):
            if self._direction is not direction:
                break
            if direction is MachineRunDirection.FORWARD:
                self._last_results = self.machine.cycle_forward()
                self._transitions += 1
                completed += 1
                if any(
                    result.status is not MachineExecutionStatus.RUNNING
                    for result in self._last_results
                ):
                    self._direction = MachineRunDirection.PAUSED
                    break
            else:
                try:
                    self.machine.cycle_backward()
                except IndexError:
                    self._direction = MachineRunDirection.PAUSED
                    break
                self._last_results = ()
                self._transitions += 1
                completed += 1
        return completed

    def tick(self, transitions: int = 1) -> int:
        """Consume one shell tick with an exact caller-selected transition budget."""

        if self.running:
            raise RuntimeError("external ticks cannot drive an active maximum-speed runner")
        if transitions < 0:
            raise ValueError("tick transition budget cannot be negative")
        completed = self._advance(int(transitions))
        self._publish()
        return completed

    def regulated_tick(self, clock: ExternalMachineClock, elapsed_seconds: float) -> int:
        """Consume a shell tick governed by a mutable transitions-per-second clock."""

        return self.tick(clock.budget(elapsed_seconds))

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                direction = self._direction
                if direction is MachineRunDirection.PAUSED:
                    self._stop.wait(0.001)
                    continue
                completed = self._advance(self.transitions_per_publication)
                if completed or self._direction is MachineRunDirection.PAUSED:
                    self._publish()
        except BaseException as error:  # surfaced to the owning host
            self.failure = error
            self._direction = MachineRunDirection.PAUSED


__all__ = [
    "ExternalMachineClock",
    "FreeRunningMachineRunner",
    "MachineRunDirection",
    "MachineSnapshotHeader",
    "MachineSnapshotLayout",
    "MachineSnapshotTripleBuffer",
    "MachineSnapshotView",
    "SubjectOutputBuffer",
    "SubjectOutputDescriptor",
    "SubjectOutputFormat",
    "SubjectOutputKind",
]
