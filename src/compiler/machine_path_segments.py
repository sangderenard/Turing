"""Content-addressed exact-state suffixes for possible-world read heads.

Each head references its parent's prefix through a fork position and stores
only one full anchor plus states produced in its private world. Segments are
independently decodable, so replay and cold reverse keep at most one chunk in
memory and never depend on an unflushed parent object.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import gzip
from hashlib import sha256
import json
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence

from .machine_execution import MachineExecutionState
from .machine_system_tape import decode_machine_state, encode_machine_state


PATH_STATE_STORE_SCHEMA = "turing-machine-path-state-segment-store"
PATH_STATE_SEGMENT_SCHEMA = "turing-machine-path-state-segment"


@dataclass(frozen=True, slots=True)
class MachinePathStateSegmentDescriptor:
    digest: str
    state_count: int
    first_position: int
    last_position: int

    def to_mapping(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "state_count": self.state_count,
            "first_position": self.first_position,
            "last_position": self.last_position,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]):
        return cls(
            str(value["digest"]), int(value["state_count"]),
            int(value["first_position"]), int(value["last_position"]),
        )


@dataclass(frozen=True, slots=True)
class MachinePathStateHeadDescriptor:
    head_id: str
    parent_head_id: str | None
    fork_position: int | None
    anchor_position: int
    segments: tuple[MachinePathStateSegmentDescriptor, ...]
    constraints: tuple[Mapping[str, Any], ...] = ()

    def to_mapping(self) -> dict[str, Any]:
        return {
            "head_id": self.head_id,
            "parent_head_id": self.parent_head_id,
            "fork_position": self.fork_position,
            "anchor_position": self.anchor_position,
            "segments": [item.to_mapping() for item in self.segments],
            "constraints": [dict(item) for item in self.constraints],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]):
        return cls(
            str(value["head_id"]),
            None if value.get("parent_head_id") is None else str(value["parent_head_id"]),
            None if value.get("fork_position") is None else int(value["fork_position"]),
            int(value["anchor_position"]),
            tuple(MachinePathStateSegmentDescriptor.from_mapping(item) for item in value["segments"]),
            tuple(MappingProxyType(dict(item)) for item in value.get("constraints", ())),
        )


class SegmentedMachinePathStateStore:
    """Thread-safe DAG storage for exact possible-world machine states."""

    def __init__(
        self,
        root: str | Path,
        *,
        create: bool = False,
        states_per_segment: int = 256,
    ) -> None:
        if states_per_segment <= 0:
            raise ValueError("path-state segment capacity must be positive")
        self.root = Path(root)
        self.states_per_segment = int(states_per_segment)
        self._objects = self.root / "objects"
        self._lock = RLock()
        self._tails: dict[str, list[dict[str, Any]]] = {}
        self._previous: dict[str, MachineExecutionState] = {}
        self._last_positions: dict[str, int] = {}
        self._cached_digest: str | None = None
        self._cached_records: tuple[Mapping[str, Any], ...] = ()
        manifest_path = self.root / "manifest.json"
        if create:
            if manifest_path.exists():
                raise FileExistsError(f"path-state store already exists at {self.root}")
            self._objects.mkdir(parents=True, exist_ok=True)
            self._heads: dict[str, MachinePathStateHeadDescriptor] = {}
            self._write_manifest()
            return
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != PATH_STATE_STORE_SCHEMA or manifest.get("version") != 1:
            raise ValueError("unsupported machine path-state segment store")
        self.states_per_segment = int(manifest["states_per_segment"])
        self._heads = {
            str(item["head_id"]): MachinePathStateHeadDescriptor.from_mapping(item)
            for item in manifest.get("heads", ())
        }
        self._validate_graph()

    @property
    def heads(self) -> Mapping[str, MachinePathStateHeadDescriptor]:
        return MappingProxyType(dict(self._heads))

    @property
    def cached_state_count(self) -> int:
        return len(self._cached_records)

    @property
    def resident_head_ids(self) -> tuple[str, ...]:
        """Heads whose mutable append cursor is currently retained in RAM."""

        return tuple(sorted(self._previous))

    def clear_read_cache(self) -> None:
        """Drop the one decoded immutable segment retained for nearby reads."""

        with self._lock:
            self._cached_digest = None
            self._cached_records = ()

    def release_head(self, head_id: str | int) -> None:
        """Seal and evict one dormant head's mutable state from memory.

        The manifest and immutable content-addressed objects remain sufficient
        to reopen, reverse, or append the head later.  Re-appending lazily
        reconstructs only its final state and then starts a fresh segment.
        """

        identity = str(head_id)
        with self._lock:
            if identity not in self._heads:
                raise KeyError(f"unknown machine path-state head {identity}")
            self._flush_head_locked(identity)
            self._tails.pop(identity, None)
            self._previous.pop(identity, None)
            self._last_positions.pop(identity, None)
            self._write_manifest()

    def release_all(self) -> None:
        """Seal every head and retain only the small persistent DAG index."""

        with self._lock:
            for identity in tuple(self._heads):
                self._flush_head_locked(identity)
            self._tails.clear()
            self._previous.clear()
            self._last_positions.clear()
            self._cached_digest = None
            self._cached_records = ()
            self._write_manifest()

    def _validate_graph(self) -> None:
        for identity in self._heads:
            active: str | None = identity
            seen: set[str] = set()
            while active is not None:
                if active in seen:
                    raise ValueError("machine path-state head graph contains a cycle")
                seen.add(active)
                head = self._heads.get(active)
                if head is None:
                    raise ValueError("machine path-state head references an unknown parent")
                if (head.parent_head_id is None) != (head.fork_position is None):
                    raise ValueError("only child path-state heads have fork positions")
                active = head.parent_head_id

    def _write_manifest(self) -> None:
        payload = {
            "schema": PATH_STATE_STORE_SCHEMA,
            "version": 1,
            "states_per_segment": self.states_per_segment,
            "heads": [self._heads[key].to_mapping() for key in sorted(self._heads)],
        }
        temporary = self.root / "manifest.json.tmp"
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.root / "manifest.json")

    def create_head(
        self,
        head_id: str | int,
        anchor_position: int,
        anchor_state: MachineExecutionState,
        *,
        parent_head_id: str | int | None = None,
        fork_position: int | None = None,
        constraints: Sequence[Mapping[str, Any]] = (),
    ) -> MachinePathStateHeadDescriptor:
        """Create one head with a full exact anchor, never a copied prefix."""

        identity = str(head_id)
        parent = None if parent_head_id is None else str(parent_head_id)
        with self._lock:
            if identity in self._heads:
                raise ValueError(f"machine path-state head {identity} already exists")
            if parent is None:
                if fork_position is not None:
                    raise ValueError("root path-state head cannot have a fork position")
            elif parent not in self._heads:
                raise KeyError(f"unknown parent path-state head {parent}")
            elif fork_position is None:
                raise ValueError("child path-state head requires a fork position")
            descriptor = MachinePathStateHeadDescriptor(
                identity, parent, None if fork_position is None else int(fork_position),
                int(anchor_position), (),
                tuple(MappingProxyType(dict(item)) for item in constraints),
            )
            self._heads[identity] = descriptor
            self._tails[identity] = []
            self._previous[identity] = anchor_state
            self._last_positions[identity] = int(anchor_position) - 1
            self._append_locked(
                identity, int(anchor_position), anchor_state,
                event="anchor", metadata={"full_state_anchor": True},
            )
            self._flush_head_locked(identity)
            self._validate_graph()
            self._write_manifest()
            return self._heads[identity]

    def _begin_append(self, identity: str) -> None:
        if identity in self._previous:
            return
        last = None
        for last in self.iter_states(identity, include_metadata=True):
            pass
        if last is None:
            raise ValueError(f"machine path-state head {identity} has no anchor")
        self._previous[identity] = last[1]
        self._last_positions[identity] = int(last[0])
        self._tails[identity] = []

    def append_state(
        self,
        head_id: str | int,
        position: int,
        state: MachineExecutionState,
        *,
        event: str = "forward",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        identity = str(head_id)
        with self._lock:
            if identity not in self._heads:
                raise KeyError(f"unknown machine path-state head {identity}")
            self._begin_append(identity)
            self._append_locked(identity, int(position), state, event=event, metadata=metadata)

    def _append_locked(
        self,
        identity: str,
        position: int,
        state: MachineExecutionState,
        *,
        event: str,
        metadata: Mapping[str, Any] | None,
    ) -> None:
        previous_position = self._last_positions[identity]
        if position <= previous_position:
            raise ValueError("machine path-state positions must increase within a head")
        tail = self._tails[identity]
        record = {
            "position": position,
            "event": str(event),
            # Every object begins with a complete state and is independently decodable.
            "checkpoint": not tail,
            "state": encode_machine_state(
                state, None if not tail else self._previous[identity],
            ),
        }
        if metadata:
            record["metadata"] = dict(metadata)
        tail.append(record)
        self._previous[identity] = state
        self._last_positions[identity] = position
        if len(tail) >= self.states_per_segment:
            self._flush_head_locked(identity)

    def _flush_head_locked(self, identity: str) -> None:
        records = self._tails.get(identity, [])
        if not records:
            return
        payload = {
            "schema": PATH_STATE_SEGMENT_SCHEMA,
            "version": 1,
            "records": records,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest = sha256(encoded).hexdigest()
        path = self._objects / f"{digest}.json.gz"
        if not path.exists():
            with gzip.open(path, "wb", compresslevel=6) as stream:
                stream.write(encoded)
        segment = MachinePathStateSegmentDescriptor(
            digest, len(records), int(records[0]["position"]),
            int(records[-1]["position"]),
        )
        head = self._heads[identity]
        self._heads[identity] = replace(head, segments=(*head.segments, segment))
        self._tails[identity] = []
        self._write_manifest()

    def flush(self) -> Path:
        with self._lock:
            for identity in tuple(self._heads):
                self._flush_head_locked(identity)
            self._write_manifest()
        return self.root

    def _load_segment(
        self, descriptor: MachinePathStateSegmentDescriptor,
    ) -> tuple[Mapping[str, Any], ...]:
        if descriptor.digest == self._cached_digest:
            return self._cached_records
        with gzip.open(self._objects / f"{descriptor.digest}.json.gz", "rb") as stream:
            encoded = stream.read()
        if sha256(encoded).hexdigest() != descriptor.digest:
            raise ValueError("machine path-state segment digest mismatch")
        payload = json.loads(encoded)
        if payload.get("schema") != PATH_STATE_SEGMENT_SCHEMA or payload.get("version") != 1:
            raise ValueError("unsupported machine path-state segment")
        records = tuple(MappingProxyType(dict(item)) for item in payload["records"])
        if len(records) != descriptor.state_count or (
            int(records[0]["position"]) != descriptor.first_position
            or int(records[-1]["position"]) != descriptor.last_position
        ):
            raise ValueError("machine path-state segment descriptor mismatch")
        self._cached_digest = descriptor.digest
        self._cached_records = records
        return records

    def _iter_local_states(self, identity: str):
        head = self._heads[identity]
        for segment in head.segments:
            previous = None
            for record in self._load_segment(segment):
                if bool(record["checkpoint"]):
                    previous = None
                state = decode_machine_state(record["state"], previous)
                previous = state
                yield int(record["position"]), state, record
        previous = None
        for record in self._tails.get(identity, ()):
            if bool(record["checkpoint"]):
                previous = None
            state = decode_machine_state(record["state"], previous)
            previous = state
            yield int(record["position"]), state, MappingProxyType(record)

    def iter_states(
        self, head_id: str | int, *, include_metadata: bool = False,
    ) -> Iterator[Any]:
        """Stream the shared prefix and exact local suffix with one chunk resident."""

        identity = str(head_id)
        head = self._heads.get(identity)
        if head is None:
            raise KeyError(f"unknown machine path-state head {identity}")
        last_position: int | None = None
        if head.parent_head_id is not None:
            assert head.fork_position is not None
            for value in self.iter_states(head.parent_head_id, include_metadata=True):
                position, state, record = value
                if position <= head.fork_position:
                    last_position = position
                    yield value if include_metadata else (position, state)
        for position, state, record in self._iter_local_states(identity):
            if last_position is not None and position <= last_position:
                if position == last_position:
                    continue
                raise ValueError("machine path-state suffix precedes its shared prefix")
            last_position = position
            yield (position, state, record) if include_metadata else (position, state)

    def states_range(
        self, head_id: str | int, start: int, end: int,
    ) -> tuple[MachineExecutionState, ...]:
        selected = {
            int(position): state
            for position, state in self.iter_states(head_id)
            if int(start) <= int(position) < int(end)
        }
        cursor = int(end) - 1
        suffix = []
        while cursor >= int(start) and cursor in selected:
            suffix.append(selected[cursor])
            cursor -= 1
        return tuple(reversed(suffix))

    def latest_state(self, head_id: str | int) -> tuple[int, MachineExecutionState]:
        latest = None
        for latest in self.iter_states(head_id):
            pass
        if latest is None:
            raise IndexError("machine path-state head has no state")
        return latest


__all__ = [
    "MachinePathStateHeadDescriptor", "MachinePathStateSegmentDescriptor",
    "SegmentedMachinePathStateStore",
]
