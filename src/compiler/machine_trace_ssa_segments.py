"""Content-addressed, memory-bounded storage for branching trace SSA paths.

Each head names a parent head and a fork sequence.  The parent prefix is shared
by reference; only operations after the fork are written into the child's own
immutable chunks.  Loading a head streams the shared prefix and local suffix
while retaining at most one decoded chunk in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .machine_trace_ssa import MachineTraceSSAProgram


TRACE_SSA_STORE_SCHEMA = "turing-machine-trace-ssa-segment-store"
TRACE_SSA_SEGMENT_SCHEMA = "turing-machine-trace-ssa-segment"


@dataclass(frozen=True, slots=True)
class MachineTraceSSASegmentDescriptor:
    digest: str
    operation_count: int
    first_sequence: int
    last_sequence: int

    def to_mapping(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "operation_count": self.operation_count,
            "first_sequence": self.first_sequence,
            "last_sequence": self.last_sequence,
        }

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any],
    ) -> "MachineTraceSSASegmentDescriptor":
        return cls(
            str(value["digest"]), int(value["operation_count"]),
            int(value["first_sequence"]), int(value["last_sequence"]),
        )


@dataclass(frozen=True, slots=True)
class MachineTraceSSAHeadDescriptor:
    head_id: str
    parent_head_id: str | None
    fork_sequence: int | None
    core: int
    specialization: str
    segments: tuple[MachineTraceSSASegmentDescriptor, ...]
    final_values: Mapping[str, str]
    reduction_witness: Mapping[str, Any] | None
    constraints: tuple[Mapping[str, Any], ...]

    def to_mapping(self) -> dict[str, Any]:
        result = {
            "head_id": self.head_id,
            "parent_head_id": self.parent_head_id,
            "fork_sequence": self.fork_sequence,
            "core": self.core,
            "specialization": self.specialization,
            "segments": [item.to_mapping() for item in self.segments],
            "final_values": dict(self.final_values),
            "constraints": [dict(item) for item in self.constraints],
        }
        if self.reduction_witness is not None:
            result["reduction_witness"] = dict(self.reduction_witness)
        return result

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any],
    ) -> "MachineTraceSSAHeadDescriptor":
        witness = value.get("reduction_witness")
        return cls(
            str(value["head_id"]),
            None if value.get("parent_head_id") is None else str(value["parent_head_id"]),
            None if value.get("fork_sequence") is None else int(value["fork_sequence"]),
            int(value["core"]), str(value["specialization"]),
            tuple(MachineTraceSSASegmentDescriptor.from_mapping(item) for item in value["segments"]),
            MappingProxyType({str(key): str(item) for key, item in value.get("final_values", {}).items()}),
            None if witness is None else MappingProxyType(dict(witness)),
            tuple(MappingProxyType(dict(item)) for item in value.get("constraints", ())),
        )


class SegmentedMachineTraceSSAStore:
    """A DAG of SSA path heads backed by deduplicated immutable chunks."""

    def __init__(self, root: str | Path, *, create: bool = False) -> None:
        self.root = Path(root)
        self._objects = self.root / "objects"
        self._cached_digest: str | None = None
        self._cached_operations: tuple[Mapping[str, Any], ...] = ()
        manifest_path = self.root / "manifest.json"
        if create:
            if manifest_path.exists():
                raise FileExistsError(f"trace SSA segment store already exists at {self.root}")
            self._objects.mkdir(parents=True, exist_ok=True)
            self._heads: dict[str, MachineTraceSSAHeadDescriptor] = {}
            self._write_manifest()
            return
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != TRACE_SSA_STORE_SCHEMA or manifest.get("version") != 1:
            raise ValueError("unsupported trace SSA segment store")
        self._heads = {
            str(item["head_id"]): MachineTraceSSAHeadDescriptor.from_mapping(item)
            for item in manifest.get("heads", ())
        }
        self._validate_head_graph()

    @property
    def heads(self) -> Mapping[str, MachineTraceSSAHeadDescriptor]:
        return MappingProxyType(dict(self._heads))

    @property
    def cached_operation_count(self) -> int:
        return len(self._cached_operations)

    def clear_read_cache(self) -> None:
        """Evict the sole decoded SSA chunk; descriptors remain reloadable."""

        self._cached_digest = None
        self._cached_operations = ()

    def _validate_head_graph(self) -> None:
        for identity in self._heads:
            seen: set[str] = set()
            active: str | None = identity
            while active is not None:
                if active in seen:
                    raise ValueError("trace SSA head graph contains a cycle")
                seen.add(active)
                head = self._heads.get(active)
                if head is None:
                    raise ValueError("trace SSA head references an unknown parent")
                if head.parent_head_id is None and head.fork_sequence is not None:
                    raise ValueError("root trace SSA head cannot have a fork sequence")
                if head.parent_head_id is not None and head.fork_sequence is None:
                    raise ValueError("child trace SSA head requires a fork sequence")
                active = head.parent_head_id

    def _write_manifest(self) -> None:
        payload = {
            "schema": TRACE_SSA_STORE_SCHEMA,
            "version": 1,
            "heads": [self._heads[key].to_mapping() for key in sorted(self._heads)],
        }
        temporary = self.root / "manifest.json.tmp"
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.root / "manifest.json")

    def _write_segment(
        self, operations: Sequence[Mapping[str, Any]],
    ) -> MachineTraceSSASegmentDescriptor:
        payload = {
            "schema": TRACE_SSA_SEGMENT_SCHEMA,
            "version": 1,
            "operations": list(operations),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest = sha256(encoded).hexdigest()
        path = self._objects / f"{digest}.json.gz"
        if not path.exists():
            with gzip.open(path, "wb", compresslevel=6) as stream:
                stream.write(encoded)
        return MachineTraceSSASegmentDescriptor(
            digest, len(operations), int(operations[0]["sequence"]),
            int(operations[-1]["sequence"]),
        )

    def add_head(
        self,
        head_id: str | int,
        program: MachineTraceSSAProgram,
        *,
        parent_head_id: str | int | None = None,
        fork_sequence: int | None = None,
        constraints: Sequence[Mapping[str, Any]] = (),
        operations_per_segment: int = 256,
    ) -> MachineTraceSSAHeadDescriptor:
        """Persist one head, sharing its parent prefix instead of copying it."""

        return self.add_operation_stream(
            head_id,
            (item.to_mapping() for item in program.operations),
            core=program.core,
            specialization=program.specialization,
            final_values=program.final_values,
            reduction_witness=program.reduction_witness,
            parent_head_id=parent_head_id,
            fork_sequence=fork_sequence,
            constraints=constraints,
            operations_per_segment=operations_per_segment,
        )

    def add_operation_stream(
        self,
        head_id: str | int,
        operations: Iterable[Mapping[str, Any]],
        *,
        core: int,
        specialization: str,
        final_values: Mapping[str, str],
        reduction_witness: Mapping[str, Any] | None = None,
        parent_head_id: str | int | None = None,
        fork_sequence: int | None = None,
        constraints: Sequence[Mapping[str, Any]] = (),
        operations_per_segment: int = 256,
    ) -> MachineTraceSSAHeadDescriptor:
        """Seal a possibly unbounded operation iterator into bounded chunks."""

        if operations_per_segment <= 0:
            raise ValueError("trace SSA segment capacity must be positive")
        identity = str(head_id)
        if identity in self._heads:
            raise ValueError(f"trace SSA head {identity} already exists")
        parent = None if parent_head_id is None else str(parent_head_id)
        if parent is None:
            if fork_sequence is not None:
                raise ValueError("root trace SSA head cannot specify a fork sequence")
        else:
            if parent not in self._heads:
                raise KeyError(f"unknown parent trace SSA head {parent}")
            if fork_sequence is None:
                raise ValueError("child trace SSA head requires a fork sequence")
        descriptors: list[MachineTraceSSASegmentDescriptor] = []
        chunk: list[Mapping[str, Any]] = []
        previous_sequence: int | None = None
        for operation in operations:
            value = dict(operation)
            sequence = int(value["sequence"])
            if parent is not None and sequence <= int(fork_sequence):
                continue
            if previous_sequence is not None and sequence <= previous_sequence:
                raise ValueError("trace SSA head operations must increase by sequence")
            previous_sequence = sequence
            chunk.append(value)
            if len(chunk) >= operations_per_segment:
                descriptors.append(self._write_segment(chunk))
                chunk = []
        if chunk:
            descriptors.append(self._write_segment(chunk))
        descriptor = MachineTraceSSAHeadDescriptor(
            identity, parent, fork_sequence, int(core), str(specialization),
            tuple(descriptors), MappingProxyType(dict(final_values)),
            reduction_witness,
            tuple(MappingProxyType(dict(item)) for item in constraints),
        )
        self._heads[identity] = descriptor
        self._validate_head_graph()
        self._write_manifest()
        return descriptor

    def _load_segment(
        self, descriptor: MachineTraceSSASegmentDescriptor,
    ) -> tuple[Mapping[str, Any], ...]:
        if descriptor.digest == self._cached_digest:
            return self._cached_operations
        with gzip.open(self._objects / f"{descriptor.digest}.json.gz", "rb") as stream:
            encoded = stream.read()
        if sha256(encoded).hexdigest() != descriptor.digest:
            raise ValueError("trace SSA segment digest mismatch")
        payload = json.loads(encoded)
        if payload.get("schema") != TRACE_SSA_SEGMENT_SCHEMA or payload.get("version") != 1:
            raise ValueError("unsupported trace SSA segment")
        operations = tuple(MappingProxyType(dict(item)) for item in payload["operations"])
        if len(operations) != descriptor.operation_count:
            raise ValueError("trace SSA segment operation count mismatch")
        if operations and (
            int(operations[0]["sequence"]) != descriptor.first_sequence
            or int(operations[-1]["sequence"]) != descriptor.last_sequence
        ):
            raise ValueError("trace SSA segment sequence bounds mismatch")
        self._cached_digest = descriptor.digest
        self._cached_operations = operations
        return operations

    def iter_operations(self, head_id: str | int) -> Iterator[Mapping[str, Any]]:
        """Stream a complete path: shared ancestry followed by local suffixes."""

        head = self._heads.get(str(head_id))
        if head is None:
            raise KeyError(f"unknown trace SSA head {head_id}")
        if head.parent_head_id is not None:
            assert head.fork_sequence is not None
            for operation in self.iter_operations(head.parent_head_id):
                if int(operation["sequence"]) <= head.fork_sequence:
                    yield operation
        for segment in head.segments:
            yield from self._load_segment(segment)

    def operation_count(self, head_id: str | int) -> int:
        return sum(1 for _item in self.iter_operations(head_id))


__all__ = [
    "MachineTraceSSAHeadDescriptor", "MachineTraceSSASegmentDescriptor",
    "SegmentedMachineTraceSSAStore",
]
