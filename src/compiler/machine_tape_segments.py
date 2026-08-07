"""Content-addressed, memory-bounded segments for machine tape subpaths."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import gzip
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping

from .machine_execution import MachineExecutionState, MachineExternalReference
from .machine_system_tape import (
    MachineTapeAnnotation,
    MachineTapeDependencyNode,
    _ANNOTATION_RGBA8,
    _binding_from_mapping,
    _binding_mapping,
    _module_from_mapping,
    _module_mapping,
    decode_machine_state,
    encode_machine_state,
)


SEGMENT_STORE_SCHEMA = "turing-machine-tape-segment-store"
SEGMENT_SCHEMA = "turing-machine-tape-segment"


@dataclass(frozen=True, slots=True)
class MachineTapeSegmentDescriptor:
    digest: str
    start_sequence: int
    end_sequence: int
    record_count: int
    parent_digest: str | None
    checkpoints: tuple[int, ...]
    core_sequences: tuple[tuple[int, int, int], ...]

    def contains(self, sequence: int) -> bool:
        return self.start_sequence <= sequence <= self.end_sequence

    def to_mapping(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "start_sequence": self.start_sequence,
            "end_sequence": self.end_sequence,
            "record_count": self.record_count,
            "parent_digest": self.parent_digest,
            "checkpoints": list(self.checkpoints),
            "core_sequences": [
                {"core": core, "first": first, "last": last}
                for core, first, last in self.core_sequences
            ],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MachineTapeSegmentDescriptor":
        return cls(
            str(value["digest"]),
            int(value["start_sequence"]),
            int(value["end_sequence"]),
            int(value["record_count"]),
            None if value.get("parent_digest") is None else str(value["parent_digest"]),
            tuple(int(item) for item in value.get("checkpoints", ())),
            tuple(
                (int(item["core"]), int(item["first"]), int(item["last"]))
                for item in value.get("core_sequences", ())
            ),
        )


class SegmentedMachineTapeStore:
    """Load only bounded content-addressed segments on one requested lineage."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema") != SEGMENT_STORE_SCHEMA or manifest.get("version") != 1:
            raise ValueError("unsupported segmented machine tape store")
        self.core_count = int(manifest["core_count"])
        self.checkpoint_interval = int(manifest["checkpoint_interval"])
        self.record_count = int(manifest["record_count"])
        self.subject_digest = str(manifest["subject_digest"])
        self.segments = tuple(
            MachineTapeSegmentDescriptor.from_mapping(item)
            for item in manifest["segments"]
        )
        self.external_references = [MachineExternalReference(
            int(item["reference_id"]), int(item["target_address"]),
            str(item["domain"]), str(item["library"]), str(item["symbol"]),
        ) for item in manifest.get("external_references", ())]
        self.linked_modules = []
        for item in manifest.get("linked_modules", ()):
            digest = str(item["digest"])
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("invalid segmented linked module digest")
            payload = (self.root / "modules" / f"{digest}.bin").read_bytes()
            self.linked_modules.append(_module_from_mapping(item, payload))
        self.import_bindings = [
            _binding_from_mapping(item)
            for item in manifest.get("import_bindings", ())
        ]
        self._runtime_dispatch_indexed = "runtime_dispatch_targets" in manifest
        self.runtime_dispatch_targets = tuple(
            int(item) for item in manifest.get("runtime_dispatch_targets", ())
        )
        self.annotations = [
            MachineTapeAnnotation.from_mapping(item)
            for item in manifest.get("annotations", ())
        ]
        self.origin_receipt = MappingProxyType(dict(manifest.get("origin_receipt", {})))
        self._cached_digest: str | None = None
        self._cached_records: tuple[dict[str, Any], ...] = ()
        self._tail_records: list[dict[str, Any]] = []
        self._append_last_states: dict[int, MachineExecutionState] = {}
        self._append_last_sequences: dict[int, int] = {}
        self._append_request_sequences: dict[tuple[int, int], int] = {}
        if self.record_count != sum(item.record_count for item in self.segments):
            raise ValueError("segmented tape manifest record count is inconsistent")
        expected = 0
        previous = None
        for segment in self.segments:
            if segment.start_sequence != expected or segment.parent_digest != previous:
                raise ValueError("segmented tape chain is not contiguous")
            expected = segment.end_sequence + 1
            previous = segment.digest

    @property
    def records(self):
        store = self

        class Records:
            def __len__(self):
                return store.record_count

            def __getitem__(self, index):
                if isinstance(index, slice):
                    return tuple(store.record(item) for item in range(*index.indices(store.record_count)))
                active = int(index)
                if active < 0:
                    active += store.record_count
                return store.record(active)

            def __iter__(self):
                for sequence in range(store.record_count):
                    yield store.record(sequence)

        return Records()

    @property
    def runtime_dispatch_indexed(self) -> bool:
        return self._runtime_dispatch_indexed

    @property
    def subject_binary(self) -> bytes:
        payload = (self.root / "subject.bin").read_bytes()
        if sha256(payload).hexdigest() != self.subject_digest:
            raise ValueError("segmented tape subject digest mismatch")
        return payload

    def _descriptor(self, sequence: int) -> MachineTapeSegmentDescriptor:
        if not 0 <= sequence < self.record_count:
            raise IndexError("segmented tape sequence is out of range")
        low, high = 0, len(self.segments)
        while low < high:
            middle = (low + high) // 2
            segment = self.segments[middle]
            if sequence < segment.start_sequence:
                high = middle
            elif sequence > segment.end_sequence:
                low = middle + 1
            else:
                return segment
        raise IndexError("segmented tape has no descriptor for sequence")

    def _load(self, descriptor: MachineTapeSegmentDescriptor) -> tuple[dict[str, Any], ...]:
        if self._cached_digest == descriptor.digest:
            return self._cached_records
        path = self.root / "segments" / f"{descriptor.digest}.json.gz"
        with gzip.open(path, "rb") as stream:
            encoded = stream.read()
        if sha256(encoded).hexdigest() != descriptor.digest:
            raise ValueError("machine tape segment digest mismatch")
        value = json.loads(encoded)
        if value.get("schema") != SEGMENT_SCHEMA or value.get("version") != 1:
            raise ValueError("unsupported machine tape segment")
        records = tuple(value["records"])
        if len(records) != descriptor.record_count:
            raise ValueError("machine tape segment record count mismatch")
        self._cached_digest = descriptor.digest
        self._cached_records = records
        return records

    def record(self, sequence: int) -> Mapping[str, Any]:
        if self._tail_records and sequence >= self._tail_records[0]["sequence"]:
            index = int(sequence) - int(self._tail_records[0]["sequence"])
            if 0 <= index < len(self._tail_records):
                return MappingProxyType(self._tail_records[index])
        descriptor = self._descriptor(int(sequence))
        records = self._load(descriptor)
        record = records[int(sequence) - descriptor.start_sequence]
        if int(record["sequence"]) != int(sequence):
            raise ValueError("machine tape segment sequence index mismatch")
        return MappingProxyType(record)

    def latest_sequence(
        self,
        core: int,
        *,
        limit: int | None = None,
        position: int | None = None,
    ) -> int:
        if not 0 <= core < self.core_count:
            raise IndexError("segmented tape core index is out of range")
        maximum = self.record_count - 1 if limit is None else min(int(limit), self.record_count - 1)
        for record in reversed(self._tail_records):
            if (
                int(record["sequence"]) <= maximum
                and int(record["core"]) == core
                and (position is None or int(record["position"]) == int(position))
            ):
                return int(record["sequence"])
        for descriptor in reversed(self.segments):
            if descriptor.start_sequence > maximum:
                continue
            bounds = next((item for item in descriptor.core_sequences if item[0] == core), None)
            if bounds is None:
                continue
            for record in reversed(self._load(descriptor)):
                if (
                    int(record["sequence"]) <= maximum
                    and int(record["core"]) == core
                    and (position is None or int(record["position"]) == int(position))
                ):
                    return int(record["sequence"])
        raise IndexError("segmented tape contains no matching core state")

    def lineage(self, core: int = 0, *, sequence: int | None = None) -> tuple[int, ...]:
        target = self.latest_sequence(core, limit=sequence)
        result = []
        active = target
        while True:
            record = self.record(active)
            if int(record["core"]) != core:
                raise ValueError("segmented tape parent edge crosses virtual cores")
            result.append(active)
            for dependency in record.get("dependencies", ()):
                referenced = int(dependency["sequence"])
                if referenced >= active:
                    raise ValueError("segmented tape dependency is not earlier than its node")
                self.record(referenced)  # existence and segment digest validation
            parent = record.get("parent_sequence")
            if parent is None:
                return tuple(reversed(result))
            active = int(parent)
            if active >= result[-1]:
                raise ValueError("segmented tape parent is not an earlier node")

    def lineage_states(
        self, core: int = 0, *, sequence: int | None = None,
    ) -> tuple[tuple[MachineTapeDependencyNode, MachineExecutionState], ...]:
        sequences = self.lineage(core, sequence=sequence)
        checkpoint_index = max(
            index for index, item in enumerate(sequences)
            if bool(self.record(item)["checkpoint"])
        )
        state = None
        result = []
        for item in sequences[checkpoint_index:]:
            record = self.record(item)
            state = decode_machine_state(
                record["state"], None if record["checkpoint"] else state,
            )
            node = MachineTapeDependencyNode(
                item, int(record["core"]), str(record["event"]), int(record["position"]),
                None if record.get("parent_sequence") is None else int(record["parent_sequence"]),
                tuple(
                    (str(value["kind"]), int(value["sequence"]))
                    for value in record.get("dependencies", ())
                ),
            )
            result.append((node, state))
        return tuple(result)

    def resume_state(self, core: int = 0, *, sequence: int | None = None) -> MachineExecutionState:
        states = self.lineage_states(core, sequence=sequence)
        if not states:
            raise IndexError("segmented tape lineage has no decodable state")
        return states[-1][1]

    def begin_append(self) -> None:
        """Initialize bounded append indexes from current segment tips."""

        if self._append_last_states:
            return
        for core in range(self.core_count):
            sequence = self.latest_sequence(core)
            state = self.resume_state(core)
            self._append_last_sequences[core] = sequence
            self._append_last_states[core] = state
            for request in state.external_requests:
                self._append_request_sequences[(core, request.request_id)] = sequence

    def append(
        self,
        core: int,
        state: MachineExecutionState,
        *,
        position: int,
        event: str,
        dependencies: tuple[tuple[str, int], ...] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Append to a bounded tail and flush one immutable segment at capacity."""

        self.begin_append()
        if not 0 <= core < self.core_count:
            raise IndexError("segmented tape core index is out of range")
        previous = self._append_last_states[core]
        sequence = self.record_count
        checkpoint = sequence % self.checkpoint_interval == 0
        active_dependencies = list(dependencies)
        previous_requests = {item.request_id for item in previous.external_requests}
        current_requests = {item.request_id for item in state.external_requests}
        for request_id in sorted(previous_requests - current_requests):
            source = self._append_request_sequences.pop((core, request_id), None)
            if source is not None:
                active_dependencies.append((
                    "external_request" if event == "external_completion" else "abandons_request",
                    source,
                ))
        record = {
            "sequence": sequence,
            "core": int(core),
            "position": int(position),
            "event": str(event),
            "checkpoint": checkpoint,
            "parent_sequence": self._append_last_sequences[core],
            "dependencies": [
                {"kind": str(kind), "sequence": int(source)}
                for kind, source in dict.fromkeys(active_dependencies)
            ],
            "state": encode_machine_state(state, None if checkpoint else previous),
        }
        if metadata:
            record["metadata"] = dict(metadata)
        if event == "runtime_dispatch" and metadata:
            self.runtime_dispatch_targets = tuple(dict.fromkeys((
                *self.runtime_dispatch_targets,
                *(int(item) for item in metadata.get("targets", ())),
            )))
        self._tail_records.append(record)
        self.record_count += 1
        self._append_last_states[core] = state
        self._append_last_sequences[core] = sequence
        for request in state.external_requests:
            if request.request_id not in previous_requests:
                self._append_request_sequences[(core, request.request_id)] = sequence
        if len(self._tail_records) >= 256:
            self.flush()

    def index_runtime_dispatch_targets(self, targets) -> None:
        """Install a complete legacy-store index discovered by one bounded scan."""

        self.runtime_dispatch_targets = tuple(dict.fromkeys(
            int(item) for item in targets
        ))
        self._runtime_dispatch_indexed = True

    def catalog_external_reference(self, reference: MachineExternalReference) -> None:
        for existing in self.external_references:
            if existing == reference:
                return
            if existing.target_address == reference.target_address or existing.reference_id == reference.reference_id:
                raise ValueError("segmented tape external reference identity conflicts")
        self.external_references.append(reference)

    def annotate(
        self,
        feature: str,
        message: str,
        *,
        color: str = "amber",
        severity: str = "note",
        sequence: int | None = None,
        end_sequence: int | None = None,
        core: int | None = None,
        position: int | None = None,
        address: int | None = None,
        external_reference: str | None = None,
        metadata: Mapping[str, object] | None = None,
        supersedes: int | None = None,
    ) -> MachineTapeAnnotation:
        start = self.record_count - 1 if sequence is None else int(sequence)
        end = start if end_sequence is None else int(end_sequence)
        annotation = MachineTapeAnnotation(
            len(self.annotations), start, end, str(feature), str(message), str(color),
            str(severity), core, position, address, external_reference,
            tuple(sorted((str(key), str(value)) for key, value in (metadata or {}).items())),
            supersedes,
        )
        self.annotations.append(annotation)
        return annotation

    def annotations_at(self, sequence: int, *, core: int | None = None):
        return tuple(item for item in self.annotations if item.sequence_start <= sequence <= item.sequence_end and (core is None or item.core is None or item.core == core))

    def active_annotations_at(self, sequence: int, *, core: int | None = None):
        candidates = self.annotations_at(sequence, core=core)
        superseded = {item.supersedes for item in candidates if item.supersedes is not None}
        return tuple(item for item in candidates if item.annotation_id not in superseded)

    def annotation_color_rgba8(self, sequence: int, *, core: int | None = None) -> int:
        active = self.active_annotations_at(sequence, core=core)
        if not active:
            return 0
        color = active[-1].color
        if color.startswith("#"):
            raw = color[1:] + ("FF" if len(color) == 7 else "")
            red, green, blue, alpha = (int(raw[index:index + 2], 16) for index in range(0, 8, 2))
            return red | (green << 8) | (blue << 16) | (alpha << 24)
        return _ANNOTATION_RGBA8[color.casefold()]

    def _manifest_mapping(self) -> dict[str, Any]:
        result = {
            "schema": SEGMENT_STORE_SCHEMA,
            "version": 1,
            "core_count": self.core_count,
            "checkpoint_interval": self.checkpoint_interval,
            "record_count": self.record_count,
            "subject_digest": self.subject_digest,
            "external_references": [
                {
                    "reference_id": item.reference_id,
                    "target_address": item.target_address,
                    "domain": item.domain,
                    "library": item.library,
                    "symbol": item.symbol,
                }
                for item in self.external_references
            ],
            "linked_modules": [
                _module_mapping(module, include_binary=False)
                for module in self.linked_modules
            ],
            "import_bindings": [
                _binding_mapping(binding) for binding in self.import_bindings
            ],
            "annotations": [item.to_mapping() for item in self.annotations],
            "segments": [item.to_mapping() for item in self.segments],
        }
        if self._runtime_dispatch_indexed:
            result["runtime_dispatch_targets"] = list(self.runtime_dispatch_targets)
        if self.origin_receipt:
            result["origin_receipt"] = dict(self.origin_receipt)
        return result

    def crop(
        self,
        root: str | Path,
        *,
        sequence: int | None = None,
    ) -> "SegmentedMachineTapeStore":
        """Seal selected core states as position-zero roots in a new tape.

        Unlike a branch, a crop intentionally carries no executable ancestry.
        It retains a content-addressed receipt so provenance can be audited
        without keeping the source segments available at runtime.
        """

        self.flush()
        target = Path(root)
        if target.exists():
            raise FileExistsError(f"refusing to replace existing graph crop {target}")
        target.mkdir(parents=True)
        (target / "segments").mkdir()
        subject = self.subject_binary
        (target / "subject.bin").write_bytes(subject)
        module_mappings = []
        if self.linked_modules:
            (target / "modules").mkdir()
            for module in self.linked_modules:
                (target / "modules" / f"{module.digest}.bin").write_bytes(module.binary)
                module_mappings.append(_module_mapping(module, include_binary=False))

        source_manifest = (self.root / "manifest.json").read_bytes()
        source_sequences = tuple(
            self.latest_sequence(core, limit=sequence)
            for core in range(self.core_count)
        )
        states = tuple(
            self.resume_state(core, sequence=source_sequence)
            for core, source_sequence in enumerate(source_sequences)
        )
        encoded_states = tuple(encode_machine_state(state, None) for state in states)
        state_digest = sha256(json.dumps(
            encoded_states, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        receipt = {
            "schema": "turing-machine-graph-crop-origin-v1",
            "source_manifest_digest": sha256(source_manifest).hexdigest(),
            "source_subject_digest": self.subject_digest,
            "source_sequences": list(source_sequences),
            "source_positions": [
                int(self.record(item)["position"]) for item in source_sequences
            ],
            "state_digest": state_digest,
        }
        records = [{
            "sequence": core,
            "core": core,
            "position": 0,
            "event": "graph_crop_root",
            "checkpoint": True,
            "parent_sequence": None,
            "dependencies": [],
            "metadata": {"origin_receipt": receipt},
            "state": encoded_states[core],
        } for core in range(self.core_count)]
        payload = {
            "schema": SEGMENT_SCHEMA,
            "version": 1,
            "parent_digest": None,
            "records": records,
        }
        encoded_segment = json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        digest = sha256(encoded_segment).hexdigest()
        with gzip.open(target / "segments" / f"{digest}.json.gz", "wb", compresslevel=6) as stream:
            stream.write(encoded_segment)
        descriptor = MachineTapeSegmentDescriptor(
            digest, 0, self.core_count - 1, self.core_count, None,
            tuple(range(self.core_count)),
            tuple((core, core, core) for core in range(self.core_count)),
        )
        manifest = {
            "schema": SEGMENT_STORE_SCHEMA,
            "version": 1,
            "core_count": self.core_count,
            "checkpoint_interval": self.checkpoint_interval,
            "record_count": self.core_count,
            "subject_digest": sha256(subject).hexdigest(),
            "external_references": [{
                "reference_id": item.reference_id,
                "target_address": item.target_address,
                "domain": item.domain,
                "library": item.library,
                "symbol": item.symbol,
            } for item in self.external_references],
            "linked_modules": module_mappings,
            "import_bindings": [
                _binding_mapping(binding) for binding in self.import_bindings
            ],
            "annotations": [],
            "runtime_dispatch_targets": list(self.runtime_dispatch_targets),
            "origin_receipt": receipt,
            "segments": [descriptor.to_mapping()],
        }
        (target / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8",
        )
        return type(self)(target)

    def flush(self) -> Path:
        if self._tail_records:
            payload = {
                "schema": SEGMENT_SCHEMA,
                "version": 1,
                "parent_digest": self.segments[-1].digest if self.segments else None,
                "records": self._tail_records,
            }
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            digest = sha256(encoded).hexdigest()
            path = self.root / "segments" / f"{digest}.json.gz"
            with gzip.open(path, "wb", compresslevel=6) as stream:
                stream.write(encoded)
            by_core: dict[int, list[int]] = {}
            for record in self._tail_records:
                by_core.setdefault(int(record["core"]), []).append(int(record["sequence"]))
            descriptor = MachineTapeSegmentDescriptor(
                digest,
                int(self._tail_records[0]["sequence"]),
                int(self._tail_records[-1]["sequence"]),
                len(self._tail_records),
                self.segments[-1].digest if self.segments else None,
                tuple(int(item["sequence"]) for item in self._tail_records if item["checkpoint"]),
                tuple((core, min(values), max(values)) for core, values in sorted(by_core.items())),
            )
            self.segments = (*self.segments, descriptor)
            self._tail_records = []
        temporary = self.root / "manifest.json.tmp"
        temporary.write_text(json.dumps(self._manifest_mapping(), indent=2), encoding="utf-8")
        temporary.replace(self.root / "manifest.json")
        return self.root

    def write(self, path: str | Path) -> Path:
        if Path(path).resolve() != self.root.resolve():
            raise ValueError("segmented tape writes must target their existing store root")
        return self.flush()

    @classmethod
    def import_jsonl(
        cls,
        source: str | Path,
        root: str | Path,
        *,
        records_per_segment: int = 256,
    ) -> "SegmentedMachineTapeStore":
        """Stream a JSONL tape into bounded immutable segments."""

        if records_per_segment <= 0:
            raise ValueError("records_per_segment must be positive")
        source_path, target = Path(source), Path(root)
        manifest_path = target / "manifest.json"
        if manifest_path.exists():
            raise FileExistsError(f"segmented tape store already exists at {target}")
        target.mkdir(parents=True, exist_ok=True)
        segment_directory = target / "segments"
        segment_directory.mkdir(parents=True, exist_ok=True)
        descriptors: list[MachineTapeSegmentDescriptor] = []
        annotations: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []
        previous_digest = None
        runtime_dispatch_targets: list[int] = []

        def flush() -> None:
            nonlocal records, previous_digest
            if not records:
                return
            payload = {
                "schema": SEGMENT_SCHEMA,
                "version": 1,
                "parent_digest": previous_digest,
                "records": records,
            }
            encoded = json.dumps(
                payload, sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")
            digest = sha256(encoded).hexdigest()
            path = segment_directory / f"{digest}.json.gz"
            if not path.exists():
                with gzip.open(path, "wb", compresslevel=6) as stream:
                    stream.write(encoded)
            by_core: dict[int, list[int]] = {}
            for record in records:
                by_core.setdefault(int(record["core"]), []).append(int(record["sequence"]))
            descriptor = MachineTapeSegmentDescriptor(
                digest,
                int(records[0]["sequence"]),
                int(records[-1]["sequence"]),
                len(records),
                previous_digest,
                tuple(int(item["sequence"]) for item in records if item["checkpoint"]),
                tuple(
                    (core, min(values), max(values))
                    for core, values in sorted(by_core.items())
                ),
            )
            descriptors.append(descriptor)
            previous_digest = digest
            records = []

        with source_path.open("r", encoding="utf-8") as stream:
            header = json.loads(stream.readline())
            if header.get("schema") != "turing-machine-system-tape" or header.get("version") != 1:
                raise ValueError("unsupported source machine tape")
            subject = base64.b64decode(header["subject_binary"])
            (target / "subject.bin").write_bytes(subject)
            linked_modules = []
            module_directory = target / "modules"
            for item in header.get("linked_modules", ()):
                module = _module_from_mapping(item)
                module_directory.mkdir(parents=True, exist_ok=True)
                module_path = module_directory / f"{module.digest}.bin"
                if not module_path.exists():
                    module_path.write_bytes(module.binary)
                linked_modules.append(_module_mapping(module, include_binary=False))
            expected_sequence = 0
            for line in stream:
                value = json.loads(line)
                if value.get("record_type") == "annotation":
                    annotations.append(value)
                    continue
                if int(value.get("sequence", -1)) != expected_sequence:
                    raise ValueError("source tape records are not contiguous")
                records.append(value)
                if value.get("event") == "runtime_dispatch":
                    runtime_dispatch_targets.extend(
                        int(item)
                        for item in value.get("metadata", {}).get("targets", ())
                    )
                expected_sequence += 1
                if len(records) >= records_per_segment:
                    flush()
            flush()
        manifest = {
            "schema": SEGMENT_STORE_SCHEMA,
            "version": 1,
            "core_count": int(header["core_count"]),
            "checkpoint_interval": int(header.get("checkpoint_interval", 1024)),
            "record_count": sum(item.record_count for item in descriptors),
            "subject_digest": sha256(subject).hexdigest(),
            "external_references": header.get("external_references", []),
            "linked_modules": linked_modules,
            "import_bindings": header.get("import_bindings", []),
            "annotations": annotations,
            "runtime_dispatch_targets": list(dict.fromkeys(runtime_dispatch_targets)),
            "segments": [item.to_mapping() for item in descriptors],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return cls(target)


__all__ = [
    "MachineTapeSegmentDescriptor", "SegmentedMachineTapeStore",
]
