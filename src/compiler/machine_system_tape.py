"""Append-only, resumable tape for complete virtual-machine state.

The in-memory execution graph is ideal for immediate reverse stepping.  This
tape is the durable counterpart: chronological records survive branch changes,
external completions, filesystem effects, and process restarts.  Records use
periodic full checkpoints and copy-on-write page/file deltas between them.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping

from .amd64_machine_semantics import PagedByteMemory
from .machine_execution import (
    MachineExecutionState, MachineExternalCallRequest, MachineExternalReference,
)
from .shell_io import (
    VirtualFileSystemContract, VirtualMount, VirtualMountAccess, VirtualMountKind,
)
from .virtual_filesystem import VirtualFile, VirtualFileHandle, VirtualFileSystemState
from .machine_module_linker import MachineImportBinding


_NAMED_ANNOTATION_COLORS = frozenset({
    "red", "amber", "yellow", "green", "cyan", "blue", "violet",
    "magenta", "gray", "white",
})
_ANNOTATION_RGBA8 = {
    "red": 0xFF3030FF, "amber": 0xFF2090FF, "yellow": 0xFF30E0FF,
    "green": 0xFF40C060, "cyan": 0xFFE0C040, "blue": 0xFFFF8040,
    "violet": 0xFFFF50A0, "magenta": 0xFFFF40E0, "gray": 0xFF909090,
    "white": 0xFFFFFFFF,
}


@dataclass(frozen=True, slots=True)
class MachineTapeAnnotation:
    """A colored, machine-addressable feature attached to a tape moment/span."""

    annotation_id: int
    sequence_start: int
    sequence_end: int
    feature: str
    message: str
    color: str = "amber"
    severity: str = "note"
    core: int | None = None
    position: int | None = None
    address: int | None = None
    external_reference: str | None = None
    metadata: tuple[tuple[str, str], ...] = ()
    supersedes: int | None = None

    def __post_init__(self) -> None:
        if self.annotation_id < 0 or self.sequence_start < 0:
            raise ValueError("tape annotation identifiers cannot be negative")
        if self.sequence_end < self.sequence_start:
            raise ValueError("tape annotation span ends before it starts")
        if not self.feature or not self.message:
            raise ValueError("tape annotations require a feature and message")
        if self.severity not in {"note", "caution", "suspect", "error", "verified", "breakpoint"}:
            raise ValueError(f"unsupported tape annotation severity {self.severity!r}")
        if (
            self.color.casefold() not in _NAMED_ANNOTATION_COLORS
            and re.fullmatch(r"#[0-9a-fA-F]{6}(?:[0-9a-fA-F]{2})?", self.color) is None
        ):
            raise ValueError("annotation color must be a named tape color or #RRGGBB[AA]")

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "record_type": "annotation", "annotation_id": self.annotation_id,
            "sequence_start": self.sequence_start, "sequence_end": self.sequence_end,
            "feature": self.feature, "message": self.message,
            "color": self.color, "severity": self.severity,
            "metadata": dict(self.metadata),
        }
        for key in ("core", "position", "address", "external_reference", "supersedes"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        return result

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MachineTapeAnnotation":
        return cls(
            int(value["annotation_id"]), int(value["sequence_start"]),
            int(value.get("sequence_end", value["sequence_start"])),
            str(value["feature"]), str(value["message"]), str(value.get("color", "amber")),
            str(value.get("severity", "note")),
            None if "core" not in value else int(value["core"]),
            None if "position" not in value else int(value["position"]),
            None if "address" not in value else int(value["address"]),
            value.get("external_reference"),
            tuple(sorted((str(key), str(item)) for key, item in value.get("metadata", {}).items())),
            None if "supersedes" not in value else int(value["supersedes"]),
        )


@dataclass(frozen=True, slots=True)
class MachineTapeDependencyNode:
    """One state node and its durable prerequisites in the tape DAG."""

    sequence: int
    core: int
    event: str
    position: int
    parent_sequence: int | None
    dependencies: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True, slots=True)
class MachineTapeDependencyGraph:
    """Validated dependency graph derived from append-only state records."""

    nodes: tuple[MachineTapeDependencyNode, ...]

    def __post_init__(self) -> None:
        by_sequence = {node.sequence: node for node in self.nodes}
        if len(by_sequence) != len(self.nodes):
            raise ValueError("tape dependency graph has duplicate sequences")
        for expected, node in enumerate(self.nodes):
            if node.sequence != expected:
                raise ValueError("tape dependency graph sequences are not contiguous")
            if node.parent_sequence is not None:
                parent = by_sequence.get(node.parent_sequence)
                if parent is None or parent.sequence >= node.sequence:
                    raise ValueError("tape parent must be an earlier graph node")
                if parent.core != node.core:
                    raise ValueError("tape parent edge crosses virtual cores")
            for kind, dependency in node.dependencies:
                if not kind:
                    raise ValueError("tape dependency kind cannot be empty")
                if dependency not in by_sequence or dependency >= node.sequence:
                    raise ValueError("tape dependency must reference an earlier graph node")

    def lineage(self, sequence: int) -> tuple[int, ...]:
        """Return one core's chronological state ancestry, root first."""

        if not 0 <= sequence < len(self.nodes):
            raise IndexError("tape graph sequence is out of range")
        result = []
        node = self.nodes[sequence]
        while True:
            result.append(node.sequence)
            if node.parent_sequence is None:
                return tuple(reversed(result))
            node = self.nodes[node.parent_sequence]


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _unb64(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"))


def _reference_mapping(reference: MachineExternalReference) -> dict[str, Any]:
    return {
        "reference_id": reference.reference_id,
        "target_address": reference.target_address,
        "domain": reference.domain,
        "library": reference.library,
        "symbol": reference.symbol,
    }


def _reference_from_mapping(value: Mapping[str, Any]) -> MachineExternalReference:
    return MachineExternalReference(
        int(value["reference_id"]), int(value["target_address"]),
        str(value["domain"]), str(value["library"]), str(value["symbol"]),
    )


@dataclass(frozen=True, slots=True)
class MachineTapeLinkedModule:
    """Approved dependency bytes retained as replay authority."""

    library: str
    load_address: int
    binary: bytes

    def __post_init__(self) -> None:
        if not self.library or self.load_address < 0:
            raise ValueError("linked module tape record requires identity and address")
        object.__setattr__(self, "binary", bytes(self.binary))

    @property
    def digest(self) -> str:
        return sha256(self.binary).hexdigest()


def _module_mapping(module: MachineTapeLinkedModule, *, include_binary: bool) -> dict[str, Any]:
    result = {
        "library": module.library,
        "load_address": module.load_address,
        "digest": module.digest,
    }
    if include_binary:
        result["binary"] = _b64(module.binary)
    return result


def _module_from_mapping(value: Mapping[str, Any], binary: bytes | None = None) -> MachineTapeLinkedModule:
    payload = _unb64(value["binary"]) if binary is None else bytes(binary)
    module = MachineTapeLinkedModule(
        str(value["library"]), int(value["load_address"]), payload,
    )
    if module.digest != str(value["digest"]):
        raise ValueError("linked module tape digest mismatch")
    return module


def _binding_mapping(binding: MachineImportBinding) -> dict[str, Any]:
    return {
        "owner_library": binding.owner_library,
        "owner_base": binding.owner_base,
        "iat_rva": binding.iat_rva,
        "requested_library": binding.requested_library,
        "requested_symbol": binding.requested_symbol,
        "target_address": binding.target_address,
        "resolution_kind": binding.resolution_kind,
        "resolved_library": binding.resolved_library,
        "resolved_symbol": binding.resolved_symbol,
        "forwarder_chain": list(binding.forwarder_chain),
        "is_delay": binding.is_delay,
    }


def _binding_from_mapping(value: Mapping[str, Any]) -> MachineImportBinding:
    return MachineImportBinding(
        str(value["owner_library"]), int(value["owner_base"]),
        int(value["iat_rva"]), str(value["requested_library"]),
        str(value["requested_symbol"]), int(value["target_address"]),
        str(value["resolution_kind"]), str(value["resolved_library"]),
        str(value["resolved_symbol"]),
        tuple(str(item) for item in value.get("forwarder_chain", ())),
        bool(value.get("is_delay", False)),
    )


def _request_mapping(request: MachineExternalCallRequest) -> dict[str, Any]:
    reference = request.reference
    return {
        "request_id": request.request_id,
        "reference": _reference_mapping(reference),
        "instruction_address": request.instruction_address,
        "return_address": request.return_address,
        "arguments": list(request.arguments),
        "stack_pointer": request.stack_pointer,
        "stack_arguments": list(request.stack_arguments),
    }


def _request_from_mapping(value: Mapping[str, Any]) -> MachineExternalCallRequest:
    ref = value["reference"]
    return MachineExternalCallRequest(
        int(value["request_id"]),
        _reference_from_mapping(ref),
        int(value["instruction_address"]), int(value["return_address"]),
        tuple(int(item) for item in value["arguments"]),
        int(value["stack_pointer"]),
        tuple(int(item) for item in value.get("stack_arguments", ())),
    )


def _vfs_mapping(value: VirtualFileSystemState | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "contract": value.contract.to_mapping(),
        "current_directory": value.current_directory,
        "generation": value.generation,
        "entries": {
            path: {
                "data": _b64(entry.data), "directory": entry.directory,
                "created_time": entry.created_time, "modified_time": entry.modified_time,
            }
            for path, entry in value.entries.items()
        },
        "handles": {
            str(handle): {
                "path": item.path, "mode": item.mode, "position": item.position,
                "entries": list(item.entries),
            }
            for handle, item in value.handles.items()
        },
        "next_handle": value.next_handle,
    }


def _vfs_from_mapping(value: Mapping[str, Any] | None) -> VirtualFileSystemState | None:
    if value is None:
        return None
    raw_contract = value["contract"]
    contract = VirtualFileSystemContract(
        str(raw_contract["current_directory"]),
        tuple(VirtualMount.create(
            mount["path"], VirtualMountKind(mount["kind"]),
            access=VirtualMountAccess(mount["access"]), source=mount.get("source"),
        ) for mount in raw_contract["mounts"]),
    )
    entries = {
        path: VirtualFile(
            path, _unb64(entry["data"]), bool(entry["directory"]),
            int(entry["created_time"]), int(entry["modified_time"]),
        )
        for path, entry in value["entries"].items()
    }
    handles = {
        int(handle): VirtualFileHandle(
            int(handle), item["path"], item["mode"], int(item["position"]),
            tuple(item.get("entries", ())),
        )
        for handle, item in value.get("handles", {}).items()
    }
    return VirtualFileSystemState(
        contract, MappingProxyType(entries), str(value["current_directory"]),
        int(value["generation"]),
        MappingProxyType(handles), int(value.get("next_handle", 0x1000)),
    )


def encode_machine_state(
    state: MachineExecutionState,
    previous: MachineExecutionState | None = None,
) -> dict[str, Any]:
    memory = state.memory
    if not isinstance(memory, PagedByteMemory):
        converted = PagedByteMemory.empty()
        for address, byte in memory.items():
            converted = converted.map_bytes(int(address), bytes((int(byte) & 0xFF,)))
        memory = converted
    previous_pages = (
        previous.memory.pages if previous is not None
        and isinstance(previous.memory, PagedByteMemory)
        and previous.memory.page_size == memory.page_size else {}
    )
    pages = {
        str(index): _b64(data) for index, data in memory.pages.items()
        if previous_pages.get(index) != data
    }
    removed_pages = [int(index) for index in previous_pages if index not in memory.pages]
    vfs = _vfs_mapping(state.virtual_filesystem)
    if previous is not None and state.virtual_filesystem == previous.virtual_filesystem:
        vfs = "unchanged"
    return {
        "pc": state.pc, "registers": list(state.registers),
        "vector_registers": [str(value) for value in state.vector_registers],
        "flags": state.flags, "fs_base": state.fs_base, "gs_base": state.gs_base,
        "call_stack": list(state.call_stack), "steps": state.steps,
        "termination_requested": state.termination_requested,
        "halted": state.halted,
        "exit_code": state.exit_code,
        "external_requests": [_request_mapping(item) for item in state.external_requests],
        "system_state": dict(state.system_state),
        "environment_state": dict(state.environment_state),
        "text_state": dict(state.text_state),
        "device_state": {key: _b64(bytes(item)) for key, item in state.device_state.items()},
        "device_generations": dict(state.device_generations),
        "memory": {"page_size": memory.page_size, "pages": pages, "removed": removed_pages},
        "virtual_filesystem": vfs,
    }


def decode_machine_state(
    value: Mapping[str, Any],
    previous: MachineExecutionState | None = None,
) -> MachineExecutionState:
    memory_value = value["memory"]
    page_size = int(memory_value["page_size"])
    pages = dict(
        previous.memory.pages if previous is not None
        and isinstance(previous.memory, PagedByteMemory)
        and previous.memory.page_size == page_size else {}
    )
    for index in memory_value.get("removed", ()):
        pages.pop(int(index), None)
    for index, data in memory_value["pages"].items():
        pages[int(index)] = _unb64(data)
    raw_vfs = value.get("virtual_filesystem")
    vfs = previous.virtual_filesystem if raw_vfs == "unchanged" and previous else _vfs_from_mapping(raw_vfs)
    return MachineExecutionState(
        pc=int(value["pc"]), registers=tuple(int(item) for item in value["registers"]),
        vector_registers=tuple(int(item) for item in value["vector_registers"]),
        flags=int(value["flags"]),
        memory=PagedByteMemory(MappingProxyType(pages), page_size),
        system_state=MappingProxyType({
            str(key): int(item) for key, item in value["system_state"].items()
        }),
        virtual_filesystem=vfs,
        environment_state=MappingProxyType({
            str(key): str(item) for key, item in value.get("environment_state", {}).items()
        }),
        text_state=MappingProxyType({
            str(key): str(item) for key, item in value.get("text_state", {}).items()
        }),
        device_state=MappingProxyType({
            str(key): _unb64(item) for key, item in value.get("device_state", {}).items()
        }),
        device_generations=MappingProxyType({
            str(key): int(item) for key, item in value.get("device_generations", {}).items()
        }),
        fs_base=int(value["fs_base"]), gs_base=int(value["gs_base"]),
        call_stack=tuple(int(item) for item in value["call_stack"]),
        external_requests=tuple(_request_from_mapping(item) for item in value["external_requests"]),
        steps=int(value["steps"]),
        termination_requested=bool(value.get("termination_requested", False)),
        halted=bool(value.get("halted", False)),
        exit_code=(
            None if value.get("exit_code") is None else int(value["exit_code"])
        ),
    )


@dataclass(slots=True)
class MachineSystemTape:
    """Chronological system-tape records for every core transition."""

    subject_binary: bytes
    core_count: int
    checkpoint_interval: int = 1024
    records: list[dict[str, Any]] = field(default_factory=list)
    annotations: list[MachineTapeAnnotation] = field(default_factory=list)
    external_references: list[MachineExternalReference] = field(default_factory=list)
    linked_modules: list[MachineTapeLinkedModule] = field(default_factory=list)
    import_bindings: list[MachineImportBinding] = field(default_factory=list)
    _last_states: dict[int, MachineExecutionState] = field(default_factory=dict, repr=False)
    _last_sequences: dict[int, int] = field(default_factory=dict, repr=False)
    _request_sequences: dict[tuple[int, int], int] = field(default_factory=dict, repr=False)
    _graph_cache: MachineTapeDependencyGraph | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.subject_binary = bytes(self.subject_binary)
        if self.core_count <= 0 or self.checkpoint_interval <= 0:
            raise ValueError("system tape sizes must be positive")

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
        if not 0 <= core < self.core_count:
            raise IndexError("system tape core index is out of range")
        previous = self._last_states.get(core)
        checkpoint = previous is None or len(self.records) % self.checkpoint_interval == 0
        sequence = len(self.records)
        active_dependencies = list(dependencies)
        if previous is not None:
            previous_requests = {item.request_id for item in previous.external_requests}
            current_requests = {item.request_id for item in state.external_requests}
            for request_id in sorted(previous_requests - current_requests):
                request_sequence = self._request_sequences.get((core, request_id))
                if request_sequence is not None:
                    active_dependencies.append((
                        "external_request"
                        if event == "external_completion" else "abandons_request",
                        request_sequence,
                    ))
                    self._request_sequences.pop((core, request_id), None)
        record = {
            "sequence": sequence, "core": core, "position": int(position),
            "event": str(event), "checkpoint": checkpoint,
            "parent_sequence": self._last_sequences.get(core),
            "dependencies": [
                {"kind": str(kind), "sequence": int(dependency)}
                for kind, dependency in dict.fromkeys(active_dependencies)
            ],
            "state": encode_machine_state(state, None if checkpoint else previous),
        }
        if metadata:
            record["metadata"] = dict(metadata)
        self.records.append(record)
        self._last_states[core] = state
        self._last_sequences[core] = sequence
        previous_ids = set() if previous is None else {
            item.request_id for item in previous.external_requests
        }
        for request in state.external_requests:
            if request.request_id not in previous_ids:
                self._request_sequences[(core, request.request_id)] = sequence
        self._graph_cache = None

    def catalog_external_reference(self, reference: MachineExternalReference) -> None:
        """Persist one static or dynamically resolved symbolic link identity."""

        for existing in self.external_references:
            if existing == reference:
                return
            if existing.target_address == reference.target_address:
                raise ValueError(
                    f"external target {reference.target_address:#x} has conflicting identities"
                )
            if existing.reference_id == reference.reference_id:
                raise ValueError(
                    f"external reference id {reference.reference_id} has conflicting identities"
                )
        self.external_references.append(reference)

    def resume_state(self, core: int = 0, *, sequence: int | None = None) -> MachineExecutionState:
        limit = len(self.records) - 1 if sequence is None else int(sequence)
        target = next((
            int(record["sequence"])
            for record in reversed(self.records)
            if int(record["sequence"]) <= limit and int(record["core"]) == core
        ), None)
        if target is None:
            raise IndexError("system tape contains no matching state")
        lineage = self.dependency_graph().lineage(target)
        checkpoint_index = max(
            index for index, item in enumerate(lineage)
            if self.records[item]["checkpoint"]
        )
        state = None
        for item in lineage[checkpoint_index:]:
            record = self.records[item]
            state = decode_machine_state(
                record["state"], None if record["checkpoint"] else state,
            )
        assert state is not None
        return state

    def lineage_states(
        self, core: int = 0, *, sequence: int | None = None,
    ) -> tuple[tuple[MachineTapeDependencyNode, MachineExecutionState], ...]:
        """Decode every state on one validated ancestry exactly once."""

        limit = len(self.records) - 1 if sequence is None else int(sequence)
        target = next((
            int(record["sequence"])
            for record in reversed(self.records)
            if int(record["sequence"]) <= limit and int(record["core"]) == core
        ), None)
        if target is None:
            raise IndexError("system tape contains no matching state")
        graph = self.dependency_graph()
        lineage = graph.lineage(target)
        state = None
        result = []
        for item in lineage:
            record = self.records[item]
            state = decode_machine_state(
                record["state"], None if record["checkpoint"] else state,
            )
            result.append((graph.nodes[item], state))
        return tuple(result)

    def dependency_graph(self) -> MachineTapeDependencyGraph:
        if self._graph_cache is None:
            self._graph_cache = MachineTapeDependencyGraph(tuple(
                MachineTapeDependencyNode(
                    int(record["sequence"]), int(record["core"]),
                    str(record["event"]), int(record["position"]),
                    None if record.get("parent_sequence") is None else int(record["parent_sequence"]),
                    tuple(
                        (str(item["kind"]), int(item["sequence"]))
                        for item in record.get("dependencies", ())
                    ),
                )
                for record in self.records
            ))
        return self._graph_cache

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
        """Append a colored observation without mutating historical records."""

        if not self.records:
            raise IndexError("cannot annotate an empty system tape")
        start = len(self.records) - 1 if sequence is None else int(sequence)
        end = start if end_sequence is None else int(end_sequence)
        if not 0 <= start < len(self.records) or not 0 <= end < len(self.records):
            raise IndexError("tape annotation sequence is out of range")
        if core is not None and not 0 <= int(core) < self.core_count:
            raise IndexError("tape annotation core is out of range")
        annotation = MachineTapeAnnotation(
            len(self.annotations), start, end, str(feature), str(message),
            str(color), str(severity), None if core is None else int(core),
            None if position is None else int(position),
            None if address is None else int(address), external_reference,
            tuple(sorted((str(key), str(value)) for key, value in (metadata or {}).items())),
            supersedes,
        )
        self.annotations.append(annotation)
        return annotation

    def annotations_at(
        self, sequence: int, *, core: int | None = None,
    ) -> tuple[MachineTapeAnnotation, ...]:
        return tuple(
            annotation for annotation in self.annotations
            if annotation.sequence_start <= sequence <= annotation.sequence_end
            and (core is None or annotation.core is None or annotation.core == core)
        )

    def active_annotations_at(
        self, sequence: int, *, core: int | None = None,
    ) -> tuple[MachineTapeAnnotation, ...]:
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
            red, green, blue, alpha = (
                int(raw[index:index + 2], 16) for index in range(0, 8, 2)
            )
            return red | (green << 8) | (blue << 16) | (alpha << 24)
        return _ANNOTATION_RGBA8[color.casefold()]

    def latest_sequence(self, core: int, *, position: int | None = None) -> int:
        for record in reversed(self.records):
            if record["core"] == core and (
                position is None or record["position"] == position
            ):
                return int(record["sequence"])
        raise IndexError("system tape contains no matching core position")

    def _rebuild_dependency_indexes(self) -> None:
        """Upgrade legacy rows and rebuild append-time graph indexes."""

        self._last_sequences.clear()
        self._request_sequences.clear()
        active_requests: dict[int, set[int]] = {
            core: set() for core in range(self.core_count)
        }
        for expected, record in enumerate(self.records):
            if int(record.get("sequence", -1)) != expected:
                raise ValueError("system tape records are not contiguous")
            core = int(record["core"])
            record.setdefault("parent_sequence", self._last_sequences.get(core))
            dependencies = list(record.get("dependencies", ()))
            current = {
                int(item["request_id"])
                for item in record["state"].get("external_requests", ())
            }
            for request_id in sorted(active_requests[core] - current):
                source = self._request_sequences.pop((core, request_id), None)
                if source is not None and not any(
                    item.get("kind") in {"external_request", "abandons_request"}
                    and int(item.get("sequence", -1)) == source
                    for item in dependencies
                ):
                    dependencies.append({
                        "kind": (
                            "external_request"
                            if record.get("event") == "external_completion"
                            else "abandons_request"
                        ),
                        "sequence": source,
                    })
            for request_id in current - active_requests[core]:
                self._request_sequences[(core, request_id)] = expected
            record["dependencies"] = dependencies
            active_requests[core] = current
            self._last_sequences[core] = expected
        self._graph_cache = None
        self.dependency_graph()  # validate before any state reconstruction

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        header = {
            "schema": "turing-machine-system-tape", "version": 1,
            "core_count": self.core_count,
            "checkpoint_interval": self.checkpoint_interval,
            "subject_binary": _b64(self.subject_binary),
            "external_references": [
                _reference_mapping(reference)
                for reference in self.external_references
            ],
            "linked_modules": [
                _module_mapping(module, include_binary=True)
                for module in self.linked_modules
            ],
            "import_bindings": [
                _binding_mapping(binding) for binding in self.import_bindings
            ],
        }
        with target.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(header, separators=(",", ":")) + "\n")
            for record in self.records:
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            for annotation in self.annotations:
                stream.write(json.dumps(annotation.to_mapping(), separators=(",", ":")) + "\n")
        return target

    @classmethod
    def read(cls, path: str | Path) -> "MachineSystemTape":
        with Path(path).open("r", encoding="utf-8") as stream:
            header = json.loads(stream.readline())
            if header.get("schema") != "turing-machine-system-tape" or header.get("version") != 1:
                raise ValueError("unsupported machine system tape")
            tape = cls(
                _unb64(header["subject_binary"]), int(header["core_count"]),
                int(header["checkpoint_interval"]),
            )
            tape.external_references = [
                _reference_from_mapping(value)
                for value in header.get("external_references", ())
            ]
            tape.linked_modules = [
                _module_from_mapping(value)
                for value in header.get("linked_modules", ())
            ]
            tape.import_bindings = [
                _binding_from_mapping(value)
                for value in header.get("import_bindings", ())
            ]
            rows = [json.loads(line) for line in stream if line.strip()]
            tape.records = [row for row in rows if row.get("record_type") != "annotation"]
            tape.annotations = [
                MachineTapeAnnotation.from_mapping(row)
                for row in rows if row.get("record_type") == "annotation"
            ]
        tape._rebuild_dependency_indexes()
        for core in range(tape.core_count):
            try:
                tape._last_states[core] = tape.resume_state(core)
            except IndexError:
                pass
        return tape


__all__ = [
    "MachineSystemTape", "MachineTapeAnnotation", "MachineTapeLinkedModule",
    "decode_machine_state", "encode_machine_state",
]
