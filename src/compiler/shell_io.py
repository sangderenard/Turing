"""Host-shell IO requirements and the shared mailbox ABI.

Machine IR does not acquire a keyboard, pointer, display, or filesystem by
pretending a low-level target owns one.  It records the facilities it needs;
deployment then wraps the artifact in the nearest shell that provides them.

The ABI is deliberately mailbox-shaped.  Hosts append input events and file
completions to memory owned by the compiled program.  The program publishes
file requests and display presents through the same memory.  This works for a
JavaScript/WebAssembly host without making asynchronous browser APIs re-enter
Wasm, and it remains usable by native wrappers around LLVM or Fortran.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence


class ShellIOCapability(str, Enum):
    KEYBOARD = "keyboard"
    POINTER = "pointer"
    DISPLAY = "display_double_buffer"
    FILES = "files"
    DEVICES = "system_devices"
    # Web pages resolve only other published Turing bundles. Host-system
    # references are deliberately a separate native-only capability.
    BUNDLE_REFERENCES = "bundle_references"
    HOST_REFERENCES = "host_references"


class SystemPortKind(str, Enum):
    FILE = "file"
    DEVICE = "device"
    EXTERNAL_REFERENCE = "external_reference"


class SystemPortDirection(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    BIDIRECTIONAL = "bidirectional"
    CALL = "call"


class ExternalReferenceDomain(str, Enum):
    BUNDLE = "bundle"
    HOST_SYSTEM = "host_system"
    GUEST_BINARY = "guest_binary"


class VirtualMountKind(str, Enum):
    """Storage supplied beneath a shell-owned virtual path."""

    MEMORY = "memory"
    BUNDLE = "bundle"
    HOST_DIRECTORY = "host_directory"
    INDEXED_DB = "indexed_db"
    OPFS = "opfs"


class VirtualMountAccess(str, Enum):
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"


@dataclass(frozen=True)
class VirtualMount:
    """One explicit mapping into the program's otherwise private namespace."""

    path: str
    kind: VirtualMountKind
    access: VirtualMountAccess = VirtualMountAccess.READ_ONLY
    source: str | None = None

    @classmethod
    def create(
        cls, path: str, kind: VirtualMountKind | str, *,
        access: VirtualMountAccess | str = VirtualMountAccess.READ_ONLY,
        source: str | None = None,
    ) -> "VirtualMount":
        return cls(str(path), VirtualMountKind(kind), VirtualMountAccess(access), source)

    def __post_init__(self) -> None:
        if not self.path or not self.path.startswith("/"):
            raise ValueError("virtual mount paths must be absolute")
        if self.path != "/" and self.path.endswith("/"):
            raise ValueError("virtual mount paths must not have a trailing slash")
        if self.kind is not VirtualMountKind.MEMORY and not self.source:
            raise ValueError(f"{self.kind.value} mounts require a source")
        if self.kind in {VirtualMountKind.INDEXED_DB, VirtualMountKind.OPFS}:
            source = str(self.source)
            if source.startswith(("/", "\\")) or any(
                part in {"", ".", ".."}
                for part in source.replace("\\", "/").split("/")
            ):
                raise ValueError(
                    f"{self.kind.value} mount sources must be relative namespace paths"
                )

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "path": self.path, "kind": self.kind.value, "access": self.access.value,
        }
        if self.source is not None:
            result["source"] = self.source
        return result


@dataclass(frozen=True)
class VirtualFileSystemContract:
    """Logical filesystem presented to a program by its enclosing shell."""

    current_directory: str = "/"
    mounts: tuple[VirtualMount, ...] = (VirtualMount("/", VirtualMountKind.MEMORY, VirtualMountAccess.READ_WRITE),)

    def __post_init__(self) -> None:
        if not self.current_directory.startswith("/"):
            raise ValueError("virtual current directory must be absolute")
        paths = [mount.path.casefold() for mount in self.mounts]
        if len(paths) != len(set(paths)):
            raise ValueError("virtual filesystem contains duplicate mount paths")
        if not any(path == "/" for path in paths):
            raise ValueError("virtual filesystem requires a root mount")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": "turing-virtual-filesystem",
            "version": 1,
            "current_directory": self.current_directory,
            "mounts": [mount.to_mapping() for mount in self.mounts],
        }


@dataclass(frozen=True)
class SystemPortField:
    """Map one semantic system-port field to a compiled API parameter."""

    name: str
    parameter: str

    def to_mapping(self) -> dict[str, str]:
        return {"name": self.name, "parameter": self.parameter}


@dataclass(frozen=True)
class SystemPort:
    """A named shell boundary, independent of target-language spelling.

    File input ports conventionally bind ``data`` and ``length`` fields.
    External-reference ports name a resolution domain and may bind request or
    result fields. Web shells accept only the ``bundle`` domain.
    """

    name: str
    kind: SystemPortKind
    direction: SystemPortDirection
    entry_point: str | None = None
    fields: tuple[SystemPortField, ...] = ()
    optional: bool = False
    external_domain: ExternalReferenceDomain | None = None
    attributes: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("system port needs a non-empty name")
        field_names = [field.name for field in self.fields]
        if len(field_names) != len(set(field_names)):
            raise ValueError(f"system port {self.name!r} has duplicate fields")
        if self.fields and not self.entry_point:
            raise ValueError(
                f"system port {self.name!r} binds parameters without an entry point"
            )
        if self.kind in {SystemPortKind.FILE, SystemPortKind.DEVICE}:
            if self.external_domain is not None:
                raise ValueError(
                    f"{self.kind.value} system ports do not have an external domain"
                )
            if (
                self.kind is SystemPortKind.FILE
                and self.direction in {
                    SystemPortDirection.INPUT, SystemPortDirection.BIDIRECTIONAL,
                }
            ):
                required = {"data", "length"}
                missing = required - set(field_names)
                if missing:
                    raise ValueError(
                        f"input file port {self.name!r} lacks fields {sorted(missing)!r}"
                    )
        elif self.external_domain is None:
            raise ValueError("external-reference ports require an explicit domain")

    @classmethod
    def create(
        cls,
        name: str,
        kind: SystemPortKind | str,
        direction: SystemPortDirection | str,
        *,
        entry_point: str | None = None,
        fields: Mapping[str, str] | None = None,
        optional: bool = False,
        external_domain: ExternalReferenceDomain | str | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> "SystemPort":
        return cls(
            str(name), SystemPortKind(kind), SystemPortDirection(direction),
            entry_point,
            tuple(SystemPortField(str(key), str(value)) for key, value in (fields or {}).items()),
            bool(optional),
            None if external_domain is None else ExternalReferenceDomain(external_domain),
            tuple(sorted((attributes or {}).items())),
        )

    @property
    def capability(self) -> ShellIOCapability:
        if self.kind is SystemPortKind.FILE:
            return ShellIOCapability.FILES
        if self.kind is SystemPortKind.DEVICE:
            return ShellIOCapability.DEVICES
        if self.external_domain is ExternalReferenceDomain.BUNDLE:
            return ShellIOCapability.BUNDLE_REFERENCES
        return ShellIOCapability.HOST_REFERENCES

    def to_mapping(self) -> dict[str, Any]:
        mapping = {
            "name": self.name,
            "kind": self.kind.value,
            "direction": self.direction.value,
            "optional": self.optional,
            "fields": [field.to_mapping() for field in self.fields],
            "attributes": dict(self.attributes),
        }
        if self.entry_point is not None:
            mapping["entry_point"] = self.entry_point
        if self.external_domain is not None:
            mapping["external_domain"] = self.external_domain.value
        return mapping


@dataclass(frozen=True)
class ShellIORequest:
    """One facility requested by IR; optional facilities may be omitted."""

    capability: ShellIOCapability
    optional: bool = False
    attributes: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def create(
        cls,
        capability: ShellIOCapability | str,
        *,
        optional: bool = False,
        attributes: Mapping[str, Any] | None = None,
    ) -> "ShellIORequest":
        return cls(
            ShellIOCapability(capability),
            bool(optional),
            tuple(sorted((attributes or {}).items())),
        )


@dataclass(frozen=True)
class ShellIOBinding:
    """Bind one compiled API parameter to one shell-provided resource."""

    resource: str
    entry_point: str
    parameter: str

    def to_mapping(self) -> dict[str, str]:
        return {
            "resource": self.resource,
            "entry_point": self.entry_point,
            "parameter": self.parameter,
        }


@dataclass(frozen=True)
class ShellOption:
    """One CLI/configuration value requested by IR rather than host code."""

    name: str
    value_type: str
    default: int | float | str | bool
    help: str = ""

    def __post_init__(self) -> None:
        if self.value_type not in {"int", "float", "str", "bool"}:
            raise ValueError(f"unsupported shell option type {self.value_type!r}")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.value_type,
            "default": self.default,
            "help": self.help,
        }


@dataclass(frozen=True)
class ShellIOManifest:
    """The complete host-facing IO demand of one compiled graph."""

    requests: tuple[ShellIORequest, ...] = ()
    bindings: tuple[ShellIOBinding, ...] = ()
    options: tuple[ShellOption, ...] = ()
    system_ports: tuple[SystemPort, ...] = ()
    virtual_filesystem: VirtualFileSystemContract | None = None

    def __post_init__(self) -> None:
        kinds = [request.capability for request in self.requests]
        if len(kinds) != len(set(kinds)):
            raise ValueError("shell IO manifest contains duplicate capabilities")
        binding_keys = [
            (binding.entry_point, binding.parameter) for binding in self.bindings
        ]
        if len(binding_keys) != len(set(binding_keys)):
            raise ValueError("shell IO manifest binds an API parameter twice")
        option_names = [option.name for option in self.options]
        if len(option_names) != len(set(option_names)):
            raise ValueError("shell IO manifest contains duplicate options")
        port_names = [port.name for port in self.system_ports]
        if len(port_names) != len(set(port_names)):
            raise ValueError("shell IO manifest contains duplicate system ports")
        requested = set(kinds)
        if self.virtual_filesystem is not None and ShellIOCapability.FILES not in requested:
            raise ValueError("virtual filesystem requires the files shell capability")
        unsupported = [
            port.name for port in self.system_ports
            if port.capability not in requested and not port.optional
        ]
        if unsupported:
            raise ValueError(
                "required system ports lack matching shell capabilities: "
                + ", ".join(unsupported)
            )

    @property
    def required(self) -> frozenset[ShellIOCapability]:
        return frozenset(
            request.capability for request in self.requests if not request.optional
        )

    @property
    def optional(self) -> frozenset[ShellIOCapability]:
        return frozenset(
            request.capability for request in self.requests if request.optional
        )

    def to_mapping(self) -> dict[str, Any]:
        result = {
            "schema": "turing-shell-io-requirements",
            "version": 1,
            "requests": [
                {
                    "capability": request.capability.value,
                    "optional": request.optional,
                    "attributes": dict(request.attributes),
                }
                for request in self.requests
            ],
            "bindings": [binding.to_mapping() for binding in self.bindings],
            "options": [option.to_mapping() for option in self.options],
            "system_ports": [port.to_mapping() for port in self.system_ports],
        }
        if self.virtual_filesystem is not None:
            result["virtual_filesystem"] = self.virtual_filesystem.to_mapping()
        return result

    def specialize_options(self, values: Mapping[str, Any]) -> "ShellIOManifest":
        """Return the manifest with compile-time option defaults recorded."""

        known = {option.name for option in self.options}
        unknown = set(values) - known
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown shell specialization options: {names}")
        return replace(
            self,
            options=tuple(
                replace(option, default=values.get(option.name, option.default))
                for option in self.options
            ),
        )


@dataclass(frozen=True)
class RingBufferABI:
    """A single-producer/single-consumer ring in shared linear memory."""

    header_bytes: int = 16
    record_bytes: int = 32
    # Header contains capacity, read index, write index, and dropped count.
    header_fields: tuple[str, ...] = (
        "capacity", "read_index", "write_index", "dropped",
    )


@dataclass(frozen=True)
class DisplayDoubleBufferABI:
    """Two caller-owned pixel buffers; a display is always optional to use."""

    pixel_format: str = "rgba8"
    descriptor_fields: tuple[str, ...] = (
        "width", "height", "stride_bytes", "front_offset", "back_offset",
        "generation",
    )


@dataclass(frozen=True)
class FileBrokerABI:
    """Handle-based asynchronous file requests and completions."""

    request_record_bytes: int = 32
    completion_record_bytes: int = 32
    operations: tuple[str, ...] = (
        "open", "create", "read", "write", "close", "stat",
        "list", "mkdir", "remove", "rename", "getcwd", "chdir", "flush",
    )
    # Paths and payloads are offset/length spans in the artifact's memory.
    span_fields: tuple[str, ...] = ("memory_offset", "byte_length")
    namespace: str = "utf8-posix-absolute"
    effects: str = "ordered-journal"


@dataclass(frozen=True)
class ExternalReferenceABI:
    """Asynchronous calls across a declared bundle/host/guest boundary."""

    request_record_bytes: int = 32
    completion_record_bytes: int = 32
    operations: tuple[str, ...] = ("resolve", "call", "release")
    domains: tuple[str, ...] = tuple(domain.value for domain in ExternalReferenceDomain)


@dataclass(frozen=True)
class ShellIOABI:
    """One stable physical contract shared by web and native host shells."""

    input_events: RingBufferABI = RingBufferABI()
    file_requests: RingBufferABI = RingBufferABI()
    file_completions: RingBufferABI = RingBufferABI()
    external_requests: RingBufferABI = RingBufferABI()
    external_completions: RingBufferABI = RingBufferABI()
    display: DisplayDoubleBufferABI = DisplayDoubleBufferABI()
    files: FileBrokerABI = FileBrokerABI()
    external_references: ExternalReferenceABI = ExternalReferenceABI()
    schema_version: int = 1
    input_event_fields: tuple[str, ...] = (
        "kind", "code", "value", "x", "y", "buttons", "modifiers",
        "timestamp_ms",
    )
    file_request_fields: tuple[str, ...] = (
        "operation", "request_id", "handle", "file_offset_low",
        "file_offset_high", "memory_offset", "byte_length", "flags",
    )
    file_completion_fields: tuple[str, ...] = (
        "operation", "request_id", "handle", "status", "bytes_transferred",
        "size_low", "size_high", "flags",
    )
    external_request_fields: tuple[str, ...] = (
        "operation", "request_id", "reference_id", "arguments_offset",
        "arguments_length", "result_offset", "result_capacity", "flags",
    )
    external_completion_fields: tuple[str, ...] = (
        "operation", "request_id", "reference_id", "status",
        "result_length", "effects_offset", "effects_length", "generation",
    )

    def to_mapping(self) -> dict[str, Any]:
        def ring(value: RingBufferABI) -> dict[str, Any]:
            return {
                "header_bytes": value.header_bytes,
                "record_bytes": value.record_bytes,
                "header_fields": list(value.header_fields),
            }

        return {
            "schema": "turing-shell-io-abi",
            "version": self.schema_version,
            "input_events": ring(self.input_events),
            "file_requests": ring(self.file_requests),
            "file_completions": ring(self.file_completions),
            "external_requests": ring(self.external_requests),
            "external_completions": ring(self.external_completions),
            "display": {
                "pixel_format": self.display.pixel_format,
                "descriptor_fields": list(self.display.descriptor_fields),
                "optional": True,
            },
            "files": {
                "request_record_bytes": self.files.request_record_bytes,
                "completion_record_bytes": self.files.completion_record_bytes,
                "operations": list(self.files.operations),
                "span_fields": list(self.files.span_fields),
                "namespace": self.files.namespace,
                "effects": self.files.effects,
            },
            "external_references": {
                "request_record_bytes": self.external_references.request_record_bytes,
                "completion_record_bytes": self.external_references.completion_record_bytes,
                "operations": list(self.external_references.operations),
                "domains": list(self.external_references.domains),
                "web_domain": ExternalReferenceDomain.BUNDLE.value,
            },
            "records": {
                "input_event_i32": list(self.input_event_fields),
                "file_request_i32": list(self.file_request_fields),
                "file_completion_i32": list(self.file_completion_fields),
                "external_request_i32": list(self.external_request_fields),
                "external_completion_i32": list(self.external_completion_fields),
            },
        }


def attach_shell_io(
    api: Any,
    manifest: ShellIOManifest,
    abi: ShellIOABI = ShellIOABI(),
) -> Any:
    """Return a compiled API descriptor carrying its shell IO contract.

    ``CompiledProgramAPI.metadata`` is already the repository's extension
    surface consumed by generated pages.  Keeping shell IO there avoids a
    second descriptor and leaves artifacts with no IO demand unchanged.
    """

    manifest = resolve_shell_io_bindings(api, manifest)
    metadata = dict(getattr(api, "metadata", {}) or {})
    metadata["shell_io"] = {
        "requirements": manifest.to_mapping(),
        "abi": abi.to_mapping(),
    }
    try:
        return replace(api, metadata=metadata)
    except TypeError as error:
        raise TypeError("shell IO can only attach to a dataclass API descriptor") from error


def resolve_shell_io_bindings(api: Any, manifest: ShellIOManifest) -> ShellIOManifest:
    """Resolve source-name bindings to concrete ABI parameter names."""

    entries = {
        str(entry.name): entry for entry in getattr(api, "entry_points", ())
    }
    resolved = []
    for binding in manifest.bindings:
        entry = entries.get(binding.entry_point)
        if entry is None:
            raise ValueError(
                f"shell IO binding names unknown entry point {binding.entry_point!r}"
            )
        direct = [
            parameter for parameter in entry.parameters
            if parameter.name == binding.parameter
        ]
        semantic = [
            parameter for parameter in entry.parameters
            if parameter.source_name == binding.parameter
        ]
        matches = direct or semantic
        if len(matches) != 1:
            raise ValueError(
                f"shell IO binding {binding.parameter!r} is not a unique ABI or "
                f"source parameter of {binding.entry_point!r}"
            )
        resolved.append(ShellIOBinding(
            binding.resource,
            binding.entry_point,
            matches[0].name,
        ))
    return ShellIOManifest(
        manifest.requests,
        bindings=tuple(resolved),
        options=manifest.options,
        system_ports=tuple(
            _resolve_system_port(api, port) for port in manifest.system_ports
        ),
        virtual_filesystem=manifest.virtual_filesystem,
    )


def _resolve_system_port(api: Any, port: SystemPort) -> SystemPort:
    if not port.fields:
        return port
    entries = {str(entry.name): entry for entry in getattr(api, "entry_points", ())}
    entry = entries.get(str(port.entry_point))
    if entry is None:
        raise ValueError(
            f"system port {port.name!r} names unknown entry point {port.entry_point!r}"
        )
    resolved = []
    for field in port.fields:
        direct = [item for item in entry.parameters if item.name == field.parameter]
        semantic = [
            item for item in entry.parameters if item.source_name == field.parameter
        ]
        matches = direct or semantic
        if len(matches) != 1:
            raise ValueError(
                f"system port {port.name!r} field {field.name!r} does not name "
                f"a unique parameter of {port.entry_point!r}"
            )
        resolved.append(SystemPortField(field.name, matches[0].name))
    return replace(port, fields=tuple(resolved))


@dataclass(frozen=True)
class ShellProfile:
    """A wrapper which accepts one artifact kind and exposes another."""

    name: str
    accepts: frozenset[str]
    exposes: str
    provides: frozenset[ShellIOCapability] = frozenset()
    cost: int = 1
    mount_kinds: frozenset[VirtualMountKind] = frozenset()


@dataclass(frozen=True)
class ShellStack:
    """Selected wrappers, ordered from the compiled artifact outwards."""

    artifact_kind: str
    wrappers: tuple[ShellProfile, ...]
    provided: frozenset[ShellIOCapability]
    optional_available: frozenset[ShellIOCapability]

    @property
    def outer_kind(self) -> str:
        return self.wrappers[-1].exposes if self.wrappers else self.artifact_kind


def plan_shell_stack(
    artifact_kind: str,
    manifest: ShellIOManifest,
    profiles: Sequence[ShellProfile],
) -> ShellStack:
    """Find the lowest-cost enclosing stack satisfying required IO.

    Profiles are edges between shell kinds.  Breadth-first cost search makes
    native-library -> process and wasm -> javascript equally expressible and
    avoids teaching any compiler backend about a particular host.
    """

    required = manifest.required
    start = (str(artifact_kind), frozenset())
    queue = deque([(0, start, ())])
    best: dict[tuple[str, frozenset[ShellIOCapability]], int] = {start: 0}
    while queue:
        # The queue remains small; ordering it makes profile cost explicit
        # without adding a deployment dependency.
        queue = deque(sorted(queue, key=lambda item: item[0]))
        cost, (kind, provided), wrappers = queue.popleft()
        if required <= provided:
            requested_mounts = {
                mount.kind for mount in (
                    manifest.virtual_filesystem.mounts
                    if manifest.virtual_filesystem is not None else ()
                )
            }
            available_mounts = frozenset().union(*(
                wrapper.mount_kinds for wrapper in wrappers
            )) if wrappers else frozenset()
            if not requested_mounts <= available_mounts:
                # This state satisfies capabilities but not the requested
                # namespace backing; keep searching for another wrapper.
                continue
            return ShellStack(
                str(artifact_kind),
                wrappers,
                provided,
                manifest.optional & provided,
            )
        for profile in profiles:
            if kind not in profile.accepts:
                continue
            next_provided = provided | profile.provides
            state = (profile.exposes, next_provided)
            next_cost = cost + max(0, int(profile.cost))
            if next_cost >= best.get(state, 1 << 60):
                continue
            best[state] = next_cost
            queue.append((next_cost, state, wrappers + (profile,)))
    missing = ", ".join(sorted(capability.value for capability in required))
    raise ValueError(
        f"no shell stack can supply required IO for {artifact_kind!r}: {missing}"
    )


WEB_JAVASCRIPT_SHELL = ShellProfile(
    name="web_javascript",
    accepts=frozenset({"wasm"}),
    exposes="web_page",
    provides=frozenset({
        ShellIOCapability.KEYBOARD,
        ShellIOCapability.POINTER,
        ShellIOCapability.DISPLAY,
        ShellIOCapability.FILES,
        ShellIOCapability.DEVICES,
        ShellIOCapability.BUNDLE_REFERENCES,
    }),
    cost=1,
    mount_kinds=frozenset({
        VirtualMountKind.MEMORY, VirtualMountKind.BUNDLE,
        VirtualMountKind.INDEXED_DB, VirtualMountKind.OPFS,
    }),
)

NATIVE_PROCESS_SHELL = ShellProfile(
    name="native_process",
    accepts=frozenset({"native_library", "llvm", "fortran"}),
    exposes="native_process",
    provides=frozenset({
        ShellIOCapability.KEYBOARD,
        ShellIOCapability.POINTER,
        ShellIOCapability.DISPLAY,
        ShellIOCapability.FILES,
        ShellIOCapability.HOST_REFERENCES,
    }),
    cost=1,
    mount_kinds=frozenset({
        VirtualMountKind.MEMORY, VirtualMountKind.BUNDLE,
        VirtualMountKind.HOST_DIRECTORY,
    }),
)


__all__ = [
    "DisplayDoubleBufferABI",
    "FileBrokerABI",
    "ExternalReferenceABI",
    "NATIVE_PROCESS_SHELL",
    "RingBufferABI",
    "ShellIOABI",
    "ShellIOBinding",
    "ShellIOCapability",
    "ShellIOManifest",
    "ShellIORequest",
    "ShellOption",
    "ShellProfile",
    "ShellStack",
    "ExternalReferenceDomain",
    "SystemPort",
    "SystemPortDirection",
    "SystemPortField",
    "SystemPortKind",
    "VirtualFileSystemContract",
    "VirtualMount",
    "VirtualMountAccess",
    "VirtualMountKind",
    "WEB_JAVASCRIPT_SHELL",
    "attach_shell_io",
    "plan_shell_stack",
    "resolve_shell_io_bindings",
]
