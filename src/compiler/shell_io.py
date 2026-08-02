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
        return {
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
        }

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
    )
    # Paths and payloads are offset/length spans in the artifact's memory.
    span_fields: tuple[str, ...] = ("memory_offset", "byte_length")


@dataclass(frozen=True)
class ShellIOABI:
    """One stable physical contract shared by web and native host shells."""

    input_events: RingBufferABI = RingBufferABI()
    file_requests: RingBufferABI = RingBufferABI()
    file_completions: RingBufferABI = RingBufferABI()
    display: DisplayDoubleBufferABI = DisplayDoubleBufferABI()
    files: FileBrokerABI = FileBrokerABI()
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
            },
            "records": {
                "input_event_i32": list(self.input_event_fields),
                "file_request_i32": list(self.file_request_fields),
                "file_completion_i32": list(self.file_completion_fields),
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
    )


@dataclass(frozen=True)
class ShellProfile:
    """A wrapper which accepts one artifact kind and exposes another."""

    name: str
    accepts: frozenset[str]
    exposes: str
    provides: frozenset[ShellIOCapability] = frozenset()
    cost: int = 1


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
    provides=frozenset(ShellIOCapability),
    cost=1,
)

NATIVE_PROCESS_SHELL = ShellProfile(
    name="native_process",
    accepts=frozenset({"native_library", "llvm", "fortran"}),
    exposes="native_process",
    provides=frozenset(ShellIOCapability),
    cost=1,
)


__all__ = [
    "DisplayDoubleBufferABI",
    "FileBrokerABI",
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
    "WEB_JAVASCRIPT_SHELL",
    "attach_shell_io",
    "plan_shell_stack",
    "resolve_shell_io_bindings",
]
