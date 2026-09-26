"""Capability-gated child programs resolved to Turing executors, never host exec.

The registry is deliberately shell-owned. A guest may request a familiar
Windows path through CreateProcess, but resolution stops at declared virtual
programs. The selected bundle identity and executor identity are durable
deployment facts; an unregistered path remains unsupported.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

from .virtual_filesystem import normalize_virtual_path


@dataclass(frozen=True, slots=True)
class VirtualProgramInvocation:
    requested_path: str
    resolved_path: str
    arguments: tuple[str, ...]
    current_directory: str
    environment: tuple[tuple[str, str], ...] = ()
    standard_input: bytes = b""


@dataclass(frozen=True, slots=True)
class VirtualProgramResult:
    exit_code: int = 0
    standard_output: bytes = b""
    standard_error: bytes = b""
    execution_units: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "standard_output", bytes(self.standard_output))
        object.__setattr__(self, "standard_error", bytes(self.standard_error))
        if self.execution_units < 0:
            raise ValueError("virtual process execution units cannot be negative")


VirtualProgramExecutor = Callable[[VirtualProgramInvocation], VirtualProgramResult]


@dataclass(frozen=True, slots=True)
class VirtualChildProcessTape:
    """Canonical, content-addressed tape for one intercepted child execution."""

    deployment_id: int
    bundle_reference: str
    executor_reference: str
    invocation: VirtualProgramInvocation
    result: VirtualProgramResult

    @property
    def schema(self) -> str:
        return "turing.virtual-child-process-tape.v1"

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "deployment_id": self.deployment_id,
            "bundle_reference": self.bundle_reference,
            "executor_reference": self.executor_reference,
            "invocation": {
                "requested_path": self.invocation.requested_path,
                "resolved_path": self.invocation.resolved_path,
                "arguments": list(self.invocation.arguments),
                "current_directory": self.invocation.current_directory,
                "environment": [list(item) for item in self.invocation.environment],
                "standard_input": base64.b64encode(self.invocation.standard_input).decode("ascii"),
            },
            "events": [
                {"kind": "start"},
                {
                    "kind": "standard_output",
                    "data": base64.b64encode(self.result.standard_output).decode("ascii"),
                },
                {
                    "kind": "standard_error",
                    "data": base64.b64encode(self.result.standard_error).decode("ascii"),
                },
                {
                    "kind": "exit",
                    "exit_code": self.result.exit_code,
                    "execution_units": self.result.execution_units,
                },
            ],
        }

    @property
    def encoded(self) -> bytes:
        return json.dumps(
            self.to_mapping(), sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")

    @property
    def digest(self) -> str:
        return sha256(self.encoded).hexdigest()


def split_windows_command_line(command_line: str) -> tuple[str, ...]:
    """Apply the CommandLineToArgvW backslash/quote rules deterministically."""

    text = str(command_line)
    arguments: list[str] = []
    cursor = 0
    while cursor < len(text):
        while cursor < len(text) and text[cursor] in " \t":
            cursor += 1
        if cursor >= len(text):
            break
        value: list[str] = []
        quoted = False
        while cursor < len(text):
            if text[cursor] in " \t" and not quoted:
                break
            slashes = 0
            while cursor < len(text) and text[cursor] == "\\":
                slashes += 1
                cursor += 1
            if cursor < len(text) and text[cursor] == '"':
                value.extend("\\" * (slashes // 2))
                if slashes % 2:
                    value.append('"')
                else:
                    quoted = not quoted
                cursor += 1
                continue
            value.extend("\\" * slashes)
            if cursor < len(text) and not (text[cursor] in " \t" and not quoted):
                value.append(text[cursor])
                cursor += 1
        arguments.append("".join(value))
    return tuple(arguments)


@dataclass(frozen=True, slots=True)
class VirtualProgram:
    virtual_path: str
    bundle_reference: str
    executor_reference: str
    executor: VirtualProgramExecutor


@dataclass(frozen=True, slots=True)
class VirtualProcessDeployment:
    deployment_id: int
    program: VirtualProgram
    invocation: VirtualProgramInvocation
    result: VirtualProgramResult
    child_tape: VirtualChildProcessTape


class VirtualProgramRegistry:
    """Exact virtual-path registry for card sets and other compiled executors."""

    def __init__(self) -> None:
        self._programs: dict[str, VirtualProgram] = {}
        self._child_tapes: dict[str, VirtualChildProcessTape] = {}

    @property
    def programs(self) -> Mapping[str, VirtualProgram]:
        return MappingProxyType(dict(self._programs))

    @property
    def child_tapes(self) -> Mapping[str, VirtualChildProcessTape]:
        return MappingProxyType(dict(self._child_tapes))

    def export_child_tapes(self, root: str | Path) -> tuple[Path, ...]:
        target = Path(root)
        target.mkdir(parents=True, exist_ok=True)
        written = []
        for digest, tape in sorted(self._child_tapes.items()):
            path = target / f"{digest}.json"
            if not path.exists():
                path.write_bytes(tape.encoded)
            written.append(path)
        return tuple(written)

    def register(
        self,
        virtual_path: str,
        *,
        bundle_reference: str,
        executor_reference: str,
        executor: VirtualProgramExecutor,
    ) -> VirtualProgram:
        path = normalize_virtual_path(virtual_path, "/")
        if not bundle_reference or not executor_reference or not callable(executor):
            raise ValueError("virtual programs require bundle and executor identities")
        key = path.casefold()
        if key in self._programs:
            raise ValueError(f"virtual program {path!r} is already registered")
        program = VirtualProgram(path, bundle_reference, executor_reference, executor)
        self._programs[key] = program
        return program

    def resolve(
        self,
        requested_path: str,
        *,
        current_directory: str = "/",
        path_search: Sequence[str] = (),
        extensions: Sequence[str] = (".exe", ".cmd", ".com"),
    ) -> VirtualProgram | None:
        requested = str(requested_path).replace("\\", "/")
        has_directory = "/" in requested
        roots = (current_directory,) if has_directory else (
            current_directory, *(str(item) for item in path_search)
        )
        suffixes = ("",) if "." in requested.rsplit("/", 1)[-1] else ("", *extensions)
        for root in roots:
            normalized_root = normalize_virtual_path(root, "/")
            for suffix in suffixes:
                candidate = normalize_virtual_path(requested + suffix, normalized_root)
                program = self._programs.get(candidate.casefold())
                if program is not None:
                    return program
        return None

    def launch(
        self,
        requested_path: str,
        arguments: Sequence[str],
        *,
        deployment_id: int,
        current_directory: str = "/",
        path_search: Sequence[str] = (),
        environment: Mapping[str, str] | None = None,
        standard_input: bytes = b"",
    ) -> VirtualProcessDeployment | None:
        program = self.resolve(
            requested_path,
            current_directory=current_directory,
            path_search=path_search,
        )
        if program is None:
            return None
        invocation = VirtualProgramInvocation(
            str(requested_path), program.virtual_path, tuple(str(item) for item in arguments),
            normalize_virtual_path(current_directory, "/"),
            tuple(sorted((str(key), str(value)) for key, value in (environment or {}).items())),
            bytes(standard_input),
        )
        result = program.executor(invocation)
        if not isinstance(result, VirtualProgramResult):
            raise TypeError("virtual program executor must return VirtualProgramResult")
        child_tape = VirtualChildProcessTape(
            int(deployment_id), program.bundle_reference,
            program.executor_reference, invocation, result,
        )
        self._child_tapes.setdefault(child_tape.digest, child_tape)
        return VirtualProcessDeployment(
            int(deployment_id), program, invocation, result, child_tape,
        )


__all__ = [
    "VirtualChildProcessTape", "VirtualProcessDeployment", "VirtualProgram", "VirtualProgramExecutor",
    "VirtualProgramInvocation", "VirtualProgramRegistry", "VirtualProgramResult",
    "split_windows_command_line",
]
