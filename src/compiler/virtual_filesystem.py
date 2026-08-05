"""Immutable shell-owned virtual filesystem state and reversible effects.

Guest code never receives an ambient host path.  Native shells may materialize
an explicitly declared host-directory mount; web shells materialize bundle and
memory mounts.  Once admitted, all operations below are ordinary immutable
state transitions and can therefore travel on the machine system tape.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping

from .shell_io import VirtualFileSystemContract, VirtualMountAccess


def normalize_virtual_path(path: str, current_directory: str = "/") -> str:
    raw = str(path).replace("\\", "/")
    # Give Windows drive paths a stable guest spelling without exposing a host
    # path: C:\tools becomes /c/tools.
    if len(raw) >= 2 and raw[1] == ":":
        raw = "/" + raw[0].lower() + raw[2:]
    if not raw.startswith("/"):
        raw = current_directory.rstrip("/") + "/" + raw
    parts: list[str] = []
    for part in raw.split("/"):
        if not part or part == ".":
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/" + "/".join(parts)


def virtual_path_to_windows(path: str, *, default_drive: str = "C") -> str:
    """Render a virtual path as a guest Windows path, never a host path."""

    normalized = normalize_virtual_path(path)
    parts = normalized.strip("/").split("/") if normalized != "/" else []
    if parts and len(parts[0]) == 1 and parts[0].isalpha():
        drive = parts.pop(0).upper()
    else:
        drive = default_drive.upper()
    suffix = "\\".join(parts)
    return drive + ":\\" + suffix if suffix else drive + ":\\"


@dataclass(frozen=True, slots=True)
class VirtualFile:
    path: str
    data: bytes = b""
    directory: bool = False
    created_time: int = 0
    modified_time: int = 0
    accessed_time: int = 0
    attributes: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", normalize_virtual_path(self.path))
        object.__setattr__(self, "data", bytes(self.data))
        if self.directory and self.data:
            raise ValueError("virtual directories cannot contain byte payloads")


@dataclass(frozen=True, slots=True)
class VirtualFileEffect:
    operation: str
    path: str
    data: bytes = b""
    offset: int = 0
    destination: str | None = None
    directory: bool = False
    handle: int = 0
    entries: tuple[str, ...] = ()
    mode: str = ""
    created_time: int | None = None
    accessed_time: int | None = None
    modified_time: int | None = None
    attributes: int | None = None

    def __post_init__(self) -> None:
        if self.operation not in {
            "create", "write", "mkdir", "remove", "rename", "chdir",
            "open", "advance", "seek", "truncate", "set_times",
            "set_attributes", "close",
        }:
            raise ValueError(f"unsupported virtual filesystem effect {self.operation!r}")
        object.__setattr__(self, "data", bytes(self.data))
        if self.offset < 0:
            raise ValueError("virtual file offsets cannot be negative")


@dataclass(frozen=True, slots=True)
class VirtualFileHandle:
    handle: int
    path: str
    mode: str
    position: int = 0
    entries: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class VirtualFileSystemState:
    contract: VirtualFileSystemContract
    entries: Mapping[str, VirtualFile]
    current_directory: str
    generation: int = 0
    handles: Mapping[int, VirtualFileHandle] = field(
        default_factory=lambda: MappingProxyType({}),
    )
    next_handle: int = 0x1000

    @classmethod
    def create(
        cls,
        contract: VirtualFileSystemContract | None = None,
        *,
        files: Mapping[str, bytes] | None = None,
    ) -> "VirtualFileSystemState":
        active = contract or VirtualFileSystemContract()
        entries: dict[str, VirtualFile] = {"/": VirtualFile("/", directory=True)}
        for mount in active.mounts:
            path = normalize_virtual_path(mount.path)
            entries.setdefault(path, VirtualFile(path, directory=True))
        for path, data in (files or {}).items():
            normalized = normalize_virtual_path(path)
            _ensure_parent_directories(entries, normalized)
            entries[normalized] = VirtualFile(normalized, bytes(data))
        cwd = normalize_virtual_path(active.current_directory)
        _ensure_parent_directories(entries, cwd + "/placeholder")
        entries.setdefault(cwd, VirtualFile(cwd, directory=True))
        return cls(active, MappingProxyType(entries), cwd)

    def _mount_for(self, path: str):
        normalized = normalize_virtual_path(path, self.current_directory)
        candidates = [
            mount for mount in self.contract.mounts
            if normalized == normalize_virtual_path(mount.path)
            or normalized.startswith(normalize_virtual_path(mount.path).rstrip("/") + "/")
        ]
        if not candidates:
            raise PermissionError(f"virtual path is outside declared mounts: {normalized}")
        return max(candidates, key=lambda mount: len(normalize_virtual_path(mount.path)))

    def read(self, path: str, *, offset: int = 0, length: int | None = None) -> bytes:
        normalized = normalize_virtual_path(path, self.current_directory)
        self._mount_for(normalized)
        entry = self.entries.get(normalized)
        if entry is None or entry.directory:
            raise FileNotFoundError(normalized)
        end = None if length is None else offset + max(0, int(length))
        return entry.data[int(offset):end]

    def stat(self, path: str) -> VirtualFile:
        normalized = normalize_virtual_path(path, self.current_directory)
        self._mount_for(normalized)
        try:
            return self.entries[normalized]
        except KeyError as error:
            raise FileNotFoundError(normalized) from error

    def list(self, path: str = ".") -> tuple[VirtualFile, ...]:
        directory = normalize_virtual_path(path, self.current_directory)
        if not self.stat(directory).directory:
            raise NotADirectoryError(directory)
        prefix = directory.rstrip("/") + "/"
        return tuple(sorted(
            (entry for key, entry in self.entries.items()
             if key.startswith(prefix) and "/" not in key[len(prefix):]),
            key=lambda entry: entry.path.casefold(),
        ))

    def apply(self, effect: VirtualFileEffect) -> "VirtualFileSystemState":
        path = normalize_virtual_path(effect.path, self.current_directory)
        mount = self._mount_for(path)
        if effect.operation in {
            "create", "write", "mkdir", "remove", "rename", "truncate",
            "set_times", "set_attributes",
        } and mount.access is VirtualMountAccess.READ_ONLY:
            raise PermissionError(f"virtual mount is read-only: {mount.path}")
        entries = dict(self.entries)
        handles = dict(self.handles)
        cwd = self.current_directory
        if effect.operation == "chdir":
            if not self.stat(path).directory:
                raise NotADirectoryError(path)
            cwd = path
        elif effect.operation in {"create", "mkdir"}:
            _ensure_parent_directories(entries, path)
            entries[path] = VirtualFile(path, effect.data, effect.operation == "mkdir" or effect.directory)
        elif effect.operation == "write":
            existing = entries.get(path)
            if existing is None or existing.directory:
                raise FileNotFoundError(path)
            payload = bytearray(existing.data)
            if effect.offset > len(payload):
                payload.extend(bytes(effect.offset - len(payload)))
            end = effect.offset + len(effect.data)
            if end > len(payload):
                payload.extend(bytes(end - len(payload)))
            payload[effect.offset:end] = effect.data
            entries[path] = replace(existing, data=bytes(payload), modified_time=self.generation + 1)
        elif effect.operation == "remove":
            if path not in entries:
                raise FileNotFoundError(path)
            prefix = path.rstrip("/") + "/"
            if any(key.startswith(prefix) for key in entries):
                raise OSError("virtual directory is not empty")
            del entries[path]
        elif effect.operation == "rename":
            if not effect.destination:
                raise ValueError("rename requires a destination")
            destination = normalize_virtual_path(effect.destination, self.current_directory)
            destination_mount = self._mount_for(destination)
            if destination_mount.access is VirtualMountAccess.READ_ONLY:
                raise PermissionError(f"virtual mount is read-only: {destination_mount.path}")
            entry = entries.pop(path)
            _ensure_parent_directories(entries, destination)
            entries[destination] = replace(entry, path=destination)
        elif effect.operation == "open":
            if effect.handle <= 0 or effect.handle in handles:
                raise ValueError("virtual open effect requires a fresh positive handle")
            handles[effect.handle] = VirtualFileHandle(
                effect.handle, path,
                effect.mode or ("enumeration" if effect.entries else "file"),
                effect.offset, tuple(effect.entries),
            )
        elif effect.operation == "advance":
            handle = handles.get(effect.handle)
            if handle is None:
                raise KeyError(f"unknown virtual handle {effect.handle}")
            handles[effect.handle] = replace(handle, position=handle.position + 1)
        elif effect.operation == "seek":
            handle = handles.get(effect.handle)
            if handle is None:
                raise KeyError(f"unknown virtual handle {effect.handle}")
            handles[effect.handle] = replace(handle, position=effect.offset)
        elif effect.operation == "truncate":
            existing = entries.get(path)
            if existing is None or existing.directory:
                raise FileNotFoundError(path)
            payload = existing.data[:effect.offset]
            if effect.offset > len(payload):
                payload += bytes(effect.offset - len(payload))
            entries[path] = replace(
                existing, data=payload, modified_time=self.generation + 1,
            )
        elif effect.operation == "set_times":
            existing = entries.get(path)
            if existing is None:
                raise FileNotFoundError(path)
            entries[path] = replace(
                existing,
                created_time=(
                    existing.created_time if effect.created_time is None
                    else effect.created_time
                ),
                accessed_time=(
                    existing.accessed_time if effect.accessed_time is None
                    else effect.accessed_time
                ),
                modified_time=(
                    existing.modified_time if effect.modified_time is None
                    else effect.modified_time
                ),
            )
        elif effect.operation == "set_attributes":
            existing = entries.get(path)
            if existing is None:
                raise FileNotFoundError(path)
            if effect.attributes is None:
                raise ValueError("set_attributes requires an attribute mask")
            entries[path] = replace(existing, attributes=effect.attributes)
        elif effect.operation == "close":
            if effect.handle not in handles:
                raise KeyError(f"unknown virtual handle {effect.handle}")
            del handles[effect.handle]
        return VirtualFileSystemState(
            self.contract, MappingProxyType(entries), cwd, self.generation + 1,
            MappingProxyType(handles),
            max(self.next_handle, effect.handle + 1) if effect.operation == "open" else self.next_handle,
        )


def _ensure_parent_directories(entries: dict[str, VirtualFile], path: str) -> None:
    parts = normalize_virtual_path(path).strip("/").split("/")[:-1]
    cursor = ""
    for part in parts:
        cursor += "/" + part
        entries.setdefault(cursor, VirtualFile(cursor, directory=True))


__all__ = [
    "VirtualFile", "VirtualFileEffect", "VirtualFileHandle", "VirtualFileSystemState",
    "normalize_virtual_path", "virtual_path_to_windows",
]
