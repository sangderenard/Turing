"""Immutable capability-owned Windows registry state.

The guest sees Win32 registry handles and binary values, but never the host
registry. Every mutation is an explicit value effect suitable for exact tape
replay, reversal, branching, and content-addressed persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping


ROOT_HANDLES = MappingProxyType({
    0x80000000: "hkey_classes_root",
    0x80000001: "hkey_current_user",
    0x80000002: "hkey_local_machine",
    0x80000003: "hkey_users",
    0x80000004: "hkey_performance_data",
    0x80000005: "hkey_current_config",
})


def normalize_registry_path(path: str) -> str:
    parts = [part for part in str(path).replace("/", "\\").split("\\") if part]
    if any(part in {".", ".."} or "\x00" in part for part in parts):
        raise ValueError("invalid virtual registry path")
    return "\\".join(part.casefold() for part in parts)


@dataclass(frozen=True, slots=True)
class VirtualRegistryValue:
    name: str
    value_type: int
    data: bytes

    def __post_init__(self) -> None:
        if "\x00" in self.name or self.value_type < 0:
            raise ValueError("invalid virtual registry value")
        object.__setattr__(self, "data", bytes(self.data))


@dataclass(frozen=True, slots=True)
class VirtualRegistryKey:
    path: str
    name: str
    values: Mapping[str, VirtualRegistryValue] = field(
        default_factory=lambda: MappingProxyType({}),
    )
    last_write_time: int = 0


@dataclass(frozen=True, slots=True)
class VirtualRegistryHandle:
    handle: int
    path: str
    access: int


@dataclass(frozen=True, slots=True)
class VirtualRegistryEffect:
    operation: str
    path: str
    handle: int = 0
    access: int = 0
    value_name: str = ""
    value_type: int = 0
    data: bytes = b""

    def __post_init__(self) -> None:
        if self.operation not in {
            "create_key", "delete_key", "open", "close",
            "set_value", "delete_value",
        }:
            raise ValueError(f"unsupported virtual registry effect {self.operation!r}")
        object.__setattr__(self, "data", bytes(self.data))


@dataclass(frozen=True, slots=True)
class VirtualRegistryState:
    keys: Mapping[str, VirtualRegistryKey]
    handles: Mapping[int, VirtualRegistryHandle] = field(
        default_factory=lambda: MappingProxyType({}),
    )
    next_handle: int = 0x4000
    generation: int = 0

    @classmethod
    def create(cls) -> "VirtualRegistryState":
        keys = {
            path: VirtualRegistryKey(path, path)
            for path in ROOT_HANDLES.values()
        }
        return cls(MappingProxyType(keys))

    def path_for_handle(self, handle: int) -> str | None:
        raw = int(handle) & 0xFFFFFFFF
        root = ROOT_HANDLES.get(raw)
        if root is not None:
            return root
        item = self.handles.get(int(handle))
        return None if item is None else item.path

    def access_for_handle(self, handle: int) -> int:
        if (int(handle) & 0xFFFFFFFF) in ROOT_HANDLES:
            return 0xF003F
        item = self.handles.get(int(handle))
        return 0 if item is None else item.access

    def resolve(self, handle: int, subkey: str = "") -> str | None:
        base = self.path_for_handle(handle)
        if base is None:
            return None
        suffix = normalize_registry_path(subkey)
        return base if not suffix else base + "\\" + suffix

    def key(self, path: str) -> VirtualRegistryKey:
        normalized = normalize_registry_path(path)
        try:
            return self.keys[normalized]
        except KeyError as error:
            raise FileNotFoundError(normalized) from error

    def children(self, path: str) -> tuple[VirtualRegistryKey, ...]:
        normalized = normalize_registry_path(path)
        prefix = normalized + "\\"
        return tuple(sorted(
            (item for key, item in self.keys.items()
             if key.startswith(prefix) and "\\" not in key[len(prefix):]),
            key=lambda item: item.name.casefold(),
        ))

    def apply(self, effect: VirtualRegistryEffect) -> "VirtualRegistryState":
        path = normalize_registry_path(effect.path)
        keys = dict(self.keys)
        handles = dict(self.handles)
        generation = self.generation + 1
        if effect.operation == "create_key":
            display_parts = [
                part for part in effect.path.replace("/", "\\").split("\\")
                if part
            ]
            parts = path.split("\\")
            cursor = parts[0]
            if cursor not in keys:
                raise FileNotFoundError(cursor)
            for index, name in enumerate(parts[1:], start=1):
                cursor += "\\" + name
                keys.setdefault(cursor, VirtualRegistryKey(
                    cursor, display_parts[index], last_write_time=generation,
                ))
        elif effect.operation == "delete_key":
            if path not in keys or path in ROOT_HANDLES.values():
                raise FileNotFoundError(path)
            prefix = path + "\\"
            if any(key.startswith(prefix) for key in keys):
                raise OSError("virtual registry key has subkeys")
            if any(item.path == path for item in handles.values()):
                raise OSError("virtual registry key still has an open handle")
            del keys[path]
        elif effect.operation == "open":
            if path not in keys or effect.handle <= 0 or effect.handle in handles:
                raise ValueError("virtual registry open requires an existing key and fresh handle")
            handles[effect.handle] = VirtualRegistryHandle(
                effect.handle, path, int(effect.access),
            )
        elif effect.operation == "close":
            if effect.handle not in handles:
                raise KeyError(f"unknown virtual registry handle {effect.handle}")
            del handles[effect.handle]
        elif effect.operation in {"set_value", "delete_value"}:
            key = keys.get(path)
            if key is None:
                raise FileNotFoundError(path)
            values = dict(key.values)
            normalized_name = effect.value_name.casefold()
            if effect.operation == "set_value":
                values[normalized_name] = VirtualRegistryValue(
                    effect.value_name, int(effect.value_type), effect.data,
                )
            else:
                if normalized_name not in values:
                    raise FileNotFoundError(effect.value_name)
                del values[normalized_name]
            keys[path] = replace(
                key, values=MappingProxyType(values), last_write_time=generation,
            )
        return VirtualRegistryState(
            MappingProxyType(keys), MappingProxyType(handles),
            max(self.next_handle, effect.handle + 1)
            if effect.operation == "open" else self.next_handle,
            generation,
        )


__all__ = [
    "ROOT_HANDLES", "VirtualRegistryEffect", "VirtualRegistryHandle",
    "VirtualRegistryKey", "VirtualRegistryState", "VirtualRegistryValue",
    "normalize_registry_path",
]
