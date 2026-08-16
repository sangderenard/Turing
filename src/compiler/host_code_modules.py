"""Cached extraction of source-less host callables into repository SSA."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from copy import deepcopy
import ctypes
import copyreg
from hashlib import sha256
import inspect
import importlib
import os
from pathlib import Path
import pickle
import sys
import tempfile
import struct
from types import MappingProxyType
from typing import Any, Callable

from .cpython_compile_ssa import (
    NativeCompileSSAResult,
    lift_pe_export_to_ssa,
)
from .binary_ingestion import parse_pe_image
from .machine_dialect_ssa import repository_ssa_legalized
from ..transmogrifier.ssa import (
    IRModule, SSAMachineControlLink, SSAMachineControlTable,
    SSAMachineIndirectLink, SSAMachineIndirectTable,
)


HOST_SSA_CACHE_SCHEMA = "turing.host-ssa-module.v2"

# The 2026-08-13 change from this digest is additive: exact VINSERTF128
# decoding/lowering plus safer incomplete-unit materialization.  Complete
# scalar units produced by that implementation remain byte-for-byte valid.
# This explicit allow-list prevents an arbitrary old compiler cache from
# silently crossing an implementation boundary.
_ADDITIVE_CACHE_IMPLEMENTATION_DIGESTS = (
    "902e9f2de78e1952032bac3fc01047e2ccaf6411bf17164d80462a126459c82a",
)


def _restore_mapping_proxy(
    items: tuple[tuple[Any, Any], ...],
) -> MappingProxyType:
    """Rebuild one immutable mapping in a cached compiler artifact."""

    return MappingProxyType(dict(items))


def _reduce_mapping_proxy(value: MappingProxyType):
    # Tuple materialization snapshots the immutable view at the same cache
    # boundary as the rest of the extracted result.  Values and keys continue
    # through pickle's ordinary recursive object graph and memoization.
    return _restore_mapping_proxy, (tuple(value.items()),)


class _HostSSACachePickler(pickle.Pickler):
    """Pickle compiler IR while preserving its immutable mapping contracts."""

    dispatch_table = copyreg.dispatch_table.copy()
    dispatch_table[type(MappingProxyType({}))] = _reduce_mapping_proxy


@dataclass(frozen=True, slots=True)
class HostCodeIdentity:
    provider: str
    module_path: Path
    symbol: str
    entry_rva: int | None = None
    calling_convention: str = ""


@dataclass(frozen=True, slots=True)
class CachedHostCodeModule:
    identity: HostCodeIdentity
    result: NativeCompileSSAResult
    cache_key: str
    cache_path: Path
    cache_hit: bool


@dataclass(frozen=True, slots=True)
class HostCodeDependencyEdge:
    """One exact PE-import edge in a recursively pursued host library."""

    source_cache_key: str
    external_identity: str
    target_cache_key: str | None
    resolution: str
    source_address: int | None = None


@dataclass(frozen=True, slots=True)
class CachedHostCodeLibrary:
    """Content-addressed transitive PE/SSA library rooted at one callable."""

    root_cache_key: str
    units: tuple[CachedHostCodeModule, ...]
    dependencies: tuple[HostCodeDependencyEdge, ...]

    @property
    def unresolved_dependencies(self) -> tuple[HostCodeDependencyEdge, ...]:
        return tuple(edge for edge in self.dependencies if edge.target_cache_key is None)

    @property
    def root(self) -> CachedHostCodeModule:
        return next(
            unit for unit in self.units
            if unit.cache_key == self.root_cache_key
        )

    @property
    def materialized_root_function(self) -> str:
        return (
            f"pe_{self.root_cache_key[:16]}__"
            f"{self.root.result.root_function}"
        )

    @property
    def blockers(self) -> tuple[Any, ...]:
        """Every occurrence from every recursively reached PE unit."""

        return tuple(
            blocker
            for unit in self.units
            for blocker in unit.result.blockers
        )

    @property
    def effective_blockers(self) -> tuple[Any, ...]:
        """Blockers remaining after exact dependency occurrences are linked.

        ``blockers`` deliberately remains the immutable extraction ledger.
        A named PE import is a unit-local blocker, but ceases to block the
        assembled library only when the matching source unit, import identity,
        and callsite address has a concrete dependency target.
        """

        resolved = {
            (
                edge.source_cache_key,
                edge.external_identity,
                edge.source_address,
            )
            for edge in self.dependencies
            if edge.target_cache_key is not None
        }
        return tuple(
            blocker
            for unit in self.units
            for blocker in unit.result.blockers
            if not (
                blocker.kind == "external-machine-module"
                and (
                    unit.cache_key,
                    blocker.external_identity,
                    blocker.address,
                ) in resolved
            )
        )

    @property
    def hard_blockers(self) -> tuple[Any, ...]:
        return tuple(
            blocker for blocker in self.effective_blockers
            if blocker.kind != "lowering"
        )

    @property
    def legalization_shortfalls(self) -> tuple[Any, ...]:
        return tuple(
            blocker for blocker in self.effective_blockers
            if blocker.kind == "lowering"
        )

    @property
    def machine_bodies_complete(self) -> bool:
        """Every extracted unit has complete executable machine semantics."""

        return all(unit.result.machine_state_complete for unit in self.units)

    @property
    def dependency_context_complete(self) -> bool:
        """The recursive library supplies every required control dependency."""

        return (
            not self.unresolved_dependencies
            and not self.hard_blockers
        )

    @property
    def machine_state_complete(self) -> bool:
        """The machine bodies and their complete dependency context deploy."""

        return self.machine_bodies_complete and self.dependency_context_complete

    @property
    def repository_ssa_complete(self) -> bool:
        return (
            not self.unresolved_dependencies
            and not self.effective_blockers
            and all(
                repository_ssa_legalized(function)
                for unit in self.units
                for function in unit.result.module.functions.values()
            )
        )


def materialize_host_code_library(library: CachedHostCodeLibrary) -> IRModule:
    """Merge cached PE units into one collision-free source-linked SSA module."""

    units = {unit.cache_key: unit for unit in library.units}
    namespaces = {
        key: f"pe_{key[:16]}" for key in units
    }
    name_maps = {
        key: {
            name: f"{namespaces[key]}__{name}"
            for name in unit.result.module.functions
        }
        for key, unit in units.items()
    }
    resolved_edges: dict[
        tuple[str, str, int | None], HostCodeDependencyEdge
    ] = {}
    for edge in library.dependencies:
        resolved_edges[
            (
                edge.source_cache_key,
                edge.external_identity,
                edge.source_address,
            )
        ] = edge
    functions = {}
    tensor_tables = {}
    sequence_tables = {}
    record_tables = {}
    reference_tables = {}
    call_table = {}
    control_links = []
    indirect_links = []

    def linked_target(edge: HostCodeDependencyEdge) -> str | None:
        """Return a concrete body name, not merely a cached target identity."""

        if edge.target_cache_key is None:
            return None
        target_unit = units.get(edge.target_cache_key)
        target_mapping = name_maps.get(edge.target_cache_key)
        if target_unit is None or target_mapping is None:
            return None
        return target_mapping.get(target_unit.result.root_function)

    def qualify_table(destination, source, mapping):
        for name, table in source.items():
            destination[mapping.get(name, name)] = deepcopy(table)

    for key, unit in units.items():
        source_module = unit.result.module
        mapping = name_maps[key]
        copied = deepcopy(source_module.functions)
        for old_name, function in copied.items():
            function.name = mapping[old_name]
            for block in function.blocks.values():
                for instruction in block.instrs:
                    callee = instruction.attributes.get("callee")
                    if callee in mapping:
                        instruction.attributes["callee"] = mapping[callee]
                    external = instruction.attributes.get("external_identity")
                    if not external:
                        continue
                    lookup = (
                        key,
                        str(external),
                        instruction.attributes.get("machine_address"),
                    )
                    edge = resolved_edges.get(lookup)
                    if edge is None:
                        continue
                    if edge.target_cache_key is None:
                        continue
                    target_root = linked_target(edge)
                    if target_root is None:
                        # The dependency occurrence remains an authored PE
                        # import and its target unit's exact blockers remain in
                        # the library ledger.  A cache file is not proof that a
                        # callable body exists.
                        continue
                    instruction.attributes.update({
                        "callee": target_root,
                        "source_linked": True,
                        "native_decompiled": True,
                        "dependency_cache_key": edge.target_cache_key,
                        "indirect_target_resolved": True,
                    })
            functions[function.name] = function
        qualify_table(tensor_tables, source_module.tensor_tables, mapping)
        qualify_table(sequence_tables, source_module.sequence_tables, mapping)
        qualify_table(record_tables, source_module.record_tables, mapping)
        qualify_table(
            reference_tables,
            getattr(source_module, "reference_tables", {}),
            mapping,
        )
        qualify_table(call_table, source_module.call_table, mapping)
        for link in source_module.machine_control_table.links:
            control_links.append(SSAMachineControlLink(
                mapping.get(link.source_function, link.source_function),
                link.source_block, link.source_address, link.edge_role,
                link.target_address,
                mapping.get(link.target_function, link.target_function),
                link.target_block, link.target_kind,
            ))
        for link in source_module.machine_indirect_table.links:
            source_name = mapping.get(link.source_function, link.source_function)
            if link.target_kind == "pe-import" and link.external_identity:
                lookup = (
                    key, link.external_identity, int(link.source_address),
                )
                edge = resolved_edges.get(lookup)
                if edge is not None and edge.target_cache_key is not None:
                    target_root = linked_target(edge)
                    if target_root is None:
                        indirect_links.append(SSAMachineIndirectLink(
                            source_name, link.source_address, link.edge_kind,
                            link.operand_kind, link.slot_address,
                            link.target_kind, link.target_address,
                            mapping.get(link.target_function, link.target_function),
                            link.external_identity,
                        ))
                        continue
                    indirect_links.append(SSAMachineIndirectLink(
                        source_name, link.source_address, link.edge_kind,
                        link.operand_kind, link.slot_address,
                        "internal-function", link.target_address,
                        target_root, link.external_identity,
                    ))
                    continue
            indirect_links.append(SSAMachineIndirectLink(
                source_name, link.source_address, link.edge_kind,
                link.operand_kind, link.slot_address, link.target_kind,
                link.target_address,
                mapping.get(link.target_function, link.target_function),
                link.external_identity,
            ))
    return IRModule(
        functions,
        tensor_tables=tensor_tables,
        sequence_tables=sequence_tables,
        record_tables=record_tables,
        reference_tables=reference_tables,
        call_table=call_table,
        machine_control_table=SSAMachineControlTable(tuple(control_links)),
        machine_indirect_table=SSAMachineIndirectTable(tuple(indirect_links)),
    )


_HOST_RESOLVERS: list[
    Callable[[Any], HostCodeIdentity | None]
] = []


def register_host_code_resolver(
    resolver: Callable[[Any], HostCodeIdentity | None],
) -> None:
    if resolver not in _HOST_RESOLVERS:
        _HOST_RESOLVERS.append(resolver)


def _cpython_builtin_resolver(value: Any) -> HostCodeIdentity | None:
    """Resolve CPython builtins to their stable exported implementation ABI."""

    if value is not compile:
        return None
    path = (
        Path(sys.executable).resolve().parent
        / f"python{sys.version_info.major}{sys.version_info.minor}.dll"
    )
    return HostCodeIdentity("cpython-pe", path, "Py_CompileString")


def _cpython_pycfunction_resolver(value: Any) -> HostCodeIdentity | None:
    """Resolve a CPython builtin through the stable PyCFunction C API.

    Builtin method definitions are generally not PE exports.  Their
    ``ml_meth`` entry nevertheless belongs to the loaded CPython image and is
    recoverable without interpreting the private PyCFunctionObject layout.
    """

    if not callable(value) or inspect.isfunction(value):
        return None
    if sys.platform != "win32":
        return None
    try:
        get_function = ctypes.pythonapi.PyCFunction_GetFunction
        get_function.argtypes = [ctypes.py_object]
        get_function.restype = ctypes.c_void_p
        get_flags = ctypes.pythonapi.PyCFunction_GetFlags
        get_flags.argtypes = [ctypes.py_object]
        get_flags.restype = ctypes.c_int
        address = int(get_function(value) or 0)
        flags = int(get_flags(value))
    except (AttributeError, ctypes.ArgumentError, SystemError, TypeError, ValueError):
        return None
    if not address:
        return None
    path = (
        Path(sys.executable).resolve().parent
        / f"python{sys.version_info.major}{sys.version_info.minor}.dll"
    )
    try:
        loaded_base = int(ctypes.WinDLL(str(path))._handle)
        entry_rva = address - loaded_base
        encoded = path.read_bytes()
        image, _statistics = parse_pe_image(
            encoded, maximum_file_size=len(encoded)
        )
    except (OSError, ValueError):
        return None
    owner = image.runtime_function_for_rva(entry_rva)
    if owner is None or not (owner.begin_rva <= entry_rva < owner.end_rva):
        return None
    module = str(getattr(value, "__module__", "builtins"))
    qualname = str(getattr(value, "__qualname__", getattr(value, "__name__", "builtin")))
    return HostCodeIdentity(
        "cpython-pycfunction", path, f"{module}.{qualname}", int(entry_rva),
        f"cpython-pycfunction-flags:{flags:#x}",
    )


def _exported_extension_resolver(value: Any) -> HostCodeIdentity | None:
    """Resolve a source-less callable when its owning PE exports its symbol."""

    if not callable(value):
        return None
    module_name = str(getattr(value, "__module__", ""))
    callable_name = str(getattr(value, "__name__", ""))
    if not module_name or not callable_name:
        return None
    module = sys.modules.get(module_name)
    if module is None:
        try:
            module = importlib.import_module(module_name)
        except (ImportError, ValueError):
            return None
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return None
    path = Path(module_file).resolve()
    if path.suffix.casefold() not in {".pyd", ".dll", ".exe"}:
        return None
    try:
        encoded = path.read_bytes()
        image, _statistics = parse_pe_image(
            encoded, maximum_file_size=len(encoded)
        )
    except (OSError, ValueError):
        return None
    candidates = tuple(dict.fromkeys((
        callable_name,
        f"Py_{callable_name}",
        str(getattr(value, "__qualname__", callable_name)).replace(".", "_"),
    )))
    symbol = next((
        candidate for candidate in candidates
        if (
            (exported := image.export_by_name(candidate)) is not None
            and exported.rva is not None
        )
    ), None)
    return (
        None if symbol is None
        else HostCodeIdentity("pe-export", path, symbol)
    )


register_host_code_resolver(_cpython_builtin_resolver)
register_host_code_resolver(_cpython_pycfunction_resolver)
register_host_code_resolver(_exported_extension_resolver)


def resolve_host_code_identity(value: Any) -> HostCodeIdentity | None:
    for resolver in tuple(_HOST_RESOLVERS):
        identity = resolver(value)
        if identity is not None:
            return identity
    return None


@lru_cache(maxsize=1)
def _implementation_digest() -> str:
    from . import cpython_compile_ssa, machine_code_lifting
    from . import machine_reference_vocabulary
    from . import machine_dialect_ssa, machine_symbolic_effects
    from . import native_code_retention

    digest = sha256()
    for module in (
        cpython_compile_ssa,
        machine_code_lifting,
        machine_reference_vocabulary,
        machine_dialect_ssa,
        machine_symbolic_effects,
        native_code_retention,
    ):
        path = Path(inspect.getsourcefile(module) or "")
        digest.update(path.read_bytes())
    return digest.hexdigest()


@lru_cache(maxsize=None)
def _module_content_digest(
    resolved_path: str, size: int, modified_ns: int,
) -> str:
    """Hash one immutable-on-this-run PE image once for all of its exports."""

    path = Path(resolved_path)
    stat = path.stat()
    if stat.st_size != size or stat.st_mtime_ns != modified_ns:
        raise ValueError(f"host module changed while being extracted: {path}")
    return sha256(path.read_bytes()).hexdigest()


def _cache_root() -> Path:
    configured = os.environ.get("TURING_HOST_SSA_CACHE")
    if configured:
        return Path(configured).resolve()
    return Path(__file__).resolve().parents[2] / ".turing-cache" / "host-ssa"


def _cache_key(identity: HostCodeIdentity) -> str:
    return _cache_key_for_implementation(identity, _implementation_digest())


def _cache_key_for_implementation(
    identity: HostCodeIdentity, implementation_digest: str,
) -> str:
    path = identity.module_path.resolve()
    stat = path.stat()
    digest = sha256()
    digest.update(HOST_SSA_CACHE_SCHEMA.encode("utf-8"))
    digest.update(identity.provider.encode("utf-8"))
    digest.update(str(path).encode("utf-8"))
    digest.update(identity.symbol.encode("utf-8"))
    digest.update(str(identity.entry_rva).encode("ascii"))
    digest.update(identity.calling_convention.encode("utf-8"))
    digest.update(_module_content_digest(
        str(path), int(stat.st_size), int(stat.st_mtime_ns),
    ).encode("ascii"))
    digest.update(str(implementation_digest).encode("ascii"))
    return digest.hexdigest()


def _safe_additive_cache_result(result: NativeCompileSSAResult) -> bool:
    """Whether the additive VEX change cannot alter an old cached unit."""

    return bool(
        result.root_function in result.module.functions
        and result.machine_state_complete
        and all(
            repository_ssa_legalized(function)
            for function in result.module.functions.values()
        )
    )


def _extract_host_code_identity(
    identity: HostCodeIdentity,
    *,
    cache_directory: str | Path | None = None,
) -> CachedHostCodeModule:
    """Extract one stable PE symbol unit through the shared disk cache."""

    key = _cache_key(identity)
    root = (
        Path(cache_directory).resolve()
        if cache_directory is not None else _cache_root()
    )
    root.mkdir(parents=True, exist_ok=True)
    cache_path = root / f"{key}.pickle"
    if cache_path.exists():
        with cache_path.open("rb") as stream:
            payload = pickle.load(stream)
        if (
            isinstance(payload, dict)
            and payload.get("schema") == HOST_SSA_CACHE_SCHEMA
            and payload.get("key") == key
            and isinstance(payload.get("result"), NativeCompileSSAResult)
        ):
            return CachedHostCodeModule(
                identity, payload["result"], key, cache_path, True
            )

    # Resume the expensive recursive library build across this one declared
    # additive decoder revision.  Negative/incomplete units and every unit
    # carrying vector state are rebuilt under the current implementation.
    for implementation_digest in _ADDITIVE_CACHE_IMPLEMENTATION_DIGESTS:
        legacy_key = _cache_key_for_implementation(
            identity, implementation_digest,
        )
        legacy_path = root / f"{legacy_key}.pickle"
        if not legacy_path.exists():
            continue
        try:
            with legacy_path.open("rb") as stream:
                legacy_payload = pickle.load(stream)
        except (OSError, pickle.PickleError, EOFError, AttributeError, ValueError):
            continue
        legacy_result = legacy_payload.get("result") if isinstance(legacy_payload, dict) else None
        if (
            isinstance(legacy_payload, dict)
            and legacy_payload.get("schema") == HOST_SSA_CACHE_SCHEMA
            and legacy_payload.get("key") == legacy_key
            and isinstance(legacy_result, NativeCompileSSAResult)
            and _safe_additive_cache_result(legacy_result)
        ):
            payload = {
                "schema": HOST_SSA_CACHE_SCHEMA,
                "key": key,
                "result": legacy_result,
            }
            temporary = tempfile.NamedTemporaryFile(
                mode="wb", prefix=key + ".", suffix=".tmp",
                dir=root, delete=False,
            )
            temporary_path = Path(temporary.name)
            try:
                with temporary:
                    _HostSSACachePickler(
                        temporary, protocol=pickle.HIGHEST_PROTOCOL,
                    ).dump(payload)
                    temporary.flush()
                    os.fsync(temporary.fileno())
                os.replace(temporary_path, cache_path)
            finally:
                if temporary_path.exists():
                    temporary_path.unlink()
            return CachedHostCodeModule(
                identity, legacy_result, key, cache_path, True,
            )

    if identity.provider not in {
        "cpython-pe", "cpython-pycfunction", "pe-export", "pe-dependency",
    }:
        raise ValueError(f"unknown host-code provider {identity.provider!r}")
    result = lift_pe_export_to_ssa(
        identity.module_path,
        root_symbol=identity.symbol,
        root_rva=identity.entry_rva,
        root_calling_convention=identity.calling_convention,
    )
    payload = {
        "schema": HOST_SSA_CACHE_SCHEMA,
        "key": key,
        "result": result,
    }
    temporary = tempfile.NamedTemporaryFile(
        mode="wb", prefix=key + ".", suffix=".tmp",
        dir=root, delete=False,
    )
    temporary_path = Path(temporary.name)
    try:
        with temporary:
            _HostSSACachePickler(
                temporary, protocol=pickle.HIGHEST_PROTOCOL,
            ).dump(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, cache_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return CachedHostCodeModule(identity, result, key, cache_path, False)


def extract_host_code_module(
    value: Any,
    *,
    cache_directory: str | Path | None = None,
) -> CachedHostCodeModule | None:
    """Return a cached decompiled SSA module for a source-less callable."""

    identity = resolve_host_code_identity(value)
    if identity is None:
        return None
    return _extract_host_code_identity(
        identity, cache_directory=cache_directory,
    )


def _default_pe_dependency_path(library: str, requester: Path) -> Path | None:
    """Locate bytes without loading or executing the requested dependency."""

    name = Path(str(library)).name
    candidates = [requester.resolve().parent / name]
    executable_parent = Path(sys.executable).resolve().parent
    candidates.append(executable_parent / name)
    system_root = os.environ.get("SystemRoot")
    if system_root:
        candidates.append(Path(system_root).resolve() / "System32" / name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    for host in _windows_api_set_hosts(str(library)):
        candidate = (
            Path(system_root).resolve() / "System32" / host
            if system_root else executable_parent / host
        )
        if candidate.is_file():
            return candidate.resolve()
    return None


@lru_cache(maxsize=1)
def _windows_api_set_map() -> dict[str, tuple[str, ...]]:
    """Read Windows' API-set namespace from its PE data section.

    This parses bytes only; it neither asks the loader to resolve a symbol nor
    executes a dependency.  Version 6 is the schema used by supported modern
    Windows releases.  Unknown schema versions remain an empty exact map.
    """

    system_root = os.environ.get("SystemRoot")
    if not system_root:
        return {}
    path = Path(system_root).resolve() / "System32" / "apisetschema.dll"
    try:
        encoded = path.read_bytes()
        if encoded[:2] != b"MZ" or len(encoded) < 0x40:
            return {}
        pe_offset = struct.unpack_from("<I", encoded, 0x3C)[0]
        if encoded[pe_offset:pe_offset + 4] != b"PE\0\0":
            return {}
        coff = pe_offset + 4
        section_count = struct.unpack_from("<H", encoded, coff + 2)[0]
        optional_size = struct.unpack_from("<H", encoded, coff + 16)[0]
        section_table = coff + 20 + optional_size
        section_range = None
        for index in range(section_count):
            offset = section_table + index * 40
            if offset + 40 > len(encoded):
                return {}
            name = encoded[offset:offset + 8].split(b"\0", 1)[0]
            if name == b".apiset":
                raw_size = struct.unpack_from("<I", encoded, offset + 16)[0]
                raw_offset = struct.unpack_from("<I", encoded, offset + 20)[0]
                section_range = (raw_offset, raw_offset + raw_size)
                break
        if section_range is None or section_range[1] > len(encoded):
            return {}
    except (OSError, ValueError, struct.error):
        return {}
    start, end = section_range
    data = encoded[start:end]
    if len(data) < 28:
        return {}
    version, size, _flags, count, entry_offset, _hash_offset, _factor = (
        struct.unpack_from("<7I", data, 0)
    )
    if version != 6 or size > len(data) or entry_offset + count * 24 > size:
        return {}

    def utf16(offset: int, length: int) -> str:
        if offset < 0 or length < 0 or offset + length > size or length % 2:
            raise ValueError("API-set string range is outside namespace")
        return data[offset:offset + length].decode("utf-16-le")

    result: dict[str, tuple[str, ...]] = {}
    try:
        for index in range(count):
            (_entry_flags, name_offset, name_length, _hashed_length,
             value_offset, value_count) = struct.unpack_from(
                "<6I", data, entry_offset + index * 24,
            )
            if value_offset + value_count * 20 > size:
                raise ValueError("API-set value table is outside namespace")
            name = utf16(name_offset, name_length).casefold()
            hosts: list[str] = []
            for value_index in range(value_count):
                (_value_flags, _alias_offset, _alias_length,
                 host_offset, host_length) = struct.unpack_from(
                    "<5I", data, value_offset + value_index * 20,
                )
                host = utf16(host_offset, host_length).casefold()
                if host and host not in hosts:
                    hosts.append(host)
            result[name.removesuffix(".dll")] = tuple(hosts)
    except (UnicodeDecodeError, ValueError, struct.error):
        return {}
    return result


def _windows_api_set_hosts(library: str) -> tuple[str, ...]:
    return _windows_api_set_map().get(
        Path(str(library)).name.casefold().removesuffix(".dll"), ()
    )


def _split_external_identity(identity: str) -> tuple[str, str]:
    library, separator, symbol = str(identity).partition("!")
    if not separator or not library or not symbol:
        raise ValueError(f"invalid PE external identity {identity!r}")
    return library, symbol


def extract_host_code_library(
    value: Any,
    *,
    cache_directory: str | Path | None = None,
    dependency_provider: Callable[[str, Path], str | Path | None] | None = None,
    max_functions: int | None = None,
    max_total_bytes: int | None = None,
    max_dependency_depth: int | None = None,
) -> CachedHostCodeLibrary | None:
    """Pursue the complete statically named PE dependency surface.

    Each ``module!export`` is an independently cached SSA unit.  The worklist
    has no depth or module-count cutoff: it converges by cache identity.  A
    missing module/export is retained as an exact edge instead of truncating
    traversal or invoking the host loader.
    """

    for name, limit in (
        ("max_functions", max_functions),
        ("max_total_bytes", max_total_bytes),
        ("max_dependency_depth", max_dependency_depth),
    ):
        if limit is not None and int(limit) <= 0:
            raise ValueError(f"{name} must be positive when supplied")
    root = extract_host_code_module(value, cache_directory=cache_directory)
    if root is None:
        return None
    provider = dependency_provider or _default_pe_dependency_path
    units: dict[str, CachedHostCodeModule] = {root.cache_key: root}
    pending = [(root, 0)]
    processed: set[str] = set()
    edges: list[HostCodeDependencyEdge] = []
    counted_paths = {root.identity.module_path.resolve()}
    try:
        total_bytes = root.identity.module_path.stat().st_size
    except OSError:
        total_bytes = 0
    while pending:
        source, source_depth = pending.pop(0)
        if source.cache_key in processed:
            continue
        processed.add(source.cache_key)
        identities = tuple(
            (link.external_identity, int(link.source_address))
            for link in source.result.module.machine_indirect_table.links
            if link.target_kind == "pe-import" and link.external_identity
        )
        for external_identity, source_address in identities:
            active_identity = external_identity
            requester = source.identity.module_path
            forwarders: list[str] = []
            seen_forwarders: set[str] = set()
            path = None
            exported = None
            failure = None
            while True:
                folded = active_identity.casefold()
                if folded in seen_forwarders:
                    failure = "forwarder-cycle:" + "->".join(
                        (*forwarders, active_identity)
                    )
                    break
                seen_forwarders.add(folded)
                library, symbol = _split_external_identity(active_identity)
                supplied = provider(library, requester)
                if supplied is None:
                    failure = f"module-unavailable:{active_identity}"
                    break
                path = Path(supplied).resolve()
                try:
                    encoded = path.read_bytes()
                    image, _statistics = parse_pe_image(
                        encoded, maximum_file_size=len(encoded),
                    )
                    exported = (
                        image.export_by_ordinal(int(symbol.partition(":")[2]))
                        if symbol.startswith("ordinal:")
                        else image.export_by_name(symbol)
                    )
                except (OSError, ValueError) as exc:
                    failure = f"module-unreadable:{type(exc).__name__}:{active_identity}"
                    break
                if exported is None:
                    failure = f"export-unavailable:{active_identity}"
                    break
                if exported.forwarder is None:
                    break
                forwarder = str(exported.forwarder)
                forwarded_library, dot, forwarded_symbol = forwarder.rpartition(".")
                if not dot:
                    failure = f"invalid-forwarder:{forwarder}"
                    break
                if not forwarded_library.casefold().endswith(".dll"):
                    forwarded_library += ".dll"
                forwarders.append(active_identity)
                active_identity = f"{forwarded_library}!{forwarded_symbol}"
                requester = path
            if failure is not None:
                edges.append(HostCodeDependencyEdge(
                    source.cache_key, external_identity, None, failure,
                    source_address,
                ))
                continue
            assert path is not None and exported is not None
            assert exported.rva is not None
            next_depth = source_depth + 1
            if (
                max_dependency_depth is not None
                and next_depth > int(max_dependency_depth)
            ):
                edges.append(HostCodeDependencyEdge(
                    source.cache_key, external_identity, None,
                    f"policy-depth-limit:{max_dependency_depth}", source_address,
                ))
                continue
            if max_functions is not None and len(units) >= int(max_functions):
                edges.append(HostCodeDependencyEdge(
                    source.cache_key, external_identity, None,
                    f"policy-function-limit:{max_functions}", source_address,
                ))
                continue
            resolved_path = path.resolve()
            added_bytes = 0
            if resolved_path not in counted_paths:
                try:
                    added_bytes = resolved_path.stat().st_size
                except OSError:
                    added_bytes = 0
            if (
                max_total_bytes is not None
                and total_bytes + added_bytes > int(max_total_bytes)
            ):
                edges.append(HostCodeDependencyEdge(
                    source.cache_key, external_identity, None,
                    f"policy-byte-limit:{max_total_bytes}", source_address,
                ))
                continue
            identity = HostCodeIdentity(
                "pe-dependency", path, active_identity,
                int(exported.rva), "windows-x64-pe-import",
            )
            target = _extract_host_code_identity(
                identity, cache_directory=cache_directory,
            )
            units.setdefault(target.cache_key, target)
            if resolved_path not in counted_paths:
                counted_paths.add(resolved_path)
                total_bytes += added_bytes
            edges.append(HostCodeDependencyEdge(
                source.cache_key, external_identity, target.cache_key,
                (
                    "resolved" if not forwarders else
                    "resolved-forwarders:" + "->".join(
                        (*forwarders, active_identity)
                    )
                ),
                source_address,
            ))
            if target.cache_key not in processed:
                pending.append((target, next_depth))
    return CachedHostCodeLibrary(
        root.cache_key, tuple(units.values()), tuple(edges),
    )


__all__ = [
    "CachedHostCodeModule",
    "CachedHostCodeLibrary",
    "HOST_SSA_CACHE_SCHEMA",
    "HostCodeIdentity",
    "HostCodeDependencyEdge",
    "extract_host_code_module",
    "extract_host_code_library",
    "materialize_host_code_library",
    "register_host_code_resolver",
    "resolve_host_code_identity",
]
