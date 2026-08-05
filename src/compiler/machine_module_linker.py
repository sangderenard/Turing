"""Capability-supplied PE module linking without ambient host DLL loading."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Mapping, Sequence

from .amd64_machine_semantics import EXTERNAL_TARGET_BASE
from .machine_execution import MachineExternalReference


def _library_aliases(name: str) -> tuple[str, ...]:
    normalized = str(name).replace("/", "\\").rsplit("\\", 1)[-1].casefold()
    return (normalized,) if normalized.endswith(".dll") else (normalized, normalized + ".dll")


def _symbol_identity(symbol) -> str:
    return symbol.name if symbol.name is not None else f"ordinal:{int(symbol.ordinal)}"


@dataclass(frozen=True, slots=True)
class MachineLinkedPEModule:
    """One approved dependency image and its deterministic guest address."""

    requested_library: str
    program: object
    load_address: int

    def __post_init__(self) -> None:
        if not self.requested_library:
            raise ValueError("linked PE module requires a library identity")
        if self.load_address < 0 or self.load_address >= 1 << 64:
            raise ValueError("linked PE module address must fit unsigned 64-bit")
        encoded = bytes(getattr(self.program.image, "encoded", b""))
        if not encoded:
            raise ValueError("linked PE module requires retained image bytes")
        export_name = getattr(self.program.image, "export_name", None)
        if export_name is not None and not set(_library_aliases(export_name)).intersection(
            _library_aliases(self.requested_library)
        ):
            raise ValueError(
                f"approved module {self.requested_library!r} exports as {export_name!r}"
            )

    @property
    def encoded(self) -> bytes:
        return bytes(self.program.image.encoded)

    @property
    def digest(self) -> str:
        return sha256(self.encoded).hexdigest()

    @property
    def export_name(self) -> str:
        return str(self.program.image.export_name or self.requested_library)


@dataclass(frozen=True, slots=True)
class MachineImportBinding:
    """One IAT decision with enough provenance to replay or audit the link."""

    owner_library: str
    owner_base: int
    iat_rva: int
    requested_library: str
    requested_symbol: str
    target_address: int
    resolution_kind: str
    resolved_library: str
    resolved_symbol: str
    forwarder_chain: tuple[str, ...] = ()
    is_delay: bool = False


@dataclass(frozen=True, slots=True)
class MachinePELinkPlan:
    """Complete deterministic IAT target plan for a mapped module set."""

    modules: tuple[MachineLinkedPEModule, ...]
    bindings: tuple[MachineImportBinding, ...]
    external_references: tuple[MachineExternalReference, ...]
    import_targets: Mapping[tuple[int, int], int]
    module_handle_targets: Mapping[tuple[int, int], int]


def _image_span(image, base: int) -> tuple[int, int]:
    end_rva = max(
        (int(section.virtual_address) + max(
            int(section.virtual_size), int(section.raw_size),
        ) for section in image.sections),
        default=0,
    )
    header_size = min(
        (int(section.raw_offset) for section in image.sections),
        default=len(image.encoded),
    )
    return int(base), int(base) + max(end_rva, header_size)


def build_pe_link_plan(
    primary_program,
    *,
    primary_load_address: int,
    modules: Sequence[MachineLinkedPEModule] = (),
) -> MachinePELinkPlan:
    """Resolve supplied exports and route every other import to a capability."""

    linked = tuple(modules)
    images = (("<subject>", primary_program, int(primary_load_address)), *(
        (module.export_name, module.program, int(module.load_address))
        for module in linked
    ))
    spans = sorted(
        (*_image_span(program.image, base), owner)
        for owner, program, base in images
    )
    for left, right in zip(spans, spans[1:]):
        if left[1] > right[0]:
            raise ValueError(
                f"linked PE images {left[2]!r} and {right[2]!r} overlap"
            )

    modules_by_alias: dict[str, MachineLinkedPEModule] = {}
    for module in linked:
        for alias in {
            *_library_aliases(module.requested_library),
            *_library_aliases(module.export_name),
        }:
            previous = modules_by_alias.setdefault(alias, module)
            if previous is not module:
                raise ValueError(f"linked PE module alias {alias!r} is ambiguous")

    def resolve(library: str, symbol: str):
        chain: list[str] = []
        active_library, active_symbol = str(library), str(symbol)
        visited: set[tuple[str, str]] = set()
        for _depth in range(32):
            identity = (_library_aliases(active_library)[-1], active_symbol)
            if identity in visited:
                raise ValueError(
                    "PE export forwarder cycle: " + " -> ".join((*chain, f"{active_library}!{active_symbol}"))
                )
            visited.add(identity)
            module = next((
                modules_by_alias[alias]
                for alias in _library_aliases(active_library)
                if alias in modules_by_alias
            ), None)
            if module is None:
                return None, active_library, active_symbol, tuple(chain)
            if active_symbol.startswith("ordinal:"):
                exported = module.program.image.export_by_ordinal(
                    int(active_symbol.split(":", 1)[1]),
                )
            else:
                exported = module.program.image.export_by_name(active_symbol)
            if exported is None:
                return None, active_library, active_symbol, tuple(chain)
            if exported.rva is not None:
                return (
                    module.load_address + int(exported.rva),
                    module.export_name,
                    exported.display_name,
                    tuple(chain),
                )
            assert exported.forwarder is not None
            chain.append(f"{module.export_name}!{exported.display_name}->{exported.forwarder}")
            if "." not in exported.forwarder:
                raise ValueError(f"invalid PE export forwarder {exported.forwarder!r}")
            active_library, forwarded_symbol = exported.forwarder.rsplit(".", 1)
            active_symbol = (
                f"ordinal:{int(forwarded_symbol[1:])}"
                if forwarded_symbol.startswith("#") else forwarded_symbol
            )
        raise ValueError("PE export forwarder chain exceeds 32 modules")

    bindings: list[MachineImportBinding] = []
    references: list[MachineExternalReference] = []
    targets: dict[tuple[int, int], int] = {}
    module_handles: dict[tuple[int, int], int] = {}
    for owner, program, base in images:
        imports = (
            *((item, False) for item in getattr(program.image, "imports", ())),
            *((item, True) for item in getattr(program.image, "delay_imports", ())),
        )
        for imported, is_delay in imports:
            requested_symbol = _symbol_identity(imported)
            resolved, resolved_library, resolved_symbol, chain = resolve(
                imported.library, requested_symbol,
            )
            if resolved is None:
                reference_id = len(references) + 1
                reference = MachineExternalReference(
                    reference_id,
                    EXTERNAL_TARGET_BASE + (reference_id - 1) * 16,
                    "guest-binary",
                    resolved_library,
                    resolved_symbol,
                )
                references.append(reference)
                target = reference.target_address
                kind = "capability"
            else:
                target = int(resolved)
                kind = "mapped_export"
            slot = (base, int(imported.iat_rva))
            if slot in targets:
                raise ValueError(f"duplicate linked PE IAT slot {slot!r}")
            targets[slot] = target
            if is_delay and resolved is not None and int(imported.module_handle_rva):
                resolved_module = next(
                    modules_by_alias[alias]
                    for alias in _library_aliases(resolved_library)
                    if alias in modules_by_alias
                )
                handle_slot = (base, int(imported.module_handle_rva))
                previous_handle = module_handles.setdefault(
                    handle_slot, int(resolved_module.load_address),
                )
                if previous_handle != int(resolved_module.load_address):
                    raise ValueError("delay-import module handle has conflicting targets")
            bindings.append(MachineImportBinding(
                owner, base, int(imported.iat_rva),
                imported.library, requested_symbol, target, kind,
                resolved_library, resolved_symbol, chain, is_delay,
            ))
    return MachinePELinkPlan(
        linked,
        tuple(bindings),
        tuple(references),
        MappingProxyType(targets),
        MappingProxyType(module_handles),
    )


__all__ = [
    "MachineImportBinding",
    "MachineLinkedPEModule",
    "MachinePELinkPlan",
    "build_pe_link_plan",
]
