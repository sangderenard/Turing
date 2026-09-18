"""Context-complete retention of native machine-code modules.

Retaining native code is a terminal lowering only when the consumer can load
the original module format on the same architecture and ABI.  A linked PE
image is not mislabeled as a relocatable COFF object: it carries imports,
exports, base relocations, unwind ranges and complete image bytes together.
Targets that cannot consume that context must use machine SSA legalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from types import MappingProxyType
from typing import Mapping, Sequence

from .binary_ingestion import PEImage, PEMachine


class NativeRetentionMode(str, Enum):
    LOADABLE_IMAGE = "loadable-image"
    RELOCATABLE_OBJECT = "relocatable-object"
    TRANSLATE = "translate"


class HostImplementationKind(str, Enum):
    REPOSITORY_SSA = "repository-ssa"
    MACHINE_STATE_SSA = "machine-state-ssa"
    RETAINED_NATIVE_MODULE = "retained-native-module"
    TRANSLATION_REQUIRED = "translation-required"


@dataclass(frozen=True, slots=True)
class NativeTargetContext:
    architecture: str
    operating_system: str
    abi: str
    object_formats: frozenset[str]
    accepts_loadable_images: bool = False
    accepts_relocatable_objects: bool = True
    # Exact decompiler dialects for which this target has a real executor.
    # Empty is deliberately the default: accepting repository SSA does not
    # imply accepting a whole guest-machine state transition.
    machine_state_ssa_dialects: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class HostImplementationDecision:
    """Backend selection result for one source-linked host definition."""

    implementation: HostImplementationKind
    reason: str
    native_mode: NativeRetentionMode | None = None
    native_module: "RetainedNativeModule | None" = None

    @property
    def deployable(self) -> bool:
        return self.implementation is not HostImplementationKind.TRANSLATION_REQUIRED


@dataclass(frozen=True, slots=True)
class RetainedNativeModule:
    """One native module retained with every loader-relevant catalogue."""

    format: str
    architecture: str
    operating_system: str
    abi: str
    encoded: bytes
    image_base: int
    entrypoint_rva: int
    is_shared_library: bool
    exports: Mapping[str, int | str]
    imports: tuple[tuple[str, str, int, bool], ...]
    sections: tuple[tuple[str, int, int, int, int], ...]
    relocations: tuple[tuple[int, int], ...]
    unwind_functions: tuple[tuple[int, int, int], ...]
    source_identity: str = ""

    @property
    def digest(self) -> str:
        return sha256(self.encoded).hexdigest()

    def retention_mode(self, target: NativeTargetContext) -> NativeRetentionMode:
        if (
            target.architecture.casefold() != self.architecture.casefold()
            or target.operating_system.casefold() != self.operating_system.casefold()
            or target.abi.casefold() != self.abi.casefold()
            or self.format.casefold() not in {
                item.casefold() for item in target.object_formats
            }
        ):
            return NativeRetentionMode.TRANSLATE
        if self.format == "pe-image" and target.accepts_loadable_images:
            return NativeRetentionMode.LOADABLE_IMAGE
        if self.format in {"coff-object", "elf-object", "macho-object"} and target.accepts_relocatable_objects:
            return NativeRetentionMode.RELOCATABLE_OBJECT
        return NativeRetentionMode.TRANSLATE

    def require_compatible(self, target: NativeTargetContext) -> NativeRetentionMode:
        mode = self.retention_mode(target)
        if mode is NativeRetentionMode.TRANSLATE:
            raise ValueError(
                f"retained {self.format}/{self.architecture}/{self.abi} module "
                f"cannot be consumed by {target.operating_system}/"
                f"{target.architecture}/{target.abi}; machine SSA translation is required"
            )
        return mode

    def write(self, path) -> None:
        from pathlib import Path

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(self.encoded)


def retain_pe_image(
    image: PEImage,
    *,
    source_identity: str = "",
) -> RetainedNativeModule:
    """Retain a parsed PE as a complete loadable image, never raw code bytes."""

    if image.machine is not PEMachine.AMD64 or not image.pe32_plus:
        raise ValueError("native retention currently requires a PE32+ AMD64 image")
    exports = {}
    for item in image.exports:
        identity = item.display_name
        destination: int | str
        if item.rva is not None:
            destination = int(item.rva)
        else:
            assert item.forwarder is not None
            destination = str(item.forwarder)
        previous = exports.setdefault(identity, destination)
        if previous != destination:
            raise ValueError(f"conflicting retained PE export {identity!r}")
    imports = tuple(
        (item.library, item.name or f"ordinal:{item.ordinal}", int(item.iat_rva), delayed)
        for collection, delayed in (
            (image.imports, False), (image.delay_imports, True),
        )
        for item in collection
    )
    return RetainedNativeModule(
        "pe-image", "amd64", "windows", "windows-x64",
        bytes(image.encoded), int(image.image_base), int(image.entrypoint_rva),
        bool(image.is_dll), MappingProxyType(exports), imports,
        tuple((
            item.name, int(item.virtual_address), int(item.virtual_size),
            int(item.raw_offset), int(item.characteristics),
        ) for item in image.sections),
        tuple((int(item.type), int(item.rva)) for item in image.base_relocations),
        tuple((
            int(item.begin_rva), int(item.end_rva), int(item.unwind_info_rva),
        ) for item in image.runtime_functions),
        str(source_identity),
    )


def select_host_implementation(
    *,
    repository_ssa_complete: bool,
    machine_state_ssa_complete: bool = False,
    machine_state_ssa_dialect: str = "turing.machine-state-ssa.amd64.v1",
    retained_native_module: RetainedNativeModule | None,
    target: NativeTargetContext,
    prefer_native: bool = True,
) -> HostImplementationDecision:
    """Choose a deployable implementation without disguising incompatibility.

    Native retention wins only when the target can consume the exact artifact
    kind. An incomplete repository-SSA implementation is diagnostic material,
    not a deployable fallback. If neither representation is deployable, the
    decision explicitly requires continued machine-SSA translation.
    """

    native_mode = (
        retained_native_module.retention_mode(target)
        if retained_native_module is not None else None
    )
    native_compatible = native_mode in {
        NativeRetentionMode.LOADABLE_IMAGE,
        NativeRetentionMode.RELOCATABLE_OBJECT,
    }
    if prefer_native and native_compatible:
        return HostImplementationDecision(
            HostImplementationKind.RETAINED_NATIVE_MODULE,
            f"target consumes retained module as {native_mode.value}",
            native_mode,
            retained_native_module,
        )
    if repository_ssa_complete:
        native_reason = (
            "no retained native module is available"
            if retained_native_module is None
            else (
                f"retained {retained_native_module.format} is incompatible "
                f"with {target.operating_system}/{target.architecture}/{target.abi}"
                if not native_compatible else
                "repository SSA was explicitly preferred"
            )
        )
        return HostImplementationDecision(
            HostImplementationKind.REPOSITORY_SSA,
            native_reason,
            native_mode,
            retained_native_module,
        )
    if (
        machine_state_ssa_complete
        and machine_state_ssa_dialect in target.machine_state_ssa_dialects
    ):
        return HostImplementationDecision(
            HostImplementationKind.MACHINE_STATE_SSA,
            "repository legalization is incomplete; target executes the "
            "fully decoded machine-state SSA dialect",
            native_mode,
            retained_native_module,
        )
    if native_compatible:
        return HostImplementationDecision(
            HostImplementationKind.RETAINED_NATIVE_MODULE,
            f"repository SSA is incomplete; target consumes retained module as {native_mode.value}",
            native_mode,
            retained_native_module,
        )
    return HostImplementationDecision(
        HostImplementationKind.TRANSLATION_REQUIRED,
        (
            "repository SSA legalization is incomplete and neither the "
            "machine-state SSA dialect nor the retained native artifact is "
            "consumable by this target"
        ),
        native_mode,
        retained_native_module,
    )
WINDOWS_AMD64_NATIVE_LOADER = NativeTargetContext(
    "amd64", "windows", "windows-x64", frozenset({"pe-image", "coff-object"}),
    accepts_loadable_images=True,
)
WINDOWS_AMD64_NATIVE_LINKER = NativeTargetContext(
    "amd64", "windows", "windows-x64", frozenset({"coff-object"}),
    accepts_loadable_images=False,
)
PORTABLE_AMD64_MACHINE_SSA_VM = NativeTargetContext(
    "portable-vm", "internal", "machine-state-v1", frozenset(),
    accepts_relocatable_objects=False,
    machine_state_ssa_dialects=frozenset({
        "turing.machine-state-ssa.amd64.v1",
    }),
)


__all__ = [
    "HostImplementationDecision", "HostImplementationKind",
    "NativeRetentionMode", "NativeTargetContext", "RetainedNativeModule",
    "PORTABLE_AMD64_MACHINE_SSA_VM", "WINDOWS_AMD64_NATIVE_LINKER",
    "WINDOWS_AMD64_NATIVE_LOADER",
    "retain_pe_image", "select_host_implementation",
]
