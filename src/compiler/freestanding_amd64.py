"""Portable contract for Turing-authored freestanding AMD64 PE images.

PE is the container, not the operating-system ABI.  This personality admits a
small, explicit capability import surface and otherwise treats the image as a
self-contained x86-64-v1 machine program.  Foreign hosts may execute its
machine-state SSA or translate the ordinary repository-SSA portions; neither
route changes the authored capability contract.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable

from .machine_reference_vocabulary import DecodedInstruction, X86InstructionToken
from .native_code_retention import (
    NativeTargetContext, RetainedNativeModule, retain_pe_image,
)


FREESTANDING_AMD64_ABI = "turing-freestanding-amd64-v1"
FREESTANDING_CAPABILITY_LIBRARY = "turing-capability-v1.dll"


class FreestandingCapability(str, Enum):
    EXIT = "turing_exit"
    INPUT_POLL = "turing_input_poll"
    OUTPUT_PUBLISH = "turing_output_publish"
    CLOCK_MONOTONIC = "turing_clock_monotonic"
    MEMORY_COMMIT = "turing_memory_commit"


@dataclass(frozen=True, slots=True)
class FreestandingAMD64Shortfall:
    kind: str
    occurrence: int
    detail: str
    address: int | None = None


@dataclass(frozen=True, slots=True)
class FreestandingAMD64Validation:
    profile: str
    shortfalls: tuple[FreestandingAMD64Shortfall, ...]

    @property
    def compatible(self) -> bool:
        return not self.shortfalls

    def require_compatible(self) -> None:
        if self.shortfalls:
            raise ValueError("; ".join(item.detail for item in self.shortfalls))


# PCMPEQQ is SSE4.1.  Every other currently authoritative encoding belongs to
# the baseline integer/SSE2 surface or is an encoding form whose semantics do
# not require a post-x86-64-v1 CPU feature.  Keep this deny set explicit: a new
# token is reviewed by the permanent catalogue tests rather than inferred from
# mnemonic spelling.
_POST_X86_64_V1_TOKENS = frozenset({X86InstructionToken.PCMPEQQ_XMM_XMMM128})


def validate_freestanding_amd64_image(
    module: RetainedNativeModule,
    decoded: Iterable[DecodedInstruction] | None = None,
    *,
    executable_coverage_complete: bool = False,
) -> FreestandingAMD64Validation:
    """Prove container, ABI, imports, and byte-complete ISA compatibility.

    ``decoded`` is deliberately not enough by itself: a caller must also
    provide the executable-coverage proof produced by the owning program
    graph.  Otherwise an empty or reachable-prefix census could incorrectly
    certify arbitrary trailing executable bytes as x86-64-v1.
    """

    shortfalls: list[FreestandingAMD64Shortfall] = []

    def reject(kind: str, detail: str, address: int | None = None) -> None:
        shortfalls.append(FreestandingAMD64Shortfall(
            kind, len(shortfalls) + 1, detail, address,
        ))

    if module.format != "pe-image":
        reject("container", f"expected pe-image, received {module.format}")
    if module.architecture.casefold() != "amd64":
        reject("architecture", f"expected amd64, received {module.architecture}")
    if module.operating_system.casefold() != "turing":
        reject(
            "operating-system",
            f"expected freestanding turing environment, received {module.operating_system}",
        )
    if module.abi != FREESTANDING_AMD64_ABI:
        reject("abi", f"expected {FREESTANDING_AMD64_ABI}, received {module.abi}")

    allowed = frozenset(item.value for item in FreestandingCapability)
    for library, symbol, iat_rva, delayed in module.imports:
        if library.casefold() != FREESTANDING_CAPABILITY_LIBRARY:
            reject(
                "foreign-import",
                f"import {library}!{symbol} is outside the freestanding capability ABI",
                int(module.image_base) + int(iat_rva),
            )
        elif symbol not in allowed:
            reject(
                "unknown-capability",
                f"capability import {symbol!r} is not in {FREESTANDING_AMD64_ABI}",
                int(module.image_base) + int(iat_rva),
            )
        if delayed:
            reject(
                "delayed-capability",
                f"capability import {symbol!r} must be bound before execution",
                int(module.image_base) + int(iat_rva),
            )

    if decoded is None or not executable_coverage_complete:
        reject(
            "executable-coverage",
            "x86-64-v1 compatibility requires a byte-complete executable instruction census",
        )

    for instruction in (() if decoded is None else decoded):
        token = X86InstructionToken(instruction.token)
        if token in _POST_X86_64_V1_TOKENS:
            reject(
                "instruction-level",
                f"{token.name} requires a CPU feature beyond x86-64-v1",
                int(instruction.address),
            )

    return FreestandingAMD64Validation(
        FREESTANDING_AMD64_ABI, tuple(shortfalls),
    )


def validate_freestanding_amd64_program(
    module: RetainedNativeModule,
    program_graph,
) -> FreestandingAMD64Validation:
    """Validate against the owning graph's authoritative executable census."""

    instructions = tuple(
        instruction
        for record in program_graph.functions
        for instruction in record.report.instructions
    )
    return validate_freestanding_amd64_image(
        module,
        instructions,
        executable_coverage_complete=bool(program_graph.complete),
    )


def retain_freestanding_amd64_program(
    image,
    program_graph,
    *,
    source_identity: str = "",
) -> RetainedNativeModule:
    """Retain an authored PE only after proving the freestanding personality.

    The parsed bytes remain exact.  Operating-system and ABI fields describe
    the loader contract selected by the image's real import surface; they are
    changed only inside this validating constructor.
    """

    candidate = replace(
        retain_pe_image(image, source_identity=source_identity),
        operating_system="turing",
        abi=FREESTANDING_AMD64_ABI,
    )
    validate_freestanding_amd64_program(
        candidate, program_graph,
    ).require_compatible()
    return candidate


TURING_FREESTANDING_AMD64_LOADER = NativeTargetContext(
    "amd64", "turing", FREESTANDING_AMD64_ABI, frozenset({"pe-image"}),
    accepts_loadable_images=True,
)


__all__ = [
    "FREESTANDING_AMD64_ABI", "FREESTANDING_CAPABILITY_LIBRARY",
    "FreestandingAMD64Shortfall", "FreestandingAMD64Validation",
    "FreestandingCapability", "TURING_FREESTANDING_AMD64_LOADER",
    "retain_freestanding_amd64_program",
    "validate_freestanding_amd64_image", "validate_freestanding_amd64_program",
]
