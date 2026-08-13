"""Executable boundary for the retained AMD64 machine-state SSA dialect.

The dialect is not a native call and it is not ordinary scalar repository
SSA.  This adapter validates every recorded state transition against the
decoder (and, when present, the retained PE image), then supplies that exact
instruction catalogue to the repository's internal reversible machine VM.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
from typing import Any, Mapping

from ..transmogrifier.ssa import Function, IRModule
from .amd64_machine_semantics import (
    condition_holds, default_effect_handlers, indirect_target,
)
from .binary_ingestion import PEImage, parse_pe_image
from .machine_dialect_ssa import MACHINE_SSA_DIALECT
from .machine_execution import MachineExecutionOrchestrator
from .machine_reference_vocabulary import X86ReferenceDecoder
from .machine_symbolic_effects import symbolic_effect_for_instruction
from .native_code_retention import RetainedNativeModule


class MachineSSABoundaryError(ValueError):
    """Machine SSA cannot be admitted as an executable guest program."""


@dataclass(frozen=True, slots=True)
class MachineSSAFunctionRecord:
    """VM-facing record with the same report contract as binary ingestion."""

    name: str
    report: Any


@dataclass(frozen=True, slots=True)
class MachineSSAProgram:
    """A context-complete decoded program sourced from machine-state SSA."""

    image: Any
    functions: tuple[MachineSSAFunctionRecord, ...]
    entries: Mapping[str, int]

    def entry_address(self, function_name: str) -> int:
        try:
            return int(self.entries[str(function_name)])
        except KeyError as error:
            raise KeyError(f"no machine-SSA entry named {function_name!r}") from error

    def executor(self, **overrides: Any) -> MachineExecutionOrchestrator:
        options = {
            "effect_handlers": default_effect_handlers(),
            "predicate_handler": condition_holds,
            "indirect_target_handler": indirect_target,
        }
        options.update(overrides)
        return MachineExecutionOrchestrator(self, **options)


def _decode_transition(instruction, decoder: X86ReferenceDecoder):
    attributes = instruction.attributes
    required = {
        "machine_dialect", "machine_address", "machine_token",
        "machine_semantic", "machine_encoded", "machine_operands",
        "machine_reads", "machine_writes", "effect_domains",
    }
    missing = sorted(required - attributes.keys())
    if missing:
        raise MachineSSABoundaryError(
            f"{instruction.op} lacks machine context: {', '.join(missing)}"
        )
    if attributes["machine_dialect"] != MACHINE_SSA_DIALECT:
        raise MachineSSABoundaryError(
            f"unsupported machine SSA dialect {attributes['machine_dialect']!r}"
        )
    try:
        encoded = bytes.fromhex(str(attributes["machine_encoded"]))
    except ValueError as error:
        raise MachineSSABoundaryError("machine_encoded is not hexadecimal") from error
    if not encoded:
        raise MachineSSABoundaryError("machine transition has no encoded bytes")
    address = int(attributes["machine_address"])
    decoded, end = decoder.decode_one(
        memoryview(encoded), 0, base_address=address,
    )
    if end != len(encoded):
        raise MachineSSABoundaryError(
            f"machine transition at {address:#x} contains trailing bytes"
        )
    expected_op = f"machine.{decoded.semantic.name.lower()}"
    checks = {
        "operation": (instruction.op, expected_op),
        "token": (str(attributes["machine_token"]), decoded.token.name),
        "semantic": (str(attributes["machine_semantic"]), decoded.semantic.name),
        "operands": (
            tuple(attributes["machine_operands"]),
            tuple(repr(item) for item in decoded.operands),
        ),
    }
    for label, (observed, expected) in checks.items():
        if observed != expected:
            raise MachineSSABoundaryError(
                f"machine {label} mismatch at {address:#x}: "
                f"recorded {observed!r}, decoded {expected!r}"
            )
    effect = symbolic_effect_for_instruction(decoded)
    effect_checks = {
        "reads": (tuple(attributes["machine_reads"]), effect.reads),
        "writes": (tuple(attributes["machine_writes"]), effect.writes),
        "effect domains": (
            tuple(attributes["effect_domains"]), effect.effect_domains,
        ),
        "trap contract": (bool(attributes.get("may_trap", False)), effect.may_trap),
        "conditional contract": (
            bool(attributes.get("conditional", False)), effect.conditional,
        ),
    }
    for label, (observed, expected) in effect_checks.items():
        if observed != expected:
            raise MachineSSABoundaryError(
                f"machine {label} mismatch at {address:#x}: "
                f"recorded {observed!r}, derived {expected!r}"
            )
    return decoded


def _machine_instructions(function: Function):
    decoder = X86ReferenceDecoder()
    decoded = []
    for block in function.blocks.values():
        for instruction in block.instrs:
            if not instruction.op.startswith("machine."):
                continue
            if instruction.op in {
                "machine.PhiState", "machine.condition",
                "machine.ExternalBr", "machine.Terminate",
            }:
                continue
            decoded.append(_decode_transition(instruction, decoder))
    decoded.sort(key=lambda item: int(item.address))
    if len({int(item.address) for item in decoded}) != len(decoded):
        raise MachineSSABoundaryError(
            f"machine SSA function {function.name!r} repeats an instruction address"
        )
    expected_count = int(function.metadata.get("machine_instruction_count", -1))
    if expected_count != len(decoded):
        raise MachineSSABoundaryError(
            f"machine SSA function {function.name!r} declares {expected_count} "
            f"instructions but contains {len(decoded)}"
        )
    if not decoded:
        raise MachineSSABoundaryError(
            f"machine SSA function {function.name!r} has no transitions"
        )
    return tuple(decoded)


def _verify_retained_bytes(image: PEImage, instructions) -> None:
    for instruction in instructions:
        rva = int(instruction.address) - int(image.image_base)
        offset = image.file_offset_for_rva(rva)
        if offset is None:
            raise MachineSSABoundaryError(
                f"machine instruction {instruction.address:#x} is not file-backed "
                "by the retained PE"
            )
        observed = image.encoded[offset:offset + len(instruction.encoded)]
        if observed != instruction.encoded:
            raise MachineSSABoundaryError(
                f"retained PE bytes disagree with machine SSA at "
                f"{instruction.address:#x}"
            )


def machine_ssa_program(
    module: IRModule,
    *,
    retained_native_module: RetainedNativeModule | None = None,
    image: Any | None = None,
) -> MachineSSAProgram:
    """Validate an IR module and expose its machine functions to the VM.

    Production admission requires a context-complete retained PE.  ``image``
    is an explicit test/embedding seam for an already validated image object.
    """

    if retained_native_module is not None and image is not None:
        raise MachineSSABoundaryError(
            "supply retained_native_module or image, not both"
        )
    if retained_native_module is not None:
        if (
            retained_native_module.format != "pe-image"
            or retained_native_module.architecture.casefold() != "amd64"
        ):
            raise MachineSSABoundaryError(
                "AMD64 machine SSA currently requires a retained AMD64 PE image"
            )
        image, _statistics = parse_pe_image(
            retained_native_module.encoded,
            maximum_file_size=len(retained_native_module.encoded),
        )
        if int(image.image_base) != int(retained_native_module.image_base):
            raise MachineSSABoundaryError("retained PE image-base catalogue mismatch")
    if image is None:
        raise MachineSSABoundaryError(
            "machine SSA execution requires retained loader/image context"
        )

    records = []
    entries: dict[str, int] = {}
    all_addresses: dict[int, object] = {}
    for function in module.functions.values():
        if function.metadata.get("dialect") != MACHINE_SSA_DIALECT:
            continue
        instructions = _machine_instructions(function)
        if isinstance(image, PEImage):
            _verify_retained_bytes(image, instructions)
        for instruction in instructions:
            previous = all_addresses.setdefault(int(instruction.address), instruction)
            if previous != instruction:
                raise MachineSSABoundaryError(
                    f"conflicting machine SSA transitions at {instruction.address:#x}"
                )
        entry = int(function.metadata.get("machine_funclet_entry_address", instructions[0].address))
        if entry not in {int(item.address) for item in instructions}:
            raise MachineSSABoundaryError(
                f"machine SSA entry {entry:#x} is absent from {function.name!r}"
            )
        entries[function.name] = entry
        report = SimpleNamespace(instructions=instructions, failures=())
        records.append(MachineSSAFunctionRecord(function.name, report))
    if not records:
        raise MachineSSABoundaryError(
            f"module contains no {MACHINE_SSA_DIALECT!r} functions"
        )
    return MachineSSAProgram(
        image, tuple(records), MappingProxyType(entries),
    )


__all__ = [
    "MachineSSABoundaryError", "MachineSSAFunctionRecord",
    "MachineSSAProgram", "machine_ssa_program",
]
