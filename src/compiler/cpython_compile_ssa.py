"""Lift CPython's native ``compile`` implementation into repository SSA.

This module joins the existing PE parser, handwritten AMD64 vocabulary, CFG
lifter, and repository ``IRModule``.  It never invokes CPython's compiler and
does not leave a runtime native call behind as an implementation strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Any

from ..transmogrifier.ssa import (
    IRModule, SSAMachineControlLink, SSAMachineControlTable,
    SSAMachineIndirectLink, SSAMachineIndirectTable,
)
from .binary_ingestion import (
    PEImage,
    PERuntimeFunction,
    parse_pe_image,
    pe_runtime_function_region,
)
from .machine_code_lifting import BinaryToSSAResult, raise_binary_region_to_ssa
from .machine_reference_vocabulary import (
    EffectiveAddressOperand,
    MachineSemanticToken,
    RelativeAddressOperand,
    X86InstructionToken,
    X86ReferenceDecoder,
)
from .machine_dialect_ssa import (
    decoded_function_to_machine_ssa, repository_ssa_legalized,
)
from .native_code_retention import RetainedNativeModule, retain_pe_image


@dataclass(frozen=True, slots=True)
class NativeCompileSSABlocker:
    occurrence: int
    function_rva: int
    function_name: str
    kind: str
    address: int | None
    detail: str
    external_identity: str | None = None


@dataclass(frozen=True, slots=True)
class NativeCompileSSAResult:
    module: IRModule
    root_symbol: str
    root_function: str
    image_path: Path
    reached_function_rvas: tuple[int, ...]
    blockers: tuple[NativeCompileSSABlocker, ...]
    retained_native_module: RetainedNativeModule | None = None

    @property
    def complete(self) -> bool:
        """The complete reached machine program exists as executable SSA."""

        return not self.hard_blockers

    @property
    def hard_blockers(self) -> tuple[NativeCompileSSABlocker, ...]:
        return tuple(
            blocker for blocker in self.blockers
            if blocker.kind != "lowering"
        )

    @property
    def machine_state_blockers(self) -> tuple[NativeCompileSSABlocker, ...]:
        """Failures that prevent an executable machine-state SSA body.

        External modules and dynamic control targets are context-closure
        requirements, not missing instruction semantics.  A decoded indirect
        branch already carries its target plus the complete architectural
        state in ``IndirectBr``.  Conversely, an unknown encoding, a missing
        SSA body, or an undecodable control funclet means the machine program
        itself is incomplete and remains a blocker here.
        """

        contextual = {
            "external-machine-module",
            "indirect-call",
            "indirect-jump",
            "unresolved-call-target",
            "lowering",
        }
        return tuple(
            blocker for blocker in self.blockers
            if blocker.kind not in contextual
        )

    @property
    def legalization_shortfalls(self) -> tuple[NativeCompileSSABlocker, ...]:
        return tuple(
            blocker for blocker in self.blockers
            if blocker.kind == "lowering"
        )

    @property
    def repository_ssa_complete(self) -> bool:
        return not self.blockers and all(
            repository_ssa_legalized(function)
            for function in self.module.functions.values()
        )

    @property
    def machine_state_complete(self) -> bool:
        """Every reached body exists and no machine byte remains undecoded."""

        return not self.machine_state_blockers

    @property
    def dependency_context_complete(self) -> bool:
        """Every authored call/control dependency has a concrete provider."""

        return not self.hard_blockers

    @property
    def uses_machine_state_dialect(self) -> bool:
        return bool(self.legalization_shortfalls) or any(
            not repository_ssa_legalized(function)
            for function in self.module.functions.values()
        )


def _default_python_dll() -> Path:
    return (
        Path(sys.executable).resolve().parent
        / f"python{sys.version_info.major}{sys.version_info.minor}.dll"
    )


def _function_name(root_rva: int, root_symbol: str, rva: int) -> str:
    if int(rva) != int(root_rva):
        return f"cpython_{rva:08x}"
    normalized = re.sub(r"[^0-9A-Za-z_]", "_", str(root_symbol))
    if not normalized or normalized[0].isdigit():
        normalized = "host_" + normalized
    return normalized


def _code_owner_for_entry(
    image: PEImage, rva: int,
) -> tuple[PERuntimeFunction, bool]:
    """Return an exact code bound and whether it came from `.pdata`.

    Windows x64 permits leaf functions and export thunks with no unwind entry.
    Such an entry is admitted only when it is file-backed executable code. Its
    conservative upper bound is the first later concrete export, runtime-
    function start, or raw-backed section end. Reachable CFG decoding remains
    authoritative inside that bound and must close or report exact failures.
    """

    entry = int(rva)
    owner = image.runtime_function_for_rva(entry)
    if owner is not None:
        return owner, True
    section = image.section_for_rva(entry)
    if (
        section is None or not section.executable
        or section.file_offset_for_rva(entry) is None
    ):
        raise ValueError(
            f"PE entry RVA {entry:#x} has no unwind record and is not "
            "unique file-backed executable code"
        )
    section_end = int(section.virtual_address) + int(section.raw_size)
    boundaries = [section_end]
    boundaries.extend(
        int(function.begin_rva)
        for function in image.runtime_functions
        if entry < int(function.begin_rva) <= section_end
    )
    boundaries.extend(
        int(export.rva)
        for export in image.exports
        if export.rva is not None
        and entry < int(export.rva) <= section_end
        and section.contains_rva(int(export.rva))
    )
    end = min(boundaries)
    if end <= entry:
        raise ValueError(f"PE leaf entry RVA {entry:#x} has no positive code bound")
    return PERuntimeFunction(entry, end, 0), False


def _code_region_for_owner(
    image: PEImage, owner: PERuntimeFunction,
) -> tuple[int, bytes]:
    size = int(owner.end_rva) - int(owner.begin_rva)
    if size <= 0:
        raise ValueError("PE code owner has no positive byte extent")
    offset = image.file_offset_for_rva(int(owner.begin_rva))
    if offset is None or offset + size > len(image.encoded):
        raise ValueError("PE code owner is not completely file backed")
    return offset, bytes(image.encoded[offset:offset + size])


def _direct_function_targets(
    image: PEImage,
    owner: PERuntimeFunction,
    lifting: BinaryToSSAResult,
) -> tuple[tuple[int, PERuntimeFunction], ...]:
    targets: dict[int, PERuntimeFunction] = {}
    for instruction in lifting.decoded:
        if MachineSemanticToken(instruction.semantic) not in {
            MachineSemanticToken.DIRECT_RELATIVE_CALL,
            MachineSemanticToken.DIRECT_RELATIVE_JUMP,
            MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP,
        }:
            continue
        operand = next((
            item for item in instruction.operands
            if isinstance(item, RelativeAddressOperand)
        ), None)
        if operand is None:
            continue
        target_rva = int(operand.target_address) - int(image.image_base)
        if (
            MachineSemanticToken(instruction.semantic)
            is not MachineSemanticToken.DIRECT_RELATIVE_CALL
            and int(owner.begin_rva) <= target_rva < int(owner.end_rva)
        ):
            continue
        try:
            target, _has_unwind = _code_owner_for_entry(image, target_rva)
        except ValueError:
            continue
        targets[int(target_rva)] = target
    return tuple((key, targets[key]) for key in sorted(targets))


def _static_indirect_function_targets(
    image: PEImage,
    owner: PERuntimeFunction,
    lifting: BinaryToSSAResult,
) -> tuple[tuple[int, PERuntimeFunction], ...]:
    """Prove PE-local targets stored in fixed RIP-relative pointer slots.

    These are just as static as a relative call once the immutable input image
    is known.  Import-address-table slots are deliberately excluded: their
    identities belong to dependency modules, not this image.
    """

    import_slots = {
        int(item.iat_rva) for item in (*image.imports, *image.delay_imports)
    }
    targets: dict[int, PERuntimeFunction] = {}
    for instruction in lifting.decoded:
        if MachineSemanticToken(instruction.semantic) not in {
            MachineSemanticToken.INDIRECT_CALL,
            MachineSemanticToken.INDIRECT_JUMP,
        }:
            continue
        operand = next((
            item for item in instruction.operands
            if isinstance(item, EffectiveAddressOperand) and item.rip_relative
        ), None)
        if operand is None:
            continue
        slot_address = (
            int(instruction.address) + len(instruction.encoded)
            + int(operand.displacement)
        )
        slot_rva = slot_address - int(image.image_base)
        if slot_rva in import_slots:
            continue
        file_offset = image.file_offset_for_rva(slot_rva)
        if file_offset is None or file_offset + 8 > len(image.encoded):
            continue
        pointer = int.from_bytes(
            image.encoded[file_offset:file_offset + 8], "little"
        )
        target_rva = pointer - int(image.image_base)
        try:
            target, _has_unwind = _code_owner_for_entry(image, target_rva)
        except ValueError:
            continue
        targets[int(target_rva)] = target
    return tuple((key, targets[key]) for key in sorted(targets))


def _external_control_occurrences(function):
    for block_name, block in function.blocks.items():
        for instruction in block.instrs:
            source_address = instruction.attributes.get("machine_address")
            if instruction.op == "CondBr":
                for role in ("true", "false"):
                    target = instruction.attributes.get(f"{role}_target_address")
                    if target is not None:
                        yield block_name, int(source_address or 0), role, int(target)
            elif instruction.op == "Br":
                target = instruction.attributes.get("target_address")
                if target is not None:
                    yield block_name, int(source_address or 0), "direct", int(target)
            elif instruction.op == "machine.ExternalBr":
                for target in instruction.attributes.get("target_addresses", ()):
                    yield block_name, int(source_address or 0), "direct", int(target)


def _link_machine_control_funclets(
    image: PEImage,
    functions: dict[str, Any],
) -> tuple[SSAMachineControlTable, tuple[NativeCompileSSABlocker, ...]]:
    """Resolve tail transfers to exact full-machine-state funclet entries."""

    links: list[SSAMachineControlLink] = []
    blockers: list[NativeCompileSSABlocker] = []
    pending = [
        (function.name, *occurrence)
        for function in tuple(functions.values())
        for occurrence in _external_control_occurrences(function)
    ]
    processed_edges: set[tuple[str, str, int, str, int]] = set()
    funclets: dict[int, str] = {}

    while pending:
        source_function, source_block, source_address, role, target_address = pending.pop(0)
        edge = (
            str(source_function), str(source_block), int(source_address),
            str(role), int(target_address),
        )
        if edge in processed_edges:
            continue
        processed_edges.add(edge)
        target_rva = int(target_address) - int(image.image_base)
        owner = image.runtime_function_for_rva(target_rva)
        if owner is None:
            links.append(SSAMachineControlLink(
                str(source_function), str(source_block), int(source_address),
                str(role), int(target_address), target_kind="outside-image",
            ))
            continue

        funclet_name = funclets.get(int(target_address))
        if funclet_name is None:
            funclet_name = f"machine_funclet_{target_rva:08x}"
            offset = int(target_rva) - int(owner.begin_rva)
            _record, _file_offset, owner_region = pe_runtime_function_region(
                image, int(owner.begin_rva),
                maximum_function_size=int(owner.end_rva) - int(owner.begin_rva),
            )
            region = owner_region[offset:]
            decoder = X86ReferenceDecoder()
            report = decoder.decode_cfg_report(
                region, base_address=int(target_address),
            )
            if report.failures:
                for failure in report.failures:
                    blockers.append(NativeCompileSSABlocker(
                        len(blockers) + 1, int(owner.begin_rva), funclet_name,
                        f"machine-control-{failure.category}",
                        int(failure.address), str(failure.reason),
                    ))
                links.append(SSAMachineControlLink(
                    str(source_function), str(source_block), int(source_address),
                    str(role), int(target_address),
                    target_function=funclet_name,
                    target_kind=(
                        "runtime-function-entry"
                        if target_rva == int(owner.begin_rva)
                        else "runtime-function-interior"
                    ),
                ))
                continue
            funclet = decoded_function_to_machine_ssa(
                funclet_name, report.instructions,
                external_fallthrough_address=(
                    report.external_fallthrough_addresses[0]
                    if len(report.external_fallthrough_addresses) == 1
                    else None
                ),
            )
            funclet.metadata.update({
                "machine_funclet": True,
                "machine_funclet_entry_address": int(target_address),
                "machine_owner_rva": int(owner.begin_rva),
                "machine_unreachable_spans": tuple(
                    (int(target_address) + start, int(target_address) + end)
                    for start, end in report.unreachable_spans
                ),
            })
            functions[funclet_name] = funclet
            funclets[int(target_address)] = funclet_name
            pending.extend(
                (funclet_name, *occurrence)
                for occurrence in _external_control_occurrences(funclet)
            )

        links.append(SSAMachineControlLink(
            str(source_function), str(source_block), int(source_address),
            str(role), int(target_address), funclet_name,
            f"block_{int(target_address):016x}",
            (
                "runtime-function-entry"
                if target_rva == int(owner.begin_rva)
                else "runtime-function-interior"
            ),
        ))

    return SSAMachineControlTable(tuple(links)), tuple(blockers)


def lift_pe_export_to_ssa(
    image_path: str | Path | None = None,
    *,
    root_symbol: str = "Py_CompileString",
    root_rva: int | None = None,
    root_calling_convention: str = "",
) -> NativeCompileSSAResult:
    """Lift the complete reachable closure rooted at a PE export or RVA.

    There is intentionally no function-count or depth cutoff.  The worklist
    stabilizes only when every direct internal call/tail-call target has been
    visited.  Dynamic/indirect destinations cannot be guessed; each occurrence
    is retained as a blocker in the result.
    """

    path = Path(image_path) if image_path is not None else _default_python_dll()
    encoded = path.read_bytes()
    image, _statistics = parse_pe_image(
        encoded, maximum_file_size=len(encoded)
    )
    if root_rva is None:
        exported = image.export_by_name(root_symbol)
        if exported is None or exported.rva is None:
            raise ValueError(
                f"CPython image {path} has no concrete export {root_symbol!r}"
            )
        root_entry_rva = int(exported.rva)
    else:
        root_entry_rva = int(root_rva)
    root_owner, root_has_unwind = _code_owner_for_entry(image, root_entry_rva)

    pending = [(root_entry_rva, root_owner, root_has_unwind)]
    queued = {root_entry_rva}
    liftings: dict[int, BinaryToSSAResult] = {}
    owners: dict[int, PERuntimeFunction] = {}
    blockers: list[NativeCompileSSABlocker] = []

    while pending:
        rva, owner, owner_has_unwind = pending.pop(0)
        rva = int(rva)
        owners[rva] = owner
        _offset, region = _code_region_for_owner(image, owner)
        owner_offset = rva - int(owner.begin_rva)
        if owner_offset:
            region = region[owner_offset:]
        function_name = _function_name(root_entry_rva, root_symbol, rva)
        lifting = raise_binary_region_to_ssa(
            region,
            maximum_file_size=len(region),
            base_address=int(image.image_base) + rva,
            name=function_name,
            argument_registers=("rcx", "rdx", "r8", "r9"),
            argument_names=("arg0", "arg1", "arg2", "arg3"),
            full_vocabulary_report=True,
            cfg_decode=True,
        )
        liftings[rva] = lifting
        if rva == root_entry_rva and lifting.function is not None:
            lifting.function.metadata.update({
                "machine_entry_rva": root_entry_rva,
                "machine_owner_rva": int(owner.begin_rva),
                "machine_owner_has_unwind": bool(owner_has_unwind),
                "host_calling_convention": str(root_calling_convention),
            })
        for failure in lifting.failed_vocabulary:
            blockers.append(NativeCompileSSABlocker(
                len(blockers) + 1,
                rva,
                function_name,
                str(failure.category),
                int(failure.address),
                str(failure.reason),
            ))
        for target_rva, target in (
            *_direct_function_targets(image, owner, lifting),
            *_static_indirect_function_targets(image, owner, lifting),
        ):
            target_rva = int(target_rva)
            if target_rva not in queued:
                queued.add(target_rva)
                pending.append((
                    target_rva, target,
                    image.runtime_function_for_rva(target_rva) is not None,
                ))

    functions = {
        lifting.function.name: lifting.function
        for lifting in liftings.values()
        if lifting.function is not None
    }
    # Every missing function body is itself a retained blocker.  This makes
    # absence visible even if a later assembler sees only the IRModule rather
    # than the per-instruction decoding ledger.
    for rva, lifting in liftings.items():
        if lifting.function is None and not lifting.failed_vocabulary:
            blockers.append(NativeCompileSSABlocker(
                len(blockers) + 1,
                int(rva),
                _function_name(root_entry_rva, root_symbol, rva),
                "missing-ssa-function",
                int(image.image_base) + int(rva),
                "decoded host function did not produce repository SSA",
            ))
    address_to_name = {
        int(image.image_base) + rva: _function_name(
            root_entry_rva, root_symbol, rva
        )
        for rva in owners
    }
    import_by_slot = {
        int(item.iat_rva): item.display_name
        for item in (*image.imports, *image.delay_imports)
    }
    indirect_links: list[SSAMachineIndirectLink] = []
    for function in functions.values():
        for block in function.blocks.values():
            for instruction in block.instrs:
                if instruction.op not in {"Call", "IndirectBr"}:
                    continue
                edge_kind = "call" if instruction.op == "Call" else "jump"
                address = instruction.attributes.get("callee_address")
                if address is None:
                    source_address = int(
                        instruction.attributes.get("machine_address") or 0
                    )
                    slot_address = instruction.attributes.get(
                        "indirect_slot_address"
                    )
                    operand_kind = str(
                        instruction.attributes.get("indirect_operand")
                        or "dynamic-state"
                    )
                    if slot_address is not None:
                        slot_address = int(slot_address)
                        slot_rva = slot_address - int(image.image_base)
                        external_identity = import_by_slot.get(slot_rva)
                        if external_identity is not None:
                            indirect_links.append(SSAMachineIndirectLink(
                                function.name, source_address, edge_kind,
                                operand_kind, slot_address,
                                "pe-import", external_identity=external_identity,
                            ))
                            blockers.append(NativeCompileSSABlocker(
                                len(blockers) + 1,
                                next((
                                    rva for rva, lifting in liftings.items()
                                    if lifting.function is function
                                ), -1),
                                function.name,
                                "external-machine-module",
                                source_address,
                                f"named PE import {external_identity} requires "
                                "a retained or translated machine module",
                                external_identity,
                            ))
                            instruction.attributes.update({
                                "external_identity": external_identity,
                                "external_reference_kind": "pe-import-slot",
                                "indirect_target_resolved": True,
                            })
                            continue
                        file_offset = image.file_offset_for_rva(slot_rva)
                        raw_target = None
                        if file_offset is not None and file_offset + 8 <= len(image.encoded):
                            raw_target = int.from_bytes(
                                image.encoded[file_offset:file_offset + 8], "little"
                            )
                        target_owner = (
                            None if raw_target is None else
                            image.runtime_function_for_rva(
                                raw_target - int(image.image_base)
                            )
                        )
                        target_function = (
                            None if target_owner is None else
                            address_to_name.get(
                                int(image.image_base) + int(target_owner.begin_rva)
                            )
                        )
                        if target_function is not None:
                            indirect_links.append(SSAMachineIndirectLink(
                                function.name, source_address, edge_kind,
                                operand_kind, slot_address,
                                "internal-function", raw_target,
                                target_function,
                            ))
                            instruction.attributes.update({
                                "callee_address": raw_target,
                                "callee": target_function,
                                "source_linked": True,
                                "native_decompiled": True,
                                "indirect_target_resolved": True,
                            })
                            continue
                        indirect_links.append(SSAMachineIndirectLink(
                            function.name, source_address, edge_kind,
                            operand_kind, slot_address, "unresolved-slot",
                            raw_target,
                        ))
                        detail = (
                            f"RIP-relative slot {slot_address:#x} contains "
                            f"{raw_target:#x} without an import or code owner"
                            if raw_target is not None else
                            f"RIP-relative slot {slot_address:#x} is not file-backed"
                        )
                    else:
                        indirect_links.append(SSAMachineIndirectLink(
                            function.name, source_address, edge_kind,
                            operand_kind, target_kind="dynamic-state",
                        ))
                        detail = (
                            f"native {edge_kind} target depends on "
                            f"{operand_kind} machine state"
                        )
                    blockers.append(NativeCompileSSABlocker(
                        len(blockers) + 1,
                        next((
                            rva for rva, lifting in liftings.items()
                            if lifting.function is function
                        ), -1),
                        function.name,
                        f"indirect-{edge_kind}",
                        source_address,
                        detail,
                    ))
                    continue
                if edge_kind == "jump":
                    continue
                callee = address_to_name.get(int(address))
                if callee is None:
                    blockers.append(NativeCompileSSABlocker(
                        len(blockers) + 1,
                        next((
                            rva for rva, lifting in liftings.items()
                            if lifting.function is function
                        ), -1),
                        function.name,
                        "unresolved-call-target",
                        instruction.attributes.get("machine_address"),
                        f"direct target {int(address):#x} has no lifted entry",
                    ))
                else:
                    instruction.attributes["callee"] = callee
                    instruction.attributes["source_linked"] = True
                    instruction.attributes["native_decompiled"] = True

    machine_control_table, machine_control_blockers = (
        _link_machine_control_funclets(image, functions)
    )
    for failure in machine_control_blockers:
        blockers.append(NativeCompileSSABlocker(
            len(blockers) + 1,
            failure.function_rva,
            failure.function_name,
            failure.kind,
            failure.address,
            failure.detail,
        ))

    reached = tuple(sorted(owners))
    return NativeCompileSSAResult(
        IRModule(
            functions,
            machine_control_table=machine_control_table,
            machine_indirect_table=SSAMachineIndirectTable(tuple(indirect_links)),
        ),
        str(root_symbol),
        _function_name(
            root_entry_rva, root_symbol, root_entry_rva
        ),
        path.resolve(),
        reached,
        tuple(blockers),
        retain_pe_image(image, source_identity=str(path.resolve())),
    )


def lift_cpython_compile_to_ssa(
    image_path: str | Path | None = None,
    *,
    root_symbol: str = "Py_CompileString",
) -> NativeCompileSSAResult:
    """Compatibility spelling for CPython's exported compiler closure."""

    return lift_pe_export_to_ssa(image_path, root_symbol=root_symbol)


__all__ = [
    "NativeCompileSSABlocker",
    "NativeCompileSSAResult",
    "lift_pe_export_to_ssa",
    "lift_cpython_compile_to_ssa",
]
