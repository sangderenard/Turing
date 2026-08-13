"""Audited repository-SSA to PE instruction recompilation ledger."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable, Mapping

from ..transmogrifier.ssa import IRModule
from .binary_ingestion import (
    ReverseBinarySelectionPlan, X86EncodingFields, plan_reverse_selection,
    write_reverse_selection,
)
from .x86_tensor_read_head import (
    X86AllocatedInstruction, X86TensorReadHead,
    controlled_x86_64_read_head_profile,
)
from .binary_ingestion import PEImage, parse_pe_image
from .machine_code_lifting import _machine_group_fingerprints
from .machine_reference_vocabulary import X86ReferenceDecoder


@dataclass(frozen=True, slots=True)
class PERecompilationOccurrence:
    occurrence: int
    function_name: str
    machine_address: int
    disposition: str
    encoded: bytes | None
    target_token: int | None
    witness: str
    detail: str


@dataclass(frozen=True, slots=True)
class PERecompilationLedger:
    occurrences: tuple[PERecompilationOccurrence, ...]

    @property
    def complete(self) -> bool:
        return all(item.encoded is not None for item in self.occurrences)

    @property
    def unresolved(self) -> tuple[PERecompilationOccurrence, ...]:
        return tuple(item for item in self.occurrences if item.encoded is None)


@dataclass(frozen=True, slots=True)
class InPlacePERecompilation:
    encoded: bytes
    ledger: PERecompilationLedger
    rewritten_ranges: tuple[tuple[int, int], ...]
    image: PEImage


def _grouped(
    function,
    block_names: frozenset[str] | None = None,
) -> dict[int, tuple[object, ...]]:
    result: dict[int, list[object]] = {}
    for block_name, block in function.blocks.items():
        if block_names is not None and str(block_name) not in block_names:
            continue
        for instruction in block.instrs:
            address = instruction.attributes.get("machine_address")
            if address is not None:
                result.setdefault(int(address), []).append(instruction)
    return {address: tuple(items) for address, items in result.items()}


def _original_provenance(group: tuple[object, ...]):
    records = tuple(dict.fromkeys(
        (
            item.attributes.get("machine_token"),
            item.attributes.get("machine_bytes"),
        )
        for item in group
        if item.attributes.get("machine_bytes") is not None
    ))
    if len(records) != 1:
        return None
    raw_token, raw_bytes = records[0]
    try:
        return int(raw_token), bytes.fromhex(str(raw_bytes))
    except (TypeError, ValueError):
        return None


def build_pe_recompilation_ledger(
    module: IRModule,
    *,
    proven_facts_by_address: Mapping[int, Iterable[str]] | None = None,
    encoding_fields_by_address: Mapping[int, X86EncodingFields] | None = None,
    allocated_instructions_by_address: Mapping[int, X86AllocatedInstruction] | None = None,
    allow_new_selection: bool = False,
    selected_blocks: Iterable[tuple[str, str]] | None = None,
) -> PERecompilationLedger:
    """Classify retained machine-address SSA groups without omission.

    ``selected_blocks`` is the incremental-emission boundary.  When omitted,
    the historical whole-module behavior is unchanged.  When supplied, every
    named basic block is validated up front and only complete machine-address
    groups owned by those repository-SSA blocks enter the ledger.  The group
    fingerprints remain function-wide, so selecting a block cannot weaken the
    lift-time provenance check or turn a partial instruction expansion into an
    apparently unchanged instruction.
    """

    facts = proven_facts_by_address or {}
    encoding_fields = encoding_fields_by_address or {}
    allocated = allocated_instructions_by_address or {}
    selected: dict[str, set[str]] | None = None
    if selected_blocks is not None:
        selected = {}
        for raw_function, raw_block in selected_blocks:
            function_name = str(raw_function)
            block_name = str(raw_block)
            function = module.functions.get(function_name)
            if function is None:
                raise KeyError(f"unknown SSA function {function_name!r}")
            if block_name not in function.blocks:
                raise KeyError(
                    f"unknown SSA basic block {function_name}:{block_name}"
                )
            selected.setdefault(function_name, set()).add(block_name)
        if not selected:
            raise ValueError("incremental recompilation requires at least one block")
    write_head = X86TensorReadHead.from_profile(
        controlled_x86_64_read_head_profile(),
    )
    occurrences: list[PERecompilationOccurrence] = []
    decoder = X86ReferenceDecoder()
    for function_name, function in module.functions.items():
        if selected is not None and str(function_name) not in selected:
            continue
        original_fingerprints = dict(
            function.metadata.get("machine_group_fingerprints", ())
        )
        current_fingerprints = dict(_machine_group_fingerprints(function))
        selected_names = (
            None if selected is None
            else frozenset(selected[str(function_name)])
        )
        groups = _grouped(function, selected_names)
        selected_addresses = frozenset(groups)
        if selected_names is not None:
            authored_block_addresses = function.metadata.get(
                "machine_block_addresses"
            )
            if authored_block_addresses is None:
                raise ValueError(
                    f"{function_name} lacks lift-time block/address provenance; "
                    "scoped recompilation is not sound"
                )
            authored_by_block = {
                str(block): tuple(map(int, addresses))
                for block, addresses in authored_block_addresses
            }
            missing_catalogue = sorted(selected_names - authored_by_block.keys())
            if missing_catalogue:
                raise ValueError(
                    f"{function_name} lacks authored address catalogues for blocks "
                    f"{missing_catalogue!r}"
                )
            authored_addresses = frozenset(
                address
                for block_name in selected_names
                for address in authored_by_block[block_name]
            )
            if authored_addresses != selected_addresses:
                raise ValueError(
                    f"{function_name} selected block machine-address membership "
                    "changed since lifting"
                )
            selected_addresses = authored_addresses
        addresses = tuple(dict.fromkeys((
            *(
                address for address in original_fingerprints
                if selected_names is None or address in selected_addresses
            ),
            *(
                address for address in current_fingerprints
                if selected_names is None or address in selected_addresses
            ),
        )))
        for address in addresses:
            group = groups.get(int(address), ())
            provenance = _original_provenance(group)
            unchanged = (
                address in original_fingerprints
                and original_fingerprints[address] == current_fingerprints.get(address)
            )
            if unchanged and provenance is not None:
                token, encoded = provenance
                try:
                    decoded, end = decoder.decode_one(
                        memoryview(encoded), 0, base_address=int(address),
                    )
                except (ValueError, TypeError):
                    decoded, end = None, -1
                if (
                    decoded is not None and end == len(encoded)
                    and int(decoded.token) == token
                ):
                    witness = sha256(
                        b"exact-retention\0" + encoded
                        + original_fingerprints[address].encode("ascii")
                    ).hexdigest()
                    occurrences.append(PERecompilationOccurrence(
                        len(occurrences) + 1, str(function_name), int(address),
                        "exact-retention", encoded, token, witness,
                        "complete SSA group matches its lift-time fingerprint",
                    ))
                    continue
            selection: ReverseBinarySelectionPlan | None = None
            if allow_new_selection and group:
                selection = plan_reverse_selection(
                    group, proven_facts=facts.get(int(address), ()),
                    allow_multi_lane=True,
                )
            selected_encoded = None
            selected_disposition = None
            selected_witness = None
            if selection is not None:
                fields = encoding_fields.get(int(address))
                allocated_instruction = allocated.get(int(address))
                if allocated_instruction is not None:
                    if allocated_instruction.token != selection.selection.target_token:
                        raise ValueError(
                            f"allocated token mismatch at {int(address):#x}"
                        )
                    if (
                        allocated_instruction.address is not None
                        and allocated_instruction.address != int(address)
                    ):
                        raise ValueError(
                            f"allocated instruction address mismatch at {int(address):#x}"
                        )
                    selected_encoded = write_head.write_allocated(
                        X86AllocatedInstruction(
                            allocated_instruction.token,
                            allocated_instruction.operands,
                            int(address),
                        )
                    )
                    selected_disposition = "allocated-selection"
                    selected_witness = sha256(
                        b"allocated-selection\0" + selected_encoded
                        + selection.witness.encode("ascii")
                    ).hexdigest()
                elif selection.encoded is not None or fields is not None:
                    selected_encoded = write_reverse_selection(selection, fields)
                    selected_disposition = selection.mode
                    selected_witness = selection.witness
            occurrences.append(PERecompilationOccurrence(
                len(occurrences) + 1, str(function_name), int(address),
                (
                    (selected_disposition or selection.mode) if selection is not None
                    else "unresolved"
                ),
                selected_encoded,
                None if selection is None else selection.selection.target_token,
                (
                    sha256(
                        repr((function_name, address, current_fingerprints.get(address)))
                        .encode("utf-8")
                    ).hexdigest()
                    if selection is None else (selected_witness or selection.witness)
                ),
                (
                    "SSA group changed or lacks complete original provenance"
                    if selection is None else (
                        selection.selection.canonical_meaning
                        if selected_encoded is not None else
                        "selection proved, but allocated physical operands are absent"
                    )
                ),
            ))
    return PERecompilationLedger(tuple(occurrences))


def recompile_pe_in_place(
    image: PEImage,
    ledger: PERecompilationLedger,
) -> InPlacePERecompilation:
    """Rewrite complete same-size instruction units and revalidate the PE.

    This intentionally does not perform layout.  It is the exact in-place
    subset of PE emission and rejects unresolved, duplicate, overlapping, or
    non-file-backed occurrences.
    """

    if not ledger.complete:
        raise ValueError(
            f"PE recompilation ledger has {len(ledger.unresolved)} unresolved occurrences"
        )
    output = bytearray(image.encoded)
    ranges: list[tuple[int, int]] = []
    seen_addresses: dict[int, tuple[bytes, int | None, tuple[int, int]]] = {}
    decoder = X86ReferenceDecoder()
    for occurrence in ledger.occurrences:
        address = int(occurrence.machine_address)
        assert occurrence.encoded is not None
        decoded, end = decoder.decode_one(
            memoryview(occurrence.encoded), 0, base_address=address,
        )
        if end != len(occurrence.encoded):
            raise ValueError(f"PE rewrite at {address:#x} contains trailing bytes")
        if (
            occurrence.target_token is not None
            and int(decoded.token) != int(occurrence.target_token)
        ):
            raise ValueError(f"PE rewrite token mismatch at {address:#x}")
        rva = address - int(image.image_base)
        offset = image.file_offset_for_rva(rva)
        if offset is None:
            raise ValueError(f"PE rewrite address {address:#x} is not file-backed")
        stop = offset + len(occurrence.encoded)
        previous = seen_addresses.get(address)
        identity = (
            occurrence.encoded, occurrence.target_token, (offset, stop),
        )
        if previous is not None:
            if previous != identity:
                raise ValueError(f"conflicting duplicate PE rewrite at {address:#x}")
            # The occurrence remains in the ledger; identical physical bytes
            # are written once even when several alternate-entry SSA functions
            # own the same decoded instruction.
            continue
        if any(not (stop <= left or offset >= right) for left, right in ranges):
            raise ValueError(f"overlapping PE rewrite at {address:#x}")
        # In-place output cannot silently change the authored instruction's
        # footprint.  Relayout and branch relaxation belong to the assembler.
        original, original_end = decoder.decode_one(
            memoryview(image.encoded[offset:]), 0, base_address=address,
        )
        original_size = original_end
        if len(occurrence.encoded) != original_size:
            raise ValueError(
                f"PE rewrite at {address:#x} changes size "
                f"{original_size}->{len(occurrence.encoded)}"
            )
        output[offset:stop] = occurrence.encoded
        ranges.append((offset, stop))
        seen_addresses[address] = identity
    reparsed, _statistics = parse_pe_image(
        bytes(output), maximum_file_size=len(output),
    )
    return InPlacePERecompilation(
        bytes(output), ledger, tuple(sorted(ranges)), reparsed,
    )


__all__ = [
    "InPlacePERecompilation", "PERecompilationLedger",
    "PERecompilationOccurrence", "build_pe_recompilation_ledger",
    "recompile_pe_in_place",
]
