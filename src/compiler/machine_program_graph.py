"""Direct PE/x86 ingestion into the precompiler token multigraph.

This is the owning binary-program representation. Repository SSA is a later
projection and is deliberately absent here. Unknown x86 bytes remain explicit
failure components; the graph never advances past an unproven instruction
boundary inside a function.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable

from .binary_ingestion import PEImage, PERuntimeFunction, parse_pe_image
from .evolution_metagraph import (
    EvolutionComponentRef,
    EvolutionMetaGraph,
    TokenPathAtlas,
)
from .machine_reference_vocabulary import (
    EffectiveAddressOperand,
    HighByteRegisterOperand,
    ImmediateOperand,
    RegisterOperand,
    RelativeAddressOperand,
    MachineSemanticToken,
    VectorRegisterOperand,
    VocabularyDecodeError,
    VocabularyFailure,
    X86ReferenceDecoder,
)


class MachineGraphNamespace(IntEnum):
    STRUCTURE = 0
    ISA = 1
    SEMANTIC = 2
    OPERAND = 3
    RELATION = 4
    FAILURE = 5


class MachineStructureToken(IntEnum):
    PROGRAM = 0
    SECTION = 1
    FUNCTION = 2
    INSTRUCTION = 3
    OPERAND = 4
    UNCLASSIFIED_EXECUTABLE_RANGE = 5
    UNREACHED_FUNCTION_RANGE = 6


class MachineOperandToken(IntEnum):
    REGISTER = 0
    EFFECTIVE_ADDRESS = 1
    IMMEDIATE = 2
    RELATIVE_ADDRESS = 3
    HIGH_BYTE_REGISTER = 4
    VECTOR_REGISTER = 5


class MachineRelationToken(IntEnum):
    CONTAINS = 0
    SEQUENCE = 1
    OPERAND = 2
    CONTROL_TARGET = 3
    INTERNAL_CALL = 4


@dataclass(frozen=True, slots=True)
class MachineFunctionGraphRecord:
    runtime_function: PERuntimeFunction
    report: "MachineFunctionDecodeReport"
    component: EvolutionComponentRef


@dataclass(frozen=True, slots=True)
class MachineFunctionDecodeReport:
    instructions: tuple[Any, ...]
    failures: tuple[VocabularyFailure, ...]
    accepted_size: int
    decoded_bytes: int
    unreached_ranges: tuple[tuple[int, int], ...]

    @property
    def complete(self) -> bool:
        """Every instruction reachable from the function entry was decoded."""

        return not self.failures


@dataclass(frozen=True, slots=True)
class MachineProgramGraphStatistics:
    file_size: int
    executable_section_count: int
    executable_raw_bytes: int
    runtime_function_count: int
    runtime_described_code_bytes: int
    unclassified_executable_bytes: int
    unreached_runtime_bytes: int
    proven_function_count: int
    proven_instruction_count: int
    proven_code_bytes: int
    failed_function_count: int


@dataclass(frozen=True, slots=True)
class MachineProgramGraph:
    image: PEImage
    metagraph: EvolutionMetaGraph
    atlas: TokenPathAtlas
    functions: tuple[MachineFunctionGraphRecord, ...]
    statistics: MachineProgramGraphStatistics

    @property
    def graph(self):
        return self.metagraph.to_token_multidigraph()

    def structure_graph(self):
        """Build the byte-complete container/region inspection graph."""

        from .binary_structure_graph import build_pe_binary_structure_graph

        return build_pe_binary_structure_graph(self)

    def orchestrate(self, *, mode, executor=None, maximum_steps: int = 1_000_000):
        """Enter inspect, decode, or explicitly handled emulation mode."""

        from .machine_execution import orchestrate_machine_program

        return orchestrate_machine_program(
            self,
            mode=mode,
            executor=executor,
            maximum_steps=maximum_steps,
        )

    @property
    def complete(self) -> bool:
        return (
            self.statistics.failed_function_count == 0
            and self.statistics.unclassified_executable_bytes == 0
            and self.statistics.unreached_runtime_bytes == 0
            and self.statistics.proven_code_bytes
            == self.statistics.runtime_described_code_bytes
        )


def decode_reachable_region(
    decoder: X86ReferenceDecoder,
    code: bytes,
    *,
    base_address: int,
    entry_offsets: tuple[int, ...] = (0,),
    instruction_sink: Callable[[Any], None] | None = None,
) -> MachineFunctionDecodeReport:
    """Decode reachable instructions breadth-first and optionally stream them.

    The queue contains block-entry offsets.  Instructions within a block are
    decoded in address order; newly discovered control targets enter the FIFO
    frontier.  ``instruction_sink`` is called exactly once for each accepted
    instruction, which lets a normalization writer progress while discovery is
    still underway without weakening fail-closed decoding.
    """

    region = memoryview(code)
    decoded: dict[int, Any] = {}
    failures: list[VocabularyFailure] = []
    worklist = deque(entry_offsets)
    queued = set(entry_offsets)
    terminal_semantics = {
        MachineSemanticToken.RETURN,
        MachineSemanticToken.INDIRECT_JUMP,
        MachineSemanticToken.BREAKPOINT_TRAP,
        MachineSemanticToken.SOFTWARE_INTERRUPT,
    }

    def enqueue_target(target_address: int) -> None:
        target = target_address - base_address
        if 0 <= target < len(region) and target not in queued:
            queued.add(target)
            worklist.append(target)

    while worklist:
        cursor = worklist.popleft()
        while cursor < len(region):
            if cursor in decoded:
                break
            overlap = next((
                start for start, instruction in decoded.items()
                if start < cursor < start + len(instruction.encoded)
            ), None)
            if overlap is not None:
                failures.append(VocabularyFailure(
                    category="overlapping-control-target",
                    region_offset=cursor,
                    address=base_address + cursor,
                    encoded_preview=bytes(region[cursor:cursor + 8]),
                    reason=(
                        f"control target enters instruction at "
                        f"{base_address + overlap:#x}"
                    ),
                ))
                break
            try:
                instruction, next_cursor = decoder.decode_one(
                    region, cursor, base_address=base_address,
                )
            except VocabularyDecodeError as error:
                failures.append(VocabularyFailure(
                    category="missing-vocabulary",
                    region_offset=cursor,
                    address=base_address + cursor,
                    encoded_preview=bytes(region[cursor:cursor + 8]),
                    reason=str(error),
                ))
                break
            decoded[cursor] = instruction
            if instruction_sink is not None:
                instruction_sink(instruction)
            relative_targets = tuple(
                operand.target_address
                for operand in instruction.operands
                if isinstance(operand, RelativeAddressOperand)
            )
            if instruction.semantic is MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP:
                for target in relative_targets:
                    enqueue_target(target)
                cursor = next_cursor
                continue
            if instruction.semantic is MachineSemanticToken.DIRECT_RELATIVE_JUMP:
                for target in relative_targets:
                    enqueue_target(target)
                break
            if instruction.semantic is MachineSemanticToken.DIRECT_RELATIVE_CALL:
                for target in relative_targets:
                    enqueue_target(target)
            if instruction.semantic in terminal_semantics:
                break
            cursor = next_cursor

    ordered = tuple(decoded[offset] for offset in sorted(decoded))
    intervals = sorted(
        (offset, offset + len(instruction.encoded))
        for offset, instruction in decoded.items()
    )
    unreached: list[tuple[int, int]] = []
    cursor = 0
    for begin, end in intervals:
        if cursor < begin:
            unreached.append((cursor, begin))
        cursor = max(cursor, end)
    if cursor < len(region):
        unreached.append((cursor, len(region)))
    return MachineFunctionDecodeReport(
        instructions=ordered,
        failures=tuple(failures),
        accepted_size=len(region),
        decoded_bytes=sum(len(instruction.encoded) for instruction in ordered),
        unreached_ranges=tuple(unreached),
    )


def _unclassified_executable_ranges(image: PEImage) -> tuple[tuple[int, int, int], ...]:
    """Return section-indexed RVA gaps not described by AMD64 runtime functions."""

    gaps: list[tuple[int, int, int]] = []
    for section_index, section in enumerate(image.sections):
        if not section.executable or section.raw_size == 0:
            continue
        section_begin = section.virtual_address
        section_end = section_begin + section.raw_size
        covered = sorted(
            (
                max(section_begin, function.begin_rva),
                min(section_end, function.end_rva),
            )
            for function in image.runtime_functions
            if function.end_rva > section_begin and function.begin_rva < section_end
        )
        cursor = section_begin
        for begin, end in covered:
            if cursor < begin:
                gaps.append((section_index, cursor, begin))
            cursor = max(cursor, end)
        if cursor < section_end:
            gaps.append((section_index, cursor, section_end))
    return tuple(gaps)


def _operand_attributes(operand: Any) -> tuple[MachineOperandToken, dict[str, Any]]:
    if isinstance(operand, RegisterOperand):
        return MachineOperandToken.REGISTER, {
            "register_token": int(operand.register),
            "width": int(operand.width),
        }
    if isinstance(operand, HighByteRegisterOperand):
        return MachineOperandToken.HIGH_BYTE_REGISTER, {
            "register_token": int(operand.register),
            "width": int(operand.width),
        }
    if isinstance(operand, VectorRegisterOperand):
        return MachineOperandToken.VECTOR_REGISTER, {
            "register_token": int(operand.register),
            "width": int(operand.width),
        }
    if isinstance(operand, EffectiveAddressOperand):
        return MachineOperandToken.EFFECTIVE_ADDRESS, {
            "base_register_token": None if operand.base is None else int(operand.base),
            "index_register_token": None if operand.index is None else int(operand.index),
            "scale": int(operand.scale),
            "displacement": int(operand.displacement),
            "address_width": int(operand.address_width),
            "rip_relative": bool(operand.rip_relative),
        }
    if isinstance(operand, ImmediateOperand):
        return MachineOperandToken.IMMEDIATE, {
            "value": int(operand.value),
            "width": int(operand.width),
            "signed": bool(operand.signed),
        }
    if isinstance(operand, RelativeAddressOperand):
        return MachineOperandToken.RELATIVE_ADDRESS, {
            "displacement": int(operand.displacement),
            "width": int(operand.width),
            "target_address": int(operand.target_address),
        }
    raise TypeError(f"unsupported machine operand {type(operand).__name__}")


def raise_pe_to_token_multigraph(
    binary_region,
    *,
    maximum_file_size: int,
    decoder: X86ReferenceDecoder | None = None,
    metagraph: EvolutionMetaGraph | None = None,
    atlas: TokenPathAtlas | None = None,
) -> MachineProgramGraph:
    """Raise every PE-described AMD64 function into tokenized graph records."""

    image, pe_statistics = parse_pe_image(
        binary_region,
        maximum_file_size=maximum_file_size,
    )
    active_decoder = decoder or X86ReferenceDecoder()
    evolution = metagraph or EvolutionMetaGraph()
    tokens = atlas or TokenPathAtlas()
    program_graph = evolution.open_graph("machine-program", "PE executable")
    program = evolution.component(
        program_graph,
        "program",
        label="PE executable",
        kind="program",
        attributes={
            "image_base": int(image.image_base),
            "entrypoint_rva": int(image.entrypoint_rva),
            "machine": int(image.machine),
        },
        token_id=tokens.consume((
            int(MachineGraphNamespace.STRUCTURE),
            int(MachineStructureToken.PROGRAM),
        )),
    )
    contains_token = tokens.consume((
        int(MachineGraphNamespace.RELATION),
        int(MachineRelationToken.CONTAINS),
    ))
    section_components: list[EvolutionComponentRef] = []
    for index, section in enumerate(image.sections):
        component = evolution.component(
            program_graph,
            f"section:{index}",
            label=section.name,
            kind="section",
            attributes={
                "section_index": index,
                "virtual_address": int(section.virtual_address),
                "virtual_size": int(section.virtual_size),
                "raw_offset": int(section.raw_offset),
                "raw_size": int(section.raw_size),
                "characteristics": int(section.characteristics),
                "executable": bool(section.executable),
            },
            token_id=tokens.consume((
                int(MachineGraphNamespace.STRUCTURE),
                int(MachineStructureToken.SECTION),
            )),
        )
        section_components.append(component)
        evolution.relationship(
            program_graph, program, component,
            role="contains", role_token_id=contains_token,
        )

    unclassified_ranges = _unclassified_executable_ranges(image)
    for section_index, begin_rva, end_rva in unclassified_ranges:
        gap = evolution.component(
            program_graph,
            f"unclassified-executable:{begin_rva:x}:{end_rva:x}",
            label="unclassified executable bytes",
            kind="unclassified-executable-range",
            attributes={
                "section_index": section_index,
                "begin_rva": begin_rva,
                "end_rva": end_rva,
                "size": end_rva - begin_rva,
            },
            token_id=tokens.consume((
                int(MachineGraphNamespace.STRUCTURE),
                int(MachineStructureToken.UNCLASSIFIED_EXECUTABLE_RANGE),
            )),
        )
        evolution.relationship(
            program_graph, section_components[section_index], gap,
            role="contains", role_token_id=contains_token,
        )

    records: list[MachineFunctionGraphRecord] = []
    function_components: dict[int, EvolutionComponentRef] = {}
    instruction_components: dict[int, EvolutionComponentRef] = {}
    proven_bytes = 0
    proven_instructions = 0
    proven_functions = 0
    entry_offsets: dict[int, set[int]] = {
        function.begin_rva: {0} for function in image.runtime_functions
    }
    reports: dict[int, MachineFunctionDecodeReport] = {}
    while True:
        for runtime_function in image.runtime_functions:
            file_offset = image.file_offset_for_rva(runtime_function.begin_rva)
            if file_offset is None:
                continue
            size = runtime_function.end_rva - runtime_function.begin_rva
            reports[runtime_function.begin_rva] = decode_reachable_region(
                active_decoder,
                image.encoded[file_offset:file_offset + size],
                base_address=image.image_base + runtime_function.begin_rva,
                entry_offsets=tuple(sorted(entry_offsets[runtime_function.begin_rva])),
            )
        added_seed = False
        for report in reports.values():
            for instruction in report.instructions:
                for operand in instruction.operands:
                    if not isinstance(operand, RelativeAddressOperand):
                        continue
                    target_rva = operand.target_address - image.image_base
                    owner = image.runtime_function_for_rva(target_rva)
                    if owner is None:
                        continue
                    target_offset = target_rva - owner.begin_rva
                    owner_seeds = entry_offsets[owner.begin_rva]
                    if target_offset not in owner_seeds:
                        owner_seeds.add(target_offset)
                        added_seed = True
        if not added_seed:
            break
    for index, runtime_function in enumerate(image.runtime_functions):
        file_offset = image.file_offset_for_rva(runtime_function.begin_rva)
        if file_offset is None:
            continue
        size = runtime_function.end_rva - runtime_function.begin_rva
        code = image.encoded[file_offset:file_offset + size]
        report = reports[runtime_function.begin_rva]
        component = evolution.component(
            program_graph,
            f"function:{runtime_function.begin_rva:x}",
            label=f"sub_{runtime_function.begin_rva:08x}",
            kind="function",
            attributes={
                "function_index": index,
                "begin_rva": int(runtime_function.begin_rva),
                "end_rva": int(runtime_function.end_rva),
                "unwind_info_rva": int(runtime_function.unwind_info_rva),
                "file_offset": int(file_offset),
                "complete": bool(report.complete),
            },
            token_id=tokens.consume((
                int(MachineGraphNamespace.STRUCTURE),
                int(MachineStructureToken.FUNCTION),
            )),
        )
        function_components[runtime_function.begin_rva] = component
        evolution.relationship(
            program_graph, program, component,
            role="contains", role_token_id=contains_token,
        )
        section = image.section_for_rva(runtime_function.begin_rva)
        if section is not None:
            section_index = image.sections.index(section)
            evolution.relationship(
                program_graph, section_components[section_index], component,
                role="contains", role_token_id=contains_token,
            )
        function_graph = evolution.open_graph(
            "machine-function", f"sub_{runtime_function.begin_rva:08x}",
        )
        previous: EvolutionComponentRef | None = None
        local_instructions: dict[int, EvolutionComponentRef] = {}
        for instruction_index, instruction in enumerate(report.instructions):
            semantic_token = tokens.consume((
                int(MachineGraphNamespace.SEMANTIC), int(instruction.semantic),
            ))
            instruction_component = evolution.component(
                function_graph,
                f"{instruction.address:x}",
                label=instruction.token.name,
                kind="instruction",
                attributes={
                    "instruction_index": instruction_index,
                    "address": int(instruction.address),
                    "encoded": instruction.encoded.hex(),
                    "instruction_token": int(instruction.token),
                    "semantic_token_id": semantic_token,
                    "rex": instruction.rex,
                    "legacy_prefixes": tuple(instruction.legacy_prefixes),
                },
                consumes=(() if previous is None else (previous,)),
                token_id=tokens.consume((
                    int(MachineGraphNamespace.ISA), int(instruction.token),
                )),
            )
            instruction_components[instruction.address] = instruction_component
            local_instructions[instruction.address] = instruction_component
            if previous is not None:
                evolution.relationship(
                    function_graph, previous, instruction_component,
                    role="sequence",
                    role_token_id=tokens.consume((
                        int(MachineGraphNamespace.RELATION),
                        int(MachineRelationToken.SEQUENCE),
                    )),
                )
            previous = instruction_component
            for position, operand in enumerate(instruction.operands):
                operand_kind, attributes = _operand_attributes(operand)
                operand_component = evolution.component(
                    function_graph,
                    f"{instruction.address:x}:operand:{position}",
                    label=operand_kind.name,
                    kind="operand",
                    attributes={"position": position, **attributes},
                    token_id=tokens.consume((
                        int(MachineGraphNamespace.OPERAND), int(operand_kind),
                    )),
                )
                evolution.relationship(
                    function_graph, operand_component, instruction_component,
                    role=f"operand:{position}",
                    role_token_id=tokens.consume((
                        int(MachineGraphNamespace.RELATION),
                        int(MachineRelationToken.OPERAND), position,
                    )),
                )
        control_token = tokens.consume((
            int(MachineGraphNamespace.RELATION),
            int(MachineRelationToken.CONTROL_TARGET),
        ))
        for instruction in report.instructions:
            source = local_instructions[instruction.address]
            for operand in instruction.operands:
                if not isinstance(operand, RelativeAddressOperand):
                    continue
                target = local_instructions.get(operand.target_address)
                if target is not None:
                    evolution.relationship(
                        function_graph, source, target,
                        role="control-target", role_token_id=control_token,
                    )
        for failure_index, failure in enumerate(report.failures):
            failure_component = evolution.component(
                function_graph,
                f"failure:{failure.region_offset:x}:{failure_index}",
                label=failure.category,
                kind="vocabulary-failure",
                attributes={
                    "region_offset": int(failure.region_offset),
                    "address": int(failure.address),
                    "encoded_preview": failure.encoded_preview.hex(),
                    "reason": failure.reason,
                },
                consumes=(() if previous is None else (previous,)),
                token_id=tokens.consume((
                    int(MachineGraphNamespace.FAILURE),
                    *map(int, failure.encoded_preview[:4]),
                )),
            )
            if previous is not None:
                evolution.relationship(
                    function_graph, previous, failure_component,
                    role="sequence",
                    role_token_id=tokens.consume((
                        int(MachineGraphNamespace.RELATION),
                        int(MachineRelationToken.SEQUENCE),
                    )),
                )
        for range_index, (begin, end) in enumerate(report.unreached_ranges):
            unreached_component = evolution.component(
                function_graph,
                f"unreached:{begin:x}:{end:x}:{range_index}",
                label="unreached function bytes",
                kind="unreached-function-range",
                attributes={
                    "region_offset": begin,
                    "address": image.image_base + runtime_function.begin_rva + begin,
                    "size": end - begin,
                    "encoded": code[begin:end].hex(),
                },
                token_id=tokens.consume((
                    int(MachineGraphNamespace.STRUCTURE),
                    int(MachineStructureToken.UNREACHED_FUNCTION_RANGE),
                )),
            )
            evolution.relationship(
                function_graph, component, unreached_component,
                role="contains", role_token_id=contains_token,
            )
        evolution.bind_artifact(report, function_graph)
        evolution.close_graph(function_graph)
        records.append(MachineFunctionGraphRecord(runtime_function, report, component))
        proven_bytes += report.decoded_bytes
        proven_instructions += len(report.instructions)
        if report.complete:
            proven_functions += 1

    internal_call_token = tokens.consume((
        int(MachineGraphNamespace.RELATION),
        int(MachineRelationToken.INTERNAL_CALL),
    ))
    for record in records:
        for instruction in record.report.instructions:
            for operand in instruction.operands:
                if not isinstance(operand, RelativeAddressOperand):
                    continue
                target_rva = operand.target_address - image.image_base
                target = function_components.get(target_rva)
                source = instruction_components.get(instruction.address)
                if source is not None and target is not None:
                    evolution.relationship(
                        program_graph, source, target,
                        role="internal-call", role_token_id=internal_call_token,
                    )
    evolution.close_graph(program_graph)
    failed_functions = len(records) - proven_functions
    runtime_described_bytes = sum(
        record.runtime_function.end_rva - record.runtime_function.begin_rva
        for record in records
    )
    unclassified_bytes = sum(end - begin for _, begin, end in unclassified_ranges)
    unreached_runtime_bytes = sum(
        end - begin
        for record in records
        for begin, end in record.report.unreached_ranges
    )
    return MachineProgramGraph(
        image=image,
        metagraph=evolution,
        atlas=tokens,
        functions=tuple(records),
        statistics=MachineProgramGraphStatistics(
            file_size=pe_statistics.file_size,
            executable_section_count=pe_statistics.executable_section_count,
            executable_raw_bytes=pe_statistics.executable_raw_bytes,
            runtime_function_count=len(records),
            runtime_described_code_bytes=runtime_described_bytes,
            unclassified_executable_bytes=unclassified_bytes,
            unreached_runtime_bytes=unreached_runtime_bytes,
            proven_function_count=proven_functions,
            proven_instruction_count=proven_instructions,
            proven_code_bytes=proven_bytes,
            failed_function_count=failed_functions,
        ),
    )


__all__ = [
    "MachineFunctionGraphRecord",
    "MachineFunctionDecodeReport",
    "MachineGraphNamespace",
    "MachineOperandToken",
    "MachineProgramGraph",
    "MachineProgramGraphStatistics",
    "MachineRelationToken",
    "MachineStructureToken",
    "decode_reachable_region",
    "raise_pe_to_token_multigraph",
]
