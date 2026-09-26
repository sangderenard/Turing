"""Byte-complete structural inspection graph for a raised PE program.

The machine token multigraph explains semantic and control relationships. This
companion graph explains file-region mechanics: which container structures,
sections, functions, instructions, gaps, and classifications cover every byte.
No disassembly text is used as identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from types import MappingProxyType
from typing import Any, Mapping

import networkx as nx


class BinaryRegionKind(IntEnum):
    FILE = 0
    DOS_HEADER = 1
    DOS_STUB = 2
    PE_SIGNATURE = 3
    COFF_HEADER = 4
    OPTIONAL_HEADER = 5
    SECTION_TABLE = 6
    SECTION_RAW = 7
    RUNTIME_FUNCTION = 8
    INSTRUCTION = 9
    UNREACHED_FUNCTION = 10
    UNCLASSIFIED_EXECUTABLE = 11
    FILE_PARTITION = 12


class BinaryRegionRelation(IntEnum):
    CONTAINS = 0
    MAPS_RVA = 1
    COVERS = 2
    NEXT_FILE_REGION = 3


@dataclass(frozen=True, slots=True)
class BinaryRegion:
    id: str
    kind: BinaryRegionKind
    file_start: int
    file_end: int
    rva_start: int | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return self.file_end - self.file_start


@dataclass(frozen=True, slots=True)
class BinaryStructureStatistics:
    file_size: int
    region_count: int
    partition_count: int
    covered_bytes: int
    uncovered_bytes: int
    multiply_described_bytes: int


@dataclass(frozen=True, slots=True)
class BinaryStructureGraph:
    graph: nx.MultiDiGraph
    regions: tuple[BinaryRegion, ...]
    partitions: tuple[BinaryRegion, ...]
    statistics: BinaryStructureStatistics

    def regions_covering_file_offset(self, offset: int) -> tuple[BinaryRegion, ...]:
        point = int(offset)
        return tuple(
            region for region in self.regions
            if region.file_start <= point < region.file_end
        )


def build_pe_binary_structure_graph(program) -> BinaryStructureGraph:
    """Materialize nested regions and a complete non-overlapping file partition."""

    image = program.image
    data = image.encoded
    file_size = len(data)
    regions: list[BinaryRegion] = []
    parents: list[tuple[str, str]] = []

    def add(
        region_id: str,
        kind: BinaryRegionKind,
        start: int,
        end: int,
        *,
        rva: int | None = None,
        parent: str | None = None,
        **attributes: Any,
    ) -> BinaryRegion:
        if not 0 <= start <= end <= file_size:
            raise ValueError(
                f"binary region {region_id!r} [{start}, {end}) exceeds file size {file_size}"
            )
        region = BinaryRegion(
            region_id, kind, start, end, rva, MappingProxyType(dict(attributes)),
        )
        regions.append(region)
        if parent is not None:
            parents.append((parent, region_id))
        return region

    file_region = add("file", BinaryRegionKind.FILE, 0, file_size)
    pe_offset = int.from_bytes(data[0x3C:0x40], "little")
    coff_offset = pe_offset + 4
    optional_size = int.from_bytes(data[coff_offset + 16:coff_offset + 18], "little")
    optional_offset = coff_offset + 20
    section_table_offset = optional_offset + optional_size
    section_table_end = section_table_offset + 40 * len(image.sections)
    add("dos-header", BinaryRegionKind.DOS_HEADER, 0, 0x40, parent="file")
    if pe_offset > 0x40:
        add("dos-stub", BinaryRegionKind.DOS_STUB, 0x40, pe_offset, parent="file")
    add("pe-signature", BinaryRegionKind.PE_SIGNATURE, pe_offset, pe_offset + 4, parent="file")
    add("coff-header", BinaryRegionKind.COFF_HEADER, coff_offset, coff_offset + 20, parent="file")
    add("optional-header", BinaryRegionKind.OPTIONAL_HEADER, optional_offset, optional_offset + optional_size, parent="file")
    add("section-table", BinaryRegionKind.SECTION_TABLE, section_table_offset, section_table_end, parent="file")

    section_ids: dict[int, str] = {}
    for index, section in enumerate(image.sections):
        section_id = f"section:{index}"
        section_ids[index] = section_id
        add(
            section_id,
            BinaryRegionKind.SECTION_RAW,
            section.raw_offset,
            section.raw_end,
            rva=section.virtual_address,
            parent="file",
            name=section.name,
            executable=section.executable,
            virtual_size=section.virtual_size,
            characteristics=section.characteristics,
        )

    function_ids: dict[int, str] = {}
    for record in program.functions:
        runtime = record.runtime_function
        start = image.file_offset_for_rva(runtime.begin_rva)
        if start is None:
            continue
        end = start + runtime.end_rva - runtime.begin_rva
        function_id = f"function:{runtime.begin_rva:x}"
        function_ids[runtime.begin_rva] = function_id
        section = image.section_for_rva(runtime.begin_rva)
        section_index = None if section is None else image.sections.index(section)
        add(
            function_id,
            BinaryRegionKind.RUNTIME_FUNCTION,
            start,
            end,
            rva=runtime.begin_rva,
            parent=None if section_index is None else section_ids[section_index],
            unwind_info_rva=runtime.unwind_info_rva,
            reachable_complete=record.report.complete,
        )
        for instruction in record.report.instructions:
            instruction_rva = instruction.address - image.image_base
            instruction_offset = image.file_offset_for_rva(instruction_rva)
            if instruction_offset is None:
                continue
            add(
                f"instruction:{instruction.address:x}",
                BinaryRegionKind.INSTRUCTION,
                instruction_offset,
                instruction_offset + len(instruction.encoded),
                rva=instruction_rva,
                parent=function_id,
                instruction_token=int(instruction.token),
                semantic_token=int(instruction.semantic),
                encoded=instruction.encoded.hex(),
            )
        for range_index, (begin, end_offset) in enumerate(record.report.unreached_ranges):
            add(
                f"unreached:{runtime.begin_rva:x}:{range_index}",
                BinaryRegionKind.UNREACHED_FUNCTION,
                start + begin,
                start + end_offset,
                rva=runtime.begin_rva + begin,
                parent=function_id,
                encoded=data[start + begin:start + end_offset].hex(),
            )

    described = sorted((f.begin_rva, f.end_rva) for f in image.runtime_functions)
    for section_index, section in enumerate(image.sections):
        if not section.executable or not section.raw_size:
            continue
        cursor = section.virtual_address
        section_end = cursor + section.raw_size
        gap_index = 0
        for begin, end in described:
            if end <= cursor or begin >= section_end:
                continue
            if cursor < begin:
                raw = section.raw_offset + cursor - section.virtual_address
                add(
                    f"unclassified-exec:{section_index}:{gap_index}",
                    BinaryRegionKind.UNCLASSIFIED_EXECUTABLE,
                    raw,
                    raw + begin - cursor,
                    rva=cursor,
                    parent=section_ids[section_index],
                )
                gap_index += 1
            cursor = max(cursor, min(end, section_end))
        if cursor < section_end:
            raw = section.raw_offset + cursor - section.virtual_address
            add(
                f"unclassified-exec:{section_index}:{gap_index}",
                BinaryRegionKind.UNCLASSIFIED_EXECUTABLE,
                raw,
                raw + section_end - cursor,
                rva=cursor,
                parent=section_ids[section_index],
            )

    graph = nx.MultiDiGraph()
    for region in regions:
        graph.add_node(
            region.id,
            node_type="region",
            region_kind_token=int(region.kind),
            file_start=region.file_start,
            file_end=region.file_end,
            size=region.size,
            rva_start=region.rva_start,
            **dict(region.attributes),
        )
    for parent, child in parents:
        graph.add_edge(
            parent, child,
            relation_token=int(BinaryRegionRelation.CONTAINS),
            relation="contains",
        )

    boundaries = sorted({0, file_size, *(r.file_start for r in regions), *(r.file_end for r in regions)})
    starts: dict[int, list[BinaryRegion]] = {}
    ends: dict[int, list[BinaryRegion]] = {}
    for region in regions:
        starts.setdefault(region.file_start, []).append(region)
        ends.setdefault(region.file_end, []).append(region)
    active: dict[str, BinaryRegion] = {}
    partitions: list[BinaryRegion] = []
    previous_partition: str | None = None
    multiply_described = 0
    for position, next_position in zip(boundaries, boundaries[1:]):
        for region in ends.get(position, ()):
            active.pop(region.id, None)
        for region in starts.get(position, ()):
            active[region.id] = region
        if next_position <= position:
            continue
        coverage = tuple(sorted(active))
        partition = BinaryRegion(
            f"partition:{position:x}:{next_position:x}",
            BinaryRegionKind.FILE_PARTITION,
            position,
            next_position,
            attributes=MappingProxyType({"coverage": coverage}),
        )
        partitions.append(partition)
        graph.add_node(
            partition.id,
            node_type="partition",
            region_kind_token=int(partition.kind),
            file_start=position,
            file_end=next_position,
            size=partition.size,
            coverage=coverage,
        )
        if len(coverage) > 2:  # file plus at least two structural descriptions
            multiply_described += partition.size
        for region_id in coverage:
            graph.add_edge(
                region_id, partition.id,
                relation_token=int(BinaryRegionRelation.COVERS),
                relation="covers",
            )
        if previous_partition is not None:
            graph.add_edge(
                previous_partition, partition.id,
                relation_token=int(BinaryRegionRelation.NEXT_FILE_REGION),
                relation="next-file-region",
            )
        previous_partition = partition.id

    covered_bytes = sum(partition.size for partition in partitions)
    return BinaryStructureGraph(
        graph=graph,
        regions=tuple(regions),
        partitions=tuple(partitions),
        statistics=BinaryStructureStatistics(
            file_size=file_size,
            region_count=len(regions),
            partition_count=len(partitions),
            covered_bytes=covered_bytes,
            uncovered_bytes=file_size - covered_bytes,
            multiply_described_bytes=multiply_described,
        ),
    )


__all__ = [
    "BinaryRegion",
    "BinaryRegionKind",
    "BinaryRegionRelation",
    "BinaryStructureGraph",
    "BinaryStructureStatistics",
    "build_pe_binary_structure_graph",
]
