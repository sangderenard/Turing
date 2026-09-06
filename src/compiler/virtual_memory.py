"""Immutable Windows-style virtual-memory allocation metadata."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


MEM_COMMIT = 0x1000
MEM_RESERVE = 0x2000
MEM_FREE = 0x10000
MEM_PRIVATE = 0x20000
MEM_IMAGE = 0x1000000
PAGE_READWRITE = 0x04
PAGE_EXECUTE_READWRITE = 0x40


@dataclass(frozen=True, slots=True)
class VirtualMemoryRegion:
    base: int
    size: int
    allocation_base: int
    allocation_protect: int
    state: int
    protect: int
    kind: int = MEM_PRIVATE
    managed: bool = False

    @property
    def end(self) -> int:
        return self.base + self.size

    @property
    def executable(self) -> bool:
        return bool(self.protect & 0xF0)


@dataclass(frozen=True, slots=True)
class VirtualMemoryEffect:
    operation: str
    base: int
    size: int
    protection: int = 0

    def __post_init__(self) -> None:
        if self.operation not in {"allocate", "release"}:
            raise ValueError(f"unsupported virtual-memory effect {self.operation!r}")
        if self.base < 0 or self.size <= 0:
            raise ValueError("virtual-memory effects require a positive range")


@dataclass(frozen=True, slots=True)
class VirtualMemoryState:
    regions: Mapping[int, VirtualMemoryRegion]
    allocation_cursor: int = 0x0000010000000000
    generation: int = 0
    page_size: int = 4096

    @classmethod
    def create(cls) -> "VirtualMemoryState":
        return cls(MappingProxyType({}))

    @classmethod
    def from_mapped_pages(
        cls,
        pages,
        *,
        executable_pages=(),
        image_pages=(),
        page_size: int = 4096,
    ) -> "VirtualMemoryState":
        executable = frozenset(int(item) for item in executable_pages)
        images = frozenset(int(item) for item in image_pages)
        descriptors = []
        for page in sorted(int(item) for item in pages):
            descriptors.append((
                page,
                PAGE_EXECUTE_READWRITE if page in executable else PAGE_READWRITE,
                MEM_IMAGE if page in images else MEM_PRIVATE,
            ))
        regions: dict[int, VirtualMemoryRegion] = {}
        index = 0
        while index < len(descriptors):
            page, protection, kind = descriptors[index]
            end = index + 1
            while (
                end < len(descriptors)
                and descriptors[end][0] == descriptors[end - 1][0] + 1
                and descriptors[end][1:] == (protection, kind)
            ):
                end += 1
            base = page * page_size
            size = (end - index) * page_size
            regions[base] = VirtualMemoryRegion(
                base, size, base, protection, MEM_COMMIT,
                protection, kind, False,
            )
            index = end
        return cls(MappingProxyType(regions), page_size=page_size)

    def region_at(self, address: int) -> VirtualMemoryRegion | None:
        target = int(address)
        for region in self.regions.values():
            if region.base <= target < region.end:
                return region
        return None

    def is_executable(self, address: int) -> bool:
        region = self.region_at(address)
        return bool(region and region.state == MEM_COMMIT and region.executable)

    def choose_base(self, size: int, requested: int = 0) -> int | None:
        aligned_size = (int(size) + self.page_size - 1) & -self.page_size
        if requested:
            base = int(requested) & -self.page_size
            return base if self.range_free(base, aligned_size) else None
        base = (self.allocation_cursor + 0xFFFF) & -0x10000
        while base + aligned_size < 0x00007FFF00000000:
            if self.range_free(base, aligned_size):
                return base
            conflicts = [
                region.end for region in self.regions.values()
                if not (region.end <= base or region.base >= base + aligned_size)
            ]
            base = ((max(conflicts, default=base + 0x10000) + 0xFFFF) & -0x10000)
        return None

    def range_free(self, base: int, size: int) -> bool:
        end = int(base) + int(size)
        return all(region.end <= base or region.base >= end for region in self.regions.values())

    def apply(self, effect: VirtualMemoryEffect) -> "VirtualMemoryState":
        regions = dict(self.regions)
        if effect.operation == "allocate":
            if effect.base % self.page_size or effect.size % self.page_size:
                raise ValueError("virtual allocation must be page aligned")
            if not self.range_free(effect.base, effect.size):
                raise ValueError("virtual allocation overlaps an existing region")
            regions[effect.base] = VirtualMemoryRegion(
                effect.base, effect.size, effect.base, effect.protection,
                MEM_COMMIT, effect.protection, MEM_PRIVATE, True,
            )
            cursor = max(self.allocation_cursor, effect.base + effect.size)
        else:
            region = regions.get(effect.base)
            if (
                region is None or region.allocation_base != effect.base
                or region.size != effect.size or not region.managed
            ):
                raise ValueError("virtual release requires one complete private allocation")
            del regions[effect.base]
            cursor = self.allocation_cursor
        return VirtualMemoryState(
            MappingProxyType(regions), cursor, self.generation + 1, self.page_size,
        )


__all__ = [
    "MEM_COMMIT", "MEM_FREE", "MEM_IMAGE", "MEM_PRIVATE", "MEM_RESERVE",
    "PAGE_EXECUTE_READWRITE", "PAGE_READWRITE", "VirtualMemoryEffect",
    "VirtualMemoryRegion", "VirtualMemoryState",
]
