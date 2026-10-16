"""Exact integer allocation for physical process-graph records."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import gcd
from typing import Callable, Iterable


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return ((int(value) + alignment - 1) // alignment) * alignment


def allocation_quantum(alignments: Iterable[int]) -> int:
    result = 1
    for alignment in alignments:
        alignment = int(alignment)
        if alignment <= 0:
            raise ValueError("alignments must be positive")
        result = result * alignment // gcd(result, alignment)
    return result


@dataclass(frozen=True, order=True)
class Allocation:
    start: int
    size: int
    label: str

    @property
    def end(self) -> int:
        return self.start + self.size


class IntegerRegionAllocator:
    """Allocate aligned byte intervals; labels own sets, not rigid bands."""

    def __init__(self, capacity: int, *, reserved_prefix: int = 0,
                 growth_quantum: int = 4096,
                 on_grow: Callable[[int], None] | None = None) -> None:
        capacity, reserved_prefix = int(capacity), int(reserved_prefix)
        if capacity < 0 or not 0 <= reserved_prefix <= capacity:
            raise ValueError("invalid arena capacity or reserved prefix")
        self.capacity = capacity
        self.growth_quantum = max(1, int(growth_quantum))
        self._on_grow = on_grow
        self._free = [(reserved_prefix, capacity)] if reserved_prefix < capacity else []
        self._by_label: dict[str, dict[int, Allocation]] = defaultdict(dict)

    @property
    def free_intervals(self) -> tuple[tuple[int, int], ...]:
        return tuple(self._free)

    def allocations(self, label: str | None = None) -> tuple[Allocation, ...]:
        if label is not None:
            return tuple(sorted(self._by_label.get(str(label), {}).values()))
        return tuple(sorted(record for records in self._by_label.values()
                            for record in records.values()))

    def offsets(self, label: str) -> tuple[int, ...]:
        return tuple(record.start for record in self.allocations(label))

    def allocate(self, label: str, size: int, *, alignment: int = 1) -> int:
        label, size, alignment = str(label), int(size), int(alignment)
        if size <= 0 or alignment <= 0:
            raise ValueError("allocation size and alignment must be positive")
        while True:
            for index, (left, right) in enumerate(self._free):
                start, end = align_up(left, alignment), 0
                end = start + size
                if end <= right:
                    replacement = []
                    if left < start:
                        replacement.append((left, start))
                    if end < right:
                        replacement.append((end, right))
                    self._free[index:index + 1] = replacement
                    self._by_label[label][start] = Allocation(start, size, label)
                    return start
            self._grow(size + alignment - 1)

    def release(self, start: int, *, label: str | None = None) -> Allocation:
        start = int(start)
        labels = ((str(label), self._by_label.get(str(label), {})),) if label is not None else tuple(self._by_label.items())
        for owner, records in labels:
            record = records.pop(start, None)
            if record is not None:
                if not records:
                    self._by_label.pop(owner, None)
                self._insert_free(record.start, record.end)
                return record
        raise KeyError(start)

    def _grow(self, minimum_extra: int) -> None:
        extra = align_up(max(1, int(minimum_extra)), self.growth_quantum)
        old_capacity, new_capacity = self.capacity, self.capacity + extra
        if self._on_grow is not None:
            self._on_grow(new_capacity)
        self.capacity = new_capacity
        self._insert_free(old_capacity, new_capacity)

    def _insert_free(self, left: int, right: int) -> None:
        merged: list[tuple[int, int]] = []
        for start, end in sorted((*self._free, (int(left), int(right)))):
            if start >= end:
                continue
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        self._free = merged
