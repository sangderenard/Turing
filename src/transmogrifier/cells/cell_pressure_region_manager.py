"""Utilities to extract cell occupancy information from ``cell_pressure``.

The classic :class:`LinearCells` helper exposes ``dump_cells`` and
``quanta_map`` for visualisations.  The pressure-based simulator stores
occupancy in a :class:`BitBitBuffer` instead of per-cell ``obj_map`` buffers.
This module replicates the LinearCells API so existing animation code can
operate on pressure–managed cells without any further changes.
"""
from __future__ import annotations

from typing import Dict, Iterable, List

from .cell_consts import Cell
from .bitbitbuffer import BitBitBuffer


class CellPressureRegionManager:
    """Expose ``dump_cells`` and ``quanta_map`` for the pressure simulator.

    Parameters
    ----------
    bitbuffer:
        The :class:`BitBitBuffer` holding cell occupancy bits.
    cells:
        Iterable of :class:`Cell` objects describing the regions managed by the
        simulator.  Each cell's ``left``/``right`` and ``stride`` attributes are
        interpreted as bit offsets.
    """

    def __init__(self, bitbuffer: BitBitBuffer, cells: Iterable[Cell]):
        self.bitbuffer = bitbuffer
        self.cells = list(cells)

    # ------------------------------------------------------------------
    def quanta_map(self, *, coalesce_free: bool = True) -> List[Dict]:
        """Build a detailed occupancy map for every cell.

        The return structure mirrors ``LinearCells.quanta_map``.  ``addr`` and
        ``size`` entries are expressed in bits to match the ``BitBitBuffer``
        layout.
        """

        result: List[Dict[str, object]] = []

        for c in self.cells:
            cell_info = {
                "label": getattr(c, "label", None),
                "stride": c.stride,
                "used": [],  # type: ignore[list-item]
                "free": [],  # type: ignore[list-item]
            }

            q_tot = (c.right - c.left) // c.stride if c.stride else 0
            if q_tot == 0:
                result.append(cell_info)
                continue

            run_start = None
            run_used = None

            def flush_run(idx: int) -> None:
                nonlocal run_start, run_used
                if run_start is None:
                    return
                addr = c.left + run_start * c.stride
                size = (idx - run_start) * c.stride
                key = "used" if run_used else "free"
                cell_info[key].append((addr, size))

            for q in range(q_tot):
                start = c.left + q * c.stride
                end = start + c.stride
                this_used = any(int(self.bitbuffer[b]) for b in range(start, end))
                if run_start is None:
                    run_start, run_used = q, this_used
                elif this_used != run_used or not coalesce_free:
                    flush_run(q)
                    run_start, run_used = q, this_used

            flush_run(q_tot)
            result.append(cell_info)

        return result

    # ------------------------------------------------------------------
    @staticmethod
    def dump_cells(mgr: "CellPressureRegionManager") -> Dict[str, object]:
        """Return a full snapshot compatible with ``LinearCells.dump_cells``."""
        cells = mgr.quanta_map()
        free_spaces = [
            (c["label"], addr, size)  # type: ignore[index]
            for c in cells
            for addr, size in c["free"]
        ]
        occupied = [
            (c["label"], addr, size)  # type: ignore[index]
            for c in cells
            for addr, size in c["used"]
        ]
        return {
            "cells": cells,
            "free_spaces": free_spaces,
            "occupied_spaces": occupied,
        }
