from typing import List, Dict, TYPE_CHECKING
if TYPE_CHECKING:
    from ..simulator import Simulator


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
    
def dump_cells(mgr: "Simulator") -> Dict[str, object]:
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