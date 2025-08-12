from __future__ import annotations
from math import gcd
from typing import Sequence, Iterable

class BitBufferAdapter:
    """Lightweight bridge to BitBitBuffer.expand.

    Accepts per-cell volume changes and translates them to stride/LCM aligned
    expand events. If ``bitbuffer`` is ``None`` the calls become no-ops so the
    engine can operate without a placement backend.
    """
    def __init__(self, bitbuffer=None):
        self.bitbuffer = bitbuffer
        self.mask_size = getattr(bitbuffer, "mask_size", 0) if bitbuffer else 0

    def intceil(self, x: int, lcm: int) -> int:
        if lcm <= 0:
            return int(x)
        return ((int(x) + lcm - 1) // lcm) * lcm

    def _lcm(self, cells: Sequence) -> int:
        L = 1
        for c in cells:
            s = max(1, getattr(c, "stride", 1))
            L = L * s // gcd(L, s)
        return L

    def expand(self, dV_total: Iterable[int], cells: Sequence, proposals):
        """Expand bitbuffer to accommodate the requested volume changes.

        ``dV_total`` is an iterable of per-cell bit counts. Each positive value
        triggers an insertion to the right edge of that cell. Insert sizes are
        rounded up to the system LCM to preserve alignment guarantees.
        """
        if self.bitbuffer is None:
            return proposals
        lcm = self._lcm(cells)
        events = []
        for cell, dV in zip(cells, dV_total):
            if dV <= 0:
                continue
            size = self.intceil(dV, lcm)
            events.append((getattr(cell, "label", None), cell.right - 1, size))
        if not events:
            return proposals
        return self.bitbuffer.expand(events, cells, proposals)
