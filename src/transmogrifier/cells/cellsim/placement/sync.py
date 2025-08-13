# cellsim/placement/sync.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Dict, Tuple, Optional
from math import gcd

# Minimal duck-typed requirements from your codebase:
# - Each cell has: label, left, right, stride, V, organelles (list)
# - BitBitBuffer-like backend: .expand(events, cells, proposals) and optional .shrink(...)
# - Proposal type is cellsim.data.proposals.CellProposal (but we can operate on plain dicts too)

def _lcm(a: int, b: int) -> int:
    return abs(a*b) // gcd(a, b) if a and b else max(a, b)

def lcm_all(strides: Iterable[int]) -> int:
    L = 1
    for s in strides:
        L = _lcm(L, max(1, int(s)))
    return max(1, L)

def intceil(x: float, quantum: int = 1) -> int:
    q = max(1, int(quantum))
    n = int(x // q)
    return (n*q) if (x == n*q) else ((n+1)*q)

def intfloor(x: float, quantum: int = 1) -> int:
    q = max(1, int(quantum))
    n = int(x // q)
    return n*q

@dataclass
class BackingSyncCfg:
    # Convert physical volume (V) to backing "bits" (or slots). Keep =1.0 if 1 V == 1 bit.
    V_to_bits: float = 1.0

    # Reserve free (cytosolic) capacity relative to data organelles:
    #   target_free_bits >= free_per_data_unit * data_units + free_min_bits
    free_per_data_unit: int = 0
    free_min_bits: int = 0

    # Incompressible organelles: lock their movement to stride grid (toggleable).
    stride_lock_incompressible: bool = True

    # Drift semantics when cells move:
    # - "relative": organelle keeps the same offset from cell.left (then snapped to stride if needed)
    # - "absolute": organelle keeps absolute backing indices (then snapped to stride)
    # - "none": organelles are repacked left-justified on every reconcile
    organelle_drift_mode: str = "relative"  # "relative" | "absolute" | "none"

    # When snapping to stride, choose rounding direction:
    # - "nearest", "down", or "up"
    snap_mode: str = "nearest"

    # Optional hard LCM override; if 0, derive from cells' strides
    system_lcm_override: int = 0


@dataclass
class OrgMapping:
    # Relocation mapping for a single cell
    sources: List[int]
    destinations: List[int]


class BackingSynchronizer:
    """
    Computes target widths from cell volumes (with reserves), expands/contracts
    the backing, and builds stride-aware relocation maps for organelles.
    """

    def __init__(self, bitbuffer=None, cfg: Optional[BackingSyncCfg] = None):
        self.bitbuffer = bitbuffer
        self.cfg = cfg or BackingSyncCfg()

    # ----- public entrypoint -------------------------------------------------

    def reconcile_and_apply(
        self,
        cells: List,                          # cells with .left,.right,.stride,.V,.organelles
        proposals: Optional[list] = None      # list[CellProposal], will be forwarded
    ) -> Dict[str, OrgMapping]:
        """
        - Align each cell's region width to volume+reserve (LCM aware).
        - Expand/contract backing as needed.
        - Build organelle relocation maps per cell (stride-aware drift policy).

        Returns: {cell_label: OrgMapping}
        """
        if proposals is None:
            proposals = []
        lcm = self.cfg.system_lcm_override or lcm_all(getattr(c, "stride", 1) for c in cells)

        # 1) Compute target widths (bits) per cell from physical volume + reserves.
        targets = [self._target_width_bits(c, lcm) for c in cells]

        # 2) Plan expand/contract events (right-edge policy to preserve left).
        expand_events, shrink_events = self._diff_to_events(cells, targets, lcm)

        # 3) Apply to backing (no-op if backend missing or method absent).
        self._apply_size_changes(expand_events, shrink_events, cells, proposals)

        # 4) Build organelle relocation maps (stride-aware).
        #    These maps do not move bytes themselves; they are instructions
        #    for your higher layer to perform the data moves.
        mapping: Dict[str, OrgMapping] = {}
        for c in cells:
            mp = self._build_organelle_mapping(c, lcm)
            if mp.sources:  # only include if there is something to move
                mapping[str(getattr(c, "label", ""))] = mp

        return mapping

    # ----- sizes / reserves --------------------------------------------------

    def _target_width_bits(self, cell, lcm: int) -> int:
        """Required backing width = ceil_to_lcm( V_to_bits * V + solid + reserves )."""
        # Solid/lumen accounting from organelles:
        solid_bits = 0
        lumen_bits = 0
        data_units = 0

        for o in getattr(cell, "organelles", []):
            # duck-typing fields: V_solid, V_lumen(), volume_total, data_units, incompressible
            Vsol = getattr(o, "V_solid", 0.0)
            Vlum = o.V_lumen() if hasattr(o, "V_lumen") else max(getattr(o, "_V_lumen", 0.0), 0.0)
            solid_bits += intceil(self.cfg.V_to_bits * Vsol, lcm)
            lumen_bits += intceil(self.cfg.V_to_bits * Vlum, lcm)
            data_units += int(getattr(o, "data_units", 0))

        # cytosol (free solution) = total V - (organelles.solid + organelles.lumen)
        V_total = float(getattr(cell, "V", 0.0))
        V_occupied = (solid_bits + lumen_bits) / max(self.cfg.V_to_bits, 1e-12)
        V_free = max(V_total - V_occupied, 0.0)

        free_bits_actual = intceil(self.cfg.V_to_bits * V_free, lcm)
        free_bits_required = max(
            self.cfg.free_min_bits,
            self.cfg.free_per_data_unit * data_units
        )
        # Final target is the *max* of what physics says and what policy requires:
        physics_bits = intceil(self.cfg.V_to_bits * V_total, lcm)
        policy_bits  = solid_bits + lumen_bits + max(free_bits_actual, free_bits_required)
        return intceil(max(physics_bits, policy_bits), lcm)

    def _diff_to_events(
        self, cells: List, targets: List[int], lcm: int
    ) -> Tuple[List[Tuple[Optional[str], int, int]], List[Tuple[Optional[str], int, int]]]:
        """
        Compute delta between current widths and target widths.
        Returns (expands, shrinks) with each event as (label, anchor_bit, size_bits).

        Policy: adjust at the RIGHT edge to preserve left anchors/stability.
        """
        expands: List[Tuple[Optional[str], int, int]] = []
        shrinks: List[Tuple[Optional[str], int, int]] = []

        for c, want in zip(cells, targets):
            left = int(getattr(c, "left", 0))
            right = int(getattr(c, "right", left))
            cur = max(0, right - left)
            if want > cur:
                expands.append((getattr(c, "label", None), right - 1, intceil(want - cur, lcm)))
            elif want < cur:
                # shrink from right
                shrinks.append((getattr(c, "label", None), right - 1, intceil(cur - want, lcm)))
        return expands, shrinks

    def _apply_size_changes(
        self,
        expands: List[Tuple[Optional[str], int, int]],
        shrinks: List[Tuple[Optional[str], int, int]],
        cells: List,
        proposals: List
    ) -> None:
        """Best-effort application against whatever adapter/bitbuffer is present."""
        bb = self.bitbuffer
        if not bb:
            return
        # Expand if implemented (BitBitBufferAdapter.expand or raw .expand)
        if hasattr(bb, "expand"):
            _ = bb.expand(expands, cells, proposals)
        # Shrink/contract if implemented.
        # Provide compatibility with either .shrink or .contract
        if shrinks:
            if hasattr(bb, "shrink"):
                _ = bb.shrink(shrinks, cells, proposals)
            elif hasattr(bb, "contract"):
                _ = bb.contract(shrinks, cells, proposals)
            # else: silently skip if backend cannot contract (design-time choice)

    # ----- stride-aware organelle relocation --------------------------------

    def _build_organelle_mapping(self, cell, lcm: int) -> OrgMapping:
        """
        Construct a relocation plan for data-carrying organelles under stride rules.

        We assume each organelle carries a set of backing indices (o.backing_indices: List[int])
        or a count (o.data_units) we can pack left-to-right on the cell's stride grid.

        Drift policy:
          - relative: keep offset from cell.left, then snap by stride/LCM
          - absolute: keep absolute index, then snap by stride/LCM
          - none:     repack densely from cell.left on each reconcile
        """
        left = int(getattr(cell, "left", 0))
        right = int(getattr(cell, "right", left))
        stride = int(max(1, getattr(cell, "stride", lcm)))
        grid = _lcm(stride, lcm)

        sources: List[int] = []
        destinations: List[int] = []

        # helper: snap
        def snap(idx: int) -> int:
            if self.cfg.snap_mode == "down":
                return intfloor(idx, grid)
            if self.cfg.snap_mode == "up":
                return intceil(idx, grid)
            # nearest
            down = intfloor(idx, grid)
            up = intceil(idx, grid)
            return down if (idx - down) <= (up - idx) else up

        # Gather declared items (prefer explicit indices; otherwise synthesize from data_units)
        declared: List[int] = []
        for o in getattr(cell, "organelles", []):
            if getattr(o, "incompressible", False) and self.cfg.stride_lock_incompressible:
                # Collect declared indices if present
                inds = list(getattr(o, "backing_indices", []))
                if inds:
                    declared.extend(inds)
                else:
                    # synthesize placeholders to keep count & stride shape
                    declared.extend([None] * int(max(0, getattr(o, "data_units", 0))))

        if not declared:
            return OrgMapping(sources=[], destinations=[])

        # Build targets
        if self.cfg.organelle_drift_mode == "none":
            # left-pack under stride/grid
            dst_iter = range(left, right, grid)
            for k, maybe_src in enumerate(declared):
                try:
                    dst = next(i for i in dst_iter)
                except StopIteration:
                    break
                if maybe_src is None:
                    continue
                sources.append(int(maybe_src))
                destinations.append(int(dst))
        elif self.cfg.organelle_drift_mode == "absolute":
            for maybe_src in declared:
                if maybe_src is None:
                    continue
                sources.append(int(maybe_src))
                destinations.append(int(snap(maybe_src)))
        else:
            # relative (default)
            for maybe_src in declared:
                if maybe_src is None:
                    continue
                rel = int(maybe_src) - left
                dst = snap(left + rel)
                # clamp into [left,right)
                dst = max(left, min(dst, right - 1))
                sources.append(int(maybe_src))
                destinations.append(int(dst))

        return OrgMapping(sources=sources, destinations=destinations)
