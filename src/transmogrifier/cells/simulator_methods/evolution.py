from ...bitbitbuffer import CellProposal


def evolution_tick(self, cells, max_iters: int = 10):
    """Advance the hydraulic model until cell widths stabilise."""
    proposals = []
    prev_widths = [c.right - c.left for c in cells]
    for _ in range(max_iters):
        self.update_s_p_expressions(cells)
        fractions = self.equilibrium_fracs(0.0)
        total_space = self.bitbuffer.mask_size
        proposals = []
        new_widths = []
        print(f"Fractions: {[float(frac) for frac in fractions]}")
        print(f"LCM: {self.system_lcm}")
        print(f"Total space: {total_space}")
        for cell, frac in zip(cells, fractions):
            new_width = self.bitbuffer.intceil(1+max(
                self.bitbuffer.intceil(cell.salinity, cell.stride),
                self.bitbuffer.intceil(int(total_space * frac), cell.stride),
            ), self.system_lcm)
            print(f"Line 18: Cell {cell.label} with stride: {cell.stride} new_width is {new_width}")
            assert new_width % cell.stride == 0
            proposal = CellProposal(cell)
            proposal.right = proposal.left + new_width
            assert proposal.right > proposal.left, "Invalid cell proposal"
            proposals.append(proposal)
            new_widths.append(new_width)
        assert len(cells) > 0, "No cells provided to snap walls"
        assert len(proposals) > 0, "No proposals provided to snap walls"
        self.snap_cell_walls(cells, proposals)
        if new_widths == prev_widths:
            break
        prev_widths = new_widths

    return proposals


def step(self, cells):
    """Run one simulation step with size resolution preceding writes."""
    proposals = self.evolution_tick(cells)

    return self.minimize(cells)
