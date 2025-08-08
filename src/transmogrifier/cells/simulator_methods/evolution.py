from ...bitbitbuffer import CellProposal


def evolution_tick(self, cells, max_iters: int = 10, *, flush: bool = True):
    """Advance the hydraulic model until cell widths stabilise."""
    proposals = []
    prev_widths = [c.right - c.left for c in cells]
    for _ in range(max_iters):
        self.update_s_p_expressions(cells)
        if self.engine is None:
            raise AttributeError("Saline engine not initialized; call run_saline_sim first")
        fractions = self.engine.equilibrium_fracs(0.0)
        total_space = self.bitbuffer.mask_size
        proposals = []
        new_widths = []
        for cell, frac in zip(cells, fractions):
            new_width = max(
                self.bitbuffer.intceil(cell.salinity, cell.stride),
                self.bitbuffer.intceil(int(total_space * frac), cell.stride),
            )
            assert new_width % cell.stride == 0
            proposal = CellProposal(cell)
            proposal.right = proposal.left + new_width
            proposals.append(proposal)
            new_widths.append(new_width)
        self.snap_cell_walls(cells, proposals)
        if new_widths == prev_widths:
            break
        prev_widths = new_widths
    if flush:
        self.flush_pending_writes()
    return proposals


def step(self, cells):
    """Run one simulation step with size resolution preceding writes."""
    self.evolution_tick(cells)
    return self.minimize(cells)
