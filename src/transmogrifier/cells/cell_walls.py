import math
from .cell_consts import LEFT_WALL, RIGHT_WALL

def snap_cell_walls(self, cells, proposals):

    # ``expand`` is provided by :class:`Simulator`; no rebinding needed here.
    #self.bar()
    #print("Snapping cell walls...")
        
    # Initialize fixed extents if they don't exist
    for cell in cells:
        if not hasattr(cell, 'leftmost') or cell.leftmost is None:
            #print(f"Line 648: Cell {cell.label} leftmost is None, setting to left {cell.left}")
            cell.leftmost = cell.left
        if not hasattr(cell, 'rightmost') or cell.rightmost is None:
            #print(f"Line 651: Cell {cell.label} rightmost is None, setting to right - 1: {cell.right - 1}")
            cell.rightmost = cell.right - 1  # rightmost is inclusive, so we subtract 1 to make it exclusive

    # Initialize fixed extents if they don't exist
    for proposal in proposals:
        if not hasattr(proposal, 'leftmost') or proposal.leftmost is None:
            #print(f"Line 657: Proposal {proposal.label} leftmost is None, setting to left {proposal.left}")
            proposal.leftmost = proposal.left
        if not hasattr(proposal, 'rightmost') or proposal.rightmost is None:
                
            # it's this one:
            #print(f"Line 662: Proposal {proposal.label} rightmost is None, setting to right - 1: {proposal.right - 1}")
            proposal.rightmost = proposal.right - 1

    for c in [LEFT_WALL, RIGHT_WALL]:
        if getattr(c, "leftmost", None) is None:
            #print(f"Line 667: Wall {c.label} leftmost is None, setting to left {c.left}")
            c.leftmost = c.left
        if getattr(c, "rightmost", None) is None:
            #print(f"Line 670: Wall {c.label} rightmost is None, setting to right - 1: {c.right - 1}")
            c.rightmost = c.right - 1

    # filter empty cells and proposals
    # Note: rightmost is inclusive. A single-width cell has leftmost == rightmost and is valid.
    empty_cells = [
        c for c in cells
        if ((c.leftmost > c.rightmost) or (c.left >= c.right)) and (c not in (LEFT_WALL, RIGHT_WALL))
    ]


    print("Empty Cell Report:")
    for c in empty_cells:
        print(f" - Cell {c.label}: {c.leftmost} to {c.rightmost} data in {c.left} to {c.right} (stride {c.stride})")

    cells = [
        c for c in cells
        if ((c.leftmost <= c.rightmost) and (c.left < c.right)) or (c in (LEFT_WALL, RIGHT_WALL))
    ]
    

    
    empty_proposals = [
        p for p in proposals
        if ((p.leftmost > p.rightmost) or (p.left >= p.right)) and (p not in (LEFT_WALL, RIGHT_WALL))
    ]
    
    proposals = [
        p for p in proposals
        if ((p.leftmost <= p.rightmost) and (p.left < p.right)) or (p in (LEFT_WALL, RIGHT_WALL))
    ]
    sorted_cells = sorted(cells, key=lambda c: c.leftmost)
    sorted_proposals = sorted(proposals, key=lambda p: p.leftmost)
    cells = [LEFT_WALL] + sorted_cells + [RIGHT_WALL]
    proposals = [LEFT_WALL] + sorted_proposals + [RIGHT_WALL]



    # --- Pass 1: Calculate LCM-aligned boundaries ---
    # Invariant: The beginning (left) of every cell MUST be aligned to the system LCM grid.
    # The right of the previous cell is snapped to the same grid line.
    boundary_updates = []
    max_needed = self.bitbuffer.mask_size
    print(f"strides are:")
    for cell in sorted_cells + empty_cells:
        print(f"Cell {cell.label} stride: {cell.stride}")
    system_lcm = self.lcm(sorted_cells + empty_cells)
    print(f"System LCM is: {system_lcm}")
    assert system_lcm > 0, "System LCM must be greater than zero"
    #this isn't always true assert system_lcm % 2 == 0, "System LCM must be even for proper alignment"
    for cell in sorted_cells:
        assert system_lcm % cell.stride == 0, f"Cell {cell.label} stride {cell.stride} must align with system LCM {system_lcm}"
    self.system_lcm = system_lcm  # ensure metadata uses the same grid

    # Keep RIGHT_WALL tracking the current mask extent to start, it may move right
    RIGHT_WALL.leftmost = RIGHT_WALL.right = RIGHT_WALL.left = self.bitbuffer.mask_size

    # Build LCM-aligned boundary for each adjacent pair (including LEFT_WALL->first and last->RIGHT_WALL)
    for i in range(1, len(proposals)):
        prev = proposals[i - 1]
        curr = proposals[i]

        # base constraint: cannot violate fixed insides (prev.rightmost, curr.leftmost)
        # and must not move boundary left of what already exists (prev.right)
        base = max(
            prev.right,
            getattr(prev, 'rightmost', prev.right) + 1,
            curr.leftmost,
        )
        b = ((base + system_lcm - 1) // system_lcm) * system_lcm

        boundary_updates.append({'index': i, 'b': b})
        max_needed = max(max_needed, b)



    # --- Pass 2: Apply all calculated changes ---
    for update in boundary_updates:
        i = update['index']
        prev = proposals[i - 1]
        curr = proposals[i]

        # preserve original lengths for pressure reweighting
        orig_a_len = prev.right - prev.left
        orig_b_len = curr.right - curr.left

        b = update['b']

        # Ensure non-negative widths while enforcing boundary on the LCM grid
        if b > curr.right:
            curr.right = (b + system_lcm - 1) // system_lcm * system_lcm
        prev.right = b
        curr.left = b

        # Recompute proportional pressures based on new sub-lengths
        new_a_len = prev.right - prev.left
        new_b_len = curr.right - curr.left

        new_p_a = (prev.pressure * new_a_len) // orig_a_len if orig_a_len > 0 else 0
        new_p_b = (curr.pressure * new_b_len) // orig_b_len if orig_b_len > 0 else 0

        self.system_pressure += (new_p_a + new_p_b) - (prev.pressure + curr.pressure)
        prev.pressure = new_p_a
        curr.pressure = new_p_b

        # If the boundary is against RIGHT_WALL, keep it a zero-width wall at that grid line
        if self.closed:
            if curr is RIGHT_WALL:
                curr.left = curr.leftmost = curr.right = b
                curr.rightmost = b - 1


    cells.pop()
    cells.pop(0)  # Remove LEFT_WALL and RIGHT_WALL from the cells list
    proposals.pop()
    proposals.pop(0)  # Remove LEFT_WALL and RIGHT_WALL from the proposals



    # destribute empty cells and proposals into empty space

    #self.bar()
    #print("Snapping empty cells and proposals to leftmost/rightmost boundaries...")


    for empty_proposal in empty_proposals:
        #print(f"Snapping empty proposal {empty_proposal.label} to left {max_needed} and right {max_needed + empty_proposal.stride}")
        empty_proposal.left = max_needed
        empty_proposal.right = max_needed + empty_proposal.stride
        empty_proposal.leftmost = empty_proposal.left
        #print(f"Line 773: Empty proposal {empty_proposal.label} leftmost set to {empty_proposal.leftmost}")
        empty_proposal.rightmost = empty_proposal.right - 1
        max_needed += empty_proposal.stride

    #print("Done snapping empty cells and proposals.")
    self.bar()
    # Diagnostic print
    print(f"Snapped cell walls: {[f'{cell.label}: {cell.left}-{cell.right} (stride {cell.stride})' for cell in proposals]}")


    proposals = proposals# + empty_proposals
    cells = cells# + empty_cells

    # --- Intermission: Perform a single, system-wide expansion if needed ---
    if max_needed > self.bitbuffer.mask_size:
        print(f"Had to expand bitbuffer mask size from {self.bitbuffer.mask_size} to {max_needed} bits for snapping cell walls")
        # This triggers the desired fallback logic in build_metadata to distribute the new space
        self.expand(self.bitbuffer.mask_size, self.bitbuffer.intceil(max_needed - self.bitbuffer.mask_size, system_lcm), cells, proposals)


    most_right = max(cell.right for cell in proposals)
    if most_right + 1 > self.bitbuffer.mask_size:
        #print(f"Expanding data buffer to accommodate last cell's right boundary: {cells[-1].right * MASK_BITS_TO_DATA_BITS} bits")
        self.expand(self.bitbuffer.mask_size, self.bitbuffer.intceil(most_right - self.bitbuffer.mask_size, system_lcm), cells, proposals, warp=False)


    if self.system_pressure > 0:
            
        #print(f"System pressure after snapping cell walls: {self.system_pressure}")
        self.expand(self.bitbuffer.mask_size, self.bitbuffer.intceil(self.system_pressure, system_lcm), cells, proposals, warp=False)

    #print("Done snapping cell walls.")
    #self.bar()

def build_metadata(self, offset_bits, size_bits, cells):
    assert len(cells) > 0, "No cells provided to build_metadata"
    events = []
    # make sure these are lists
    offs = offset_bits if isinstance(offset_bits, (list,tuple)) else [offset_bits]
    szs  = size_bits   if isinstance(size_bits,   (list,tuple)) else [size_bits]

    for offset in offs:
        assert isinstance(offset, int), f"Offset {offset} is not an integer"
        assert offset >= 0, f"Offset {offset} is negative, must be non-negative"
        assert offset <= self.bitbuffer.mask_size, f"Offset {offset} exceeds mask size {self.bitbuffer.mask_size}"

    for off, sz in zip(offs, szs):
        # 1) try to find a cell that contains `off`
        for cell in cells:
            # Modification 4: auto-heal pathological “left > right”
            if cell.right < cell.left:
                cell.right = cell.left
            if cell.left <= off < cell.right:
                # Modification 1: robust centre selection (align to LCM within cell)
                raw_mid = (cell.left + cell.right) // 2
                aligned = raw_mid - (raw_mid % self.system_lcm)
                center  = max(cell.left, min(aligned, cell.right - 1))
                events.append((cell.label, center, sz))
                break
        else:
            # 2) fallback – split `sz` among cells
            n = len(cells)
            base = sz // n
            rem  = sz % n
            for idx, cell in enumerate(cells):
                share  = self.bitbuffer.intceil(base + (1 if idx < rem else 0), self.system_lcm)
                raw_mid = (cell.left + cell.right) // 2
                center  = raw_mid - (raw_mid % self.system_lcm)
                center  = max(cell.left, min(center, max(cell.right - self.system_lcm, cell.left)))
                events.append((cell.label, center, share))

        
    final = [(label, pos, share) for label, pos, share in events]
    return sorted(final, key=lambda e: e[1])
    
    

def expand(self, offset_bits, size_bits, cells, proposals, warp=True):
    """
    Build the event list exactly as before, then hand it off to BitBitBuffer.
    """
    #print(f"Expanding bitbuffer mask size from {self.bitbuffer.mask_size} to accommodate {size_bits} bits at offsets {offset_bits}")
    if isinstance(offset_bits, int):
        offset_bits = [offset_bits]
    for offset in offset_bits:
        assert isinstance(offset, int), f"Offset {offset} is not an integer"
        assert offset >= 0, f"Offset {offset} is negative, must be non-negative"
        assert offset <= self.bitbuffer.mask_size, f"Offset {offset} exceeds mask size {self.bitbuffer.mask_size}"

    # Ensure metadata alignment grid is always current
    #self.system_lcm = self.lcm(proposals)

    events = self.build_metadata(offset_bits, size_bits, cells)
    #for label, pos, share in events:
        #print(f"Expanding cell {label} at position {pos} with share {share} bits")
    proposals = self.bitbuffer.expand(events, cells, proposals)

    for new_cell in proposals:
        for cell in cells:
            if cell.label == new_cell.label:
                cell.left = new_cell.left
                cell.right = new_cell.right
                cell.leftmost = new_cell.leftmost
                cell.rightmost = new_cell.rightmost
                break