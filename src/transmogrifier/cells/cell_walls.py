import math
from .cell_consts import LEFT_WALL, RIGHT_WALL

def snap_cell_walls(self, cells, proposals):
    """
    Determines and applies new cell boundaries using a stable, two-pass approach.
    1. Calculation Pass: Determines all new boundaries and the total required buffer size.
    2. Execution Pass: Expands the buffer once (triggering the desired global distribution)
    and then applies the new boundaries to the cells.
    """
    import math
    self.bar()
    print("Snapping cell walls...")
        
    # Initialize fixed extents if they don't exist
    for cell in cells:
        if not hasattr(cell, 'leftmost') or cell.leftmost is None:
            print(f"Line 648: Cell {cell.label} leftmost is None, setting to left {cell.left}")
            cell.leftmost = cell.left
        if not hasattr(cell, 'rightmost') or cell.rightmost is None:
            print(f"Line 651: Cell {cell.label} rightmost is None, setting to right - 1: {cell.right - 1}")
            cell.rightmost = cell.right - 1  # rightmost is inclusive, so we subtract 1 to make it exclusive

    # Initialize fixed extents if they don't exist
    for proposal in proposals:
        if not hasattr(proposal, 'leftmost') or proposal.leftmost is None:
            print(f"Line 657: Proposal {proposal.label} leftmost is None, setting to left {proposal.left}")
            proposal.leftmost = proposal.left
        if not hasattr(proposal, 'rightmost') or proposal.rightmost is None:
                
            # it's this one:
            print(f"Line 662: Proposal {proposal.label} rightmost is None, setting to right - 1: {proposal.right - 1}")
            proposal.rightmost = proposal.right - 1

    for c in [LEFT_WALL, RIGHT_WALL]:
        if getattr(c, "leftmost", None) is None:
            print(f"Line 667: Wall {c.label} leftmost is None, setting to left {c.left}")
            c.leftmost = c.left
        if getattr(c, "rightmost", None) is None:
            print(f"Line 670: Wall {c.label} rightmost is None, setting to right - 1: {c.right - 1}")
            c.rightmost = c.right - 1

    # filter empty cells and proposals
    cells = [c for c in cells if c.leftmost < c.rightmost and c.left < c.right or c == LEFT_WALL or c == RIGHT_WALL]
    empty_cells = [c for c in cells if c.leftmost > c.rightmost or c.left >= c.right and c != LEFT_WALL and c != RIGHT_WALL]
    proposals = [p for p in proposals if p.leftmost < p.rightmost and p.left < p.right or p == LEFT_WALL or p == RIGHT_WALL]
    empty_proposals = [p for p in proposals if p.leftmost > p.rightmost or p.left >= p.right and p != LEFT_WALL and p != RIGHT_WALL]
    sorted_cells = sorted(cells, key=lambda c: c.leftmost)
    sorted_proposals = sorted(proposals, key=lambda p: p.leftmost)
    cells = [LEFT_WALL] + sorted_cells + [RIGHT_WALL]
    proposals = [LEFT_WALL] + sorted_proposals + [RIGHT_WALL]



    # --- Pass 1: Calculate all desired changes ---
    boundary_updates = []
    max_needed = self.bitbuffer.mask_size
    system_lcm = self.lcm(proposals)

    for i in range(len(proposals) + 1):
        prev = proposals[i - 1] if i > 0 else LEFT_WALL
        curr = proposals[i]     if i < len(proposals) else RIGHT_WALL

        if i == len(proposals):
            # push RIGHT_WALL to the very end
            RIGHT_WALL.leftmost = RIGHT_WALL.right = RIGHT_WALL.left = self.bitbuffer.mask_size

        # envelope [low, high]
        low  = min(prev.right, curr.leftmost)
        high = max(prev.right, curr.leftmost)

        # ----- START ROBUST FIX -----
        # First, determine the ideal right boundary for the previous cell ('a0').
        # It's clamped within the [low, high] envelope and aligned to its own stride.
        s_prev = prev.stride
        k_min = math.ceil(low / s_prev)
        k_max = math.floor(high / s_prev)
        k0 = prev.right // s_prev
        k_best = min(max(k0, k_min), k_max)
        a0 = k_best * s_prev

        # Now, determine the left boundary for the current cell ('b0').
        # It MUST be at or after 'a0'. We find the first position >= a0 that
        # is correctly aligned to the current cell's stride.
        s_curr = curr.stride
        b0 = ((a0 + s_curr - 1) // s_curr) * s_curr
        # ----- END ROBUST FIX -----
        boundary_updates.append({'index': i, 'a': a0, 'b': b0})

        #boundary_updates.append({'index': i, 'a': a0, 'b': b0})
        max_needed = max(max_needed, a0, b0)



    # --- Pass 2: Apply all calculated changes ---
    for update in boundary_updates:
        i = update['index']
        prev = proposals[i - 1] if i > 0 else LEFT_WALL
        curr = proposals[i] if i < len(proposals) else RIGHT_WALL
        # now safe to compute pressure adjustments
        orig_a_len = prev.right - prev.left
        orig_b_len = curr.right - curr.left
            
        # Apply the new boundaries, but clamp so width ≥ 0
        a_best = update['a']
        b_best = update['b']

        # enforce prev.right ≥ prev.left, and curr.left ≤ curr.right
        print(f'Line 743: Updating cell {prev.label} from left {prev.left} to right {prev.right} with new leftmost {prev.leftmost}, stride {prev.stride}, and pressure {prev.pressure}')
        print(f'Line 744: Updating cell {curr.label} from left {curr.left} to right {curr.right} with new leftmost {curr.leftmost}, stride {curr.stride}, and pressure {curr.pressure}')
        prev.right = max((curr.leftmost - prev.stride)//prev.stride * prev.stride, max(prev.rightmost+1, a_best))
        curr.left  = min(curr.leftmost, min(curr.right, b_best))
        print(f'Line 747: Updating cell {prev.label} from left {prev.left} to right {prev.right} with new leftmost {prev.leftmost}, stride {prev.stride}, and pressure {prev.pressure}')
        print(f'Line 748: Updating cell {curr.label} from left {curr.left} to right {curr.right} with new leftmost {curr.leftmost}, stride {curr.stride}, and pressure {curr.pressure}')

        # Recompute proportional pressures based on new sub-lengths
        new_a_len = prev.right - prev.left
        new_b_len = curr.right - curr.left
            
        # Prevent division by zero
            
        new_p_a = (prev.pressure * new_a_len) // orig_a_len if orig_a_len > 0 else 0
        new_p_b = (curr.pressure * new_b_len) // orig_b_len if orig_b_len > 0 else 0
            
        self.system_pressure += (new_p_a + new_p_b) - (prev.pressure + curr.pressure)
        prev.pressure = new_p_a
        curr.pressure = new_p_b


    cells.pop()
    cells.pop(0)  # Remove LEFT_WALL and RIGHT_WALL from the cells list
    proposals.pop()
    proposals.pop(0)  # Remove LEFT_WALL and RIGHT_WALL from the proposals

    # destribute empty cells and proposals into empty space

    self.bar()
    print("Snapping empty cells and proposals to leftmost/rightmost boundaries...")


    for empty_proposal in empty_proposals:
        print(f"Snapping empty proposal {empty_proposal.label} to left {max_needed} and right {max_needed + empty_proposal.stride}")
        empty_proposal.left = max_needed
        empty_proposal.right = max_needed + empty_proposal.stride
        empty_proposal.leftmost = empty_proposal.left
        print(f"Line 773: Empty proposal {empty_proposal.label} leftmost set to {empty_proposal.leftmost}")
        empty_proposal.rightmost = empty_proposal.right - 1
        max_needed += empty_proposal.stride

    print("Done snapping empty cells and proposals.")
    self.bar()
    # Diagnostic print
    #print(f"Snapped cell walls: {[f'{cell.label}: {cell.left}-{cell.right} (stride {cell.stride})' for cell in proposals]}")


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

    print("Done snapping cell walls.")
    self.bar()
def build_metadata(self, offset_bits, size_bits, cells):
        
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
    events = self.build_metadata(offset_bits, size_bits, cells)
    #for label, pos, share in events:
        #print(f"Expanding cell {label} at position {pos} with share {share} bits")
    self.bitbuffer.expand(events, cells, proposals)

