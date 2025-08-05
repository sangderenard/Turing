import string
from typing import Union
from sympy import Integer
from .cell_consts import Cell, CellFlags, LeftWallFlags, RightWallFlags, SystemFlags, MASK_BITS_TO_DATA_BITS, TEST_SIZE_STRIDE_TIMES_UNITS, CELL_COUNT, STRIDE, RIGHT_WALL, LEFT_WALL
from .salinepressure import SalineHydraulicSystem
from .bitbitbuffer import BitBitBuffer, CellProposal
from .bitstream_search import BitStreamSearch


class Simulator:
    FORCE_THRESH = .5
    LOCK = 0x1
    ELASTIC = 0x2

    def __init__(self, cells):
        self.assignable_gaps = {}
        self.pid_list = []
        self.cells = cells
        self.input_queues = {}
        self.system_pressure = 0
        self.elastic_coeff = 0.1
        #for cell in self.cells:
        #    print(f"Simulator: Initializing cell {cell.label} with left={cell.left}, right={cell.right}, stride={cell.stride}")
        self.system_lcm   = self.lcm(cells)                       # ← uses your helper
        required_end = max(c.right for c in cells)           # highest bit any cell uses
        #print(f"Simulator: required end is {required_end} bits")
        mask_size    = BitBitBuffer._intceil(required_end, self.system_lcm)
        #print(f"Simulator: mask size is {mask_size} bits, system LCM is {self.system_lcm}")
        self.bitbuffer = BitBitBuffer(mask_size=mask_size, caster=bytes,
                                    bitsforbits=MASK_BITS_TO_DATA_BITS)
        self.bitbuffer.register_pid_buffer(cells=self.cells)
        self.locked_data_regions = []

        self.search = BitStreamSearch()

        self.s_exprs = [Integer(0) for _ in range(CELL_COUNT)]
        self.p_exprs = [Integer(1) for _ in range(CELL_COUNT)]

        self.engine = None
        self.fractions = None

        self.run_saline_sim()

    def update_s_p_expressions(self, cells):
        """
        Update the salinity and pressure expressions for each cell.
        This is called before running the saline simulation.
        """
        self.s_exprs = [Integer(cell.salinity) for cell in cells]
        self.p_exprs = [Integer(cell.pressure) for cell in cells]

    def run_saline_sim(self):
        # 1) Instantiate engine with your per‐cell salinity & pressure expressions (or plain numbers)
        self.update_s_p_expressions(self.cells)
        self.engine = SalineHydraulicSystem(
            self.s_exprs,           # e.g. [Integer(s0), Integer(s1), …]
            self.p_exprs,           # e.g. [Integer(p0), Integer(p1), …]
            width=self.bitbuffer.mask_size, # the total bit‐space you’re dividing
            chars=[chr(97+i) for i in range(CELL_COUNT)],
            tau=5, math_type='int',
            int_method='adams',
            protect_under_one=True,
            bump_under_one=True
        )
        for cell in self.cells:
            if cell.leftmost is None:
                print(f"Line 67: Cell {cell.label} leftmost is None, setting to left {cell.left}")
                cell.leftmost = cell.left
            if cell.rightmost is None:
                print(f"Line 70: Cell {cell.label} rightmost is None, setting to right - 1: {cell.right - 1}")
                cell.rightmost = cell.right - 1
        # 2) Ask for the equilibrium fractions at t=0
        self.fractions = self.engine.equilibrium_fracs(0.0)
        #for cell in self.cells:
            #if cell.salinity == 0:
                #cell.salinity = 1

        necessary_size = self.bitbuffer.intceil(sum(cell.salinity for cell in self.cells if hasattr(cell, 'salinity') and cell.salinity > 0), self.system_lcm)
        
        if self.bitbuffer.mask_size < necessary_size:
            offsets = [self.bitbuffer.intceil((cell.rightmost - cell.leftmost)//2+cell.leftmost, cell.stride) for cell in self.cells if hasattr(cell, 'leftmost')]
            sizes = [(cell.salinity) for cell in self.cells if hasattr(cell, 'salinity') and cell.salinity > 0]
            size_and_offsets = sorted(list(zip(sizes, offsets)), reverse=True, key=lambda x: x[1])
            for size, offset in size_and_offsets:
                self.expand([offset], self.bitbuffer.intceil(size, self.lcm(self.cells)), self.cells, self.cells)

        self.snap_cell_walls(self.cells, self.cells)

    def get_cell_mask(self, cell: Cell) -> bytearray:
        return self.bitbuffer[cell.left:cell.right]

    def set_cell_mask(self, cell: Cell, mask: bytearray) -> None:
        self.bitbuffer[cell.left:cell.right] = mask

    def pull_cell_mask(self, cell):
        cell._buf = self.get_cell_mask(cell)
    def push_cell_mask(self, cell):
        self.set_cell_mask(cell, cell._buf)
    

    def evolution_tick(self, cells):
        # Use the saline pressure system to set cell proportions
        # inside minimize(…) or evolution_tick(…), once every cell.pressure & .salinity are up‑to‑date:

        # rebuild the engine’s callables so they always return the current attributes
        self.engine.s_funcs = [
            (lambda _t, s=cell.salinity: s)
            for cell in cells
        ]
        self.engine.p_funcs = [
            (lambda _t, p=cell.pressure: p)
            for cell in cells   
        ]



        proposals = []
        fractions = self.engine.equilibrium_fracs(0.0)
        total_space = self.bitbuffer.mask_size
        current_left = 0
        for cell, frac in zip(cells, fractions):
            new_width = max(self.bitbuffer.intceil(cell.salinity,cell.stride), self.bitbuffer.intceil(int(total_space * frac), cell.stride))
            assert new_width % cell.stride == 0, f"New width {new_width} for cell {cell.label} is not aligned with stride {cell.stride}"
            assert cell.stride > 0, f"Cell {cell.label} has non-positive stride {cell.stride}"
            proposal = CellProposal(cell)
            proposals.append(proposal)
            #cell.pressure = 0  # reset pressure after reallocation
            current_left = cell.right
            print(f"Cell {cell.label} resized to {cell.left} - {cell.right} ({new_width} bits)")
        self.snap_cell_walls(cells, proposals)
        self.print_system(cells)
        return proposal


    def print_system(self, cells, width=80):
        """
        Draw the entire address space scaled to `width` characters,
        but whenever a column has N cells in it, emit N glyphs—
        lower‐case if no data in that slice, ramp‐mapped if data.
        """
        total_bits = self.bitbuffer.mask_size
        if total_bits == 0:
            print("<empty>")
            return

        # 1) Build bit_info: (bit_index, cell_idx_or_None, mask_bit)
        bit_info = []
        for b in range(total_bits):
            cell_idx = None
            for idx, cell in enumerate(cells):
                if cell.left <= b < cell.right:
                    cell_idx = idx
                    break
            mask_bit = bool(int(self.bitbuffer[b]))
            bit_info.append((b, cell_idx, mask_bit))

        # 2) Fragmentation (unchanged)
        free_bits = sum(1 for _, idx, m in bit_info if idx is not None and not m)
        runs, run = [], 0
        for _, idx, m in bit_info:
            if idx is not None and not m:
                run += 1
            elif run:
                runs.append(run); run = 0
        if run: runs.append(run)
        max_run = max(runs) if runs else 0
        frag_pct = (1 - max_run / free_bits) * 100 if free_bits else 0.0

        size_string = (
            f"Total size: {total_bits} bits "
            f"({total_bits/8:.2f} bytes, mask bits: {self.bitbuffer.mask_size})"
        )
        free_string = f"Free: {free_bits} bits; fragmentation: {frag_pct:.2f}%"

        # 3) Prepare labels & ramp
        labels     = string.ascii_lowercase
        alpha_len  = len(labels)
        glyph_bases = [
            ord('a'),    # level 1
            ord('A'),    # level 2
            0x03B1,      # level 3 α
            0x0391,      # level 4 Α
            0x0430,      # level 5 а
            0x0410,      # level 6 А
        ]
        max_level = len(glyph_bases)

        # 4) Columnize
        bits_per_col = (total_bits + width - 1) // width
        output = []
        for col in range(width):
            start = col * bits_per_col
            end   = min(start + bits_per_col, total_bits)

            # track which cells appear, and how many data‐hits each has
            cell_presence = set()
            cell_counts   = {}
            for b, idx, m in bit_info[start:end]:
                if idx is None:
                    continue
                cell_presence.add(idx)
                # only count “data” at stride‐anchor positions
                cell = cells[idx]
                if m and ((b - cell.left) % cell.stride) == 0:
                    cell_counts[idx] = cell_counts.get(idx, 0) + 1

            if not cell_presence:
                # entirely outside all cells
                output.append('.')
            else:
                # for each cell in this column, emit one glyph
                for idx in sorted(cell_presence):
                    cnt = cell_counts.get(idx, 0)
                    if cnt == 0:
                        # inside a cell but no data → lower‐case label
                        ch = labels[idx % alpha_len]
                    else:
                        # data present → map 1→level1, 2→level2, ..., clamp
                        level = min(cnt, max_level) - 1
                        base  = glyph_bases[level]
                        ch    = chr(base + (idx % alpha_len))
                    output.append(ch)

        # 5) Print map + stats
        print(''.join(output))
        print(size_string, free_string)



    def contiguate(self, raw, pattern, fragmented_slice, stride, rev=False):
        # fragmented slice contains mixed data, indices not already spoken for
        # as recipients for new data

        # pattern contains tuples of (bit, count) where bit is 0 or 1
        # and count is the number of consecutive bits of that type
        
        

        
        contiguous_strides = [pattern[i][1] for i in range(len(pattern)) if pattern[i][0] == 1]
        contiguous_strides = sorted(contiguous_strides, reverse=True)
        output = []
        junk = []

        i = 0
        
        print(f"Fragmented slice: {fragmented_slice}")
        print(f"Pattern: {pattern}, stride: {stride}, raw length: {len(raw)}")
        for cluster in pattern:
            
            if cluster[0] == 1:# when you get back to this tomorrow, only the right path creates errors, the reverse one
                # it's working pretty close to okay now but still causes a write outside of range error
                this_range = range(cluster[1] // stride) if not rev else range(cluster[1] // stride, -1, -1)
                is_junk = cluster[1] % stride
                if is_junk == 0:
                    print(f"Contiguate: cluster={cluster}, stride={stride}, raw length={len(raw)}")
                    for j in this_range:
                        base_off   = raw.mask_index
                        rel_off    = i + j * stride
                        if not rev:
                            # forwards
                            pointer_offset = base_off + rel_off
                        else:
                            # backwards – safe, never crosses the right edge
                            pointer_offset = base_off + max(0, len(raw) - (i + (j + 1) * stride))
                        
                        print(f"i: {i}, j: {j}, base_off: {base_off}, rel_off: {rel_off}, pointer_offset: {pointer_offset}")
                        print(f"Contiguate: pointer_offset={pointer_offset}, stride={stride}, raw length={len(raw)}")
                        print(f"Pointer Offset to stride alignment: {pointer_offset % stride} == 0")
                        
                        mask_data    = self.bitbuffer[pointer_offset: pointer_offset + stride]
                        backing_data = self.bitbuffer._data_access[ pointer_offset: pointer_offset + stride ]

                        print(f"Mask data: {mask_data.hex()}, \nBacking data: {backing_data.hex()}")
                        
                        output.append((backing_data, stride))
                        i += stride
                        
                else:
                    #print(f"Contiguate: cluster={cluster}, stride={stride}, raw length={len(raw)}")
                    junk.append(raw[i:i + cluster[1]])
                    i += cluster[1]
                    assert False, "Junk data found in contiguate function, this should not happen with the current algorithm"
            else:
                # This is a gap, we don't care about it
                i += cluster[1]

        # output is by definition oddball data, because our stride is the length
        # of the objects we're dealing with, so if we are here, with a pattern
        # of 1s, if it's not a perfect integer multiple of the stride,
        # something is wrong with it.
        return output, junk
                
        
    def minimize(self, cells):
        
        system_pressure = 0
        raws = {}
        for i, cell in enumerate(cells):
            self.pull_cell_mask(cell)  # Ensure the cell's mask is up-to-date
            print(f"Mask state at the beginning of minimize: {self.bitbuffer.mask.hex()}")
        
            raw = self.bitbuffer[cell.left:cell.right]
            print(f"raw state at the beginning of minimize: {raw.hex()}")
        
            #calculate forces into pressures
            #add to volumetric pressure
            #keep the metaphor loose because this is actually a simple swap algorithm
            known_gaps = []
            #if cell.obj_map is None:
                ###print(f"Cell {cell.label} has no object map, skipping.")
                #continue
            
            left_resistive_force = 0
            right_resistive_force = 0
            center_chances = 0
            pressure = 0
            ##print(f"cell injection queue: {cell.injection_queue}")
            ###print(f"Processing cell {cell.label} with raw data: {raw.hex()}")
            assert cell.left % cell.stride == 0, f"Cell {cell.label} left {cell.left} is not aligned with stride {cell.stride}"
            assert cell.right % cell.stride == 0, f"Cell {cell.label}   right {cell.right} is not aligned with stride {cell.stride}"
            # Trust BitBitSlice’s own bookkeeping instead of recomputing
            # with an implicit 8‑bit alignment.
            self.padding = raw.padding            # bits of alignment filler
            assert self.padding >= 0, "negative padding is impossible"
            right_flat_length = left_flat_length = self.bitbuffer.intceil(self.bitbuffer.intceil(cell.right - cell.left,4)//4, cell.stride)
            assert left_flat_length % cell.stride == 0, f"Left flat length {left_flat_length} for cell {cell.label} is not aligned with stride {cell.stride}"
            assert left_flat_length >= 0, f"Left flat length {left_flat_length} for cell {cell.label} is negative, this should not happen"
            assert right_flat_length >= 0, f"Right flat length {right_flat_length} for cell {cell.label} is negative, this should not happen"
            assert cell.left + left_flat_length <= cell.right, f"Cell {cell.label} left + left_flat_length {cell.left + left_flat_length} exceeds right {cell.right}, this should not happen"
            assert cell.right - right_flat_length >= cell.left, f"Cell {cell.label} right - right_flat_length {cell.right - right_flat_length} is less than left {cell.left}, this should not happen"
            assert left_flat_length + right_flat_length <= (cell.right - cell.left), f"Cell {cell.label} left_flat_length + right_flat_length {left_flat_length + right_flat_length} exceeds right - left {cell.right - cell.left}, this should not happen"
            if cell.left != cell.right:
                # Replace existing calls to self.count_from with BitStreamSearch runs:
                left_pattern, right_pattern = self.bitbuffer.tuplepattern(cell.left, cell.right, left_flat_length, "bi")
                assert len(left_pattern) > 0, f"Left pattern for cell {cell.label} is empty, this should not happen"
                left_gaps = BitStreamSearch.find_aligned_zero_runs(left_pattern, cell.stride)
                left_gaps = [cell.left + gap for gap in left_gaps if gap < left_flat_length]
                right_gaps = BitStreamSearch.find_aligned_zero_runs(right_pattern, cell.stride)
                assert cell.right % cell.stride == 0, f"Cell {cell.label} right {cell.right} is not aligned with stride {cell.stride}"
                right_gaps = [cell.right - (gap + cell.stride) for gap in right_gaps] # +1 is for the fact that just the stride puts us at the end of another stride
                print(f"Cell {cell.label} left pattern: {left_pattern}, right pattern: {right_pattern}")
                #print(f"Cell {cell.label} left gaps: {left_gaps}, right gaps: {right_gaps}")
                #exit()


                for i, pattern in enumerate(left_pattern):
                    assert pattern[1] % cell.stride == 0, f"Left pattern {pattern} for cell {cell.label} is not aligned with stride {cell.stride}"
                for i, pattern in enumerate(right_pattern):
                    assert pattern[1] % cell.stride == 0, f"Right pattern {pattern} for cell {cell.label} is not aligned with stride {cell.stride}"

                # Set leftmost and rightmost based on the patterns
                # If the leftmost/rightmost are not set, use the left/right borders
                if cell.leftmost is None:
                    print(f"Line 362: Cell {cell.label} leftmost is None, setting to left {cell.left}")
                    cell.leftmost = cell.left
                if cell.rightmost is None:
                    print(f"Line 364: Cell {cell.label} rightmost is None, setting to right - 1: {cell.right - 1}")
                    cell.rightmost = cell.right - 1

                # Find the first 1 in the left pattern and right pattern
                for i, pattern in enumerate(left_pattern):
                    if pattern[0] == 1:
                        print(f"Line 370: Cell {cell.label} leftmost before adjustment: {cell.leftmost}, left: {cell.left}, i: {i}, pattern[1]: {pattern[1]}")
                        cell.leftmost = cell.left + (i * cell.stride) + (cell.stride - pattern[1] if pattern[1] < cell.stride else 0)
                        break
                for i, pattern in enumerate(right_pattern):
                    if pattern[0] == 1:
                        print(f"Line 374: Cell {cell.label} rightmost before adjustment: {cell.rightmost}, right: {cell.right}, i: {i}, pattern[1]: {pattern[1]}")
                        cell.rightmost = cell.right - ((i) * cell.stride) - (cell.stride - pattern[1] if pattern[1] < cell.stride else 0) - 1
                        break


                center_gap = (cell.right - cell.left) - left_flat_length - right_flat_length
                center_chances = max(0, center_gap // cell.stride)
                assert center_chances >= 0, f"Cell {cell.label} center chances {center_chances} is negative, check stride and gap calculation"
                #print(f"Cell {cell.label} left pattern: {left_pattern}, right pattern: {right_pattern}, center chances: {center_chances}")
                if len(left_pattern) == 1:
                    cell.compressible = raw[0] == 0
                    if False and cell.compressible:
                        ##print(f"Cell {cell.label} is compressible, setting left/right flags.")
                        
                        known_gaps = list(range((cell.left + cell.stride - 1)//cell.stride * cell.stride, cell.left+left_flat_length, cell.stride))+list(range(((cell.right - cell.left - right_flat_length)+cell.stride-1)//cell.stride * cell.stride, cell.right // cell.stride * cell.stride, cell.stride))
                if False and cell.compressible == 0:
                    pressure = 0
                    cell.l_flags = cell.l_flags | self.LOCK
                    cell.r_flags = cell.r_flags | self.LOCK
                else:
                    left_resistive_force = (len(left_pattern)-1) * cell.l_solvent_permiability
                    right_resistive_force = (len(right_pattern)-1) * cell.r_solvent_permiability
                    pressure += left_resistive_force + right_resistive_force
                    left_neighbor_stride_equiv = (cells[i-1].stride + cell.stride - 1) // cell.stride if i > 0 else 0
                    right_neighbor_stride_equiv = (cells[i+1].stride + cell.stride - 1) // cell.stride if i < len(cells) - 1 else 0
                    if right_neighbor_stride_equiv < len(right_gaps) and right_neighbor_stride_equiv > 0:
                        cell.r_wall_flags = cell.r_wall_flags | self.ELASTIC
                        pressure -= len(right_gaps) // right_neighbor_stride_equiv
                    if left_neighbor_stride_equiv < len(left_gaps) and left_neighbor_stride_equiv > 0:
                        cell.l_wall_flags = cell.l_wall_flags | self.ELASTIC
                        pressure -= len(left_gaps) // left_neighbor_stride_equiv

                
                # Collect known gaps from left and right patterns
                                
                gap_pairs = list(zip(left_gaps, right_gaps)) #this makes tuples we want a straight interleave of left and right gaps
                for left_gap, right_gap in gap_pairs[::-1]:
                    known_gaps.append(left_gap)
                    known_gaps.append(right_gap)

                ##print(f"known gaps for cell {cell.label}: {known_gaps}")
                indices_to_zero = set()
                #if len(known_gaps) == 0:
                    ##print(f"Cell {cell.label} has no known gaps, skipping.")

                #for i, cluster in enumerate(left_pattern):
                #    cell.leftmost = i + cell.left
                #    if cluster[0] == 1:
                #        break

                #for i, cluster in enumerate(right_pattern):
                #    cell.rightmost = cell.right - i
                #    if cluster[0] == 1:
                #        break
                contiguating = False
                if contiguating:
                    if left_resistive_force > self.FORCE_THRESH:
                        spoken_for_slice = { bit
                            for gap in left_gaps
                            for bit in range(gap, gap + cell.stride) }
                        fragmented_slice = set(range(0, left_flat_length, cell.stride)) - spoken_for_slice
                        
                        compacted_strides, junk = self.contiguate(raw[:left_flat_length], left_pattern, fragmented_slice, cell.stride)
                        #print(f"Compacted strides for cell {cell.label}: {compacted_strides}, junk: {junk}")

                        #if junk:
                            ##print(f"Junk data found in cell {cell.label}: {junk}")
                        if cell.label not in self.input_queues:
                            self.input_queues[cell.label] = set()
                        #print(f"Compacted strides for cell {cell.label}: {compacted_strides}")
                        
                        self.input_queues[cell.label].extend(compacted_strides)
                        cell.injection_queue += len(compacted_strides)
                        indices_to_zero.update(fragmented_slice)

                    if right_resistive_force > self.FORCE_THRESH:
                        
                        spoken_for_slice = { bit 
                            for gap in right_gaps
                            for bit in range(gap, gap + cell.stride) }
                        fragmented_slice = set(range(0, right_flat_length, cell.stride)) - spoken_for_slice
                        right_reverse = right_pattern[::-1]
                        compacted_strides, junk = self.contiguate(raw[-right_flat_length::-1], right_reverse, fragmented_slice, cell.stride, rev=True)
                        
                        #if junk:
                            ##print(f"Junk data found in cell {cell.label}: {junk}")
                        if cell.label not in self.input_queues:
                            self.input_queues[cell.label] = set()
                        self.input_queues[cell.label].extend(compacted_strides)
                        cell.injection_queue += len(compacted_strides)
                        indices_to_zero.update(fragmented_slice)

                    
                    raw = self.bitbuffer.stamp(raw, indices_to_zero, 1, 0)
                    #print(f"Cell {cell.label} raw data after stamping: {raw.hex()}")
                    pressure -= len(indices_to_zero) // cell.stride


                
                center_gaps = []
                if center_chances > 0 and cell.injection_queue > 0:
                    
                    center_start_bit = left_flat_length
                    center_end_bit = (cell.right - cell.left) - right_flat_length
                    center_bit_length = center_end_bit - center_start_bit
                    assert center_bit_length >= 0, f"Center bit length {center_bit_length} for cell {cell.label} is negative, this should not happen"
                    assert center_start_bit >= 0, f"Center start bit {center_start_bit} for cell {cell.label} is negative, this should not happen"
                    assert center_end_bit <= len(raw), f"Center end bit {center_end_bit} for cell {cell.label} exceeds raw length {len(raw)}"
                    assert center_start_bit + center_bit_length <= len(raw), f"Center start bit {center_start_bit} + center bit length {center_bit_length} for cell {cell.label} exceeds raw length {len(raw)}"
                    center_alignment_offset = cell.left + left_flat_length

                    #print(f"padding: {padding}")
                    #print(len(trimmed_byte_string)*8, center_bit_length, center_start_bit, cell.left, left_flat_length, cell.right, right_flat_length)
                    #trimmed_byte_string = raw[(left_flat_length + 8 - 1)//8:((cell.right-cell.left)-right_flat_length)//8]
                    #center_alignment_offset = cell.left + left_flat_length
                    ##print(f"byte string: {trimmed_byte_string.hex()}")
                    # since extract_bit_region(...) produced exactly `center_bit_length` bits:
                    #assert len(trimmed_byte_string)*8 == center_bit_length, f"Trimmed byte string length {len(trimmed_byte_string)*8} does not match expected center bit length {center_bit_length}"

                    center_slice = raw[center_start_bit : center_start_bit + center_bit_length]
                    _, center_gaps = self.search.detect_stride_gaps(
                            center_slice, cell.stride, sort_order='center-out')                    
                
                    center_gaps = [gap + center_alignment_offset for gap in center_gaps if gap < center_bit_length]

                # CORRECTED: Use .update() to add elements from the list to the set.
                known_gaps = center_gaps + known_gaps
                for gap in known_gaps:
                    assert gap % cell.stride == 0, f"Gap {gap} in cell {cell.label} is not aligned with stride {cell.stride}"
                    assert gap >= 0, f"Gap {gap} in cell {cell.label} is negative, this should not happen"
                    assert cell.left <= gap < cell.right - cell.stride + 1, f"Gap {gap} in cell {cell.label} is out of bounds, should be between {cell.left} and {cell.right}"
                    
                    
                pressure += cell.injection_queue                    


                relative_gaps = [gap - cell.left for gap in known_gaps]
                gap_pids = []
                best_gaps = known_gaps[:cell.injection_queue] if cell.injection_queue > 0 else []
                if len(best_gaps) > 0:
                    #print(f"Cell {cell.label} best gaps: {best_gaps}")
                    if cell.label not in self.assignable_gaps:
                        self.assignable_gaps[cell.label] = []
                    self.assignable_gaps[cell.label].extend(best_gaps)
                    cell.injection_queue -= len(best_gaps)
                    gap_pids = self.bitbuffer.pid_buffers[cell.label].get_pids(best_gaps)
                    
                

                cell.pressure = pressure
                cell.salinity = len(self.input_queues[cell.label]) if cell.label in self.input_queues else 0
                system_pressure += pressure
                if cell.label in self.input_queues and len(self.input_queues[cell.label]) > 0 and cell.label in self.assignable_gaps and len(self.assignable_gaps[cell.label]) > 0:
                    #print(f"Injecting data into cell {cell.label} with injection queue: {cell.injection_queue} and queue: {self.input_queues[cell.label]}")
                    #print(f"self.input_queues: {self.input_queues}")
                    #print(f"assignable gaps: {self.assignable_gaps}")
                    #print(f"Cell {cell.label} assignable gaps: {self.assignable_gaps[cell.label]}")
                    #print(f"data size: self.bitbuffer.data_size: {self.bitbuffer.data_size}, self.bitbuffer.bittobyte(self.bitbuffer.data_size): {self.bitbuffer.bittobyte(self.bitbuffer.data_size)}, self.bitbuffer.data_size:{self.bitbuffer.data_size}, self.bitbuffer.mask_size:{self.bitbuffer.mask_size}")
                    #print(f"left pattern: {left_pattern}, right pattern: {right_pattern}")
                    relative_consumed_gaps, consumed_gaps, self.input_queues[cell.label] = self.injection(self.input_queues[cell.label], self.assignable_gaps[cell.label], gap_pids, 0)
                    pids = gap_pids[:len(relative_consumed_gaps)]
                    for i, pid in enumerate(pids):
                        gap_idx = self.bitbuffer.get_by_pid(cell.label, pid)
                        print(f"Trying to retrieve data in cell {cell.label} with pid {pid} at gap index {gap_idx}")
                        stride  = cell.stride              # or store this with the PID tuple
                        payload = self.bitbuffer._data_access[gap_idx : gap_idx + stride]
                        print(f"Cell {cell.label} injecting data with pid {pid} at relative gaps {relative_consumed_gaps} and absolute gaps {consumed_gaps}")
                        print(f"Data: {payload.hex()}")
                        for item in self.input_queues[cell.label]:
                            if item[0] == payload and item[1] == stride:
                                i = self.input_queues[cell.label].index(item)
                                break
                        print(f"Original data: {self.input_queues[cell.label][i][0].hex()}")

                        self.input_queues[cell.label].remove((payload, stride)) # remove the consumed gaps from the input queue
                    relative_consumed_gaps = [gap - cell.left for gap in relative_consumed_gaps]



                    # For deterministic checking, it's best to iterate over sorted lists.
                    for relative_gap, absolute_gap in zip(sorted(list(relative_consumed_gaps)), sorted(list(consumed_gaps))):
                        # --- Assertions for Cell Integrity ---
                        # These checks ensure the cell itself is in a valid state before checking the gaps.
                        assert cell.stride > 0, f"FATAL: Cell {cell.label} has zero or negative stride: {cell.stride}"
                        assert cell.right >= cell.left, f"FATAL: Cell {cell.label} has inverted boundaries: left={cell.left}, right={cell.right}"
                        assert (cell.right - cell.left) % cell.stride == 0, f"FATAL: Cell {cell.label} width ({cell.right - cell.left}) is not a multiple of its stride {cell.stride}"

                        # --- Assertions for the Relative Gap ---
                        # A relative gap is an offset from the beginning of the cell's memory.
                        cell_width = cell.right - cell.left
                        assert isinstance(relative_gap, int) and relative_gap >= 0, f"Relative gap {relative_gap} must be a non-negative integer."
                        assert relative_gap % cell.stride == 0, f"Relative gap {relative_gap} in cell {cell.label} is not aligned to its stride {cell.stride}."
                        assert relative_gap < cell_width, f"Relative gap {relative_gap} must be less than the cell's total width of {cell_width}."
                        assert (relative_gap + cell.stride) <= cell_width, f"The end of the relative gap ({relative_gap + cell.stride}) exceeds the cell's width of {cell_width}."

                        # --- Assertions for the Absolute Gap ---
                        # An absolute gap is a direct memory address in the main buffer.
                        assert isinstance(absolute_gap, int) and absolute_gap >= 0, f"Absolute gap {absolute_gap} must be a non-negative integer."
                        assert absolute_gap >= cell.left, f"Absolute gap {absolute_gap} cannot be less than the cell's left boundary of {cell.left}."
                        assert absolute_gap < cell.right, f"Absolute gap {absolute_gap} must be less than the cell's right boundary of {cell.right}."
                        assert (absolute_gap + cell.stride) <= cell.right, f"The end of the absolute gap ({absolute_gap + cell.stride}) exceeds the cell's right boundary of {cell.right}."

                        # --- Core Relational Assertion ---
                        # This is the most critical check: the absolute and relative gaps must correspond perfectly.
                        assert absolute_gap == (cell.left + relative_gap), \
                            f"Fatal mismatch in cell {cell.label}: absolute_gap ({absolute_gap}) does not equal cell.left ({cell.left}) + relative_gap ({relative_gap})."
                    print(f"Line 571: Cell {cell.label} leftmost before adjustment: {cell.leftmost}, left: {cell.left}, relative_consumed_gaps: {relative_consumed_gaps}, consumed_gaps: {consumed_gaps}")
                    cell.leftmost = min(cell.leftmost, min(consumed_gaps)) if consumed_gaps else cell.leftmost
                    assert cell.leftmost >= cell.left, f"Cell {cell.label} leftmost {cell.leftmost} is less than left {cell.left}"
                    assert cell.leftmost < cell.right, f"Cell {cell.label} leftmost {cell.leftmost} is not less than right {cell.right}"
                    assert cell.leftmost % cell.stride == 0, f"Cell {cell.label} leftmost {cell.leftmost} is not aligned with stride {cell.stride}"
                    
                    
                    print(f"Line 578: Cell {cell.label} rightmost before adjustment: {cell.rightmost}, right: {cell.right}, relative_consumed_gaps: {relative_consumed_gaps}, consumed_gaps: {consumed_gaps}")
                    cell.rightmost = max(cell.rightmost, max(consumed_gaps)+cell.stride - 1) if consumed_gaps else cell.rightmost
                    assert cell.rightmost >= cell.left, f"Cell {cell.label} rightmost {cell.rightmost} is less than left {cell.left}"
                    assert cell.rightmost < cell.right, f"Cell {cell.label} rightmost {cell.rightmost} is not less than right {cell.right}"
                    assert (cell.rightmost + 1) % cell.stride == 0, f"Cell {cell.label} rightmost {cell.rightmost} is not aligned with stride {cell.stride}"
                    #this reduction should already occur above
                    #cell.injection_queue -= len(consumed_gaps)
                    ##print(f"Cell {cell.label} processed with raw data: {raw.hex()}")
                    raw = self.bitbuffer.stamp(raw, relative_consumed_gaps, cell.stride, 1) # this is because raw is cell local
                    ##print(f"Cell {cell.label} processed with raw data: {raw.hex()}")
                    #for consumed_gap in consumed_gaps:
                        ##print(f"known_gaps: {known_gaps}")
                        ##print(f"Cell {cell.label} consumed gap at {consumed_gap}")
                        #print(f"assignable_gaps: {self.assignable_gaps}")
                        # this was already removed by .pop in a pass by ref
                        #self.assignable_gaps[cell.label].remove(consumed_gap)
                    ##print(f"Cell {cell.label} processed with raw data: {raw.hex()}")
                
                #assert cell.injection_queue == 0, f"Cell {cell.label} injection queue is not empty after processing: {cell.injection_queue}"
                raws[cell.label] = raw
                #print(f"Cell {cell.label} processed with raw data: {raw.hex()}")

                byte_len = len(cell._buf)                    # same as (cell.len+7)//8
                #print(byte_len)
                
                
                if cell.injection_queue > 0:
                    #print(f"Cell {cell.label} still has injection queue: {cell.injection_queue}")
                    if self.assignable_gaps.get(cell.label):
                        #print(f"Cell {cell.label} has assignable gaps: {self.assignable_gaps[cell.label]}")
                        assert False, "Cell has assignable gaps but injection queue is not empty"

            else:
                print(f"Cell {cell.label} has no left/right distinction, skipping.")
                continue
            self.push_cell_mask(cell)
        
        self.system_pressure = system_pressure
        print(f"Mask state at the end of minimize: {self.bitbuffer.mask.hex()}")
        
        #self.snap_cell_walls(cells, cells)
        
            #else:
                #print(f"Cell {cell.label} has no left/right distinction, skipping.")
            #print(f"after cell {cell.label}, data: {self.data.hex()}")
        
        return system_pressure, raws
    def lcm(self, cells):
        """
        Calculate the least common multiple of all cell strides.
        This is used to ensure that all cells are aligned correctly.
        """
        from math import gcd
        from functools import reduce

        def lcm(a, b):
            return a * b // gcd(a, b)

        return reduce(lcm, (cell.stride for cell in cells if hasattr(cell, 'stride')), 1)
    def bar(self, number=2, width=80):
        for _ in range(number):
            print("#" * width)
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


    def actual_data_hook(self, payload: bytes, dst_bits: int, length_bits: int):
        """
        Write `length_bits` from `payload` directly into our data plane,
        at bit-offset `dst_bits`.  `payload` must be exactly
        ceil(length_bits / 8) bytes long.
        """
        # sanity-check length
        # This sanity check is not in line with bitbit philosophy of 
        # allowing arbitrary payloads, so it is commented out.
        #expected_bytes = (length_bits + 7) // 8
        #assert len(payload) == expected_bytes, (
        #    f"Payload length {len(payload)} != expected for {length_bits} bits ({expected_bytes} bytes)"
        #)
        # direct slice‐assign into BitBitBuffer’s data plane
        self.bitbuffer._data_access[dst_bits : dst_bits + length_bits] = payload

# In cell_pressure.py, inside the Simulator class

    def write_data(self, cell_label: str, payload: bytes):
        """
        Enqueue a (bytes, stride) tuple for later injection.
        Validates that the payload size is correct for the cell's stride.
        """
        # Find the matching cell to get its stride
        try:
            cell = next(c for c in self.cells if c.label == cell_label)
            stride = cell.stride
        except StopIteration:
            raise KeyError(f"No cell with label {cell_label!r}")

        # Calculate the exact number of bytes required for the data plane
        expected_bytes = (stride * self.bitbuffer.bitsforbits + 7) // 8
        
        # Enforce strict size matching
        if len(payload) != expected_bytes:
            raise ValueError(
                f"Payload for cell '{cell_label}' has incorrect size. "
                f"Expected {expected_bytes} bytes for stride {stride}, but got {len(payload)}."
            )

        # Enqueue (payload, stride)
        self.input_queues.setdefault(cell_label, []).append((payload, stride))

        # bump the cell’s injection counter
        cell.injection_queue = getattr(cell, "injection_queue", 0) + 1


    def injection(self, queue, known_gaps, gap_pids, left_offset=0):
        consumed_gaps = []
        relative_consumed_gaps = []
        data_copy = queue.copy()
        for i, (payload, stride) in enumerate(data_copy):
            if len(known_gaps) > 0:
                gap = known_gaps.pop()
                if gap >= self.bitbuffer.data_size:
                    #print(f"Gap {gap} exceeds data bit length {self.bitbuffer.data_size}, skipping")
                    exit()
                relative_consumed_gaps.append(gap)
                gap += left_offset
                consumed_gaps.append(gap)
                #queue.remove((payload, stride))
                self.pid_list.append((gap, gap_pids[i]))
                #print(f"Injecting data at gap {gap} with stride {stride}")
                #print(f"data size: len(self.bitbuffer.data): {len(self.bitbuffer.data)}, data_bit_length:{self.bitbuffer.data_size}, mask_bit_length:{self.bitbuffer.mask_size}")
                #print(f"data in hex: {self.bitbuffer.data.hex()}")
                assert stride == len(payload) / self.bitbuffer.bitsforbits * 8, (
                    f"Stride {stride} does not match payload length {len(payload)}"
                )
                gap_data = self.bitbuffer._data_access[gap: gap + stride]
                
                # Replace actual_data_hook call using slice assignment
                self.actual_data_hook(payload, gap, stride)
            else:
                break
        return relative_consumed_gaps, consumed_gaps, queue

    def step(self, cells):
        # Coordinate one simulation step
        sp, mask = self.minimize(cells)
        self.evolution_tick(cells)
        return sp, mask

# ====== Begin fast + focused tests ======
import random, pytest

# ---------- 1.  smoke‑check every supported stride ----------
@pytest.mark.parametrize(
    "stride",
    [1, 2, 3, 4, 5, 7, 8, 9, 11, 13, 16, 17, 19, 23,
     29, 31, 64, 128, 256, 512, 1024]
)
def test_simulation_stride_basic(stride):
    """
    One‑step sanity check per stride.  Catches
    obvious alignment / boundary mishaps fast.
    """
    random.seed(0)
    CELL_COUNT = random.randint(1, 5)          # ≤5 keeps it snappy
    WIDTH      = stride * 8                    # 8×stride bits per cell
    cells = [Cell(stride=stride,
                  left=i * WIDTH,
                  len=WIDTH,
                  right=i * WIDTH + WIDTH)
             for i in range(CELL_COUNT)]

    sim = Simulator(cells)
    sp, _ = sim.step(cells)                    # **single** step   :contentReference[oaicite:2]{index=2}
    assert isinstance(sp, (int, float))

    # quick mask‑length sanity
    for c in cells:
        assert len(sim.get_cell_mask(c)) == c.right - c.left
# ---------- 2.  deep injection stress at a single odd prime stride ----------
# In cell_pressure.py

def test_injection_mixed_prime7():
    """
    Simplified public injection test: deposit payloads using write_data()
    and then run a few simulation ticks.
    """
    stride = 7
    CELL_COUNT = 3
    WIDTH = stride * 20
    cells = [Cell(stride=stride,
                  left=i * WIDTH,
                  len=WIDTH,
                  right=i * WIDTH + WIDTH,
                  label=f"cell{i}")
             for i in range(CELL_COUNT)]

    sim = Simulator(cells)

    # Calculate the correct data payload size in bytes.
    # This must match the space allocated in the data plane for 'stride' mask bits.
    data_bytes_per_stride = (stride * sim.bitbuffer.bitsforbits + 7) // 8

    # Create payloads with the correct, validated size.
    payloads = [
        b'\xff' * data_bytes_per_stride,
        b'\xaa' * data_bytes_per_stride,
        b'\x55' * data_bytes_per_stride
    ]

    # Deposit payloads to cell0 via the new public write command.
    for p in payloads:
        sim.write_data(cells[0].label, p)
    
    # Drive several simulation ticks.
    for _ in range(10):
        sp, _ = sim.step(cells)
    
    sim.print_system(cells)
    # In a successful injection cycle, the injection queue should be empty.
    assert cells[0].injection_queue == 0

# Add 'import os' to the top of cell_pressure.py
import os


def test_sustained_random_injection():
    """
    A more rigorous stress test involving sustained, randomized injections
    across multiple cells with different strides over many simulation steps.
    """
    print("\n--- Starting Sustained Random Injection Stress Test ---")
    
    # 1. Define test parameters
    # Using different, prime strides helps stress LCM and alignment logic
    CELL_STRIDES = [7, 11, 13, 17]
    CELL_COUNT = len(CELL_STRIDES)
    INITIAL_WIDTH_PER_CELL = 300  # Initial bit-width for each cell
    SIMULATION_STEPS = 50         # Total number of simulation steps to run
    WRITES_PER_STEP = 50           # Number of random write operations to queue each step



    INITIAL_TARGET = 300          # keep the same “about‑300‑bits” idea

    cells = [
        Cell(
            stride=s,
            left=i * BitBitBuffer._intceil(INITIAL_TARGET, s),
            len =BitBitBuffer._intceil(INITIAL_TARGET, s),
            right=(i + 1) * BitBitBuffer._intceil(INITIAL_TARGET, s),
            label=f"cell_{s}",
        )
        for i, s in enumerate(CELL_STRIDES)
    ]

    sim = Simulator(cells)
    print("Initial System State:")
    

    # 3. Main simulation loop
    for step in range(SIMULATION_STEPS):
        print(f"\n[Step {step + 1}/{SIMULATION_STEPS}] Queuing {WRITES_PER_STEP} new data chunks...")
        
        # 4. In each step, queue multiple new writes to random cells
        for _ in range(random.randint(1, WRITES_PER_STEP)):
            # Randomly select a target cell
            target_cell = random.choice(cells)
            
            # Generate a correctly-sized payload of random bytes
            # os.urandom is great for creating unpredictable data
            data_bytes = (target_cell.stride * sim.bitbuffer.bitsforbits + 7) // 8
            payload = os.urandom(data_bytes)
            
            # Write the data. The write_data method will validate the payload size.
            sim.write_data(target_cell.label, payload)

        # 5. Execute one full simulation step to process the queue and rebalance
        print("Stepping simulation to process queue and rebalance memory...")
        sim.step(cells)
        
    # 6. Final assertions after the test loop completes
    print("\n--- Test Complete. Final Assertions ---")
    total_remaining_items = 0
    for cell in cells:
        # The per-cell counter should be zero
        assert cell.injection_queue == 0, (
            f"Error: Cell {cell.label} has a non-empty injection queue "
            f"({cell.injection_queue}) after the test."
        )
        # The central queue for that cell should also be empty
        remaining_in_queue = len(sim.input_queues.get(cell.label, []))
        assert remaining_in_queue == 0, (
            f"Error: Simulator input queue for {cell.label} still contains "
            f"{remaining_in_queue} items."
        )
        total_remaining_items += remaining_in_queue
    
    assert total_remaining_items == 0, "The global input queue is not fully drained."

    print("✅ PASSED: All injection queues are empty and all data was processed.")

if __name__ == '__main__':
    test_sustained_random_injection()
    #test_injection_mixed_prime7()
    #test_simulation_stride_basic(7)
#    pytest.main([__file__])
# ====== End new tests ======