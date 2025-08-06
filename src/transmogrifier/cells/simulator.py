from typing import Union
from sympy import Integer
from .cell_consts import Cell, MASK_BITS_TO_DATA_BITS, CELL_COUNT, RIGHT_WALL, LEFT_WALL
from .salinepressure import SalineHydraulicSystem
from .bitbitbuffer import BitBitBuffer, CellProposal
from .bitstream_search import BitStreamSearch
from .cell_walls import snap_cell_walls, build_metadata, expand
import math
import random
import os

class Simulator:
    FORCE_THRESH = .5
    LOCK = 0x1
    ELASTIC = 0x2
    snap_cell_walls = snap_cell_walls
    build_metadata = build_metadata
    expand = expand

    def __init__(self, cells):
        self.assignable_gaps = {}
        self.pid_list = []
        self.cells = cells
        self.input_queues = {}
        self.system_pressure = 0
        self.elastic_coeff = 0.1
        self.system_lcm   = self.lcm(cells)
        required_end = max(c.right for c in cells)
        mask_size    = BitBitBuffer._intceil(required_end, self.system_lcm)
        self.bitbuffer = BitBitBuffer(mask_size=mask_size, caster=bytes,
                                    bitsforbits=MASK_BITS_TO_DATA_BITS)
        self.bitbuffer.register_pid_buffer(cells=self.cells)
        self.locked_data_regions = []
        self.search = BitStreamSearch()
        self.s_exprs = [Integer(0) for _ in range(CELL_COUNT)]
        self.p_exprs = [Integer(1) for _ in range(CELL_COUNT)]
        self.engine = None
        self.fractions = None
        # Call ``run_saline_sim`` to enable the full saline pressure model.

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------
    def print_system(self, width=80):
        """Print the current memory layout using :mod:`visualization`."""
        from .visualization import print_system as _print_system

        _print_system(self, width)

    def bar(self, number=2, width=80):
        from .visualization import bar as _bar

        _bar(self, number, width)

    def get_cell_mask(self, cell: Cell) -> bytearray:
        return self.bitbuffer[cell.left:cell.right]

    def set_cell_mask(self, cell: Cell, mask: bytearray) -> None:
        self.bitbuffer[cell.left:cell.right] = mask

    def pull_cell_mask(self, cell):
        cell._buf = self.get_cell_mask(cell)
    def push_cell_mask(self, cell):
        self.set_cell_mask(cell, cell._buf)

    def evolution_tick(self, cells):
        update_s_p_expressions(self, cells)
        proposals = []
        fractions = equilibrium_fracs(self, 0.0)
        total_space = self.bitbuffer.mask_size
        for cell, frac in zip(cells, fractions):
            new_width = max(self.bitbuffer.intceil(cell.salinity,cell.stride), self.bitbuffer.intceil(int(total_space * frac), cell.stride))
            assert new_width % cell.stride == 0
            assert cell.stride > 0
            proposal = CellProposal(cell)
            proposals.append(proposal)
        self.snap_cell_walls(cells, proposals)
        self.print_system()
        return proposals

    def write_data(self, cell_label: str, payload: bytes):
        try:
            cell = next(c for c in self.cells if c.label == cell_label)
            stride = cell.stride
        except StopIteration:
            raise KeyError(f"No cell with label {cell_label!r}")
        expected_bytes = (stride * self.bitbuffer.bitsforbits + 7) // 8
        if len(payload) != expected_bytes:
            raise ValueError(
                f"Payload for cell '{cell_label}' has incorrect size. "
                f"Expected {expected_bytes} bytes for stride {stride}, but got {len(payload)}."
            )
        self.input_queues.setdefault(cell_label, []).append((payload, stride))
        cell.injection_queue = getattr(cell, "injection_queue", 0) + 1

    def injection(self, queue, known_gaps, gap_pids, left_offset=0):
        consumed_gaps = []
        relative_consumed_gaps = []
        data_copy = queue.copy()
        for i, (payload, stride) in enumerate(data_copy):
            if len(known_gaps) > 0:
                gap = known_gaps.pop()
                if gap >= self.bitbuffer.data_size:
                    exit()
                relative_consumed_gaps.append(gap)
                gap += left_offset
                consumed_gaps.append(gap)
                self.pid_list.append((gap, gap_pids[i]))
                assert stride == len(payload) / self.bitbuffer.bitsforbits * 8
                self.actual_data_hook(payload, gap, stride)
            else:
                break
        return relative_consumed_gaps, consumed_gaps, queue

    def actual_data_hook(self, payload: bytes, dst_bits: int, length_bits: int):
        self.bitbuffer._data_access[dst_bits : dst_bits + length_bits] = payload

    def step(self, cells):
        sp, mask = self.minimize(cells)
        self.evolution_tick(cells)
        return sp, mask

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
                    # Copy the original input queue to track consumed payloads
                    original_queue = self.input_queues[cell.label].copy()
                    # Perform injection, updating the queue returned
                    relative_consumed_gaps, consumed_gaps, queue = self.injection(
                        self.input_queues[cell.label], self.assignable_gaps[cell.label], gap_pids, 0
                    )
                    # Store back the possibly updated queue
                    self.input_queues[cell.label] = queue
                    pids = gap_pids[:len(relative_consumed_gaps)]
                    # Remove consumed entries by their original queue index
                    for idx, pid in enumerate(pids):
                        gap_idx = self.bitbuffer.get_by_pid(cell.label, pid)
                        print(f"Trying to retrieve data in cell {cell.label} with pid {pid} at gap index {gap_idx}")
                        stride = cell.stride
                        data_payload = self.bitbuffer._data_access[gap_idx : gap_idx + stride]
                        print(f"Cell {cell.label} injecting data with pid {pid} at relative gaps {relative_consumed_gaps} and absolute gaps {consumed_gaps}")
                        print(f"Data: {data_payload.hex()}")
                        # Retrieve and print original payload from the copied queue
                        orig_payload, _ = original_queue[idx]
                        print(f"Original data: {orig_payload.hex()}")
                        # Remove the first entry from the live queue
                        queue.pop(0)
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
        from math import gcd
        from functools import reduce
        def lcm(a, b):
            return a * b // gcd(a, b)
        return reduce(lcm, (cell.stride for cell in cells if hasattr(cell, 'stride')), 1)


# Attach pressure-model helpers
from .pressure_model import run_saline_sim, update_s_p_expressions, equilibrium_fracs

Simulator.run_saline_sim = run_saline_sim
Simulator.update_s_p_expressions = update_s_p_expressions



def _print_system_basic(self, width: int = 80):  # pragma: no cover - debug helper
    from .visualization import print_system as _print_system
    _print_system(self, width)


Simulator.print_system = _print_system_basic

def main():
    """
    A simple demonstration of the cell simulation.
    This function is intended to be run as a script.
    """
    from .cell_consts import Cell
    cells = [
        Cell(label='A', left=0, right=16, stride=4),
        Cell(label='B', left=16, right=32, stride=4),
    ]
    sim = Simulator(cells)

    for _ in range(5):
        sim.write_data('A', b'\\xde\\xad')
        sim.write_data('B', b'\\xbe\\xef')

        sim.step(cells)
        sim.print_system()

if __name__ == "__main__":
    main()
