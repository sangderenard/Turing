from ..bitstream_search import BitStreamSearch

def minimize(self, cells):
    system_pressure = 0
    raws = {}

    print("Cell stat table")
    print(f"{'Index':<5} {'Label':<10} {'Left':<10} {'Right':<10} {'Raw':<10}")
    print("=" * 55)
    self.run_balanced_saline_sim()
    for i, cell in enumerate(cells):
        self.pull_cell_mask(cell)  # Ensure the cell's mask is up-to-date
        #print(f"Mask state at the beginning of minimize: {self.bitbuffer.mask.hex()}")
        
        
        raw = self.bitbuffer[cell.left:cell.right]
        assert not (raw is None), f"Line 10: Cell {cell.label} raw is None or empty, sort this out before calling"

        #print(f"raw state at the beginning of minimize: {raw.hex()}")
        known_gaps = []
        left_resistive_force = 0
        right_resistive_force = 0
        center_chances = 0
        pressure = 0
        assert cell.left % cell.stride == 0, f"Cell {cell.label} left {cell.left} is not aligned with stride {cell.stride}"
        assert cell.right % cell.stride == 0, f"Cell {cell.label}   right {cell.right} is not aligned with stride {cell.stride}"
        self.padding = raw.padding            # bits of alignment filler
        assert self.padding >= 0, "negative padding is impossible"
        right_flat_length = left_flat_length = self.bitbuffer.intceil(self.bitbuffer.intceil(cell.right - cell.left,4)//4, cell.stride)
        if right_flat_length + left_flat_length > cell.right-cell.left:
            right_flat_length = (right_flat_length - cell.stride + 1)//cell.stride*cell.stride
            left_flat_length = (left_flat_length + cell.stride - 1)//cell.stride*cell.stride
        assert left_flat_length % cell.stride == 0, f"Left flat length {left_flat_length} for cell {cell.label} is not aligned with stride {cell.stride}"
        assert left_flat_length >= 0, f"Left flat length {left_flat_length} for cell {cell.label} is negative, this should not happen"
        assert right_flat_length >= 0, f"Right flat length {right_flat_length} for cell {cell.label} is negative, this should not happen"
        assert cell.left + left_flat_length <= cell.right, f"Cell {cell.label} left + left_flat_length {cell.left + left_flat_length} exceeds right {cell.right}, this should not happen"
        assert cell.right - right_flat_length >= cell.left, f"Cell {cell.label} right - right_flat_length {cell.right - right_flat_length} is less than left {cell.left}, this should not happen"
        assert left_flat_length + right_flat_length <= (cell.right - cell.left), f"Cell {cell.label} left_flat_length + right_flat_length {left_flat_length + right_flat_length} exceeds right - left {cell.right - cell.left}, this should not happen"
        if cell.left != cell.right:
            print(f"Cell {cell.label} left: {cell.left}, right: {cell.right}, stride: {cell.stride}, raw length: {len(raw)}")
            left_pattern, right_pattern = self.bitbuffer.tuplepattern(cell.left, cell.right, left_flat_length, "bi")
            print(f"Cell {cell.label} left pattern: {left_pattern}, right pattern: {right_pattern}")
            if not (len(left_pattern) + len(right_pattern) == 0):
                
                assert len(left_pattern) > 0, f"Left pattern for cell {cell.label} is empty, this should not happen"
                left_gaps = BitStreamSearch.find_aligned_zero_runs(left_pattern, cell.stride)
                left_gaps = [cell.left + gap for gap in left_gaps if gap < left_flat_length]
                right_gaps = BitStreamSearch.find_aligned_zero_runs(right_pattern, cell.stride)
                assert cell.right % cell.stride == 0, f"Cell {cell.label} right {cell.right} is not aligned with stride {cell.stride}"
                right_gaps = [cell.right - (gap + cell.stride) for gap in right_gaps] # +1 is for the fact that just the stride puts us at the end of another stride
                print(f"Cell {cell.label} left pattern: {left_pattern}, right pattern: {right_pattern}")
                print(f"left gaps: {left_gaps}, right gaps: {right_gaps}")
                for i, pattern in enumerate(left_pattern):
                    assert pattern[1] % cell.stride == 0, f"Left pattern {pattern} for cell {cell.label} is not aligned with stride {cell.stride}"
                for i, pattern in enumerate(right_pattern):
                    assert pattern[1] % cell.stride == 0, f"Right pattern {pattern} for cell {cell.label} is not aligned with stride {cell.stride}"
                if cell.leftmost is None:
                    print(f"Line 362: Cell {cell.label} leftmost is None, setting to left {cell.left}")
                    cell.leftmost = cell.left
                if cell.rightmost is None:
                    print(f"Line 364: Cell {cell.label} rightmost is None, setting to right - 1: {cell.right - 1}")
                    cell.rightmost = cell.right - 1
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
                if len(left_pattern) == 1:
                    cell.compressible = raw[0] == 0
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
                gap_pairs = list(zip(left_gaps, right_gaps))
                for left_gap, right_gap in gap_pairs[::-1]:
                    known_gaps.append(left_gap)
                    known_gaps.append(right_gap)
                print(f"known_gaps: {known_gaps}")
                indices_to_zero = set()
                contiguating = False
                if contiguating:
                    if left_resistive_force > self.FORCE_THRESH:
                        spoken_for_slice = { bit for gap in left_gaps for bit in range(gap, gap + cell.stride) }
                        fragmented_slice = set(range(0, left_flat_length, cell.stride)) - spoken_for_slice
                        compacted_strides, junk = self.contiguate(raw[:left_flat_length], left_pattern, fragmented_slice, cell.stride)
                        if cell.label not in self.input_queues:
                            self.input_queues[cell.label] = set()
                        self.input_queues[cell.label].extend(compacted_strides)
                        cell.injection_queue += len(compacted_strides)
                        indices_to_zero.update(fragmented_slice)
                    if right_resistive_force > self.FORCE_THRESH:
                        spoken_for_slice = { bit for gap in right_gaps for bit in range(gap, gap + cell.stride) }
                        fragmented_slice = set(range(0, right_flat_length, cell.stride)) - spoken_for_slice
                        right_reverse = right_pattern[::-1]
                        compacted_strides, junk = self.contiguate(raw[-right_flat_length::-1], right_reverse, fragmented_slice, cell.stride, rev=True)
                        if cell.label not in self.input_queues:
                            self.input_queues[cell.label] = set()
                        self.input_queues[cell.label].extend(compacted_strides)
                        cell.injection_queue += len(compacted_strides)
                        indices_to_zero.update(fragmented_slice)
                    raw = self.bitbuffer.stamp(raw, indices_to_zero, 1, 0)
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
                    center_slice = raw[center_start_bit : center_start_bit + center_bit_length]
                    _, center_gaps = self.search.detect_stride_gaps(center_slice, cell.stride, sort_order='center-out')
                    center_gaps = [gap + center_alignment_offset for gap in center_gaps if gap < center_bit_length]
                print(f"Center gaps for cell {cell.label}: {center_gaps}")
                print(f"Known gaps for cell {cell.label}: {known_gaps}")
                known_gaps = center_gaps + known_gaps
                for gap in known_gaps:
                    assert gap % cell.stride == 0, f"Gap {gap} in cell {cell.label} is not aligned with stride {cell.stride}"
                    assert gap >= 0, f"Gap {gap} in cell {cell.label} is negative, this should not happen"
                    assert cell.left <= gap < cell.right - cell.stride + 1, f"Gap {gap} in cell {cell.label} is out of bounds, should be between {cell.left} and {cell.right}"
                pressure += cell.injection_queue
                gap_pids = []
                best_gaps = known_gaps[:cell.injection_queue] if cell.injection_queue > 0 else []
                print(f"Best gaps for cell {cell.label}: {best_gaps}")
                if len(best_gaps) > 0:
                    if cell.label not in self.assignable_gaps:
                        self.assignable_gaps[cell.label] = []
                    self.assignable_gaps[cell.label].extend(best_gaps)
                    cell.injection_queue -= len(best_gaps)
                    # PIDBuffer expects absolute bit positions; pass the system-level
                    # gaps directly rather than cell-relative offsets.
                    
                    gap_pids = self.bitbuffer.pid_buffers[cell.label].get_pids(best_gaps)
                cell.pressure = pressure
                if cell.label in self.input_queues:
                    cell.salinity += sum(stride for _, stride in self.input_queues[cell.label])
                print(f"Checking cell {cell.label}:")
                system_pressure += pressure
                print(f"input_queues: {self.input_queues}")

                if cell.label in self.input_queues and len(self.input_queues[cell.label]) > 0 and cell.label in self.assignable_gaps and len(self.assignable_gaps[cell.label]) > 0:
                    print(f"Cell {cell.label} has input queues and assignable gaps, proceeding with injection.")
                    original_queue = self.input_queues[cell.label].copy()
                    
                    relative_consumed_gaps, consumed_gaps, queue = self.injection(
                        self.input_queues[cell.label], self.assignable_gaps[cell.label], gap_pids, 0
                    )
                    self.input_queues[cell.label] = queue
                    pids = gap_pids[:len(relative_consumed_gaps)]
                    for idx, pid in enumerate(pids):
                        gap_idx = self.bitbuffer.get_by_pid(cell.label, pid)
                        print(f"Trying to retrieve data in cell {cell.label} with pid {pid} at gap index {gap_idx}")
                        stride = cell.stride
                        data_payload = self.bitbuffer._data_access[gap_idx : gap_idx + stride]
                        print(f"Cell {cell.label} injecting data with pid {pid} at relative gaps {relative_consumed_gaps} and absolute gaps {consumed_gaps}")
                        print(f"Data: {data_payload.hex()}")
                        orig_payload, _ = original_queue[idx]
                        print(f"Original data: {orig_payload.hex()}")
                        queue.pop(0)
                    relative_consumed_gaps = [gap - cell.left for gap in relative_consumed_gaps]
                    for relative_gap, absolute_gap in zip(sorted(list(relative_consumed_gaps)), sorted(list(consumed_gaps))):
                        assert cell.stride > 0, f"FATAL: Cell {cell.label} has zero or negative stride: {cell.stride}"
                        assert cell.right >= cell.left, f"FATAL: Cell {cell.label} has inverted boundaries: left={cell.left}, right={cell.right}"
                        assert (cell.right - cell.left) % cell.stride == 0, f"FATAL: Cell {cell.label} width ({cell.right - cell.left}) is not a multiple of its stride {cell.stride}"
                        cell_width = cell.right - cell.left
                        assert isinstance(relative_gap, int) and relative_gap >= 0, f"Relative gap {relative_gap} must be a non-negative integer."
                        assert relative_gap % cell.stride == 0, f"Relative gap {relative_gap} in cell {cell.label} is not aligned to its stride {cell.stride}."
                        assert relative_gap < cell_width, f"Relative gap {relative_gap} must be less than the cell's total width of {cell_width}."
                        assert (relative_gap + cell.stride) <= cell_width, f"The end of the relative gap ({relative_gap + cell.stride}) exceeds the cell's width of {cell_width}."
                        assert isinstance(absolute_gap, int) and absolute_gap >= 0, f"Absolute gap {absolute_gap} must be a non-negative integer."
                        assert absolute_gap >= cell.left, f"Absolute gap {absolute_gap} cannot be less than the cell's left boundary of {cell.left}."
                        assert absolute_gap < cell.right, f"Absolute gap {absolute_gap} must be less than the cell's right boundary of {cell.right}."
                        assert (absolute_gap + cell.stride) <= cell.right, f"The end of the absolute gap ({absolute_gap + cell.stride}) exceeds the cell's right boundary of {cell.right}."
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
                    raw = self.bitbuffer.stamp(raw, relative_consumed_gaps, cell.stride, 1)
                raws[cell.label] = raw
                byte_len = len(cell._buf)
                if cell.injection_queue > 0:
                    if self.assignable_gaps.get(cell.label):
                        assert False, "Cell has assignable gaps but injection queue is not empty"
        else:
            if cell.label in self.input_queues:
                cell.salinity += sum(stride for _, stride in self.input_queues[cell.label])
                print(f"Cell {cell.label} has no left/right distinction, skipping.")
            continue
        self.push_cell_mask(cell)
    self.system_pressure = system_pressure
    print(f"Mask state at the end of minimize: {self.bitbuffer.mask.hex()}")
    return system_pressure, raws
