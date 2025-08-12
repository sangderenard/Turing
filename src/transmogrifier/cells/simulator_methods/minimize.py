from itertools import zip_longest
from ..bitstream_search import BitStreamSearch
from .logutil import logger, analysis_logger

def minimize(self, cells, verify=False):
    system_pressure = 0
    raws = {}

    logger.info("Cell stat table")
    logger.info(f"{'Index':<5} {'Label':<10} {'Left':<10} {'Right':<10} {'Raw':<10}")
    logger.info("=" * 55)
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
        # stride-clean, slot-based split of the cell width into left/right/center
        width  = cell.right - cell.left
        slots  = width // cell.stride  # width is stride-aligned by asserts above
        left_slots   = slots // 4
        right_slots  = slots // 4
        center_chances = slots - left_slots - right_slots
        left_flat_length  = left_slots  * cell.stride
        right_flat_length = right_slots * cell.stride
        # Define regions and take BEFORE snapshot for analysis
        def _region_bounds():
            whole = (cell.left, cell.right)
            left  = (cell.left, cell.left + left_flat_length)
            right = (cell.right - right_flat_length, cell.right)
            center = (left[1], right[0])
            return {"whole": whole, "left": left, "center": center, "right": right}

        regions = _region_bounds()

        def _build_snapshot(raw_bits):
            snap = {}
            pb = self.bitbuffer.pid_buffers.get(cell.label) if hasattr(self.bitbuffer, 'pid_buffers') else None
            for name, (start, end) in regions.items():
                length = max(0, end - start)
                # raw_bits is cell-local; map absolute to relative
                rel_start = max(0, start - cell.left)
                rslc = raw_bits[rel_start:rel_start+length] if length > 0 else raw_bits[0:0]
                try:
                    mask_hex = rslc.hex()
                except Exception:
                    mask_hex = "<nohex>"

                # Count mask 1s at stride anchors in this region
                mask_count = 0
                anchors = []
                if length > 0 and cell.stride > 0:
                    a = ((start + cell.stride - 1)//cell.stride) * cell.stride
                    while a < end:
                        anchors.append(a)
                        try:
                            mask_count += int(self.bitbuffer[a])
                        except Exception:
                            pass
                        a += cell.stride

                # PID mask/data sampling
                pid_mask_count = 0
                samples = []
                if pb is not None and hasattr(pb, 'pids') and anchors:
                    for a in anchors:
                        # Translate absolute bit position 'a' to PID mask index without stride rounding
                        try:
                            idx = (a - pb.domain_left) // max(1, getattr(pb, 'domain_stride', cell.stride))
                            bit_on = int(pb.pids[idx]) if 0 <= idx < pb.pids.mask_size else 0
                        except Exception:
                            bit_on = 0
                        pid_mask_count += bit_on
                        if bit_on and len(samples) < 3:
                            # Sample the main data payload at this anchor without creating new PIDs
                            try:
                                payload = self.bitbuffer._data_access[a : a + cell.stride]
                                samples.append(payload.hex())
                            except Exception:
                                # Best-effort sampling; continue on failure
                                pass

                snap[name] = {
                    "range": (start, end),
                    "mask_hex": mask_hex,
                    "mask_anchors_set": mask_count,
                    "pid_mask_anchors_set": pid_mask_count,
                    "pid_data_samples": samples,
                }
            return snap

        before_snap = _build_snapshot(raw)
        analysis_logger.analysis(f"Cell {cell.label} SNAPSHOT BEFORE → {before_snap}")
        assert left_flat_length % cell.stride == 0, f"Left flat length {left_flat_length} for cell {cell.label} is not aligned with stride {cell.stride}"
        assert left_flat_length >= 0, f"Left flat length {left_flat_length} for cell {cell.label} is negative, this should not happen"
        assert right_flat_length >= 0, f"Right flat length {right_flat_length} for cell {cell.label} is negative, this should not happen"
        assert cell.left + left_flat_length <= cell.right, f"Cell {cell.label} left + left_flat_length {cell.left + left_flat_length} exceeds right {cell.right}, this should not happen"
        assert cell.right - right_flat_length >= cell.left, f"Cell {cell.label} right - right_flat_length {cell.right - right_flat_length} is less than left {cell.left}, this should not happen"
        assert left_flat_length + right_flat_length <= (cell.right - cell.left), f"Cell {cell.label} left_flat_length + right_flat_length {left_flat_length + right_flat_length} exceeds right - left {cell.right - cell.left}, this should not happen"
        if cell.left != cell.right:
            logger.debug(f"Cell {cell.label} left: {cell.left}, right: {cell.right}, stride: {cell.stride}, raw length: {len(raw)}")
            left_pattern, right_pattern = self.bitbuffer.tuplepattern(cell.left, cell.right, left_flat_length, "bi")
            logger.debug(f"Cell {cell.label} left pattern: {left_pattern}, right pattern: {right_pattern}")
            
                
            
            left_gaps = BitStreamSearch.find_aligned_zero_runs(left_pattern, cell.stride)
            left_gaps = [cell.left + gap for gap in left_gaps if gap < left_flat_length]
            right_gaps = BitStreamSearch.find_aligned_zero_runs(right_pattern, cell.stride)
            assert cell.right % cell.stride == 0, f"Cell {cell.label} right {cell.right} is not aligned with stride {cell.stride}"
            right_gaps = [cell.right - (gap + cell.stride) for gap in right_gaps] # +1 is for the fact that just the stride puts us at the end of another stride
            logger.debug(f"Cell {cell.label} left pattern: {left_pattern}, right pattern: {right_pattern}")
            logger.debug(f"left gaps: {left_gaps}, right gaps: {right_gaps}")
            # Analysis snapshot of initial gap discovery
            analysis_logger.analysis(
                    f"Cell {cell.label} gaps — left: {left_gaps}, right: {right_gaps}"
                )
            for i, pattern in enumerate(left_pattern):
                assert pattern[1] % cell.stride == 0, f"Left pattern {pattern} for cell {cell.label} is not aligned with stride {cell.stride}"
            for i, pattern in enumerate(right_pattern):
                assert pattern[1] % cell.stride == 0, f"Right pattern {pattern} for cell {cell.label} is not aligned with stride {cell.stride}"
            if cell.leftmost is None:
                logger.debug(f"Line 362: Cell {cell.label} leftmost is None, setting to left {cell.left}")
                cell.leftmost = cell.left
            if cell.rightmost is None:
                logger.debug(f"Line 364: Cell {cell.label} rightmost is None, setting to right - 1: {cell.right - 1}")
                cell.rightmost = cell.right - 1
            for i, pattern in enumerate(left_pattern):
                if pattern[0] == 1:
                    logger.debug(f"Line 370: Cell {cell.label} leftmost before adjustment: {cell.leftmost}, left: {cell.left}, i: {i}, pattern[1]: {pattern[1]}")
                    cell.leftmost = cell.left + (i * cell.stride) + (cell.stride - pattern[1] if pattern[1] < cell.stride else 0)
                    break
            for i, pattern in enumerate(right_pattern):
                if pattern[0] == 1:
                    logger.debug(f"Line 374: Cell {cell.label} rightmost before adjustment: {cell.rightmost}, right: {cell.right}, i: {i}, pattern[1]: {pattern[1]}")
                    cell.rightmost = cell.right - ((i) * cell.stride) - (cell.stride - pattern[1] if pattern[1] < cell.stride else 0) - 1
                    break

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
                

            def interleave_keep_leftovers(a, b):
                out = []
                for x, y in zip_longest(a, b, fillvalue=None):
                    if x is not None: out.append(x)
                    if y is not None: out.append(y)
                return out

            known_gaps = interleave_keep_leftovers(left_gaps, right_gaps)
            # then known = center_gaps + known  (if you still want center-first)

            logger.debug(f"known_gaps: {known_gaps}")
            analysis_logger.analysis(
                f"Cell {cell.label} gaps — known(before center): {known_gaps}"
            )
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
            logger.debug(f"Center gaps for cell {cell.label}: {center_gaps}")
            logger.debug(f"Known gaps for cell {cell.label}: {known_gaps}")
            analysis_logger.analysis(
                f"Cell {cell.label} gaps — center: {center_gaps}"
            )
            known_gaps = center_gaps + known_gaps
            analysis_logger.analysis(
                f"Cell {cell.label} gaps — known(after center): {known_gaps}"
            )
            for gap in known_gaps:
                assert gap % cell.stride == 0, f"Gap {gap} in cell {cell.label} is not aligned with stride {cell.stride}"
                assert gap >= 0, f"Gap {gap} in cell {cell.label} is negative, this should not happen"
                assert cell.left <= gap < cell.right - cell.stride + 1, f"Gap {gap} in cell {cell.label} is out of bounds, should be between {cell.left} and {cell.right}"
            pressure += cell.injection_queue
            gap_pids = []
            best_gaps = known_gaps[:cell.injection_queue] if cell.injection_queue > 0 else []
            logger.debug(f"Best gaps for cell {cell.label}: {best_gaps}")
            analysis_logger.analysis(
                f"Cell {cell.label} gaps — assignable: {best_gaps} (queue={cell.injection_queue})"
            )
            if len(best_gaps) > 0:
                if cell.label not in self.assignable_gaps:
                    self.assignable_gaps[cell.label] = []
                self.assignable_gaps[cell.label].extend(best_gaps)
                cell.injection_queue -= len(best_gaps)
                # PIDBuffer expects absolute bit positions; pass the system-level
                # gaps directly rather than cell-relative offsets.
                    
                gap_pids = self.bitbuffer.pid_buffers[cell.label].get_pids(best_gaps)
                analysis_logger.analysis(
                    f"Cell {cell.label} pids — gap_pids: {gap_pids} for assignable {best_gaps}"
                )
            cell.pressure = pressure
            if cell.label in self.input_queues:
                cell.salinity += sum(stride for _, stride in self.input_queues[cell.label])
            logger.debug(f"Checking cell {cell.label}:")
            system_pressure += pressure
            logger.debug(f"input_queues: {self.input_queues}")

            relative_consumed_gaps = []

            if cell.label in self.input_queues and len(self.input_queues[cell.label]) > 0 and cell.label in self.assignable_gaps and len(self.assignable_gaps[cell.label]) > 0:
                # Active development: log grid/data placement and PID search
                logger.debug(f"[GRID] Cell {cell.label}: Assigning data to gaps {self.assignable_gaps[cell.label]}")
                logger.debug(f"[PID] Cell {cell.label}: PID search for data, gap_pids: {gap_pids}")
                original_queue = self.input_queues[cell.label].copy()
                relative_consumed_gaps, consumed_gaps, queue = self.injection(
                    self.input_queues[cell.label], self.assignable_gaps[cell.label], gap_pids, 0
                )
                analysis_logger.analysis(
                        f"Cell {cell.label} injection — consumed rel={relative_consumed_gaps}, abs={consumed_gaps}"
                )
                self.input_queues[cell.label] = queue
                pids = gap_pids[:len(relative_consumed_gaps)]
                for idx, pid in enumerate(pids):
                    gap_idx = self.bitbuffer.get_by_pid(cell.label, pid)
                    logger.debug(f"[PID] Cell {cell.label}: Data placed at gap index {gap_idx} (pid {pid})")
                    stride = cell.stride
                    data_payload = self.bitbuffer._data_access[gap_idx : gap_idx + stride]
                    logger.debug(f"[DATA] Cell {cell.label}: Injected data at relative gaps {relative_consumed_gaps} and absolute gaps {consumed_gaps}")
                    logger.debug(f"[DATA] Cell {cell.label}: Data payload: {data_payload.hex()}")
                    orig_payload, _ = original_queue[idx]
                    logger.debug(f"[DATA] Cell {cell.label}: Original payload: {orig_payload.hex()}")
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
                logger.debug(f"Line 571: Cell {cell.label} leftmost before adjustment: {cell.leftmost}, left: {cell.left}, relative_consumed_gaps: {relative_consumed_gaps}, consumed_gaps: {consumed_gaps}")
                cell.leftmost = min(cell.leftmost, min(consumed_gaps)) if consumed_gaps else cell.leftmost
                assert cell.leftmost >= cell.left, f"Cell {cell.label} leftmost {cell.leftmost} is less than left {cell.left}"
                assert cell.leftmost < cell.right, f"Cell {cell.label} leftmost {cell.leftmost} is not less than right {cell.right}"
                assert cell.leftmost % cell.stride == 0, f"Cell {cell.label} leftmost {cell.leftmost} is not aligned with stride {cell.stride}"
                logger.debug(f"Line 578: Cell {cell.label} rightmost before adjustment: {cell.rightmost}, right: {cell.right}, relative_consumed_gaps: {relative_consumed_gaps}, consumed_gaps: {consumed_gaps}")
                cell.rightmost = max(cell.rightmost, max(consumed_gaps)+cell.stride - 1) if consumed_gaps else cell.rightmost
                assert cell.rightmost >= cell.left, f"Cell {cell.label} rightmost {cell.rightmost} is less than left {cell.left}"
                assert cell.rightmost < cell.right, f"Cell {cell.label} rightmost {cell.rightmost} is not less than right {cell.right}"
                assert (cell.rightmost + 1) % cell.stride == 0, f"Cell {cell.label} rightmost {cell.rightmost} is not aligned with stride {cell.stride}"
            raw = self.bitbuffer.stamp(raw, relative_consumed_gaps, cell.stride, 1)
        else:
            if cell.label in self.input_queues:
                cell.salinity += sum(stride for _, stride in self.input_queues[cell.label])

                logger.debug(f"Cell {cell.label} has no left/right distinction, skipping.")
            continue
        # AFTER snapshot once stamping/injection for this cell is applied
        after_snap = _build_snapshot(raw)
        analysis_logger.analysis(f"Cell {cell.label} SNAPSHOT AFTER  → {after_snap}")

        raws[cell.label] = raw
        byte_len = len(cell._buf)
        if cell.injection_queue > 0:
            if self.assignable_gaps.get(cell.label):
                assert False, "Cell has assignable gaps but injection queue is not empty"
        self.push_cell_mask(cell)

    self.system_pressure = system_pressure
    logger.info(f"Mask state at the end of minimize: {self.bitbuffer.mask.hex()}")
    if verify:
        self.crosscheck()
    return system_pressure, raws
