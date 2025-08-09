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

    # Track queued bits as salinity and trigger expansion if nearing capacity
    queued_bits = sum(s for _, s in self.input_queues[cell_label])
    cell.salinity += queued_bits
    available_bits = cell.right - cell.left - cell.salinity
    threshold = int(available_bits * (1 - self.SALINE_BUFFER))
    print(f"Cell '{cell_label}' queued bits: {queued_bits}/{available_bits} (threshold: {threshold})")
    if cell.salinity > threshold:
        print(f"Cell '{cell_label}' nearing capacity ({queued_bits}/{available_bits} bits). Expanding...")
        self.run_saline_sim()


def actual_data_hook(self, payload: bytes, dst_bits: int, length_bits: int):
    self.bitbuffer._data_access[dst_bits : dst_bits + length_bits] = payload
