from .cell_consts import Cell


def snap_cell_walls(self, cells, proposals, left_boundary=0, right_boundary=None):
    if right_boundary is None:
        right_boundary = self.bitbuffer.mask_size
    system_lcm = self.lcm(cells)
    assert system_lcm > 0, "System LCM must be greater than zero"
    self.system_lcm = system_lcm

    sorted_cells = sorted(cells, key=lambda c: c.left)
    prev = left_boundary
    for c in sorted_cells:
        if c.left < prev:
            raise AssertionError("LCM alignment requires non-overlapping cells")
        prev = c.right

    for p in proposals:
        if getattr(p, 'leftmost', None) is None:
            p.leftmost = p.left
        if getattr(p, 'rightmost', None) is None:
            p.rightmost = p.right - 1

    sorted_props = sorted(proposals, key=lambda p: p.left)
    current = (left_boundary // system_lcm) * system_lcm
    for p in sorted_props:
        orig_left = p.left
        data_width = p.rightmost - orig_left + 1
        width = max(p.stride, p.right - orig_left, data_width)
        p.left = current
        p.right = ((p.left + width + system_lcm - 1) // system_lcm) * system_lcm
        delta = p.left - orig_left
        p.leftmost += delta
        p.rightmost += delta
        current = p.right

    for cell, prop in zip(sorted(cells, key=lambda c: c.left), sorted_props):
        cell.left, cell.right = prop.left, prop.right

    final_extent = max(p.right for p in sorted_props) if sorted_props else right_boundary
    if final_extent > self.bitbuffer.mask_size:
        self.expand(self.bitbuffer.mask_size, final_extent - self.bitbuffer.mask_size, cells, sorted_props)


def build_metadata(self, offset_bits, size_bits, cells):
    assert len(cells) > 0, "No cells provided to build_metadata"
    events = []
    offs = offset_bits if isinstance(offset_bits, (list, tuple)) else [offset_bits]
    szs = size_bits if isinstance(size_bits, (list, tuple)) else [size_bits]

    for offset in offs:
        assert isinstance(offset, int), f"Offset {offset} is not an integer"
        assert offset >= 0, f"Offset {offset} is negative, must be non-negative"
        assert offset <= self.bitbuffer.mask_size, f"Offset {offset} exceeds mask size {self.bitbuffer.mask_size}"

    for off, sz in zip(offs, szs):
        for cell in cells:
            if cell.right < cell.left:
                cell.right = cell.left
            if cell.left <= off < cell.right:
                raw_mid = (cell.left + cell.right) // 2
                aligned = raw_mid - (raw_mid % self.system_lcm)
                center = max(cell.left, min(aligned, cell.right - 1))
                events.append((cell.label, center, sz))
                break
        else:
            n = len(cells)
            base = sz // n
            rem = sz % n
            for idx, cell in enumerate(cells):
                share = self.bitbuffer.intceil(base + (1 if idx < rem else 0), self.system_lcm)
                raw_mid = (cell.left + cell.right) // 2
                center = raw_mid - (raw_mid % self.system_lcm)
                center = max(cell.left, min(center, max(cell.right - self.system_lcm, cell.left)))
                events.append((cell.label, center, share))

    return sorted(events, key=lambda e: e[1])


def expand(self, offset_bits, size_bits, cells, proposals, warp=True):
    if isinstance(offset_bits, int):
        offset_bits = [offset_bits]
    for offset in offset_bits:
        assert isinstance(offset, int), f"Offset {offset} is not an integer"
        assert offset >= 0, f"Offset {offset} is negative, must be non-negative"
        assert offset <= self.bitbuffer.mask_size, f"Offset {offset} exceeds mask size {self.bitbuffer.mask_size}"

    events = self.build_metadata(offset_bits, size_bits, cells)
    proposals = self.bitbuffer.expand(events, cells, proposals)

    for new_cell in proposals:
        for cell in cells:
            if cell.label == new_cell.label:
                cell.left = new_cell.left
                cell.right = new_cell.right
                cell.leftmost = new_cell.leftmost
                cell.rightmost = new_cell.rightmost
                break
