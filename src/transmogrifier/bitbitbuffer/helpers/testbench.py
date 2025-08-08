# 3️⃣  A fluent, slice‑only test‑bench
import random
from .bitbitindexer import BitBitIndexer
from ..bitbitbuffer import BitBitBuffer

def main():
    # — basic mask + data round‑trip —
    BitBitIndexer.configure(enabled=True, verbosity=9)
    buf = BitBitBuffer(mask_size=16, bitsforbits=8)

    # mask                                   data
    buf[0:8]          = [1, 0, 1, 1, 0, 0, 1, 1]
    buf._data_access[0:8] = bytes(range(1, 9))

    assert buf[0:8].hex()            == 'b3'              # mask check
    assert buf._data_access[0:8]     == bytes(range(1, 9))  # data check

    # reversed views (mask reverses, data preserves byte order)
    assert buf[7::-1].hex()          == 'cd'
    assert buf._data_access[7::-1]   == bytes(range(8, 0, -1))

    # — stamp, expand, pid, etc. (unchanged logic, just fluent access) ——
    buf2 = BitBitBuffer(mask_size=9, bitsforbits=3)
    view = buf2[0:3]
    buf2.stamp(view, [0], 3)
    assert buf2[0:3].hex() == 'e0'


    from ...cells.cell_consts import Cell

    # 1) Create two test cells with stride=4, covering bits [1,4) and [4,7):
    cell1 = Cell(stride=4, left=1, right=4, len=3, label='c1')
    cell2 = Cell(stride=4, left=4, right=7, len=3, label='c2')
    cells = [cell1, cell2]

    # 2) New buffer covering 8 mask‑bits with 4 data‑bits per mask‑bit:
    buf3 = BitBitBuffer(mask_size=8, bitsforbits=4)

    # 3) Register your cells for PID tracking (optional for expand):
    buf3.register_pid_buffer(cells=cells, stride=cell1.stride)

    # 4) Expand at offset=2 by inserting 4 zero bits, tagged with cell1’s label:
    events = [
        ('c1', 2, 4),   # label, insert‑at‐mask‑index, #bits
    ]
    proposals = buf3.expand(events, cells=cells)

    # 5) Verify that each cell’s left/right moved as expected:
    assert (proposals[0].left, proposals[0].right) == (1, 8), f"Cell c1 should have moved right by 4 bits {proposals[0].label}, {proposals[0].left}, {proposals[0].right}"
    assert (proposals[1].left, proposals[1].right) == (8, 11), f"Cell c2 should have moved right by 4 bits {proposals[1].label}, {proposals[1].left}, {proposals[1].right}"
    print("expand test passed!")

    for proposal in proposals:
        for cell in cells:
            if proposal.label == cell.label:
                # Update the cell with the new left/right values
                assert proposal.left >= cell.left, f"Cell {cell.label} left should not move left: {proposal.left} < {cell.left}"
                assert proposal.right >= cell.right, f"Cell {cell.label} right should not move left: {proposal.right} < {cell.right}"
                cell.left = proposal.left
                cell.right = proposal.right
                cell.leftmost = proposal.leftmost
                cell.rightmost = proposal.rightmost
    # --- PID System Round-Trip Test ---
    # 1) Get the PIDBuffer for the 'test' domain.
    pb = buf3.pid_buffers['c1']

    # 2) Define test values and the target index.
    #    Since bitsforbits=4, these are 4-bit values.
    initial_value = 0xA  # (binary 1010)
    new_value = 0x5      # (binary 0101)
    test_index = 4

    # 3) Write the initial data value to the main buffer at the target index.
    buf3._data_access[test_index] = initial_value

    # 4) Create a PID that points to this location.
    pid = pb.create_id(test_index)

    # 5) Use the PID to GET the initial value and verify it's correct.
    retrieved_view = pb[pid]
    value = int.from_bytes(retrieved_view, 'big')
    assert value == initial_value, f"Expected initial value {initial_value}, got {value}"

    # 6) Use the PID to SET a new value.
    pb[pid] = new_value

    # 7) Use the PID to GET the new value back and verify the change was successful.
    final_view = pb[pid]

    assert int.from_bytes(final_view, 'big') == new_value


    # 8) As a final check, verify the data was changed in the underlying main buffer.
    assert int.from_bytes(buf3._data_access[test_index], 'big') == new_value
    print("PID system set/get round-trip test passed!")

    events = [
        ('c1', 2, 4),   # label, insert‑at‑mask‑index, #bits
        ('c2', 6, 2),   # label, insert‑at‑mask‑index, #bits
    ]
    # Expand the buffer with the events and cells
    proposals = buf3.expand(events, cells=cells)  # re-apply to update PIDs

    for proposal in proposals:
        for cell in cells:
            if proposal.label == cell.label:
                # Update the cell with the new left/right values
                assert proposal.left >= cell.left, f"Cell {cell.label} left should not move left: {proposal.left} < {cell.left}"
                assert proposal.right >= cell.right, f"Cell {cell.label} right should not move left: {proposal.right} < {cell.right}"
                cell.left = proposal.left
                cell.right = proposal.right
                cell.leftmost = proposal.leftmost
                cell.rightmost = proposal.rightmost

    pbuf = pb.pids
    pbuf[0] = 1
    orig   = int(pbuf[0])
    pbuf.move(0, 2, 1);  assert int(pbuf[2]) == orig
    pbuf.swap(0, 2, 1);  assert int(pbuf[0]) == orig

    # Use the existing 'buf' (mask_size=16, bitsforbits=8) and fully seed it.
    # Mask: all ones; Data: 1..16 (predictable)
    buf[0:16] = [1] * 16
    buf._data_access[0:16] = bytes(range(1, 17))
    expected = list(bytes(range(1, 17)))  # python-side mirror for verification

    # 1) Overlapping swaps (windows of length 1..4), many times
    for _ in range(96):
        a = random.randrange(0, 16)
        b = random.randrange(0, 16)
        if a == b:
            continue
        if a > b:
            a, b = b, a
        lmax = min(4, 16 - b)  # keep in-bounds
        if lmax == 0:
            continue
        l = random.randrange(1, lmax + 1)
        buf.swap(a, b, l)
        for k in range(l):
            expected[a + k], expected[b + k] = expected[b + k], expected[a + k]

    # 2) Overlapping moves (windows of length 1..4), both directions
    for _ in range(96):
        l = random.randrange(1, 5)
        if l > 16:
            continue
        src = random.randrange(0, 16 - l + 1)
        dst = random.randrange(0, 16 - l + 1)
        if src == dst:
            continue
        buf.move(src, dst, l)
        block = expected[src:src + l]
        del expected[src:src + l]
        if dst > src:
            dst -= l
        expected[dst:dst] = block

    # 3) Full reverse via BitBitSlice assignment (tests slice-get + slice-set)
    rev = buf[::-1]     # BitBitSlice view
    buf[:] = rev
    expected.reverse()

    # 4) Verify: data coupling + mask integrity
    got = list(bytes(buf._data_access[0:16]))
    assert got == expected, f"shuffle data mismatch: {got} != {expected}"
    assert all(int(buf[i]) == 1 for i in range(16)), "mask not preserved as all-1 after shuffle"

    print("\nAll checks passed!")

if __name__ == "__main__":
    main()
else:
    print("This module is intended to be run as a script, not imported.")