from src.bitbitbuffer.bitbitbuffer import BitBitBuffer


def test_nested_slice_offsets():
    buf = BitBitBuffer(mask_size=256)
    buf.mask[:] = bytes(range(32))  # 0x00..0x1f

    outer = buf[8:40]
    assert outer.hex() == bytes(range(1, 5)).hex()

    inner = outer[8:24]
    assert inner.hex() == bytes(range(2, 4)).hex()

    for i in [0, 6, 15]:
        assert inner[i] == int(buf[16 + i])


def test_cell_slice_does_not_overlap():
    buf = BitBitBuffer(mask_size=256)
    buf[0:128] = [1] * 128

    cell1 = buf[128:256]
    assert cell1.hex() == '00' * 16
    assert cell1[0:128].hex() == '00' * 16


def test_bitbititem_as_index():
    buf = BitBitBuffer(mask_size=8)
    buf[0:8] = [0, 1, 0, 1, 0, 1, 0, 1]
    index = buf[1]
    values = ['zero', 'one']
    assert values[index] == 'one'
