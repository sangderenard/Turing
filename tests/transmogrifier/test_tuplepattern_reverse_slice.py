from src.bitbitbuffer.bitbitbuffer import BitBitBuffer


def test_tuplepattern_handles_full_reverse_slice():
    buf = BitBitBuffer(mask_size=32, bitsforbits=16)
    buf[0:10] = [1,0,1,1,0,0,1,0,1,0]
    left, right = buf.tuplepattern(0, 10, 5, "bi")
    assert left == [[1, 1], [0, 1], [1, 2], [0, 1]]
    assert right == [[0, 1], [1, 1], [0, 1], [1, 1], [0, 1]]
