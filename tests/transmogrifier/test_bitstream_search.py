from src.transmogrifier.bitbitbuffer.helpers.bitstream_search import BitStreamSearch


class DummyBitslice:
    def __init__(self, bits):
        self.bits = bits
        self.padding = 0

    def __getitem__(self, idx):
        return self.bits[idx]

    def __len__(self):
        return len(self.bits)


def test_bitstream_search_runs():
    bs = DummyBitslice([0, 0, 1, 1, 0, 0, 0, 1, 0, 0])
    pattern = BitStreamSearch.count_runs(bs)
    assert pattern == [(0, 2), (1, 2), (0, 3), (1, 1), (0, 2)]

    gaps = BitStreamSearch.find_aligned_zero_runs(pattern, stride=2)
    assert gaps == [0, 4, 8]

    _, center_gaps = BitStreamSearch.detect_stride_gaps(bs, stride=2, sort_order="center-out")
    assert center_gaps == [4, 8, 0]
