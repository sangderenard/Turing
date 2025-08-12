from typing import List, Tuple, Set

# Assumes BitBitSlice implements __getitem__(int) -> int and exposes:
# - bit_length: int
# - padding: int

class BitStreamSearch:
    """
    A stride-aware, BitBitSlice-native bit stream search utility.
    Bit-level logic is strictly centralized to BitBit; all access is via bit index only.
    """
from typing import List, Tuple, Set

# Assumes BitBitSlice implements __getitem__(int) -> int and exposes:
# - bit_length: int
# - padding: int

class BitStreamSearch:
    """
    A stride-aware, BitBitSlice-native bit stream search utility.
    Bit-level logic is strictly centralized to BitBit; all access is via bit index only.
    """

    @staticmethod
    def count_runs(bitslice) -> List[Tuple[int, int]]:
        """
        Run-length encode a BitBitSlice using bit-index access only.
        """
        pattern = []
        last_bit = None
        count = 0
        for i in range(len(bitslice)):
            bit = int(bitslice[i])
            if bit == last_bit:
                count += 1
            else:
                if last_bit is not None:
                    pattern.append((last_bit, count))
                last_bit = bit
                count = 1
        if count > 0:
            pattern.append((last_bit, count))
        return pattern

    @staticmethod
    def find_aligned_zero_runs(pattern: List[Tuple[int, int]], stride: int) -> Set[int]:
        """
        Identify stride-aligned zero-run starts using run-length pattern.
        This implementation correctly and efficiently finds all valid starting positions.
        All returned offsets are local bit indices within the slice.
        """
        offsets = []
        cursor = 0
        for bit, count in pattern:
            if bit == 0:
                run_end = cursor + count
                # Calculate the first potential aligned starting position within this run.
                # This is the first multiple of 'stride' that is >= cursor.
                if cursor % stride == 0:
                    start_pos = cursor
                else:
                    start_pos = cursor + (stride - (cursor % stride))

                # Iterate through all aligned positions within the run
                for pos in range(start_pos, run_end, stride):
                    # Ensure a full stride-length of zeros exists from this position
                    if pos + stride <= run_end:
                        offsets.append(pos)
            cursor += count
        return offsets

    @classmethod
    def detect_stride_gaps(cls, bitslice, stride: int, sort_order: str) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        High-level entry point: returns run-length pattern and a sorted list of stride-aligned 0-gaps.
        sort_order can be 'center-out' or a default ascending sort.
        """
        pattern = cls.count_runs(bitslice)
        gaps_set = cls.find_aligned_zero_runs(pattern, stride)
        gaps = list(gaps_set) # Convert to list for sorting

        if sort_order == 'center-out':
            # This sort key achieves the 'center-out' order.
            # 1. Primary sort key: `abs(gap - center)`. This sorts by distance from the middle, closest first.
            # 2. Secondary sort key: `gap`. For gaps equidistant from the center, this sorts by the
            #    bit index itself, ensuring a consistent order (i.e., the "left" gap before the "right" gap).
            # This combination delivers the centermost gaps first and then expands outwards.
            center = len(bitslice) // 2
            gaps.sort(key=lambda gap: (abs(gap - center), gap))
        else: # Default behavior: sort gaps in ascending order by bit index.
            gaps.sort()

        return pattern, gaps
