# Token Orchard's ``validate_batch_envelope`` reached through ``intern_batch``,
# reduced.  The offsets loop's ``if not isinstance(offset, int) or
# isinstance(offset, bool): return 1, 0`` is disproved for a list[int], but
# only by the structural fold that runs after the loop was composed.  The
# loop still placed that return, and its removed ``1, 0`` literals became
# untyped caller storage ("declared call storage ... has no recorded
# dtype").  The second call takes the bounds-check return (status 2).
def validate(values: list[int], offsets: list[int]) -> tuple[int, int]:
    previous = 0
    for offset in offsets:
        if not isinstance(offset, int) or isinstance(offset, bool):
            return 1, 0
        if offset < previous or offset > len(values):
            return 2, 0
        previous = offset
    return 0, len(offsets) - 1


def intern_batch(values: list[int], offsets: list[int]) -> int:
    status, count = validate(values, offsets)
    return status * 100 + count


def root(rounds: int):
    checksum = 0
    index = 0
    while index < rounds:
        checksum += intern_batch([65 + index, 66], [0, 2])
        index += 1
    checksum += intern_batch([65 + rounds], [1, 0])
    return checksum

# INPUTS [{'rounds': 2}, {'rounds': 0}]
