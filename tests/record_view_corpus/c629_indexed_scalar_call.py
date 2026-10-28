def count(begin: int, end: int):
    i = begin
    total = 0
    while i < end:
        total += i
        i += 1
    return total


def root(offsets: list[int], index: int):
    return count(offsets[index], offsets[index + 1])
