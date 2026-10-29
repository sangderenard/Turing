def intern(n: int, a: int):
    return 0, [0], [n + a, 5]


def root(rounds: int) -> int:
    checksum = 0
    index = 0
    while index < rounds:
        envelope, statuses, tokens = intern(1, index)
        if envelope != 0 or statuses[0] != 0:
            return -1
        t = tokens[0]
        checksum += t + 1
        index += 1
    return checksum
# INPUTS [{"rounds": 3}]
