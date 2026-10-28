# A name rebound after a ``while`` loop that carries it.  The loop's result
# was filed as the name's newest version, after ``checksum += ...`` below
# the loop, so ``return checksum`` returned the loop's value and the add was
# silently dropped (native 3 for Python's 303).
def bump(n: int) -> int:
    return n + 1


def root(rounds: int):
    checksum = 0
    index = 0
    while index < rounds:
        checksum += bump(index)
        index += 1
    checksum += bump(rounds) * 100
    return checksum

# INPUTS [{'rounds': 2}, {'rounds': 0}]
