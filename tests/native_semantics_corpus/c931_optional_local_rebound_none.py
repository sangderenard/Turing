# An Optional local born with an int and rebound to None in one arm: the
# None rebinding is one of the declaration's two classes, not a contradiction.
def root(n: int) -> int:
    k: int | None = 5
    if n > 2:
        k = None
    if k is None:
        return n
    return k + n

# INPUTS [{'n': 0}, {'n': 4}]
