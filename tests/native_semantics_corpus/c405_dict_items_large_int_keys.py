# Lane E (E23): the first target of ``for k, w in d.items()`` is a key of
# ``d``.  A large int key read through float64 loses its low digits.
def root(n: int) -> int:
    d = {36028797018963969: n + 2, 5: n}
    s = 0
    for k, w in d.items():
        s = s + (k % 1000) * w
    return s

# INPUTS [{'n': 0}, {'n': 3}, {'n': 11}]
