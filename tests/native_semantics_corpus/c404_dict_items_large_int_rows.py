# Lane E (E23): the targets of ``for k, w in d.items()`` are a key and a row
# of ``d``.  A large int row read through float64 comes back rounded.
def root(n: int) -> int:
    d1 = {1: n + 36028797018963969, 3: 5}
    s1 = 0
    for k1, w1 in d1.items():
        s1 += k1 * 10 + w1
    return s1

# INPUTS [{'n': 0}, {'n': 2}, {'n': -7}]
