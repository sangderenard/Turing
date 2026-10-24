def root(n: int) -> int:
    # A ``del`` an ``if`` holds runs only when its test does, at its place:
    # inside the loop when the ``if`` is in the loop body (its arm owns no
    # numerical region), and before a later store and length read.
    d = {1: n, 2: 5}
    for i in range(3):
        if i == n:
            del d[2]
            d[2] = i
    d[7] = 1
    e = {1: n, 2: 5, 3: 4}
    for j in range(4):
        if j == n:
            del e[3]
    a = len(d) * 100 + d[2] * 10 + d[7]
    return a * 100 + len(e) * 10 + e.get(3, 0)

# INPUTS [{'n': -3}, {'n': 0}, {'n': 1}, {'n': 2}]
