def root(n: int) -> int:
    # A dict's or set's len read before a later delete keeps the length it
    # had where it was authored.
    d1 = {1: n, 2: 3, 3: -1}
    t = len(d1) + n
    del d1[2]
    g = d1[n % 4] if n % 4 in d1 else t
    s = {n, 3, 13}
    a = len(s)
    s.add(n + 20)
    return len(d1) * 10000 + t * 100 + g * 10 + len(s) * 1000 + a

# INPUTS [{'n': -3}, {'n': 0}, {'n': 1}, {'n': 3}, {'n': 5}, {'n': 13}]
