def root(n: int) -> int:
    # Comprehensions whose row is the loop target run after the loop that
    # fills their source.
    values = [i % 4 for i in range(n)]
    a = [v for v in values]
    b = {w for w in a}
    c = {u for u in values if u >= 1}
    s = 0
    for x in b:
        s += x
    return len(a) * 1000 + len(b) * 100 + len(c) * 10 + s

# INPUTS [{'n': 0}, {'n': 1}, {'n': 3}, {'n': 7}]
