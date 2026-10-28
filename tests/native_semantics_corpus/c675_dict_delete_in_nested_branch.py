def root(n: int) -> int:
    # A keyed delete two ``if``s deep with no loop and no region of its own.
    d = {1: n, 2: 5, 3: 8}
    if n > 0:
        if n > 1:
            del d[3]
    return len(d) * 10 + d.get(3, -1)

# INPUTS [{'n': -3}, {'n': 0}, {'n': 1}, {'n': 2}]
