def root(n: int) -> int:
    # A dict's ``clear`` an ``if`` holds, read back with ``get`` and ``len``.
    d = {1: n, 2: 5}
    if n > 100:
        d.clear()
    return len(d) * 100 + d.get(1, -1) * 10 + d.get(2, -1)

# INPUTS [{'n': -3}, {'n': 0}, {'n': 1}, {'n': 200}]
