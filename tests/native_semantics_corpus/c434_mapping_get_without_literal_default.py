def root(n: int) -> int:
    # ``d.get(k)`` is None for an absent key, and ``d.get(k, m)`` is ``m``;
    # a numeric literal default keeps its own helper.
    d = {1: n, 2: 5}
    v = d.get(n)
    m = n * 3
    w = d.get(n + 1, m)
    u = d.get(n, 9)
    a = -1 if v is None else v
    return a * 10000 + w * 100 + u

# INPUTS [{'n': -3}, {'n': 0}, {'n': 1}, {'n': 2}]
