def put(d: dict[int, int], k: int) -> None:
    d[k] = k * 3


def root(n: int) -> int:
    d = {1: 10}
    put(d, n)
    return len(d) * 100 + d[n]

# INPUTS [{'n': 1}, {'n': 5}]
