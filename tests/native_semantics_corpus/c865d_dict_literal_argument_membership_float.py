def weigh(d: dict[int, float], k: int) -> float:
    if k in d:
        return d[k]
    return 0.5


def root(n):
    return weigh({1: 1.5, 2: n * 0.5}, n % 3) + weigh({7: n * 2.0}, 7)
# INPUTS [{"n": 0}, {"n": 4}, {"n": 5}]
