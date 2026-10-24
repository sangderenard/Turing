def pick(n: int) -> list[int]:
    if n > 2:
        return [n]
    return []


def root(n: int) -> int:
    t = pick(n)
    return len(t) * 10 + (t[0] if t else 0)

# INPUTS [{"n": 0}, {"n": 5}, {"n": 3}]
