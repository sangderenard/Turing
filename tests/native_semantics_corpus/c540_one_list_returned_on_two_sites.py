def build(n: int) -> list[int]:
    out = [n]
    if n > 2:
        out.append(1)
        return out
    out.append(2)
    out.append(3)
    return out


def root(n: int) -> int:
    t = build(n)
    return len(t) * 1000 + t[0] * 10 + t[-1]

# INPUTS [{"n": 0}, {"n": 5}, {"n": 2}]
