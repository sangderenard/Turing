def choose(n: int) -> list[int]:
    a = [n, n + 1]
    b = [7]
    b.append(n * 2)
    b.append(a[0] + 5)
    if n % 2 == 0:
        return a
    return b


def root(n: int) -> int:
    t = choose(n)
    t.append(100)
    return len(t) * 10000 + t[0] * 100 + t[-2] + t[-1]

# INPUTS [{"n": 2}, {"n": 3}, {"n": 0}, {"n": 7}]
