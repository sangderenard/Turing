def collect(n: int) -> list[int]:
    out = []
    for i in range(n):
        if i % 2 == 0:
            out.append(i)
    return out


def root(n: int) -> int:
    xs = collect(n)
    return len(xs) * 100 + sum(xs)

# INPUTS [{'n': 0}, {'n': 5}, {'n': 10}]
