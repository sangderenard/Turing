def add_all(dst: list[int], src: list[int]) -> None:
    for v in src:
        dst.append(v)


def root(n: int) -> int:
    a = [n]
    b = [2, 3]
    add_all(a, b)
    return len(a) * 100 + a[2] + sum(b)

# INPUTS [{'n': 1}, {'n': 5}]
