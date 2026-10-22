def g(x) -> int:
    return len(f"v={int(x)}")


def root(n: int) -> int:
    return g(n * 7)

# INPUTS [{'n': 0}, {'n': 4}, {'n': 40}]
