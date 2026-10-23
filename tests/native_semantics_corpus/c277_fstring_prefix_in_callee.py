def label(x) -> int:
    return len(f"value={int(x)}")


def root(n: int) -> int:
    return label(n * 13)

# INPUTS [{'n': 0}, {'n': 4}, {'n': 40}]
