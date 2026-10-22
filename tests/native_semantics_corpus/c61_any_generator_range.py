def root(n: int) -> int:
    if any(i * i > n for i in range(5)):
        return 1
    return 0

# INPUTS [{'n': 0}, {'n': 15}, {'n': 16}, {'n': 40}]
