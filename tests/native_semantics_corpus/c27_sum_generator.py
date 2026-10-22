def root(k: int) -> int:
    total = sum(i * k for i in range(6))
    flagged = sum(i % 2 == 0 for i in range(7))
    shifted = sum((i - k for i in range(4)), 10)
    return total * 10000 + flagged * 100 + shifted
# INPUTS [{'k': 0}, {'k': 3}, {'k': -2}]
