def root(n: int) -> int:
    # A list born as [0] * 5 has five rows; a store into one keeps them.
    a = [0] * 5
    a[2] = n
    return a[2] * 10 + a[4] + len(a)

# INPUTS [{'n': 3}, {'n': -1}]
