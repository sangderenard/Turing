def root(n: int, t: int) -> int:
    # A for body rebinds a parameter it never reads, and a later expression
    # consumes the loop's result: zero iterations keep the caller's value.
    for i in range(n):
        t = i * 3
    return t + n

# INPUTS [{'n': 0, 't': 5}, {'n': 1, 't': 5}, {'n': 3, 't': 5}]
