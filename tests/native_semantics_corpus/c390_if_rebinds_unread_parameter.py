def root(n: int, t: int = 1) -> int:
    # The arm rebinds a parameter it never reads; the false edge keeps the
    # caller's value, so the merge needs the parameter's entry version.
    if n > 0:
        t = n * 2
    return t + n

# INPUTS [{'n': -2, 't': 1}, {'n': 0, 't': 5}, {'n': 3, 't': 1}]
