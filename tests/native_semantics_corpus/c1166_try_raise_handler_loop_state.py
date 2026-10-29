# A protected update is live on the exceptional edge.  Each iteration's
# handler must read the value written before raise, then carry its result to
# the next iteration.
def root(n: int) -> int:
    x = 0
    for i in range(n):
        try:
            x = x + i
            raise ValueError("bad")
        except ValueError:
            x = x + 2
    return x


# INPUTS [{'n': 0}, {'n': 1}, {'n': 3}, {'n': 5}]
