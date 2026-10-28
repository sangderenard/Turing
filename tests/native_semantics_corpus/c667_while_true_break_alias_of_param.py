# A name bound only on the break edge to a loop-carried parameter reads the
# current value at the break, not the seed (also what a tail call normalizes to).
def f(v: int) -> int:
    while True:
        if v <= 0:
            r = v
            break
        v = v - 1
        continue
    return r


def root(n: int) -> int:
    return f(n) + 1
# INPUTS [{'n': 0}, {'n': 7}, {'n': -4}]
