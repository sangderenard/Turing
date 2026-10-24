# Two names enter an if/else as one value and each arm rebinds a different
# one.  The arms were keyed by the shared entry value, so one name's arm
# overwrote the other's.
def root(k: int):
    a = k * 2
    b = a
    if k > 1:
        b = b - 1
    else:
        a = a + 10
    return b, a
# INPUTS [{'k': 0}, {'k': 1}, {'k': 3}]
