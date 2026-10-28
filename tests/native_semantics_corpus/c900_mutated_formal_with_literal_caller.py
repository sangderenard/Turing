# A callee that mutates its list formal before looping over it, called with
# a runtime list and with an all-literal list.  The literal argument
# specialized the callee (and, with one caller, folded the formal in the
# catalogue): the loop unrolled over the authored four elements and ignored
# ``values.pop()`` (native 103 for Python's 63 in the literal call).
def trim(values: list[int]) -> int:
    values.pop()
    total = 0
    for v in values:
        total += v
    return total * 10 + len(values)


def root(rounds: int):
    a = trim([65 + rounds, 66, 70])
    b = trim([1, 2, 3, 4])
    return a * 10000 + b

# INPUTS [{'rounds': 2}, {'rounds': 0}]
