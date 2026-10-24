# ``b = a`` then ``b`` rebound: ``a``'s value was published as an earlier
# identity of ``b``'s output, so ``Ret`` returned ``b`` twice.
def root(k: int):
    a = k + 2
    b = a
    b += 5
    c = a * 3
    d = c
    if k > 2:
        d = d - 1
    return a, b, c, d
# INPUTS [{'k': 0}, {'k': 3}, {'k': 5}]
