# Lane X2 (2026-10-28): the Wasm core wrote a loop header's Phis one by one,
# so the second read the first's new value instead of the old one (c125 and
# the record swaps c427/c537 came back wrong). A swap through the back edge
# is the smallest program that needs the Phis copied in parallel.
def root(a: float, b: float) -> float:
    for _ in range(3):
        a, b = b + 0.5, a
    return a * 10.0 + b
# INPUTS [{'a': 1.0, 'b': 2.0}, {'a': -0.25, 'b': 4.5}]
