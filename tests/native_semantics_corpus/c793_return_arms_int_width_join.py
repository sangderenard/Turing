# Token Orchard's ``_fingerprint_span``: one return is the constant pair
# ``0, 0`` (int64 by class) and the other the full-width combines, proven
# [0, 2**64-1] (uint64).  Python returns an int from either.  The return
# slot joins the arms' intervals and takes the narrowest width that holds
# the union; the narrower arm converts on its own edge.
def fingerprint(mode: int, n: int) -> tuple[int, int]:
    if mode == 1:
        return 0, 0
    high = (n * 2654435761 + 12345) % 4294967296
    low = (n * 40503 + 777) % 4294967296
    first = high * (1 << 32) + low
    second = low * (1 << 32) + high
    return first, second


def root(mode: int, n: int):
    first, second = fingerprint(mode, n)
    if first < 0 or second < 0:
        return -1
    return first % 1000000007 + second % 1000000009

# INPUTS [{'mode': 1, 'n': 3}, {'mode': 0, 'n': 0}, {'mode': 0, 'n': 1}, {'mode': 0, 'n': 3}, {'mode': 2, 'n': 7}]
