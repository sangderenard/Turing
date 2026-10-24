# Lane E (Fortran list growth): a grown list of bools is reached through a
# logical pointer view.
def root(n: int) -> int:
    bs = []
    for i in range(n % 6 + 2):
        bs.append(i % 3 == 0)
    c = 0
    for b in bs:
        if b:
            c += 1
    return len(bs) * 10 + c

# INPUTS [{'n': 0}, {'n': 4}, {'n': 11}]
