def root(n: int) -> int:
    # A loop over a tuple display with a runtime member: the display is
    # born as a local sequence of its members, like a list display.
    r = 0
    for f in (True, False, n > 2):
        if f:
            r += 1
    s = 0.0
    for g in (1, n * 0.5):
        s += g
    t = (n - 1, n)
    for h in t:
        if h > 3:
            break
        r += h * 10
    return r * 1000 + int(s * 10) + t[1] * 7

# INPUTS [{'n': -3}, {'n': 0}, {'n': 2}, {'n': 5}]
