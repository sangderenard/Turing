def root(xs: list[int]) -> int:
    t = 0
    for x in xs:
        t += x * 2
    return t
# INPUTS [{'xs': [1, 2, 3]}, {'xs': [5, 7]}]
