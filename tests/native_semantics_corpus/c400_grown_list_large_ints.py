# Lane E (Fortran list growth): a list grown in a loop holds int64 values
# past float64's exact integers, so an element store or load that passes
# through a real would be visible in the result.
def root(n: int) -> int:
    xs = []
    for i in range(n % 5 + 3):
        xs.append(36028797018963969 + i * 7 + n)
    t = 0
    for x in xs:
        t = t + x % 1000
    return len(xs) * 1000000 + t * 10 + xs[-1] % 10

# INPUTS [{'n': 0}, {'n': 4}, {'n': 9}]
