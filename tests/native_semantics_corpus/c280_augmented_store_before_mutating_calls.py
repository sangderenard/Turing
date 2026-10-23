def push(xs: list[int], v: int) -> None:
    xs.append(v)


def root(n: int) -> int:
    xs0 = [5, n, 0]
    xs0[1] = 8
    xs0[0] += n
    xs1 = [7, 7]
    push(xs1, n)
    push(xs1, 2)
    return (xs0[0] + xs0[1]) + (len(xs1) * 100 + xs1[2] + xs1[3])
# INPUTS [{'n':-3},{'n':0},{'n':4}]
