def same(a):
    return a


def root(n: int) -> int:
    a = [n, 2, 3]
    t = same(a)
    t.append(5)
    return len(a) * 100 + len(t) * 10 + a[-1] + t[0]
# INPUTS [{'n': 0}, {'n': 1}, {'n': 7}]
