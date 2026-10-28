def mk(k: int) -> list[int]:
    return [k, k + 1]
def root(n: int) -> int:
    t = mk(3)
    t.append(n)
    return len(t)*100 + t[0]*10 + t[2]
# INPUTS [{'n': 1}, {'n': 4}]
