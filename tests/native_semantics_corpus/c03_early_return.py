def find(n: int, k: int) -> int:
    i = 0
    while i < n:
        if i * i >= k:
            return i
        i += 1
    return -1

def root(n: int, k: int) -> int:
    return find(n, k) * 10 + find(n, k + 5)
# INPUTS [{'n':10,'k':17},{'n':3,'k':50},{'n':0,'k':0}]
