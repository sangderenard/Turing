def fib(n: int) -> int:
    a = 0
    b = 1
    for _ in range(n):
        a, b = b, a + b
    return a

def root(n: int) -> int:
    return fib(n) + fib(n // 2)
# INPUTS [{'n':10},{'n':1},{'n':0}]
