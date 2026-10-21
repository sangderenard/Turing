def step(x: float, v: float, dt: float):
    a = -x
    return x + v * dt, v + a * dt

def root(x0: float, v0: float, n: int) -> float:
    x = x0
    v = v0
    for _ in range(n):
        x, v = step(x, v, 0.1)
    return x * 1000.0 + v
# INPUTS [{'x0':1.0,'v0':0.0,'n':5},{'x0':0.0,'v0':1.0,'n':0}]
