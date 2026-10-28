def damp(value):
    return value * 0.5 + 1.0


def root(x: float, k: float) -> float:
    return damp(x) + damp(k * 2.0)

# INPUTS [{'x': 1.5, 'k': -2.0}, {'x': 0.0, 'k': 3.25}]
