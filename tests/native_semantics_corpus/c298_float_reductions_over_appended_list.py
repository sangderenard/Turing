def root(x: float) -> float:
    xs = [x, 1.5, x * 0.25]
    xs.append(2.0 - x)
    return max(xs) * 100.0 + min(xs) + sum(xs)

# INPUTS [{'x': 0.0}, {'x': 3.25}, {'x': -2.5}]
