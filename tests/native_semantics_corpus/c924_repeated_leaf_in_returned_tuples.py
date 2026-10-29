def root(x: float):
    # One value at two positions of a nested returned tuple: each position
    # is its own Ret slot.  The inner ``x`` was dropped as "already
    # published" by the outer one, so every backend returned (x, 1.0) for
    # Python's leaves (x, x, 1.0).
    y = x * 2.0
    return (x, (x, 1.0), (y, x))

# INPUTS [{'x': 2.5}, {'x': -7.25}, {'x': 0.0}, {'x': 1e300}]
