import math


def root(x: float):
    # A region holding only ``math.floor`` of a parameter was retired as a
    # host boundary that was never installed: ``return math.floor(x),
    # int(x)`` returned one value, not two.
    return math.floor(x), int(x)

# INPUTS [{'x': 2.7}, {'x': 0.6}, {'x': -1.5}, {'x': 2.5}, {'x': -2.7}, {'x': 4503599627370497.0}]
