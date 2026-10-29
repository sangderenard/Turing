import math


def root(x: float):
    # ``round`` of a float is the nearest int, ties to even, of either sign;
    # ``floor``/``ceil``/``trunc`` are ints too.  Each rounds in the float
    # domain and only then converts: LLVM's single-function lane truncated
    # the float first, so round(2.7) gave 2 and round(-1.5) gave -1.
    return round(x), math.floor(x), math.ceil(x), math.trunc(x)

# INPUTS [{'x': 0.5}, {'x': 1.5}, {'x': 2.5}, {'x': 3.5}, {'x': -0.5}, {'x': -1.5}, {'x': -2.5}, {'x': -3.5}, {'x': 2.7}, {'x': 0.6}, {'x': -2.7}, {'x': -0.6}, {'x': 0.49999999999999994}, {'x': 4503599627370497.0}, {'x': -0.0}, {'x': 3000000000.7}, {'x': -9.2e18}]
