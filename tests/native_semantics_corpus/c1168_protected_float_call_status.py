# A protected native conversion must check its status before reading the
# region's result. The normal edge publishes the result; the exceptional
# edge is retained for a failed conversion and is checked separately.
def root(n: int) -> float:
    value = 0.0
    for index in range(n):
        try:
            value = float(index)
        except (TypeError, ValueError, OverflowError):
            raise RuntimeError("conversion failed")
    return value


# INPUTS [{'n': 0}, {'n': 1}, {'n': 3}]
