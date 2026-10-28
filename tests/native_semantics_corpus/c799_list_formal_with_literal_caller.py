# One list formal, two callers: a runtime list and an all-literal list.  The
# literal call's specialization declared the formal as a retained arena with
# live flags; the schema survey joins every variant of ``width``, so the
# generic variant counted live flags its runtime caller never wrote and
# ``len(values)`` read 0 (native 3 for Python's 23).
def width(values: list[int]) -> int:
    return len(values)


def root(rounds: int):
    a = width([65 + rounds, 66])
    b = width([65, 67, 68])
    return a * 10 + b

# INPUTS [{'rounds': 2}, {'rounds': 0}]
