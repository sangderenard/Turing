class Acc:
    def __init__(self, k: int) -> None:
        self.total = k

    def add(self, t: int) -> int:
        self.total += t
        return self.total


def root(n: int) -> int:
    # The reads of ``acc.total`` and ``other.total`` are operands of the
    # call that stores into that record: Python evaluates them before the
    # call runs.
    acc = Acc(n)
    first = acc.add(acc.total * 2)
    other = Acc(n + 1)
    return other.add(t=other.total - first + 5) * 10 + first

# INPUTS [{'n': 0}, {'n': 3}, {'n': -4}, {'n': 7}]
