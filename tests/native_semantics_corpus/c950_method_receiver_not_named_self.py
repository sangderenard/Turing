# Python binds a method's receiver to its first positional parameter, whatever
# it is named; ``me`` and ``this`` are the instance as surely as ``self`` is.
class Bag:
    def __init__(me, n: int):
        me.xs = [n, n + 1, n + 2]
        me.seen = 0

    def total(this) -> int:
        this.seen = this.seen + 1
        return this.xs[0] + this.xs[2] + this.seen

    def twice(me) -> int:
        return me.total() + me.total()


def root(n: int) -> int:
    b = Bag(n)
    return b.twice()

# INPUTS [{'n': 3}, {'n': -5}, {'n': 0}]
