# An instance method called through its class with an explicit self
# (A.val(self, n)): the first argument is the receiver of A.val itself.  The
# prepass refused it (bootstrap: IdentityBook._page(self, name)).
class A:
    def __init__(self, k: int) -> None:
        self.k = k

    def val(self, n: int) -> int:
        return self.k + n

    def twice(self, n: int) -> int:
        return A.val(self, n) * 2


def root(n: int) -> int:
    a = A(n)
    return a.twice(3) * 100 + a.val(4)

# INPUTS [{'n': 0}, {'n': 5}]
