"""A scalar library entry with private constructed record workspace."""


class Counter:
    def __init__(self):
        self.total = 0

    def add(self, value: int):
        self.total = self.total + value

    def read(self) -> int:
        return self.total
