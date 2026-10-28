class Builder:
    def __init__(self, first_value_id: int):
        self.next_value_id = first_value_id

    def fresh(self) -> int:
        value = self.next_value_id
        self.next_value_id += 1
        return value

    def delegated(self) -> int:
        return self.fresh()


def root(first: int):
    builder = Builder(first)
    a = builder.fresh()
    b = builder.delegated()
    return a * 10000 + b * 100 + builder.next_value_id
# INPUTS [{'first': 7}, {'first': 0}, {'first': -3}]
