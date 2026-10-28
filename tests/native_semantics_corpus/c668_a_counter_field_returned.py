class Builder:
    def __init__(self, first_value_id: int):
        self.next_value_id = first_value_id

    def fresh(self) -> int:
        value = self.next_value_id
        self.next_value_id += 1
        return value


def root(first: int) -> int:
    builder = Builder(first)
    a = builder.fresh()
    return a * 100 + builder.next_value_id
# INPUTS [{'first': 7}, {'first': 0}, {'first': -3}]
