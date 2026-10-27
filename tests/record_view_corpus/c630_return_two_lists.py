class Batch:
    def __init__(self):
        self.epoch = 0

    def envelope(self, values: list[int]):
        return self.epoch, len(values)

    def run(self, values: list[int]):
        status, count = self.envelope(values)
        statuses = [0] * count
        tokens = [1] * count
        if status == 0:
            i = 0
            while i < count:
                statuses[i] = status
                tokens[i] = values[i]
                i += 1
        return status, statuses, tokens


def root(values: list[int]):
    batch = Batch()
    return batch.run(values)
