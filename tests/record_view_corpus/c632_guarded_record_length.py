from array import array


class Ledger:
    def __init__(self):
        self.domains = array("B")

    def add(self, domain: int):
        if len(self.domains) >= 16:
            return 0
        token = 100 + len(self.domains)
        if token > 200:
            return 0
        row = len(self.domains)
        self.domains.append(domain)
        return 100 + row


def root(domain: int):
    ledger = Ledger()
    return ledger.add(domain)
