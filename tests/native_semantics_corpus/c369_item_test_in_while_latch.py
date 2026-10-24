from src.common.tensors.abstraction import AbstractTensor


def root(n: float) -> float:
    lim = AbstractTensor.tensor(n)
    total = AbstractTensor.tensor(0.0)
    while (lim - total).item() > 0.5:
        total = total + 1.0
    return float(total.item())
# INPUTS [{'n':0.0},{'n':3.2},{'n':7.0}]
