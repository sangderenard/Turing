from src.common.tensors.abstraction import AbstractTensor


def root(n: float) -> float:
    lim = AbstractTensor.tensor(n)
    total = AbstractTensor.tensor(0.0)
    step = AbstractTensor.tensor(0.25)
    while (lim - total).item() > step.item():
        total = total + step
        step = step * 2.0
    return float(total.item()) + float(step.item())
# INPUTS [{'n':0.0},{'n':3.2},{'n':40.5}]
