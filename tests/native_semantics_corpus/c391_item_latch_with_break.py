from src.common.tensors.abstraction import AbstractTensor


def root(n: float, cap: int) -> float:
    total = AbstractTensor.tensor(0.0)
    count = 0
    while (total * 2.0).item() < n:
        total = total + 1.5
        count += 1
        if count >= cap:
            break
    return float(total.item()) + count
# INPUTS [{'n':0.0,'cap':3},{'n':9.0,'cap':2},{'n':9.0,'cap':10}]
