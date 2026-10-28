from src.common.tensors.abstraction import AbstractTensor


def root(n: float, cap: int) -> float:
    acc = AbstractTensor.tensor(1.0)
    hops = 0
    while (acc * 3.0).item() < n:
        acc = acc + 2.0
        hops += 1
        if hops >= cap:
            break
    return float(acc.item()) * 10.0 + hops
# INPUTS [{'n':0.0,'cap':3},{'n':20.0,'cap':2},{'n':20.0,'cap':9},{'n':-5.0,'cap':4}]
