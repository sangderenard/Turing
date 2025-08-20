from __future__ import annotations

import random
from typing import Any, Callable, List, Tuple

import numpy as np

try:  # pragma: no cover - torch is optional
    import torch
except Exception:  # pragma: no cover - torch is optional
    torch = None

from ..abstraction import AbstractTensor


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (optionally) PyTorch RNGs.

    Parameters
    ----------
    seed:
        The seed value used for all available random number generators.
    """

    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

def as_list(t: AbstractTensor) -> list:
    return t.tolist()

def from_list_like(data: list, like: AbstractTensor) -> AbstractTensor:
    return type(like).tensor_from_list(data, dtype=None, device=None)

def zeros_like(t: AbstractTensor) -> AbstractTensor:
    def zmap(x):
        if isinstance(x, list):
            return [zmap(v) for v in x]
        return 0.0
    return from_list_like(zmap(as_list(t)), like=t)

def ones_like(t: AbstractTensor) -> AbstractTensor:
    def omap(x):
        if isinstance(x, list):
            return [omap(v) for v in x]
        return 1.0
    return from_list_like(omap(as_list(t)), like=t)

def map_unary(fn: Callable[[float], float], t: AbstractTensor) -> AbstractTensor:
    def rec(x):
        if isinstance(x, list):
            return [rec(v) for v in x]
        return fn(float(x))
    return from_list_like(rec(as_list(t)), like=t)

def map_binary(fn: Callable[[float, float], float], a: AbstractTensor, b: AbstractTensor) -> AbstractTensor:
    la, lb = as_list(a), as_list(b)
    def rec(x, y):
        if isinstance(x, list) and isinstance(y, list):
            return [rec(xi, yi) for xi, yi in zip(x, y)]
        if isinstance(x, list):
            return [rec(xi, y) for xi in x]
        if isinstance(y, list):
            return [rec(x, yi) for yi in y]
        return fn(float(x), float(y))
    return from_list_like(rec(la, lb), like=a)

def transpose2d(t: AbstractTensor) -> AbstractTensor:
    L = as_list(t)
    if not L or not isinstance(L[0], list):
        L = [[v] for v in L]
    T = [list(row) for row in zip(*L)]
    return from_list_like(T, like=t)

def shape_from_list_like(t: AbstractTensor) -> Tuple[int, ...]:
    L = as_list(t)
    if not isinstance(L, list):
        return ()
    if not L or not isinstance(L[0], list):
        return (len(L),)
    return (len(L), len(L[0]))
