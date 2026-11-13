"""Modular paper strategy experiments over a shared market snapshot.

Baseline policies make dry-run proposals. The tensor model predicts separate
outcome heads in one AbstractTensor batch, with a mask for observations whose
outcomes are not yet known. This module never places an order.
"""
from __future__ import annotations

import math

import numpy as np

from meowdoku import paper_arena

FEATURES = ("return_1", "return_3", "return_8", "volatility", "btc_return_3", "spy_return_3")
HEADS = paper_arena.OUTCOMES
ESTIMATED_ROUNDTRIP_COST = 0.004


def _prices(bars):
    values = []
    for bar in bars:
        close = bar.get("c") if isinstance(bar, dict) else None
        if isinstance(close, (int, float)) and not isinstance(close, bool) and math.isfinite(close) and close > 0:
            values.append(float(close))
    return values


def features(history: dict) -> dict[str, np.ndarray]:
    """Use only observed closes; incomplete series produce no proposal."""
    bars = history.get("bars", {}) if isinstance(history, dict) else {}
    if not isinstance(bars, dict):
        return {}
    observed = {symbol: _prices(series) for symbol, series in bars.items() if isinstance(series, list)}

    def change(symbol, lag):
        values = observed.get(symbol, ())
        return values[-1] / values[-1 - lag] - 1 if len(values) > lag else 0.0

    btc3, spy3 = change("BTC/USD", 3), change("SPY", 3)
    out = {}
    for symbol, values in observed.items():
        if symbol not in paper_arena.SYMBOLS or len(values) < 9:
            continue
        returns = np.diff(np.log(np.asarray(values[-9:], dtype=np.float64)))
        out[symbol] = np.asarray((change(symbol, 1), change(symbol, 3), change(symbol, 8),
                                  float(np.std(returns)), btc3, spy3), dtype=np.float64)
    return out


def baseline(policy: str, vector: np.ndarray) -> dict:
    if policy not in ("momentum", "mean_reversion"):
        return {"ready": False, "reason": "this policy needs its own runner or trained weights"}
    direction = 1 if policy == "momentum" else -1
    signal = direction * float(vector[1]) - 0.5 * float(vector[3])
    score = signal - ESTIMATED_ROUNDTRIP_COST
    return {"ready": True, "score": score,
            "proposal": "buy" if score > 0 else "hold",
            "estimated_roundtrip_cost_fraction": ESTIMATED_ROUNDTRIP_COST}


def proposals(seats: list[dict], history: dict, *, offset: int = 0, limit: int = 100) -> dict:
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 200:
        raise paper_arena.ArenaError("invalid proposal page")
    market = features(history)
    page = seats[offset:offset + limit]
    rows = []
    for seat in page:
        symbols = [{"symbol": symbol, **baseline(seat["policy"], vector)}
                   for symbol, vector in market.items()]
        rows.append({"strategy_id": seat["id"], "policy": seat["policy"],
                     "outcomes": seat["outcomes"], "symbols": symbols})
    return {"strategies": rows, "next_offset": offset + limit if offset + limit < len(seats) else None,
            "feature_names": FEATURES, "market_count": len(market),
            "execution": "dry-run proposals only; no order was submitted"}


class TensorOutcomeNet:
    """One shared batched AbstractTensor model with independent outcome heads."""

    def __init__(self, hidden: int = 32, seed: int = 0):
        from src.common.tensors import AbstractTensor as AT
        if type(hidden) is not int or not 1 <= hidden <= 256:
            raise ValueError("hidden must be 1-256")
        rng = np.random.default_rng(seed)
        self.hidden = hidden
        with AT.use_backend("numpy"):
            self.W1 = AT.tensor(rng.normal(0, 1 / math.sqrt(len(FEATURES)), (len(FEATURES), hidden)))
            self.b1 = AT.tensor(np.zeros((1, hidden)))
            self.W2 = AT.tensor(rng.normal(0, 1 / math.sqrt(hidden), (hidden, len(HEADS))))
            self.b2 = AT.tensor(np.zeros((1, len(HEADS))))

    def forward(self, x):
        return ((x @ self.W1 + self.b1).tanh() @ self.W2 + self.b2).tanh()

    def values(self, x: np.ndarray) -> np.ndarray:
        from src.common.tensors import AbstractTensor as AT
        x = np.asarray(x, dtype=np.float64).reshape(-1, len(FEATURES))
        with AT.use_backend("numpy"), AT.autograd.no_grad():
            return np.asarray(self.forward(AT.tensor(x)).numpy(), dtype=np.float64)

    def train_step(self, x: np.ndarray, targets: np.ndarray, known: np.ndarray, lr: float = 0.01) -> float:
        """Train only observed outcome heads; caller owns provenance and maturity checks."""
        from src.common.tensors import AbstractTensor as AT
        x = np.asarray(x, dtype=np.float64).reshape(-1, len(FEATURES))
        targets = np.asarray(targets, dtype=np.float64).reshape(-1, len(HEADS))
        known = np.asarray(known, dtype=np.float64).reshape(-1, len(HEADS))
        if (x.shape[0] == 0 or targets.shape[0] != x.shape[0] or known.shape != targets.shape
                or not np.isfinite(x).all() or not np.isfinite(targets).all()
                or not np.isin(known, (0, 1)).all() or known.sum() == 0
                or not math.isfinite(lr) or lr <= 0):
            raise ValueError("finite feature rows, targets, observed-head mask and positive learning rate required")
        with AT.use_backend("numpy"):
            params = (self.W1, self.b1, self.W2, self.b2)
            for param in params:
                param.requires_grad_()
            output = self.forward(AT.tensor(x))
            error = output - AT.tensor(targets)
            loss = ((error * error) * AT.tensor(known)).mean() * (known.size / known.sum())
            grads = AT.autograd.grad(loss, params, retain_graph=False)
            with AT.autograd.no_grad():
                for param, grad in zip(params, grads):
                    AT.copyto(param, param - grad * lr)
            return float(np.asarray(loss.numpy()))
