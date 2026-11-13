"""Independent paper-trading outcome evaluators for modular experiments.

They consume explicit fill and reference facts. Missing facts stay unobserved;
the tensor trainer's mask excludes them. No evaluator treats order acceptance as
a profit label.
"""
from __future__ import annotations

import math

from meowdoku import paper_arena


def _positive(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def fixed_horizon(fact: dict) -> float | None:
    entry = _positive(fact.get("entry_price"))
    mark = _positive(fact.get("later_mark"))
    entry_time = _positive(fact.get("entry_time"))
    mark_time = _positive(fact.get("mark_time"))
    horizon = _positive(fact.get("horizon_seconds"))
    fee = _positive(fact.get("total_cost_fraction"))
    side = fact.get("side")
    if None in (entry, mark, entry_time, mark_time, horizon) or side not in ("buy", "sell"):
        return None
    if mark_time < entry_time + horizon:
        return None
    direction = 1 if side == "buy" else -1
    return direction * (mark / entry - 1) - (fee or 0.0)


def closed_trade(fact: dict) -> float | None:
    entry = _positive(fact.get("buy_price"))
    exit_price = _positive(fact.get("sell_price"))
    if entry is None or exit_price is None:
        return None
    fee = _positive(fact.get("total_cost_fraction")) or 0.0
    return exit_price / entry - 1 - fee


def fill_quality(fact: dict) -> float | None:
    reference = _positive(fact.get("reference_mark"))
    execution = _positive(fact.get("execution_price"))
    side = fact.get("side")
    if reference is None or execution is None or side not in ("buy", "sell"):
        return None
    fee = _positive(fact.get("fee_fraction")) or 0.0
    direction = 1 if side == "buy" else -1
    return -direction * (execution / reference - 1) - fee


EVALUATORS = {"fixed_horizon": fixed_horizon,
              "closed_trade": closed_trade,
              "fill_quality": fill_quality}
assert tuple(EVALUATORS) == paper_arena.OUTCOMES


def observe(facts: dict[str, dict], selected: list[str]) -> tuple[list[float], list[int]]:
    """Return all tensor-head values and a mask for selected, mature outcomes."""
    if not isinstance(facts, dict) or not isinstance(selected, list) or any(x not in EVALUATORS for x in selected):
        raise ValueError("known outcome modules required")
    values, known = [], []
    for name in paper_arena.OUTCOMES:
        value = EVALUATORS[name](facts.get(name, {})) if name in selected else None
        values.append(value if value is not None else 0.0)
        known.append(1 if value is not None else 0)
    return values, known
