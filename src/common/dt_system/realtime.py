# -*- coding: utf-8 -*-
"""Realtime mode utilities.

Provides a small timing ledger that assigns per-engine time budgets each frame
based on error penalties and moving-average processing costs. Engines/controllers
use this to shape their next dt to keep the whole system live.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple

from .dt_scaler import Metrics
from .dt_controller import Targets


@dataclass
class RealtimeConfig:
    budget_ms: float = 16.7
    slack: float = 0.9
    beta: float = 1.0  # penalty exponent for weighting
    w_floor: float = 0.1
    ms_floor: float = 1e-20
    ms_cap: float = 1e20
    ema_alpha: float = 0.2


@dataclass
class RealtimeState:
    # Moving average processing time per engine id
    proc_ms_ma: Dict[str, float] = field(default_factory=dict)
    # Moving average penalty per engine id
    penalty_ma: Dict[str, float] = field(default_factory=dict)

    def update_proc_ms(self, engine_id: str, proc_ms: float, alpha: float) -> float:
        prev = self.proc_ms_ma.get(engine_id, proc_ms)
        val = (1.0 - alpha) * prev + alpha * proc_ms
        self.proc_ms_ma[engine_id] = min(val, 1e6)
        return self.proc_ms_ma[engine_id]

    def update_penalty(self, engine_id: str, penalty: float, alpha: float) -> float:
        prev = self.penalty_ma.get(engine_id, penalty)
        val = (1.0 - alpha) * prev + alpha * penalty
        self.penalty_ma[engine_id] = min(val, 1e6)
        return self.penalty_ma[engine_id]


def compile_allocations(
    cfg: RealtimeConfig,
    st: RealtimeState,
    engine_ids: Iterable[str],
) -> Dict[str, float]:
    """Compute per-engine ms allocations using a minimum compute-time budget.

    Algorithm
    ---------
    - Baseline per-engine allocation is its moving-average processing cost
      (``proc_ms_ma``) clamped to [ms_floor, ms_cap]. The sum across engines is
      the minimum budget. If this exceeds the frame budget, we still grant it;
      the frame will simply take longer (liveness over target FPS).
    - Any remaining slack (budget - minimum) is distributed by normalized
      penalty weights (penalty^beta with a small floor) to improve stability
      by allowing engines to use smaller dt in the next realtime step.

    Returns a mapping engine_id -> ms_alloc for this frame.
    """
    # 1) Minimum budget from processing costs
    base: Dict[str, float] = {}
    min_total = 0.0
    for eid in engine_ids:
        ms = float(st.proc_ms_ma.get(eid, 0.0))
        ms = max(cfg.ms_floor, min(ms, cfg.ms_cap))
        base[eid] = ms
        min_total += ms

    # 2) Extra pool from target budget (can be zero if over budget)
    target_pool = max(cfg.slack * cfg.budget_ms, 0.0)
    extra = max(target_pool - min_total, 0.0)

    # 3) Weights from penalty^beta with a floor, then normalized
    weights = compute_normalized_weights(cfg, st, engine_ids)

    # 4) Final allocations = baseline + weighted extras
    alloc: Dict[str, float] = {}
    for eid in engine_ids:
        w = float(weights.get(eid, 0.0))
        extra_ms = extra * w
        ms = base[eid] + extra_ms
        alloc[eid] = max(cfg.ms_floor, min(ms, cfg.ms_cap))
    return alloc


__all__ = [
    "RealtimeConfig",
    "RealtimeState",
    "compile_allocations",
    "compute_normalized_weights",
    "compute_minimum_budget",
    "compute_penalty",
    "compute_global_penalty",
]


def compute_penalty(metrics: Metrics, targets: Targets) -> float:
    """Map metrics to a scalar penalty ≥ 0 for time weighting.

    Uses normalized deviations from targets; 1.0 is neutral. Flags can be
    encoded by increasing the penalty further upstream if desired.
    """
    base = max(
        metrics.div_inf / max(targets.div_max, 1e-30),
        metrics.mass_err / max(targets.mass_max, 1e-30),
        1.0,
    )
    # Oscillation/stiffness hints modestly increase urgency.
    if metrics.osc_flag:
        base *= 1.25
    if metrics.stiff_flag:
        base *= 1.25
    return float(max(base, 0.0))


def compute_normalized_weights(
    cfg: RealtimeConfig,
    st: RealtimeState,
    engine_ids: Iterable[str],
) -> Dict[str, float]:
    """Return penalty-based weights normalized to sum to 1.0.

    Weight_i = w_floor + penalty_i^beta, then normalized; engines missing a
    penalty default to neutral (1.0).
    """
    raw: Dict[str, float] = {}
    for eid in engine_ids:
        p = max(float(st.penalty_ma.get(eid, 1.0)), 0.0)
        raw[eid] = cfg.w_floor + (p ** cfg.beta)
    total = sum(raw.values()) or 1.0
    return {eid: (w / total) for eid, w in raw.items()}


def compute_minimum_budget(
    cfg: RealtimeConfig,
    st: RealtimeState,
    engine_ids: Iterable[str],
) -> Tuple[Dict[str, float], float]:
    """Compute baseline per-engine allocations and their total (ms).

    Baseline is the clamped moving-average processing time per engine.
    Returns (per_engine_ms, total_ms).
    """
    base: Dict[str, float] = {}
    total = 0.0
    for eid in engine_ids:
        ms = float(st.proc_ms_ma.get(eid, 0.0))
        ms = max(cfg.ms_floor, min(ms, cfg.ms_cap))
        base[eid] = ms
        total += ms
    return base, total


def compute_global_penalty(
    cfg: RealtimeConfig,
    st: RealtimeState,
    engine_ids: Iterable[str],
) -> float:
    """Return a single normalized penalty scalar for the set of engines.

    Defined as the mean of ``penalty^beta`` across engines (penalty>=1). This
    yields 1.0 when all engines are at or below target, and increases as a
    smooth measure of global urgency.
    """
    vals = []
    for eid in engine_ids:
        p = max(float(st.penalty_ma.get(eid, 1.0)), 0.0)
        vals.append(p ** cfg.beta)
    if not vals:
        return 1.0
    return float(sum(vals) / len(vals))
