# -*- coding: utf-8 -*-
"""Real-time preview mode utilities.

Provides a small timing ledger that assigns per-engine time budgets each frame
based on error penalties and moving-average processing costs. Engines/controllers
use this to shape their next dt to keep the whole system live.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable

from .dt_scaler import Metrics
from .dt_controller import Targets


@dataclass
class RTPreviewConfig:
    budget_ms: float = 16.7
    slack: float = 0.9
    beta: float = 1.0  # penalty exponent for weighting
    w_floor: float = 0.1
    ms_floor: float = 0.25
    ms_cap: float = 100.0
    ema_alpha: float = 0.2


@dataclass
class RTPreviewState:
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
    cfg: RTPreviewConfig,
    st: RTPreviewState,
    engine_ids: Iterable[str],
) -> Dict[str, float]:
    """Compute per-engine ms allocations based on penalties.

    Returns a mapping engine_id -> ms_alloc for this frame.
    """
    # Weights from penalty^beta with a floor.
    weights = {}
    for eid in engine_ids:
        p = max(float(st.penalty_ma.get(eid, 0.0)), 0.0)
        w = cfg.w_floor + (p ** cfg.beta)
        weights[eid] = w
    total_w = sum(weights.values()) or 1.0
    pool = max(cfg.slack * cfg.budget_ms, 0.0)

    alloc = {}
    for eid, w in weights.items():
        ms = pool * (w / total_w)
        ms = max(cfg.ms_floor, min(ms, cfg.ms_cap))
        alloc[eid] = ms
    return alloc


__all__ = [
    "RTPreviewConfig",
    "RTPreviewState",
    "compile_allocations",
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
