"""Modular policy and outcome heads over shared market features."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from meowdoku import paper_outcomes, paper_policies  # noqa: E402


def test_baseline_modules_share_features_without_submitting_orders():
    history = {"bars": {"SPY": [{"c": 100 + i} for i in range(9)],
                        "BTC/USD": [{"c": 50000 + 100 * i} for i in range(9)]}}
    seats = [{"id": "a", "policy": "momentum", "outcomes": ["fixed_horizon", "closed_trade"]},
             {"id": "b", "policy": "mean_reversion", "outcomes": ["fill_quality"]},
             {"id": "c", "policy": "tensor", "outcomes": ["fixed_horizon"]}]
    result = paper_policies.proposals(seats, history)
    assert result["market_count"] == 2 and result["next_offset"] is None
    assert result["strategies"][0]["symbols"][0]["ready"] is True
    assert result["strategies"][2]["symbols"][0]["ready"] is False
    assert result["execution"].startswith("dry-run")


def test_outcome_modules_mask_unknown_and_unselected_labels():
    facts = {"fixed_horizon": {"entry_price": 100, "later_mark": 102, "entry_time": 1000,
                               "mark_time": 4601, "horizon_seconds": 3600, "side": "buy", "total_cost_fraction": .005},
             "fill_quality": {"reference_mark": 100, "execution_price": 100.1, "side": "buy", "fee_fraction": .001}}
    values, known = paper_outcomes.observe(facts, ["fixed_horizon", "closed_trade", "fill_quality"])
    assert known == [1, 0, 1]
    assert values[0] == pytest.approx(.015) and values[1] == 0
    assert values[2] < 0
    assert paper_outcomes.fixed_horizon({**facts["fixed_horizon"], "mark_time": 2000}) is None


def test_abstract_tensor_network_trains_only_observed_heads():
    net = paper_policies.TensorOutcomeNet(hidden=8, seed=4)
    x = np.asarray([[.01, .02, .03, .01, .005, .01],
                    [-.01, -.02, -.03, .02, -.005, -.01]], dtype=np.float64)
    before = net.values(x)
    assert before.shape == (2, 3)
    targets = np.asarray([[.1, 0, 0], [-.1, 0, 0]], dtype=np.float64)
    mask = np.asarray([[1, 0, 0], [1, 0, 0]], dtype=np.float64)
    loss = net.train_step(x, targets, mask)
    assert np.isfinite(loss)
    assert not np.allclose(before[:, 0], net.values(x)[:, 0])
    with pytest.raises(ValueError):
        net.train_step(x, targets, np.zeros_like(mask))
