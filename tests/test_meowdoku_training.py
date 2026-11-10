"""The live Lichess reader and background learner share one value network."""

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku.chess_net import ConvValueNet, encode  # noqa: E402
from meowdoku.games import Chess  # noqa: E402


def test_legacy_chess_weights_gain_trainable_residual_layer(tmp_path):
    net = ConvValueNet(channels=2, hidden=8)
    game = Chess("legacy-weights")
    boards = np.stack([encode(game.pieces, game.turn)] * 2)
    original = net.values(boards)
    legacy = tmp_path / "legacy.npz"
    np.savez(legacy, **{f"p{i}": np.asarray(p.numpy()) for i, p in enumerate(net.parameters()[:8])})
    loaded = ConvValueNet(channels=2, hidden=8, seed=7)
    assert loaded.load(legacy)
    np.testing.assert_array_equal(loaded.values(boards), original)
    assert np.count_nonzero(loaded.W5.numpy()) == 0
    loaded.train_step(boards, np.array([0.5, 0.5]))
    assert np.count_nonzero(loaded.W5.numpy()) > 0
    current = tmp_path / "current.npz"
    loaded.save(current)
    restored = ConvValueNet(channels=2, hidden=8, seed=8)
    assert restored.load(current)
    np.testing.assert_array_equal(restored.values(boards), loaded.values(boards))


def test_lichess_evaluation_does_not_disable_learner_gradients():
    net = ConvValueNet(channels=2, hidden=8)
    game = Chess("training-concurrency")
    boards = np.stack([encode(game.pieces, game.turn)] * 2)
    entered, release = threading.Event(), threading.Event()
    forward = net.forward

    def held_forward(x):
        if not entered.is_set():
            entered.set()
            assert release.wait(5)
        return forward(x)

    net.forward = held_forward
    with ThreadPoolExecutor(max_workers=2) as pool:
        evaluation = pool.submit(lambda: net.values(boards))
        assert entered.wait(5)
        training = pool.submit(lambda: net.train_step(boards, np.zeros(2)))
        try:
            # Evaluation holds the process-wide no_grad scope. Training must
            # wait for it rather than record an empty autograd graph.
            assert not training.done()
        finally:
            release.set()
        assert evaluation.result(timeout=5).shape == (2,)
        assert np.isfinite(training.result(timeout=5))
