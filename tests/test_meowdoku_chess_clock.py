"""Chess against the network: the user's move returns before the net's
search finishes, and the server-owned chess clock (increment, flag fall, the
net's search budget following its own remaining time)."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from meowdoku import games  # noqa: E402
from meowdoku.chess_net import POLICY  # noqa: E402


class SlowNet:
    """A network stand-in: uniform priors, a fixed delay per forward batch."""

    def __init__(self, delay=0.0):
        self.delay = delay

    def evaluate(self, boards, masks):
        time.sleep(self.delay)
        n = len(boards)
        return np.zeros(n), np.asarray(masks, dtype=np.float64) + 1e-3

    def values(self, boards):
        return np.zeros(len(boards))


class Player:
    def __init__(self, net):
        self.net = net

    def learn_from(self, *args):
        return None


class FakeTime:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def _game(monkeypatch, clock="5+0", delay=0.0):
    monkeypatch.setattr(games, "self_play", lambda: Player(SlowNet(delay)))
    game = games.ChessVsNet(clock=clock)
    game.SIMULATIONS = 96
    game.fake = FakeTime()
    game._now = game.fake
    return game


def _play(game, uci):
    game.uci_move(uci)


def _wait(game, seconds=30.0):
    end = time.perf_counter() + seconds
    while game.thinking and time.perf_counter() < end:
        time.sleep(0.01)
    assert not game.thinking


def test_user_move_returns_before_the_search_completes(monkeypatch):
    game = _game(monkeypatch, "untimed", delay=0.4)
    started = time.perf_counter()
    _play(game, "e2e4")
    reply = time.perf_counter() - started
    st = game.state()
    assert st["thinking"] and game.turn == "black" and st["status"].startswith("network thinking")
    assert reply < 0.2                                         # the request did not wait on the forward passes
    _wait(game)
    total = time.perf_counter() - started
    assert total > 2 * reply and game.turn == "white" and game.moves == 2
    print(f"user-move response {reply * 1000:.1f} ms; net move landed after {total * 1000:.0f} ms")


def test_input_is_ignored_while_the_net_thinks(monkeypatch):
    game = _game(monkeypatch, "untimed", delay=0.2)
    _play(game, "e2e4")
    game.cell(0, 1, "primary")                                # a click on a black piece mid-search
    assert game.selected is None
    _wait(game)
    assert game.moves == 2


def test_clock_decrements_with_increment(monkeypatch):
    game = _game(monkeypatch, "3+2")
    assert game.state()["clock"]["white"] == 180.0 and game.state()["clock"]["running"] is None   # starts with White's first move
    game.fake.t += 50
    _play(game, "e2e4")                                       # the first move is free; +2
    assert game.clock["white"] == 182.0
    st = game.state()["clock"]
    assert st["running"] == "black" or game.thinking
    _wait(game)
    game.fake.t += 7.5
    st = game.state()["clock"]
    assert st["running"] == "white" and abs(st["white"] - (182.0 - 7.5)) < 1e-9
    _play(game, "d2d4")                                       # -7.5 +2
    assert abs(game.clock["white"] - 176.5) < 1e-9


def test_flag_fall_ends_the_game(monkeypatch):
    game = _game(monkeypatch, "1+0")
    _play(game, "e2e4")
    _wait(game)
    game.fake.t += 61
    st = game.state()
    assert st["clock"]["flagged"] == "white" and st["outcome"] == "black" and st["lost"]
    assert "wins on time" in st["status"] and st["clock"]["white"] == 0.0
    moves = game.moves
    _play_ignored = game.cell(6, 3, "primary"); game.cell(4, 3, "primary")
    assert game.moves == moves                                 # no moves after the flag
    assert "on time" in game.result()["detail"]


def test_net_search_budget_follows_its_clock(monkeypatch):
    game = _game(monkeypatch, "5+0")
    _play(game, "e2e4")
    _wait(game)
    rich = game.last_think["budget"]
    game.clock["black"] = 20.0
    game.stamp = game.fake.t
    _play(game, "d2d4")
    _wait(game)
    poor = game.last_think["budget"]
    assert abs(rich - 300.0 / 30.0) < 0.5 or rich == game.MAX_THINK    # a share of the time left, capped
    assert poor < rich and abs(poor - 20.0 / 30.0) < 0.1
    assert game.last_think["seconds"] < poor + 1.5
    untimed = _game(monkeypatch, "untimed")
    assert untimed.think_budget() is None


def test_reset_drops_a_stale_search(monkeypatch):
    game = _game(monkeypatch, "untimed", delay=0.3)
    _play(game, "e2e4")
    game.reset()
    time.sleep(1.0)
    assert game.moves == 0 and game.turn == "white" and not game.thinking
