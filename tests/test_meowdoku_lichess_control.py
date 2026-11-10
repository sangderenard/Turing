"""Lichess can finish a live game without seeking another one."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku.lichess import LichessBot  # noqa: E402


class FakeApi:
    def __init__(self):
        self.posts = []

    def post(self, path, data=None):
        self.posts.append((path, data))
        return {}


def test_pause_finishes_live_game_then_cancels_pending_and_stays_idle():
    bot = LichessBot("unused", log=lambda _message: None)
    bot.api = FakeApi()
    bot.me = {"id": "self"}
    bot.games["current"] = {"status": "started"}
    bot.pending = {"id": "next", "to": "other", "at": 0.0}

    bot.pause_at_idle()
    bot._maybe_challenge()
    assert bot.control_state() == {"pause_requested": True, "live": True, "pending": False}
    assert bot.api.posts == [("/api/challenge/next/cancel", None)]
    bot.accept_or_decline({"id": "incoming", "challenger": {"id": "other"},
                           "variant": {"key": "standard"}, "speed": "blitz"})
    assert bot.api.posts[-1] == ("/api/challenge/incoming/decline", {"reason": "generic"})

    bot.games["current"]["status"] = "mate"
    bot._maybe_challenge()
    assert len(bot.api.posts) == 2
    assert bot.control_state() == {"pause_requested": True, "live": False, "pending": False}
    bot.challenge_a_bot()
    assert len(bot.api.posts) == 2

    requested = []
    bot.challenge_a_bot = lambda: requested.append(True)
    bot.resume()
    bot._maybe_challenge()
    assert requested == [True]
