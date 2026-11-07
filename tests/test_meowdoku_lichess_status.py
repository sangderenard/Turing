"""The status CLI reads an account without enrolling it or starting play."""

import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from meowdoku import chess_net, lichess  # noqa: E402


@pytest.fixture
def offline_api(monkeypatch):
    # Never read credentials or send a request, including in the red baseline.
    monkeypatch.setattr(lichess, "token", lambda: "offline-test-token")
    monkeypatch.setattr(lichess.urllib.request, "urlopen", Mock(
        side_effect=AssertionError("network must remain offline")))
    api = Mock(spec=lichess.Api)
    factory = Mock(return_value=api)
    monkeypatch.setattr(lichess, "Api", factory)
    player = Mock()
    monkeypatch.setattr(chess_net, "SelfPlay", player)
    return api, factory, player


@pytest.mark.parametrize("title", [None, "BOT"])
@pytest.mark.parametrize("extra", [[], ["--random"], ["--challenge", "other-bot"]])
def test_status_only_reads_account(title, extra, offline_api, monkeypatch, capsys):
    api, factory, player = offline_api
    account = {"id": "offline", "username": "Offline", "perfs": {"blitz": {"rating": 1500}}}
    if title is not None:
        account["title"] = title
    api.get.return_value = account
    start = Mock(side_effect=AssertionError("status must not start a bot"))
    monkeypatch.setattr(lichess.LichessBot, "start", start)

    assert lichess.main(["--status", *extra]) == 0

    api.post.assert_not_called()
    api.get.assert_called_once_with("/api/account")
    factory.assert_called_once_with("offline-test-token")
    player.assert_not_called()
    start.assert_not_called()
    assert json.loads(capsys.readouterr().out) == {
        key: account.get(key) for key in ("id", "username", "title", "perfs")
    }


def test_status_without_token_does_not_create_client(offline_api, monkeypatch, capsys):
    _api, factory, player = offline_api
    monkeypatch.setattr(lichess, "token", lambda: None)

    assert lichess.main(["--status"]) == 1

    factory.assert_not_called()
    player.assert_not_called()
    assert "no token" in capsys.readouterr().out


@pytest.mark.parametrize("title", [None, "BOT"])
def test_play_connection_keeps_existing_enrollment_behavior(title, offline_api):
    api, _factory, _player = offline_api
    account = {"id": "offline", "username": "Offline", "title": title}
    upgraded = {**account, "title": "BOT"}
    api.get.side_effect = [account, upgraded]
    client = lichess.LichessBot("offline-test-token", log=lambda _message: None)

    assert client.connect() == (upgraded if title is None else account)

    if title is None:
        api.post.assert_called_once_with("/api/bot/account/upgrade")
        assert api.get.call_count == 2
    else:
        api.post.assert_not_called()
        api.get.assert_called_once_with("/api/account")


@pytest.mark.parametrize("random_play", [False, True])
def test_play_cli_still_connects_and_starts(random_play, offline_api, monkeypatch):
    _api, _factory, player = offline_api
    client = Mock()
    make_client = Mock(return_value=client)
    monkeypatch.setattr(lichess, "LichessBot", make_client)
    monkeypatch.setattr(lichess.time, "sleep", Mock(side_effect=KeyboardInterrupt))
    args = ["--challenge", "other-bot"]
    if random_play:
        args.append("--random")

    assert lichess.main(args) == 0

    client.connect.assert_called_once_with()
    client.start.assert_called_once_with()
    client.api.post.assert_called_once_with("/api/challenge/other-bot", {
        "rated": "false", "clock.limit": "300", "clock.increment": "3", "variant": "standard"})
    make_client.assert_called_once_with(
        "offline-test-token", player=None if random_play else player.return_value)
    assert player.call_count == (0 if random_play else 1)
