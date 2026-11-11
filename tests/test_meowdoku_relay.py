"""The relay, with real UDP sockets on loopback and a real SQLite database."""
import base64
import json
import socket
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from meowdoku import relay  # noqa: E402

ANN, BOB = 2, 3


def free_ports(count):
    """A run of UDP ports that are free right now."""
    for first in range(47900, 60000, 97):
        held = []
        try:
            for port in range(first, first + count):
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.bind(("127.0.0.1", port))
                held.append(sock)
            return first
        except OSError:
            continue
        finally:
            for sock in held:
                sock.close()
    raise RuntimeError("no free ports")


@pytest.fixture
def world(tmp_path):
    first = free_ports(3)
    config = {**relay.DEFAULTS, "host": "127.0.0.1", "first_port": first, "last_port": first + 2, "addresses_per_account": 2,
              "rate_per_minute": 5, "kept_per_address": 3}
    db = sqlite3.connect(tmp_path / "accounts.db", isolation_level=None)
    process = relay.Relay(tmp_path / "accounts.db", config, log=lambda line: None)
    yield {"db": db, "config": config, "relay": process, "db_file": tmp_path / "accounts.db"}
    process.close()


def body(package):
    return base64.b64decode(package["body_base64"])


def test_a_datagram_goes_out_from_an_address_and_the_reply_comes_back_to_it(world):
    db, config, process = world["db"], world["config"], world["relay"]
    mine = relay.open_address(db, user_id=ANN, username="ann", config=config)["address"]
    peer = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    peer.bind(("127.0.0.1", 0))
    peer.settimeout(5)
    relay.send(db, mine, user_id=ANN, to=f"127.0.0.1:{peer.getsockname()[1]}", body=b"hello out there", config=config)
    assert process.step() == (0, 1)
    data, source = peer.recvfrom(65535)
    assert data == b"hello out there" and source == ("127.0.0.1", mine), "it leaves from the address itself"
    peer.sendto(b"and back", source)
    assert process.step(wait=2)[0] == 1
    got = relay.receive(db, mine, user_id=ANN)
    assert [body(p) for p in got["packages"]] == [b"and back"]
    assert got["packages"][0]["from"] == f"127.0.0.1:{peer.getsockname()[1]}" and got["packages"][0]["kind"] == "datagram"
    assert relay.receive(db, mine, user_id=ANN, after=got["after"])["packages"] == []
    peer.close()


def test_anyone_may_send_to_an_address_and_nobody_sends_through_one_that_is_not_theirs(world):
    db, config, process = world["db"], world["config"], world["relay"]
    mine = relay.open_address(db, user_id=ANN, username="ann", config=config)["address"]
    process.step()
    stranger = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for text in (b"one", b"two", b"three", b"four"):
        stranger.sendto(text, ("127.0.0.1", mine))
    taken = 0
    for _ in range(20):
        taken += process.step(wait=0.2)[0]
        if taken == 4:
            break
    assert taken == 4
    assert [body(p) for p in relay.receive(db, mine, user_id=ANN)["packages"]] == [b"two", b"three", b"four"], "the oldest is dropped"
    for user in (BOB, None, True, "2"):
        with pytest.raises(relay.RelayError):
            relay.send(db, mine, user_id=user, to="127.0.0.1:9", body=b"x", config=config)
        with pytest.raises(relay.RelayError):
            relay.receive(db, mine, user_id=user)
    assert db.execute("SELECT COUNT(*) FROM relay_packages WHERE direction = 'out'").fetchone()[0] == 0
    stranger.close()


def test_a_package_goes_address_to_address_without_a_socket(world):
    db, config = world["db"], world["config"]
    a = relay.open_address(db, user_id=ANN, username="ann", config=config)["address"]
    b = relay.open_address(db, user_id=BOB, username="bob", config=config)["address"]
    sent = relay.send(db, a, user_id=ANN, to=str(b), body=b"move e4", kind="chess.move", config=config)
    assert sent["delivered"] is True
    got = relay.receive(db, b, user_id=BOB)["packages"]
    assert [(p["from"], p["kind"], body(p)) for p in got] == [(str(a), "chess.move", b"move e4")]
    relay.send(db, b, user_id=BOB, to=str(a), body=b"e5", kind="chess.move", config=config)
    assert body(relay.receive(db, a, user_id=ANN)["packages"][0]) == b"e5", "and back the other way"
    with pytest.raises(relay.RelayError) as nowhere:
        relay.send(db, a, user_id=ANN, to="1", body=b"x", config=config)
    assert nowhere.value.status == 404


def test_limits_addresses_rate_size_and_a_closed_address(world):
    db, config, process = world["db"], world["config"], world["relay"]
    first = relay.open_address(db, user_id=ANN, username="ann", config=config)["address"]
    relay.open_address(db, user_id=ANN, username="ann", config=config)
    with pytest.raises(relay.RelayError) as many:
        relay.open_address(db, user_id=ANN, username="ann", config=config)
    assert many.value.status == 429
    with pytest.raises(relay.RelayError) as big:
        relay.send(db, first, user_id=ANN, to="127.0.0.1:9", body=b"x" * (config["max_body"] + 1), config=config)
    assert big.value.status == 413
    for _ in range(5):
        relay.send(db, first, user_id=ANN, to="127.0.0.1:9", body=b"x", config=config, now=1000.0)
    with pytest.raises(relay.RelayError) as fast:
        relay.send(db, first, user_id=ANN, to="127.0.0.1:9", body=b"x", config=config, now=1001.0)
    assert fast.value.status == 429
    relay.send(db, first, user_id=ANN, to="127.0.0.1:9", body=b"x", config=config, now=1100.0)
    for bad in ("nowhere", "host:", "host:99999", ""):
        with pytest.raises(relay.RelayError):
            relay.send(db, first, user_id=ANN, to=bad, body=b"x", config=config, now=2000.0)
    process.step()
    assert first in process.sockets
    relay.close_address(db, first, user_id=ANN)
    process.step()
    assert first not in process.sockets, "a closed address stops listening"
    with pytest.raises(relay.RelayError):
        relay.receive(db, first, user_id=ANN)
    again = relay.open_address(db, user_id=BOB, username="bob", config=config)
    assert again["address"] == first and relay.receive(db, first, user_id=BOB)["packages"] == [], "a reused address starts empty"


def test_the_relay_over_rest_takes_the_caller_from_the_session(world):
    config, db_file = world["config"], world["db_file"]

    def ask(method, path, user=ANN, username="ann", payload=None):
        sent = []
        handler = SimpleNamespace(path=path, headers={"Content-Type": "application/json"},
                                  _principal=lambda: SimpleNamespace(user_id=user, username=username),
                                  _body_json=lambda limit: payload,
                                  _json=lambda document, status=200: sent.append((int(status), json.loads(json.dumps(document)))))
        relay.http_route(handler, path.split("?")[0], method, accounts_db=db_file, config=config)
        return sent[0]

    assert ask("GET", "/api/relay/addresses", user=None)[0] == 401
    a = ask("POST", "/api/relay/open", payload={"user_id": BOB})[1]["address"]
    b = ask("POST", "/api/relay/open", user=BOB, username="bob", payload={})[1]["address"]
    assert [x["address"] for x in ask("GET", "/api/relay/addresses")[1]["addresses"]] == [a]
    assert ask("POST", "/api/relay/send", payload={"address": a, "to": str(b), "text": "hi", "kind": "note"})[1]["delivered"] is True
    assert ask("POST", "/api/relay/send", user=BOB, username="bob", payload={"address": a, "to": str(b), "text": "x"})[0] == 404
    assert ask("POST", "/api/relay/send", payload={"address": a, "to": str(b), "text": "x", "body_base64": "eA=="})[0] == 400
    got = ask("GET", f"/api/relay/receive?address={b}", user=BOB, username="bob")[1]["packages"]
    assert base64.b64decode(got[0]["body_base64"]) == b"hi" and got[0]["from"] == str(a) and "username" not in got[0]
    assert ask("GET", f"/api/relay/receive?address={b}")[0] == 404 and ask("GET", "/api/relay/receive?address=x")[0] == 400
    assert ask("POST", "/api/relay/close", payload={"address": a})[1]["closed"] is True
