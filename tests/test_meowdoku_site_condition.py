"""site_condition.py: the operators' switch for the condition card at the Apache front.

The switch answers its code and its state only to an operator or an admin, and checks the role on every switch, so
nothing a visitor can rewrite in a page posts a card.
"""
import io
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from meowdoku import site_condition as condition  # noqa: E402


class Handler:
    """The parts of the server's request handler the route uses."""

    def __init__(self, user_id, username="someone", headers=None, body=None):
        self.principal = SimpleNamespace(user_id=user_id, username=username)
        self.headers = {"Host": "example.test", **(headers or {})}
        raw = json.dumps(body).encode() if body is not None else b""
        if body is not None:
            self.headers.setdefault("Content-Type", "application/json")
            self.headers["Content-Length"] = str(len(raw))
        self.rfile, self.wfile = io.BytesIO(raw), io.BytesIO()
        self.answers, self.sent = [], []
        self.close_connection, self._set_cookie = False, None

    def _principal(self):
        return self.principal

    def _is_secure(self):
        return True

    def _json(self, payload, status=200):
        self.answers.append((status, payload))

    def _body_json(self, limit=4096):
        return json.loads(self.rfile.read(int(self.headers["Content-Length"])))

    def send_response(self, status):
        self.sent.append(status)

    def send_header(self, name, value):
        self.sent.append((name, value))

    def end_headers(self):
        pass


@pytest.fixture
def accounts(tmp_path):
    path = tmp_path / "accounts.db"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE account_roles (user_id INTEGER, role TEXT)")
    db.executemany("INSERT INTO account_roles VALUES (?, ?)", [(1, "admin"), (2, "operator"), (3, "member")])
    db.commit()
    db.close()
    return str(path)


def route(accounts, folder, handler, path, method):
    condition.http_route(handler, path, method, accounts_db=accounts, folder=folder)
    return handler


def test_only_an_operator_or_admin_gets_the_controls_or_the_state(accounts, tmp_path):
    for user_id, status in ((None, 401), (3, 403)):
        for path in ("/api/condition", "/api/condition/controls.js"):
            h = route(accounts, tmp_path, Handler(user_id), path, "GET")
            assert h.answers[0][0] == status and h.wfile.getvalue() == b"", (user_id, path)
    for user_id in (1, 2):
        h = route(accounts, tmp_path, Handler(user_id), "/api/condition/controls.js", "GET")
        assert h.sent[0] == 200 and b"export async function mount" in h.wfile.getvalue()
        assert ("Cache-Control", "no-store, private") in h.sent
        assert route(accounts, tmp_path, Handler(user_id), "/api/condition", "GET").answers == [(200, {"card": None, "line": None})]


def test_a_member_cannot_switch_and_an_operator_posts_and_takes_down(accounts, tmp_path):
    folder = tmp_path / "condition"
    h = route(accounts, folder, Handler(3, body={"card": "red"}), "/api/condition", "POST")
    assert h.answers[0][0] == 403 and not (folder / "red.on").exists()
    h = route(accounts, folder, Handler(2, "op", body={"card": "yellow", "note": "mcp\nflood"}), "/api/condition", "POST")
    status, state = h.answers[0]
    assert status == 200 and state["card"] == "yellow" and state["line"].endswith(" by op: mcp flood")
    assert (folder / "yellow.html").read_bytes() == (condition.CARD_SOURCE / "yellow.html").read_bytes()
    state = route(accounts, folder, Handler(1, "boss", body={"card": "red"}), "/api/condition", "POST").answers[0][1]
    assert state["card"] == "red" and not (folder / "yellow.on").exists(), "one card at a time"
    state = route(accounts, folder, Handler(1, body={"card": "none"}), "/api/condition", "POST").answers[0][1]
    assert state == {"card": None, "line": None} and not list(folder.glob("*.on"))


def test_a_switch_must_be_same_origin_json_and_a_known_card(accounts, tmp_path):
    cases = [({"Sec-Fetch-Site": "cross-site"}, {"card": "red"}, 403),
             ({"Origin": "https://elsewhere.test"}, {"card": "red"}, 403),
             ({"Content-Type": "text/plain"}, {"card": "red"}, 415),
             ({}, {"card": "purple"}, 400)]
    for headers, body, status in cases:
        h = route(accounts, tmp_path, Handler(1, headers=headers, body=body), "/api/condition", "POST")
        assert h.answers[0][0] == status, headers
    assert not list(tmp_path.glob("*.on"))
    h = route(accounts, None, Handler(1, body={"card": "red"}), "/api/condition", "POST")
    assert h.answers[0][0] == 503, "no --condition-dir: nothing to switch"


def test_the_controls_are_not_published_with_the_site():
    assert condition.CONTROLS.parent.name == "meowdoku" and not (condition.HERE / "web" / condition.CONTROLS.name).exists()
    export = (condition.HERE / "web_export.py").read_text(encoding="utf-8")
    assert condition.CONTROLS.name not in export
