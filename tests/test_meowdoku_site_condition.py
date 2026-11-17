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
        assert route(accounts, tmp_path, Handler(user_id), "/api/condition", "GET").answers == [
            (200, {"card": None, "line": None, "notice": None})]


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


# ---- [NOTICE] the header notice


def test_anyone_reads_the_notice_without_the_posters_name(accounts, tmp_path):
    folder = tmp_path / "condition"
    assert route(accounts, folder, Handler(None), "/api/condition/notice", "GET").answers == [(200, {"notice": None})]
    assert route(accounts, None, Handler(None), "/api/condition/notice", "GET").answers == [(200, {"notice": None})]
    h = route(accounts, folder, Handler(2, "op", body={"title": "In preparation", "body": "One.\n\nTwo.", "eta": "soon"}),
              "/api/condition/notice", "POST")
    status, answer = h.answers[0]
    posted = answer["notice"]
    assert status == 200 and posted["version"] == 1 and posted["mode"] == "push" and posted["posted"].endswith(" by op")
    public = route(accounts, folder, Handler(None), "/api/condition/notice", "GET").answers[0][1]["notice"]
    assert public["title"] == "In preparation" and public["body"] == "One.\n\nTwo." and "posted" not in public
    assert "op" not in str(public), "a visitor never learns which operator posted it"
    assert route(accounts, folder, Handler(1), "/api/condition", "GET").answers[0][1]["notice"]["posted"].endswith(" by op")


def test_only_an_operator_posts_or_clears_and_every_field_is_checked(accounts, tmp_path):
    folder = tmp_path / "condition"
    for user_id, status in ((None, 401), (3, 403)):
        h = route(accounts, folder, Handler(user_id, body={"title": "x"}), "/api/condition/notice", "POST")
        assert h.answers[0][0] == status and not (folder / condition.NOTICE).exists()
    for body in ({}, {"title": "x" * 201}, {"title": "x", "mode": "sideways"}, {"title": "x", "block": True},
                 {"title": "x", "mode": "overlay", "block": "yes"}, {"title": "x", "paths": ["no-slash"]},
                 {"title": "x", "body": 5}):
        h = route(accounts, folder, Handler(2, body=body), "/api/condition/notice", "POST")
        assert h.answers[0][0] == 400, body
    h = route(accounts, folder, Handler(2, headers={"Sec-Fetch-Site": "cross-site"}, body={"title": "x"}), "/api/condition/notice", "POST")
    assert h.answers[0][0] == 403
    posted = route(accounts, folder, Handler(2, body={"title": "Closed", "mode": "overlay", "block": True, "dismissible": True}),
                   "/api/condition/notice", "POST").answers[0][1]["notice"]
    assert posted["block"] and posted["dismissible"] is False, "a blocking notice is never dismissible"
    again = route(accounts, folder, Handler(1, body={"title": "Open", "paths": ["/games"]}), "/api/condition/notice", "POST")
    assert again.answers[0][1]["notice"]["version"] == 2 and again.answers[0][1]["notice"]["paths"] == ["/games"]
    assert not list(folder.glob("*.tmp")), "written whole and renamed into place"
    cleared = route(accounts, folder, Handler(2, body={"clear": True}), "/api/condition/notice", "POST")
    assert cleared.answers[0] == (200, {"notice": None}) and not (folder / condition.NOTICE).exists()
    assert not list(folder.glob("*.on")), "the notice never posts a card"


def test_the_page_code_draws_the_notice_as_text_and_exports_it():
    nav = (condition.HERE / "web" / "site-nav.js").read_text(encoding="utf-8")
    start = nav.index("export function mountNotice")
    drawing = nav[start:nav.index("async function showSiteNotice")]
    assert "innerHTML" not in drawing and "insertAdjacentHTML" not in drawing, "notice text is never HTML"
    assert "mountNotice" in nav[nav.index("window.MeowdokuSiteNav"):], "other pages can reuse it"
    assert "keep: [header]" in nav, "a blocking notice never blocks the site header"
    controls = condition.CONTROLS.read_text(encoding="utf-8")
    assert "/api/condition/notice" in controls and "innerHTML" not in controls
