"""[CONDITION] The site's skin: the card the Apache front shows instead of the site (deploy/apache/meowdoku-proxy.conf,
block "site condition"). A flag file in the condition folder (--condition-dir; srv/state/condition) turns every new request
away at Apache: yellow.on is "We are under mild attack", red.on is "under genuine threat, or down". One card at a time.
deploy/Set-SiteCondition.ps1 switches the same folder from a shell.

    read(folder)                          -> {"card": "yellow" | "red" | None, "line": "<time> by <name>: <note>" | None}
    post(folder, card, by=, note=)        -> read(folder) once the card is posted and the other taken down
    take_down(folder)                     -> read(folder) once no card is posted
    may_switch(db, user_id)               -> True for an operator or an admin (account_roles)
    http_route(self, path, method, *, accounts_db, folder)

The controls are never public code. Their page code (condition_controls.js, kept beside this file and outside web/, so
web_export never publishes it) is answered only to a signed-in operator or admin, from GET /api/condition/controls.js;
every switch checks the role again here, so a page rewritten in a browser switches nothing. The Apache block lets exactly
/api/condition and /api/condition/controls.js through a posted card, so a card can be taken down from the card.
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import time
from pathlib import Path

CARDS = ("yellow", "red")
HERE = Path(__file__).resolve().parent
CARD_SOURCE = HERE / "deploy" / "apache" / "condition"
CONTROLS = HERE / "condition_controls.js"
MAX_NOTE = 200


class ConditionError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


def _flag(folder: Path, card: str) -> Path:
    return Path(folder) / f"{card}.on"


def read(folder) -> dict:
    """The posted card (red is read first, as Apache reads it) and its line."""
    for card in ("red", "yellow"):
        try:
            line = _flag(folder, card).read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            continue
        return {"card": card, "line": line or None}
    return {"card": None, "line": None}


def post(folder, card: str, *, by: str, note: str = "") -> dict:
    if card not in CARDS:
        raise ConditionError("card is yellow, red or none")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name in CARDS:                                   # the cards Apache serves, as the repository has them
        shutil.copyfile(CARD_SOURCE / f"{name}.html", folder / f"{name}.html")
    for other in CARDS:
        if other != card:
            _flag(folder, other).unlink(missing_ok=True)
    note = " ".join(str(note or "").split())[:MAX_NOTE]
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} by {by}" + (f": {note}" if note else "")
    temp = folder / f"{card}.on.tmp"                     # moved into place, so Apache never sees a half-written flag
    temp.write_text(line + "\n", encoding="utf-8")
    os.replace(temp, _flag(folder, card))
    return read(folder)


def take_down(folder) -> dict:
    for card in CARDS:
        _flag(folder, card).unlink(missing_ok=True)
    return read(folder)


def may_switch(db, user_id) -> bool:
    """An operator or an admin."""
    if user_id is None:
        return False
    if not db.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'account_roles'").fetchone():
        return False
    return db.execute("SELECT 1 FROM account_roles WHERE user_id = ? AND role IN ('admin', 'operator')",
                      (user_id,)).fetchone() is not None


def http_route(self, path, method, *, accounts_db, folder):
    """The operators' switch over REST; everyone else gets 401 or 403 and no code.

        GET  /api/condition                 {card, line}
        GET  /api/condition/controls.js     the switch's page code (an ES module exporting mount(target, {onCard}))
        POST /api/condition {card: "yellow" | "red" | "none", note?}   post a card or take it down -> {card, line}
    """
    self._set_cookie = []
    if method == "POST":
        self.close_connection = True
    principal = self._principal()
    if principal.user_id is None:
        return self._json({"error": "sign in first"}, 401)
    db = sqlite3.connect(accounts_db, timeout=10, isolation_level=None)
    try:
        allowed = may_switch(db, principal.user_id)
    finally:
        db.close()
    if not allowed:
        return self._json({"error": "only an operator or an admin may see or switch the site's condition"}, 403)
    if folder is None:
        return self._json({"error": "this server has no condition folder (--condition-dir)"}, 503)
    if method == "GET" and path == "/api/condition/controls.js":
        body = CONTROLS.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/javascript; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, private")
        self.end_headers()
        self.wfile.write(body)
        return None
    if path != "/api/condition":
        return self._json({"error": "not found"}, 404)
    if method == "GET":
        return self._json(read(folder))
    expected_origin = ("https://" if self._is_secure() else "http://") + self.headers.get("Host", "")
    if self.headers.get("Sec-Fetch-Site") == "cross-site" or self.headers.get("Origin") not in (None, expected_origin):
        return self._json({"error": "same-origin request required"}, 403)
    if not (self.headers.get("Content-Type") or "").lower().startswith("application/json"):
        return self._json({"error": "JSON only"}, 415)
    body = self._body_json(2048)
    if body is None:
        return None
    card = body.get("card")
    try:
        if card == "none":
            state = take_down(folder)
        else:
            state = post(folder, card, by=principal.username, note=body.get("note") or "")
    except ConditionError as error:
        return self._json({"error": error.message}, error.status)
    print(f"[condition] {principal.username}: {card} -> {state['card'] or 'open'}", flush=True)
    return self._json(state)
