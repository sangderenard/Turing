"""[CONDITION] The site's skin: the card the Apache front shows instead of the site (deploy/apache/meowdoku-proxy.conf,
block "site condition"). A flag file in the condition folder (--condition-dir; srv/state/condition) turns every new request
away at Apache: yellow.on is "We are under mild attack", red.on is "under genuine threat, or down". One card at a time.
deploy/Set-SiteCondition.ps1 switches the same folder from a shell.

    read(folder)                          -> {"card": "yellow" | "red" | None, "line": "<time> by <name>: <note>" | None}
    post(folder, card, by=, note=)        -> read(folder) once the card is posted and the other taken down
    take_down(folder)                     -> read(folder) once no card is posted
    may_switch(db, user_id)               -> True for an operator or an admin (account_roles)
    read_notice(folder) / post_notice(folder, fields, by=) / clear_notice(folder)   [NOTICE] the header notice (below)
    http_route(self, path, method, *, accounts_db, folder)

The controls are never public code. Their page code (condition_controls.js, kept beside this file and outside web/, so
web_export never publishes it) is answered only to a signed-in operator or admin, from GET /api/condition/controls.js;
every switch checks the role again here, so a page rewritten in a browser switches nothing. The Apache block lets exactly
/api/condition and /api/condition/controls.js through a posted card, so a card can be taken down from the card.

[NOTICE] The condition also has a header notice: ``notice.json`` in the same folder, written whole and renamed into place
like the flags.
- It holds a title, a body (plain text; blank lines make paragraphs) and an estimate (``eta``).
- ``mode`` is ``push`` (it takes the space at the top and pushes the page down) or ``overlay`` (it sits on top of the
  page). With ``block``, an overlay makes the page under it unusable, but never the site header or its menu.
- It can also be ``dismissible`` (never when it blocks), and limited to some path prefixes (``paths``).
- Anyone reads it (GET /api/condition/notice, no account: visitors are its audience), without the operator's name. Only
  an operator or an admin posts or clears it (POST /api/condition/notice), with the same checks as the card switch.
- site-nav.js draws it on every page with ``mountNotice``. That function is exported, so a page can block one element of
  its own the same way. Research case 9d893547745849e699c7ec3dd6f0ff9a; ticket 5f4290355eb6425a8035812b4cc78fe0.
"""
from __future__ import annotations

import json
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
NOTICE = "notice.json"                               # [NOTICE]
NOTICE_MODES = ("push", "overlay")
NOTICE_TEXT = {"title": 200, "body": 8000, "eta": 300}
NOTICE_PATHS = 20


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


def read_notice(folder) -> dict | None:
    """[NOTICE] The posted notice, or None."""
    try:
        notice = json.loads((Path(folder) / NOTICE).read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return None
    return notice if isinstance(notice, dict) else None


def public_notice(notice) -> dict | None:
    """What a visitor reads: everything but who posted it."""
    if not notice:
        return None
    return {k: notice[k] for k in ("title", "body", "eta", "mode", "block", "dismissible", "paths", "version", "posted_at")
            if k in notice}


def post_notice(folder, fields, *, by: str) -> dict:
    """[NOTICE] Validate and post the notice; returns it. A blocking notice is never dismissible."""
    if not isinstance(fields, dict):
        raise ConditionError("the notice is a JSON object")
    notice = {}
    for key, limit in NOTICE_TEXT.items():
        value = fields.get(key, "")
        if not isinstance(value, str) or len(value) > limit:
            raise ConditionError(f"{key} is text of at most {limit} characters")
        notice[key] = value.replace("\r\n", "\n").strip()
    if not notice["title"] and not notice["body"]:
        raise ConditionError("a notice needs a title or a body")
    mode = fields.get("mode", "push")
    if mode not in NOTICE_MODES:
        raise ConditionError("mode is push or overlay")
    block = fields.get("block", False)
    dismissible = fields.get("dismissible", True)
    if not isinstance(block, bool) or not isinstance(dismissible, bool):
        raise ConditionError("block and dismissible are true or false")
    if block and mode != "overlay":
        raise ConditionError("only an overlay blocks the page under it")
    paths = fields.get("paths", [])
    if (not isinstance(paths, list) or len(paths) > NOTICE_PATHS
            or any(not isinstance(p, str) or not p.startswith("/") or len(p) > 200 for p in paths)):
        raise ConditionError(f"paths is a list of at most {NOTICE_PATHS} path prefixes, each starting with /")
    previous = read_notice(folder) or {}
    now = time.time()
    notice.update(mode=mode, block=block, dismissible=dismissible and not block, paths=paths,
                  version=int(previous.get("version", 0)) + 1, posted_at=round(now, 3),
                  posted=f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(now))} by {by}")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    temp = folder / (NOTICE + ".tmp")
    temp.write_text(json.dumps(notice, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, folder / NOTICE)
    return notice


def clear_notice(folder) -> None:
    (Path(folder) / NOTICE).unlink(missing_ok=True)


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
        GET  /api/condition/notice          [NOTICE] anyone, no account: {notice: <public notice> | null}
        POST /api/condition/notice {title?, body?, eta?, mode?, block?, dismissible?, paths?} | {clear: true}
    """
    self._set_cookie = []
    if method == "GET" and path == "/api/condition/notice":      # [NOTICE] public: the visitors are the audience
        return self._json({"notice": public_notice(read_notice(folder)) if folder is not None else None})
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
    if path not in ("/api/condition", "/api/condition/notice"):
        return self._json({"error": "not found"}, 404)
    if method == "GET":
        return self._json({**read(folder), "notice": read_notice(folder)})
    expected_origin = ("https://" if self._is_secure() else "http://") + self.headers.get("Host", "")
    if self.headers.get("Sec-Fetch-Site") == "cross-site" or self.headers.get("Origin") not in (None, expected_origin):
        return self._json({"error": "same-origin request required"}, 403)
    if not (self.headers.get("Content-Type") or "").lower().startswith("application/json"):
        return self._json({"error": "JSON only"}, 415)
    body = self._body_json(2048 if path == "/api/condition" else 16384)
    if body is None:
        return None
    if path == "/api/condition/notice":                  # [NOTICE] an operator posts or clears the header notice
        try:
            if body.get("clear") is True:
                clear_notice(folder)
                notice = None
            else:
                notice = post_notice(folder, body, by=principal.username)
        except ConditionError as error:
            return self._json({"error": error.message}, error.status)
        print(f"[condition] {principal.username}: notice " + (f"v{notice['version']} ({notice['mode']}"
              + (", blocking" if notice["block"] else "") + ")" if notice else "cleared"), flush=True)
        return self._json({"notice": notice})
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
